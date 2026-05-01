"""
KernelAgent — multi-turn, tool-using agent for KernelBench, built on the
OpenAI Responses API.

Architecture
------------
- One agent instance handles one (problem, run) pair.
- Each call to run() returns a KernelTrajectory (full history + final result).
- Conversation state is stateless: the full `input` array is resent every turn.
  Reasoning items returned by the model flow back into `input` unchanged so
  the model sees its own chain-of-thought across turns. (The Responses API
  errors if reasoning items are dropped between turns with reasoning models.)
- Tools are declared natively via JSON schema. No XML parsing; the API
  returns structured `function_call` items and we reply with
  `function_call_output` items matched by `call_id`.
- The system prompt is passed as the top-level `instructions` parameter,
  separate from `input`.

Loop, in words:
    instructions := build_system_prompt(...)
    input := [problem_message]
    for turn in range(max_turns):
        response = client.responses.create(instructions, input, tools, ...)
        input += response.output            (preserves reasoning + tool calls)
        for fc in function_calls(response):
            result = execute_tool(fc)
            input.append(function_call_output(fc.call_id, result.output))
            if fc.name == 'submit_kernel' and result is a real KernelExecResult:
                return trajectory
        if no function_calls: break
    return trajectory
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

import torch
from openai import OpenAI

from kernelbench.eval import KernelExecResult
from kernelbench.agent.tools import Tool, ToolContext, ToolResult, get_tools
from kernelbench.agent.trajectory import KernelTurn, KernelTrajectory, ToolCall
from kernelbench.agent.prompt_templates import (
    build_system_prompt,
    build_problem_message,
    build_turn_warning_message,
)


def _strip_status(obj: Any) -> None:
    """Recursively pop `status` keys from a dumped Responses-API item.

    Azure's /openai/v1/ preview 400s if `status` is echoed back inside the
    `input` array; the public OpenAI API ignores it. Mutates in place.
    """
    if isinstance(obj, dict):
        obj.pop("status", None)
        for v in obj.values():
            _strip_status(v)
    elif isinstance(obj, list):
        for v in obj:
            _strip_status(v)


class KernelAgent:
    """
    Multi-turn, tool-using agent for a single KernelBench problem.

    The caller is responsible for constructing the OpenAI client. This keeps
    the agent agnostic to Azure/OpenAI-direct/compatible-gateway wiring — the
    caller passes in a configured `OpenAI(...)` instance with whatever
    `api_key` and `base_url` they need.

    Args:
        problem_id:         Integer problem ID.
        level:              Dataset level (1, 2, or 3).
        problem_name:       Human-readable problem name.
        ref_arch_src:       Reference PyTorch model source code.
        client:             An openai.OpenAI instance, pre-configured with
                            api_key and (optionally) base_url.
        model:              Model name to pass to client.responses.create.
        run_name:           Name for this run (used in trajectory metadata).
        tool_names:         Names of tools to enable. None = default set
                            (all except profile_kernel).
        max_turns:          Max number of LLM turns.
        max_tool_calls:     Max total tool calls across all turns.
        backend:            Kernel backend. Default "cuda".
        precision:          Computation precision. Default "fp32".
        device:             torch.device for evaluation.
        build_dir:          CUDA compile cache directory.
        num_correct_trials: Correctness trials for run_correctness / submit.
        num_perf_trials:    Timing trials for submit_kernel.
        timing_method:      Timing method for submit_kernel.
        reasoning_effort:   "minimal" | "low" | "medium" | "high" | None.
                            If set, passed to the API as reasoning.effort.
        warn_turns_remaining: Inject warning when this many turns remain.
        turn_delay_s:       Sleep between turns (for rate-limited APIs).
        verbose:            Verbose logging.
    """

    def __init__(
        self,
        *,
        problem_id: int,
        level: int,
        problem_name: str,
        ref_arch_src: str,
        client: OpenAI,
        model: str,
        run_name: str = "default",
        tool_names: list[str] | None = None,
        max_turns: int = 10,
        max_tool_calls: int = 30,
        backend: str = "cuda",
        precision: str = "fp32",
        device: torch.device | None = None,
        build_dir: str | None = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timing_method: str = "cuda_event",
        reasoning_effort: str | None = None,
        warn_turns_remaining: int = 2,
        turn_delay_s: float = 0.0,
        verbose: bool = False,
        api_kind: str = "openai",
        save_path: str | None = None,
    ) -> None:
        self.problem_id = problem_id
        self.level = level
        self.problem_name = problem_name
        self.ref_arch_src = ref_arch_src
        self.client = client
        self.model = model
        self.run_name = run_name
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.backend = backend
        self.precision = precision
        self.reasoning_effort = reasoning_effort
        self.warn_turns_remaining = warn_turns_remaining
        self.turn_delay_s = turn_delay_s
        self.verbose = verbose
        self.api_kind = api_kind
        self.save_path = save_path

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Resolve tools: `tool_names` filters the registry; get_tools() ensures
        # submit_kernel is always included.
        self.tools: list[Tool] = get_tools(tool_names)
        self.tool_names_enabled: list[str] = [t.name for t in self.tools]
        self.tool_map: dict[str, Tool] = {t.name: t for t in self.tools}

        # ToolContext is constructed once and reused for every tool call this run.
        self.ctx = ToolContext(
            ref_arch_src=ref_arch_src,
            backend=backend,
            precision=precision,
            device=device,
            build_dir=build_dir,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            timing_method=timing_method,
            verbose=verbose,
        )

        # Per-run mutable state.
        self._total_tool_calls: int = 0
        self._final_result: KernelExecResult | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> KernelTrajectory:
        """Execute the agent loop and return a completed KernelTrajectory."""
        if self.api_kind == "openai_chat":
            return self._run_chat_completions()
        return self._run_responses()

    def _tag(self) -> str:
        return f"[L{self.level}/P{self.problem_id}/{self.model}]"

    def _install_autosave(self, trajectory: KernelTrajectory) -> None:
        """Wrap trajectory.add_turn / finish so they flush to save_path after
        every mutation. Mid-run snapshots have outcome='in_progress'; the wrapped
        finish() overwrites that with the final outcome.
        """
        if not self.save_path:
            return
        path = self.save_path
        orig_add = trajectory.add_turn
        orig_finish = trajectory.finish

        def _save():
            try:
                trajectory.save(path)
            except Exception as e:
                print(f"[Agent] mid-run save failed: {e}", flush=True)

        def add_turn(turn):
            orig_add(turn)
            if trajectory.finished_at is None:
                trajectory.outcome = "in_progress"
            _save()

        def finish(result):
            orig_finish(result)
            _save()

        trajectory.add_turn = add_turn  # type: ignore[assignment]
        trajectory.finish = finish      # type: ignore[assignment]

    def _ingest_endpoint_progress(self) -> list[dict[str, Any]]:
        """Read new entries from {trajectory_basename}_aeprogress.jsonl.

        Endpoint adapters (currently aeproxy) write one JSON line per
        intermediate program they evaluate while a chat completion is in
        flight. We pick those up at the end of each turn, convert them into
        synthetic function_call items in the trajectory's response list, and
        advance a cursor so we don't re-ingest them next turn.

        Returns the list of synthesized response items (may be empty).
        """
        if not self.save_path:
            return []
        progress_path = self.save_path.replace(
            "_trajectory.json", "_aeprogress.jsonl"
        )
        try:
            if not os.path.isfile(progress_path):
                return []
            with open(progress_path) as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[Agent] could not read {progress_path}: {e}", flush=True)
            return []

        new_lines = lines[self._endpoint_progress_cursor:]
        self._endpoint_progress_cursor = len(lines)

        items: list[dict[str, Any]] = []
        for raw in new_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                e = json.loads(raw)
            except Exception:
                continue
            kernel = e.get("kernel_src") or ""
            outcome = (
                "correct" if e.get("correct")
                else "compiled" if e.get("compiled")
                else "fail"
            )
            speedup = float(e.get("speedup") or 0.0)
            elapsed = float(e.get("elapsed_s") or 0.0)
            args_payload = {
                "kernel_code": kernel,
                "_outcome": outcome,
                "_speedup": speedup,
                "_elapsed_s": elapsed,
                "_program": e.get("program_name", ""),
            }
            if e.get("error"):
                args_payload["_error"] = e["error"]
            items.append({
                "type": "function_call",
                "name": "evaluate_ae_candidate",
                "arguments": json.dumps(args_payload),
                "call_id": f"ae_call_{e.get('idx','?')}",
            })
        return items

    def _run_responses(self) -> KernelTrajectory:
        """Original Responses-API loop."""
        tag = self._tag()
        # NOTE: install_autosave is called below, after construction.
        trajectory = KernelTrajectory(
            problem_id=self.problem_id,
            level=self.level,
            problem_name=self.problem_name,
            run_name=self.run_name,
            model_name=self.model,
            backend=self.backend,
            precision=self.precision,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            tools_enabled=self.tool_names_enabled,
        )
        self._install_autosave(trajectory)
        if self.save_path:
            trajectory.outcome = "in_progress"
            try:
                trajectory.save(self.save_path)
            except Exception as e:
                print(f"[Agent] initial save failed: {e}", flush=True)
        self._endpoint_progress_cursor = 0

        instructions = build_system_prompt(
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            backend=self.backend,
            tool_names=self.tool_names_enabled,
        )
        problem_msg = build_problem_message(
            ref_arch_src=self.ref_arch_src,
            backend=self.backend,
            precision=self.precision,
        )

        # The input array we resend every turn. It grows monotonically as:
        #   - the model emits reasoning / function_call items (we append all
        #     of response.output)
        #   - we append function_call_output items after each tool executes
        #   - we append our own role=user messages (problem, warnings)
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": problem_msg},
        ]

        tool_schemas = [t.to_responses_schema() for t in self.tools]

        for turn_idx in range(self.max_turns):
            # Pacing for rate-limited APIs.
            if turn_idx > 0 and self.turn_delay_s > 0:
                time.sleep(self.turn_delay_s)
            print(
                f"{tag} turn {turn_idx + 1}/{self.max_turns} → LLM call...",
                flush=True,
            )

            turns_remaining = self.max_turns - turn_idx
            tool_calls_remaining = self.max_tool_calls - self._total_tool_calls

            if turns_remaining <= self.warn_turns_remaining and turn_idx > 0:
                has_profiling = bool(
                    {"profile_kernel", "disassemble_kernel", "ert_roofline"}
                    & set(self.tool_names_enabled)
                )
                input_items.append(
                    {
                        "role": "user",
                        "content": build_turn_warning_message(
                            turns_remaining,
                            tool_calls_remaining,
                            has_profiling_tools=has_profiling,
                        ),
                    }
                )

            # Snapshot the input we're about to send so the trajectory can
            # replay this exact turn. Deep-copy via json round-trip so later
            # mutations of input_items don't leak in.
            messages_in_snapshot = json.loads(json.dumps(input_items))

            # --- LLM call ---
            t0 = time.time()
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "instructions": instructions,
                    "input": input_items,
                    "tools": tool_schemas,
                }
                if self.reasoning_effort is not None:
                    create_kwargs["reasoning"] = {
                        "effort": self.reasoning_effort,
                        "summary": "auto",
                    }
                else:
                    create_kwargs["reasoning"] = {"summary": "auto"}

                response = self.client.responses.create(**create_kwargs)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(
                    f"[Agent] LLM call FAILED on turn {turn_idx} "
                    f"(problem {self.problem_id}, level {self.level}):\n"
                    f"{err_msg}\n{tb}"
                )
                failed_turn = KernelTurn(
                    turn_id=turn_idx,
                    messages_in=messages_in_snapshot,
                    response=[],
                    feedback_to_model=f"LLM call failed: {err_msg}",
                    llm_latency_s=time.time() - t0,
                    is_final=False,
                )
                trajectory.add_turn(failed_turn)
                trajectory.finish(self._final_result)
                return trajectory
            llm_latency = time.time() - t0

            # Always-on per-turn line so users can see progress in long runs.
            n_fc = sum(
                1 for it in (item.model_dump() for item in response.output)
                if it.get("type") == "function_call"
            )
            print(
                f"{tag} turn {turn_idx + 1} LLM done in {llm_latency:.1f}s "
                f"({n_fc} tool call{'s' if n_fc != 1 else ''})",
                flush=True,
            )

            # Serialize the model's output items so we can both (a) resend
            # them to the API next turn and (b) store them in the trajectory.
            # model_dump() gives us plain dicts, which is what both consumers
            # want.
            #
            # Azure's /openai/v1/ preview rejects the `status` field that the
            # SDK echoes back on reasoning/function_call items
            # (`Unknown parameter: 'input[N].status'`). The public OpenAI API
            # tolerates it. Strip it on the way back in to keep both happy.
            response_items: list[dict[str, Any]] = []
            for item in response.output:
                d = item.model_dump()
                _strip_status(d)
                response_items.append(d)
            input_items.extend(response_items)

            if self.verbose:
                n_fc = sum(
                    1 for it in response_items if it.get("type") == "function_call"
                )

                print(
                    f"\n[Agent] Turn {turn_idx} "
                    f"({llm_latency:.1f}s, {len(response_items)} output items, "
                    f"{n_fc} function calls)"
                )

                for it in response_items:
                    if it.get("type") == "function_call":
                        fc = it.get("function_call", {})
                        print(f"[fn] {fc.get('name')}({fc.get('arguments')})")

            # --- Execute tool calls ---
            function_calls = [
                it for it in response_items if it.get("type") == "function_call"
            ]

            executed_tool_calls: list[ToolCall] = []
            is_final = False
            submitted_kernel: str | None = None

            if not function_calls:
                # Model produced no tool calls. It either finished (unusual
                # without submit_kernel) or is stalling. Record the turn and
                # end the loop — there's nothing to act on.
                turn = KernelTurn(
                    turn_id=turn_idx,
                    messages_in=messages_in_snapshot,
                    response=response_items,
                    tool_calls=[],
                    feedback_to_model="",
                    llm_latency_s=llm_latency,
                    is_final=False,
                )
                trajectory.add_turn(turn)
                if self.verbose:
                    print("[Agent] No tool calls in response — ending loop.")
                break

            for fc in function_calls:
                # Enforce the total tool-call budget. We still need to reply
                # with a function_call_output for every function_call so the
                # API doesn't error on the next turn — so synthesize a
                # budget-exceeded output for any call beyond the cap.
                if self._total_tool_calls >= self.max_tool_calls:
                    output_text = (
                        "Tool call limit reached. No further tool calls will "
                        "be executed this run."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=fc.get("name", "?"),
                            args={},
                            result_text=output_text,
                            success=False,
                            metadata={"skipped": "tool_call_limit_reached"},
                        )
                    )
                    continue

                tool_name = fc.get("name", "")
                raw_args = fc.get("arguments", "{}")
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else (raw_args or {})
                    )
                except json.JSONDecodeError as e:
                    output_text = (
                        f"Tool call arguments could not be parsed as JSON: {e}. "
                        "Please re-issue the call with valid JSON arguments."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args={"_raw": raw_args},
                            result_text=output_text,
                            success=False,
                            metadata={"error": "invalid_json_arguments"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                if tool_name not in self.tool_map:
                    output_text = (
                        f"Unknown tool '{tool_name}'. Available tools: "
                        f"{', '.join(self.tool_names_enabled)}."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args=args,
                            result_text=output_text,
                            success=False,
                            metadata={"error": "unknown_tool"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                tool = self.tool_map[tool_name]
                _t_tool = time.time()
                tool_result = self._execute_tool(tool, args)
                _tool_dt = time.time() - _t_tool
                print(
                    f"{tag}   {tool_name} {'OK' if tool_result.success else 'FAIL'} "
                    f"({_tool_dt:.1f}s)",
                    flush=True,
                )
                self._total_tool_calls += 1

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc["call_id"],
                        "output": tool_result.output,
                    }
                )

                # Save full args (including kernel source) into the trajectory
                # so reviewers can see exactly what the model produced.
                executed_tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=dict(args),
                        result_text=tool_result.output,
                        success=tool_result.success,
                        metadata=tool_result.metadata,
                    )
                )

                # submit_kernel finalizes the run, but only if it returned a
                # real KernelExecResult. Transient errors (e.g. lock file
                # contention) leave `compiled`/`correctness` absent from
                # metadata — in those cases let the model retry.
                if tool_name == "submit_kernel":
                    submitted_kernel = args.get("kernel_code")
                    meta = tool_result.metadata or {}
                    if "compiled" in meta and "correctness" in meta:
                        is_final = True
                        try:
                            self._final_result = KernelExecResult(**meta)
                        except Exception:
                            self._final_result = None
                        break

            turn = KernelTurn(
                turn_id=turn_idx,
                messages_in=messages_in_snapshot,
                response=response_items,
                tool_calls=executed_tool_calls,
                feedback_to_model="",
                llm_latency_s=llm_latency,
                is_final=is_final,
                submitted_kernel=submitted_kernel,
            )
            trajectory.add_turn(turn)

            if is_final:
                if self.verbose:
                    print(f"[Agent] Final submission on turn {turn_idx}.")
                break

        trajectory.finish(self._final_result)
        return trajectory

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _execute_tool(self, tool: Tool, args: dict[str, Any]) -> ToolResult:
        """Execute a single tool, catching unexpected exceptions."""
        try:
            return tool.execute(self.ctx, **args)
        except Exception as e:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                output=(
                    f"{tool.name} FAILED: unexpected error during tool execution.\n"
                    f"{type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                ),
                metadata={"unexpected_error": str(e)},
            )

    # -----------------------------------------------------------------------
    # Chat Completions code path
    # -----------------------------------------------------------------------

    def _run_chat_completions(self) -> KernelTrajectory:
        """Agent loop that uses the OpenAI Chat Completions API.

        Used for models behind endpoints that don't speak the Responses API
        (e.g. DeepSeek-R1, Llama-Maverick, Kimi via Azure AI Inference).

        Differences vs. _run_responses():
          - State is a list of messages: {role, content, ...}.
          - Tool calls come back on the assistant message as `tool_calls=[...]`,
            and we reply with role="tool" messages keyed by tool_call_id.
          - Tool schemas use the nested {"type":"function","function":{...}}
            shape rather than the flat Responses shape.
          - No `reasoning` items; if a model returns reasoning_content (e.g.
            DeepSeek-R1) we capture it for the trajectory but don't resend it.
        """
        tag = self._tag()
        trajectory = KernelTrajectory(
            problem_id=self.problem_id,
            level=self.level,
            problem_name=self.problem_name,
            run_name=self.run_name,
            model_name=self.model,
            backend=self.backend,
            precision=self.precision,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            tools_enabled=self.tool_names_enabled,
        )
        self._install_autosave(trajectory)
        if self.save_path:
            trajectory.outcome = "in_progress"
            try:
                trajectory.save(self.save_path)
            except Exception as e:
                print(f"[Agent] initial save failed: {e}", flush=True)
        self._endpoint_progress_cursor = 0

        instructions = build_system_prompt(
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            backend=self.backend,
            tool_names=self.tool_names_enabled,
        )
        problem_msg = build_problem_message(
            ref_arch_src=self.ref_arch_src,
            backend=self.backend,
            precision=self.precision,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": problem_msg},
        ]

        # Convert each tool's flat Responses schema into the nested Chat shape.
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools
        ]

        for turn_idx in range(self.max_turns):
            if turn_idx > 0 and self.turn_delay_s > 0:
                time.sleep(self.turn_delay_s)
            print(
                f"{tag} turn {turn_idx + 1}/{self.max_turns} → LLM call...",
                flush=True,
            )

            turns_remaining = self.max_turns - turn_idx
            tool_calls_remaining = self.max_tool_calls - self._total_tool_calls
            if turns_remaining <= self.warn_turns_remaining and turn_idx > 0:
                has_profiling = bool(
                    {"profile_kernel", "disassemble_kernel", "ert_roofline"}
                    & set(self.tool_names_enabled)
                )
                messages.append(
                    {
                        "role": "user",
                        "content": build_turn_warning_message(
                            turns_remaining,
                            tool_calls_remaining,
                            has_profiling_tools=has_profiling,
                        ),
                    }
                )

            messages_in_snapshot = json.loads(json.dumps(messages))

            t0 = time.time()
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tool_schemas,
                    "tool_choice": "auto",
                }
                # Annotate every chat-completion call with run metadata so
                # endpoint-side adapters (e.g. aeproxy) can write progress
                # files into the trajectory dir and the website can render
                # per-candidate detail. Servers that don't recognize the
                # field ignore it.
                if self.save_path:
                    create_kwargs["extra_body"] = {
                        "popcornbench_run_meta": {
                            "run_name": self.run_name,
                            "level": self.level,
                            "problem_id": self.problem_id,
                            "model": self.model,
                            "turn_id": turn_idx,
                            "trajectory_dir": os.path.dirname(self.save_path),
                            "trajectory_basename": os.path.splitext(
                                os.path.basename(self.save_path)
                            )[0],
                        }
                    }
                response = self.client.chat.completions.create(**create_kwargs)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(
                    f"[Agent/chat] LLM call FAILED on turn {turn_idx} "
                    f"(problem {self.problem_id}, level {self.level}):\n"
                    f"{err_msg}\n{tb}"
                )
                trajectory.add_turn(
                    KernelTurn(
                        turn_id=turn_idx,
                        messages_in=messages_in_snapshot,
                        response=[],
                        feedback_to_model=f"LLM call failed: {err_msg}",
                        llm_latency_s=time.time() - t0,
                        is_final=False,
                    )
                )
                trajectory.finish(self._final_result)
                return trajectory
            llm_latency = time.time() - t0
            _n_fc_chat = len(response.choices[0].message.tool_calls or [])
            print(
                f"{tag} turn {turn_idx + 1} LLM done in {llm_latency:.1f}s "
                f"({_n_fc_chat} tool call{'s' if _n_fc_chat != 1 else ''})",
                flush=True,
            )

            choice = response.choices[0]
            asst = choice.message
            asst_content = asst.content or ""
            reasoning = getattr(asst, "reasoning_content", None) or ""
            raw_tool_calls = list(asst.tool_calls or [])

            # DeepSeek-R1 (and similar) emit reasoning inline as <think>...</think>
            # in the content. Split those out into the reasoning slot and strip
            # them from the visible content so they're not resent next turn.
            if "<think>" in asst_content:
                import re as _re
                think_blocks = _re.findall(r"<think>(.*?)</think>", asst_content, _re.S)
                if think_blocks:
                    reasoning = (reasoning + "\n" + "\n".join(think_blocks)).strip()
                    asst_content = _re.sub(
                        r"<think>.*?</think>", "", asst_content, flags=_re.S
                    ).strip()

            # Build the assistant message that goes back into `messages` for
            # the next turn. Chat Completions requires tool_calls (and ids) to
            # be echoed back in the assistant message; otherwise the role=tool
            # replies have nothing to anchor to.
            asst_msg: dict[str, Any] = {
                "role": "assistant",
                "content": asst_content,
            }
            if raw_tool_calls:
                asst_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in raw_tool_calls
                ]
            messages.append(asst_msg)

            # Build a Responses-shaped record for the trajectory (so the HTML
            # report doesn't have to know about both API shapes).
            response_items: list[dict[str, Any]] = []
            if reasoning:
                response_items.append(
                    {"type": "reasoning", "summary": [{"text": reasoning}]}
                )
            # Endpoint adapters (notably aeproxy) write per-step progress to
            # {trajectory_basename}_aeprogress.jsonl while the chat completion
            # is in flight. Ingest those entries here so the saved trajectory
            # records every intermediate program AE produced — not just the
            # final tool call.
            response_items.extend(self._ingest_endpoint_progress())
            if asst_content:
                response_items.append(
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": asst_content}],
                    }
                )
            for tc in raw_tool_calls:
                response_items.append(
                    {
                        "type": "function_call",
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                        "call_id": tc.id,
                    }
                )

            executed_tool_calls: list[ToolCall] = []
            is_final = False
            submitted_kernel: str | None = None

            if not raw_tool_calls:
                trajectory.add_turn(
                    KernelTurn(
                        turn_id=turn_idx,
                        messages_in=messages_in_snapshot,
                        response=response_items,
                        tool_calls=[],
                        feedback_to_model="",
                        llm_latency_s=llm_latency,
                        is_final=False,
                    )
                )
                if self.verbose:
                    print("[Agent/chat] No tool calls — ending loop.")
                break

            for tc in raw_tool_calls:
                tool_name = tc.function.name or ""
                raw_args = tc.function.arguments or "{}"

                if self._total_tool_calls >= self.max_tool_calls:
                    output_text = (
                        "Tool call limit reached. No further tool calls will "
                        "be executed this run."
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": output_text}
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args={},
                            result_text=output_text,
                            success=False,
                            metadata={"skipped": "tool_call_limit_reached"},
                        )
                    )
                    continue

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except json.JSONDecodeError as e:
                    output_text = (
                        f"Tool call arguments could not be parsed as JSON: {e}. "
                        "Please re-issue the call with valid JSON arguments."
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": output_text}
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args={"_raw": raw_args},
                            result_text=output_text,
                            success=False,
                            metadata={"error": "invalid_json_arguments"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                if tool_name not in self.tool_map:
                    output_text = (
                        f"Unknown tool '{tool_name}'. Available tools: "
                        f"{', '.join(self.tool_names_enabled)}."
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": output_text}
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args=args,
                            result_text=output_text,
                            success=False,
                            metadata={"error": "unknown_tool"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                tool = self.tool_map[tool_name]
                _t_tool = time.time()
                tool_result = self._execute_tool(tool, args)
                _tool_dt = time.time() - _t_tool
                print(
                    f"{tag}   {tool_name} {'OK' if tool_result.success else 'FAIL'} "
                    f"({_tool_dt:.1f}s)",
                    flush=True,
                )
                self._total_tool_calls += 1

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result.output,
                    }
                )

                executed_tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=dict(args),
                        result_text=tool_result.output,
                        success=tool_result.success,
                        metadata=tool_result.metadata,
                    )
                )

                if tool_name == "submit_kernel":
                    submitted_kernel = args.get("kernel_code")
                    meta = tool_result.metadata or {}
                    if "compiled" in meta and "correctness" in meta:
                        is_final = True
                        try:
                            self._final_result = KernelExecResult(**meta)
                        except Exception:
                            self._final_result = None
                        break

            trajectory.add_turn(
                KernelTurn(
                    turn_id=turn_idx,
                    messages_in=messages_in_snapshot,
                    response=response_items,
                    tool_calls=executed_tool_calls,
                    feedback_to_model="",
                    llm_latency_s=llm_latency,
                    is_final=is_final,
                    submitted_kernel=submitted_kernel,
                )
            )

            if is_final:
                if self.verbose:
                    print(f"[Agent/chat] Final submission on turn {turn_idx}.")
                break

        trajectory.finish(self._final_result)
        return trajectory
