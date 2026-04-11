"""
KernelAgent — multi-turn, tool-using agent for KernelBench.

Architecture
------------
- One agent instance handles one (problem, run) pair.
- Each call to run() returns a KernelTrajectory (full history + final result).
- The agent loop:
    for turn in range(max_turns):
        response = llm(conversation)
        parse tool calls from response
        execute tool calls (up to remaining cap)
        append tool results to conversation
        if submit_kernel was called → record result, break
- Tool calls are synchronous and executed in the order they appear in the response.
  The interface is async-ready: wrap execute() in asyncio.to_thread for async use.

Tool call parsing — multi-format
---------------------------------
Models emit tool calls in different dialects. We support three automatically:

Format A — our canonical XML (Claude, Gemini, GPT-4, etc.):
    <tool_call>
    <tool_name>compile_kernel</tool_name>
    <kernel_code>```python ... ```</kernel_code>
    </tool_call>

Format B — Nemotron / Llama function-calling XML:
    <function=compile_kernel>
    <parameter=kernel_code>```python ... ```</parameter>
    </function>

Format C — Nemotron "execute" shim (model wraps tool name in a command parameter):
    <function=execute>
    <parameter=command>compile_kernel</parameter>
    <parameter=kernel_code>```python ... ```</parameter>
    </function>

All three are tried in order on every response; duplicates are de-duped.
The system prompt shown to the model is chosen to match its preferred dialect.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from kernelbench.eval import KernelExecResult
from kernelbench.agent.tools import Tool, ToolContext, ToolResult, get_tools
from kernelbench.agent.trajectory import KernelTurn, KernelTrajectory, ToolCall
from kernelbench.agent.prompt_templates import (
    build_system_prompt,
    build_problem_message,
    build_tool_results_message,
    build_turn_warning_message,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Finds a ```...``` code block anywhere in a string (model may prepend reasoning text).
# Uses re.search so the fence doesn't have to be at the start of the string.
# Accepts any language tag (or none) so models that write ```c, ```cu, ```c++, etc.
# are handled correctly — the tag is stripped along with the fence delimiters.
_FENCE_SEARCH_RE = re.compile(
    r"```[^\n`]*\n?(.*?)```",
    re.DOTALL,
)

def _strip_fence(code: str) -> str:
    """Extract Python code from a potentially fence-wrapped string.

    Uses re.search rather than re.match so leading reasoning text (which smaller
    models sometimes inject before the code block) is correctly ignored.
    Falls back to the raw string if no fence is found.
    """
    m = _FENCE_SEARCH_RE.search(code.strip())
    return m.group(1).strip() if m else code.strip()


# Detects a plain code block (```...```) in a response that contains no tool calls.
# Must contain `class ModelNew` to be considered a real kernel attempt.
_BARE_CODE_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)

def _extract_bare_code(response: str) -> Optional[str]:
    """Return the first code block containing 'class ModelNew', or None."""
    for m in _BARE_CODE_RE.finditer(response):
        block = m.group(1).strip()
        if "class ModelNew" in block or "ModelNew" in block:
            return block
    return None


def _build_no_tool_nudge(bare_code: Optional[str], tool_format: str) -> str:
    """
    Build a targeted feedback message for when the model wrote code but didn't
    use a tool call.  If bare_code is provided, the message shows the exact
    tool-call syntax with their code already filled in so the model only needs
    to copy-paste.
    """
    if tool_format == "nemotron":
        if bare_code:
            return (
                "I can see you wrote kernel code but did not call a tool. "
                "Please wrap it exactly like this and send it again:\n\n"
                f"<function=compile_kernel>\n"
                f"<parameter=kernel_code>\n"
                f"```python\n{bare_code}\n```\n"
                f"</parameter>\n</function>"
            )
        return (
            "No tool call was detected in your response. "
            "To call a tool, use:\n"
            "<function=TOOL_NAME>\n<parameter=kernel_code>```python\n# code\n```</parameter>\n</function>"
        )
    else:  # canonical or auto
        if bare_code:
            return (
                "I can see you wrote kernel code but did not call a tool. "
                "Please wrap it exactly like this and send it again:\n\n"
                f"<tool_call>\n<tool_name>compile_kernel</tool_name>\n"
                f"<kernel_code>\n```python\n{bare_code}\n```\n</kernel_code>\n</tool_call>"
            )
        return (
            "No tool call was detected in your response. "
            "To call a tool, use:\n"
            "<tool_call>\n<tool_name>TOOL_NAME</tool_name>\n"
            "<kernel_code>```python\n# code\n```</kernel_code>\n</tool_call>"
        )


# ---------------------------------------------------------------------------
# Format A parser — canonical <tool_call> XML
# ---------------------------------------------------------------------------

_FMT_A_BLOCK_RE  = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>",  re.DOTALL | re.IGNORECASE)
_FMT_A_NAME_RE   = re.compile(r"<tool_name>\s*(.*?)\s*</tool_name>",  re.DOTALL | re.IGNORECASE)
# Accept several tag names for the code block — models use different names in practice:
#   <kernel_code>       (canonical, shown in system prompt)
#   <tool_kernel_code>  (Claude's spontaneous variant)
#   <code>              (generic fallback used by some models)
_FMT_A_CODE_RE   = re.compile(
    r"<(?:tool_)?(?:kernel_)?code>\s*(.*?)\s*</(?:tool_)?(?:kernel_)?code>",
    re.DOTALL | re.IGNORECASE,
)

def _parse_format_a(response: str) -> List[Dict[str, Any]]:
    parsed = []
    for block_match in _FMT_A_BLOCK_RE.finditer(response):
        block = block_match.group(1)
        name_match = _FMT_A_NAME_RE.search(block)
        if not name_match:
            continue
        tool_name = name_match.group(1).strip()
        args: Dict[str, Any] = {}
        code_match = _FMT_A_CODE_RE.search(block)
        if code_match:
            args["kernel_code"] = _strip_fence(code_match.group(1))
        parsed.append({"tool_name": tool_name, "args": args})
    return parsed


# ---------------------------------------------------------------------------
# Format B parser — Nemotron/Llama <function=name> XML
# ---------------------------------------------------------------------------

# Matches <function=TOOL_NAME> ... </function>
_FMT_B_BLOCK_RE = re.compile(
    r"<function=([^>]+)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
# Matches <parameter=NAME>VALUE</parameter> anywhere inside the block
_FMT_B_PARAM_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)

# Nemotron and other models may use different parameter names for the kernel code.
# We accept all of these as aliases for "kernel_code".
_CODE_PARAM_ALIASES = ("kernel_code", "code", "source", "kernel", "content", "input")

def _parse_format_b(response: str) -> List[Dict[str, Any]]:
    """
    Parse Nemotron-style <function=name><parameter=p>v</parameter></function>.
    Also handles the 'execute' shim (Format C) where the tool name lives in
    a <parameter=command> tag.

    Accepts several parameter name aliases for kernel code (kernel_code, code,
    source, kernel, content) because smaller models often ignore the exact
    parameter name shown in the system prompt.
    """
    parsed = []
    for block_match in _FMT_B_BLOCK_RE.finditer(response):
        func_name = block_match.group(1).strip()
        block     = block_match.group(2)

        # Collect all parameters from this block
        params: Dict[str, str] = {}
        for pm in _FMT_B_PARAM_RE.finditer(block):
            params[pm.group(1).strip()] = pm.group(2).strip()

        # Format C: <function=execute><parameter=command>actual_tool</parameter>...
        if func_name.lower() == "execute" and "command" in params:
            tool_name = params.pop("command").strip()
        else:
            tool_name = func_name

        args: Dict[str, Any] = {}
        # Accept any known code-parameter alias; prefer the canonical name first.
        for alias in _CODE_PARAM_ALIASES:
            if alias in params:
                args["kernel_code"] = _strip_fence(params[alias])
                break

        parsed.append({"tool_name": tool_name, "args": args})
    return parsed


# ---------------------------------------------------------------------------
# Unified parser — tries all formats, de-dupes by position
# ---------------------------------------------------------------------------

def parse_tool_calls(response: str) -> List[Dict[str, Any]]:
    """
    Parse tool calls from an LLM response, supporting multiple XML dialects.

    Returns a list of dicts:
        {"tool_name": str, "args": {"kernel_code": str | None, ...}}

    Formats recognised (tried in order, duplicates removed):
      A: <tool_call><tool_name>…</tool_name><kernel_code>…</kernel_code></tool_call>
      B: <function=name><parameter=kernel_code>…</parameter></function>
      C: <function=execute><parameter=command>name</parameter>…</function>
    """
    # Collect with source-format tag to allow de-duplication
    fmt_a = _parse_format_a(response)
    fmt_b = _parse_format_b(response)

    # Merge: prefer format A when a tool_name appears in both (format A is more
    # explicit), but keep ordering by first appearance.
    seen_names: set = set()
    merged = []
    for tc in fmt_a + fmt_b:
        key = tc["tool_name"]
        if key not in seen_names:
            seen_names.add(key)
            merged.append(tc)

    return merged


# ---------------------------------------------------------------------------
# KernelAgent
# ---------------------------------------------------------------------------

class KernelAgent:
    """
    Multi-turn, tool-using agent for a single KernelBench problem.

    Args:
        problem_id:        Integer problem ID.
        level:             Dataset level (1, 2, or 3).
        problem_name:      Human-readable problem name.
        ref_arch_src:      Reference PyTorch model source code.
        inference_fn:      Callable[List[dict]] → str — the LLM query function.
                           Takes a list of {role, content} messages and returns the response.
        run_name:          Name for this run (used for storing trajectories).
        tool_names:        Names of tools to enable. None = all except profile_kernel.
        max_turns:         Maximum number of LLM turns (responses). Default 10.
        max_tool_calls:    Maximum total tool calls across all turns. Default 50.
        backend:           Kernel backend. Default "cuda".
        precision:         Computation precision. Default "fp32".
        device:            torch.device for evaluation.
        build_dir:         CUDA compile cache directory.
        num_correct_trials: Correctness trials for run_correctness / submit_kernel.
        num_perf_trials:   Timing trials for submit_kernel.
        timing_method:     Timing method for submit_kernel. Default "cuda_event".
        include_hardware_info: Whether to prepend GPU specs to the problem message.
        warn_turns_remaining: Inject a turn-limit warning when this many turns remain.
        verbose:           Verbose logging.
    """

    def __init__(
        self,
        *,
        problem_id: int,
        level: int,
        problem_name: str,
        ref_arch_src: str,
        inference_fn: Callable[[List[Dict[str, str]]], str],
        run_name: str = "agent_run",
        tool_names: Optional[List[str]] = None,
        max_turns: int = 10,
        max_tool_calls: int = 50,
        backend: str = "cuda",
        precision: str = "fp32",
        device: Optional[torch.device] = None,
        build_dir: Optional[str] = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timing_method: str = "cuda_event",
        include_hardware_info: bool = False,
        warn_turns_remaining: int = 2,
        tool_format: str = "canonical",
        turn_delay_s: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.problem_id = problem_id
        self.level = level
        self.problem_name = problem_name
        self.ref_arch_src = ref_arch_src
        self.inference_fn = inference_fn
        self.run_name = run_name
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.backend = backend
        self.precision = precision
        self.device = device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.build_dir = build_dir
        self.warn_turns_remaining = warn_turns_remaining
        self.tool_format = tool_format
        self.turn_delay_s = turn_delay_s
        self.verbose = verbose

        # Tool setup
        self.tools: List[Tool] = get_tools(tool_names)
        self.tool_map: Dict[str, Tool] = {t.name: t for t in self.tools}
        self.tool_names_enabled = [t.name for t in self.tools]

        # Shared context passed to every tool
        self.ctx = ToolContext(
            ref_arch_src=ref_arch_src,
            backend=backend,
            precision=precision,
            device=self.device,
            build_dir=build_dir,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            timing_method=timing_method,
            verbose=verbose,
        )

        # Hardware info string (optional, prepended to problem message)
        self._hardware_info = ""
        if include_hardware_info and torch.cuda.is_available():
            self._hardware_info = torch.cuda.get_device_name(self.device)

        # State
        self._conversation: List[Dict[str, str]] = []
        self._total_tool_calls: int = 0
        self._final_result: Optional[KernelExecResult] = None
        # Last kernel code seen in any tool call — used as implicit fallback
        # when the model calls a tool without repeating the kernel_code argument
        # (e.g. calling run_correctness after compile_kernel without re-pasting code).
        self._last_kernel_code: Optional[str] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> KernelTrajectory:
        """
        Execute the full agent loop and return the trajectory.

        This is the main entry point. After run() returns, call trajectory.save()
        to persist the result.
        """
        trajectory = KernelTrajectory(
            problem_id=self.problem_id,
            level=self.level,
            problem_name=self.problem_name,
            run_name=self.run_name,
            model_name=getattr(self.inference_fn, "__name__", "unknown"),
            backend=self.backend,
            precision=self.precision,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            tools_enabled=self.tool_names_enabled,
        )

        self._init_conversation()

        for turn_idx in range(self.max_turns):
            # Honour inter-turn delay (useful for APIs with tight per-minute
            # output-token quotas, e.g. Anthropic free-tier at 4k tokens/min).
            if turn_idx > 0 and self.turn_delay_s > 0:
                time.sleep(self.turn_delay_s)

            turns_remaining = self.max_turns - turn_idx
            tool_calls_remaining = self.max_tool_calls - self._total_tool_calls

            # Inject warning when approaching limits
            if turns_remaining <= self.warn_turns_remaining and turn_idx > 0:
                warn_msg = build_turn_warning_message(turns_remaining, tool_calls_remaining)
                self._conversation.append({"role": "user", "content": warn_msg})

            # --- LLM call ---
            t0 = time.time()
            messages_snapshot = list(self._conversation)  # snapshot before adding response
            try:
                response = self.inference_fn(self._conversation)
            except Exception as e:
                import traceback
                err_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                # Always surface LLM failures — these are never expected and silence
                # makes debugging impossible.
                print(
                    f"[Agent] LLM call FAILED on turn {turn_idx} "
                    f"(problem {self.problem_id}, level {self.level}):\n{err_msg}\n{tb}"
                )
                # Record the failed turn so the trajectory captures the error state.
                failed_turn = KernelTurn(
                    turn_id=turn_idx,
                    messages_in=messages_snapshot,
                    response="",
                    feedback_to_model=f"LLM call failed: {err_msg}",
                    llm_latency_s=time.time() - t0,
                    is_final=False,
                )
                trajectory.add_turn(failed_turn)
                trajectory.finish(self._final_result)
                return trajectory
            llm_latency = time.time() - t0

            if self.verbose:
                print(f"\n[Agent] Turn {turn_idx} response ({llm_latency:.1f}s):\n{response[:200]}...")

            # Append assistant response to conversation
            self._conversation.append({"role": "assistant", "content": response})

            # --- Parse tool calls ---
            raw_tool_calls = parse_tool_calls(response)

            if not raw_tool_calls:
                # No tool calls found. Check whether the model wrote a code block
                # without wrapping it — a common failure mode on smaller models.
                bare_code = _extract_bare_code(response)
                is_last_turn = (turn_idx == self.max_turns - 1)

                if bare_code and is_last_turn:
                    # Final turn with bare code: auto-submit as best-effort so the
                    # run doesn't silently die as outcome=error.
                    if self.verbose:
                        print(f"[Agent] Last turn: auto-submitting bare code block.")
                    raw_tool_calls = [{"tool_name": "submit_kernel",
                                       "args": {"kernel_code": bare_code}}]
                else:
                    # Inject a targeted nudge that shows the exact tool-call format
                    # with the model's own code already filled in.
                    nudge = _build_no_tool_nudge(bare_code, self.tool_format)
                    self._conversation.append({"role": "user", "content": nudge})
                    turn = KernelTurn(
                        turn_id=turn_idx,
                        messages_in=messages_snapshot,
                        response=response,
                        feedback_to_model=nudge,
                        llm_latency_s=llm_latency,
                        is_final=False,
                    )
                    trajectory.add_turn(turn)
                    continue

            # --- Execute tool calls ---
            executed_tool_calls: List[ToolCall] = []
            result_texts: List[str] = []
            is_final = False
            submitted_kernel: Optional[str] = None

            for raw_tc in raw_tool_calls:
                if self._total_tool_calls >= self.max_tool_calls:
                    result_texts.append(
                        f"[System] Tool call limit ({self.max_tool_calls}) reached. "
                        "No further tool calls will be executed."
                    )
                    break

                tool_name = raw_tc["tool_name"]
                args = raw_tc["args"]

                if tool_name not in self.tool_map:
                    result_texts.append(
                        f"[Tool: {tool_name}] ERROR: Unknown tool. "
                        f"Available: {self.tool_names_enabled}"
                    )
                    continue

                # If the model called a code-requiring tool without kernel_code
                # (e.g. run_correctness after compile_kernel without re-pasting),
                # silently reuse the last kernel code seen in this session.
                if "kernel_code" not in args and self._last_kernel_code:
                    args = {**args, "kernel_code": self._last_kernel_code}

                # Track the most recent kernel code for future implicit reuse.
                if "kernel_code" in args and args["kernel_code"]:
                    self._last_kernel_code = args["kernel_code"]

                tool = self.tool_map[tool_name]
                tool_result = self._execute_tool(tool, args)
                self._total_tool_calls += 1

                executed_tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args={k: (v[:200] + "...") if isinstance(v, str) and len(v) > 200 else v
                              for k, v in args.items()},
                        result_text=tool_result.output,
                        success=tool_result.success,
                        metadata=tool_result.metadata,
                    )
                )

                header = f"[Tool: {tool_name}] {'SUCCESS' if tool_result.success else 'FAILED'}"
                result_texts.append(f"{header}\n{tool_result.output}")

                # submit_kernel → record final result and stop.
                # But if the tool rejected the submission early (e.g., CUDA C instead
                # of Python, or missing kernel_code), the metadata won't have the
                # standard KernelExecResult fields — don't finalize in that case so
                # the model can correct itself and try again.
                if tool_name == "submit_kernel":
                    submitted_kernel = args.get("kernel_code")
                    meta = tool_result.metadata or {}
                    # A real evaluation always populates 'compiled' from KernelExecResult
                    if "compiled" in meta and "correctness" in meta:
                        is_final = True
                        try:
                            self._final_result = KernelExecResult(**meta)
                        except Exception:
                            self._final_result = None
                        break
                    # else: early rejection (format error) — let the model try again

            # Append combined tool results as a user message
            combined_results = "\n\n---\n\n".join(result_texts)
            feedback_msg = build_tool_results_message(combined_results)
            self._conversation.append({"role": "user", "content": feedback_msg})

            turn = KernelTurn(
                turn_id=turn_idx,
                messages_in=messages_snapshot,
                response=response,
                tool_calls=executed_tool_calls,
                feedback_to_model=feedback_msg,
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

    def _init_conversation(self) -> None:
        """Build the initial system + user messages."""
        system_prompt = build_system_prompt(
            available_tools=self.tool_names_enabled,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            tool_format=self.tool_format,
        )
        problem_msg = build_problem_message(
            ref_arch_src=self.ref_arch_src,
            backend=self.backend,
            precision=self.precision,
            hardware_info=self._hardware_info,
        )
        self._conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_msg},
        ]

    def _execute_tool(self, tool: Tool, args: Dict[str, Any]) -> ToolResult:
        """Execute a single tool, catching unexpected exceptions."""
        try:
            return tool.execute(self.ctx, **args)
        except Exception as e:
            import traceback
            return ToolResult(
                tool_name=tool.name,
                success=False,
                output=(
                    f"Unexpected error executing tool '{tool.name}':\n"
                    f"{type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                ),
                metadata={"unexpected_error": str(e)},
            )
