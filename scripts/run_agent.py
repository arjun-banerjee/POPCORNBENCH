"""
run_agent.py — Single-problem multi-turn agent entry point.

Mirrors generate_and_eval_single_sample.py but runs the KernelAgent loop
instead of a single-shot LLM call.

Example usage:
  uv run python scripts/run_agent.py \
    dataset_src=local level=1 problem_id=1 \
    server_type=anthropic model_name=anthropic/claude-sonnet-4-6 \
    backend=cuda precision=fp32 \
    max_turns=10 max_tool_calls=30 \
    tools=compile_kernel,run_correctness,static_check,submit_kernel \
    run_name=my_agent_run

To enable nsight profiling (requires ncu + permissions):
    tools=compile_kernel,run_correctness,profile_kernel,get_gpu_specs,static_check,submit_kernel

To run with all default tools (no profiling):
    tools=default

Available tools:
    compile_kernel   — compile without running
    run_correctness  — correctness trials only (no timing)
    profile_kernel   — nsight roofline profiling (opt-in, requires ncu)
    get_gpu_specs    — hardware specs for the current GPU
    static_check     — static reward-hack pattern detector
    submit_kernel    — full eval: correctness + timing (always included)
"""

import os
import sys
import json
import torch
import pydra
from pydra import REQUIRED, Config

from kernelbench.utils import create_inference_server_from_presets, set_gpu_arch
from kernelbench.eval import get_torch_dtype_from_string
from kernelbench.agent import KernelAgent, get_tools

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AgentConfig(Config):
    def __init__(self):
        # Problem specification
        self.dataset_src = REQUIRED         # "local" or "huggingface"
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.problem_id = REQUIRED

        # Model / inference
        self.server_type = REQUIRED
        self.model_name = REQUIRED
        self.max_tokens = None
        self.temperature = None
        self.server_address = None
        self.server_port = None
        self.is_reasoning_model = False
        self.reasoning_effort = None
        self.budget_tokens = 0

        # Agent loop parameters
        self.max_turns = 10             # LLM response cap
        self.max_tool_calls = 30        # total tool call cap across all turns
        self.warn_turns_remaining = 2   # inject warning this many turns before cap
        # Tool call XML dialect shown to the model in the system prompt.
        # "canonical" (default) | "nemotron" | "auto"
        # Use "nemotron" for Nemotron / Llama function-calling models.
        self.tool_format = "canonical"

        # Tool selection:
        #   "default"   → all tools except profile_kernel
        #   "all"       → all tools including profile_kernel
        #   "a,b,c"     → explicit comma-separated list
        self.tools = "default"

        # Backend / hardware
        self.backend = "cuda"
        self.precision = "fp32"
        self.gpu_arch = ["Ada"]
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.timing_method = "cuda_event"
        self.num_correct_trials = 5
        self.num_perf_trials = 100

        # Run identity
        self.run_name = "agent_run"
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        # Logging
        self.verbose = False
        self.save_trajectory = True    # write trajectory JSON to runs/

    def __repr__(self):
        return f"AgentConfig({self.to_dict()})"


def _resolve_tools(tools_arg: str):
    """Parse the tools config argument into a list of tool names or None."""
    if isinstance(tools_arg, (list, tuple)):
        # pydra may have already parsed a comma-separated value into a list
        return list(tools_arg)
    tools_arg = str(tools_arg).strip().lower()
    if tools_arg == "default":
        return None          # get_tools(None) → all except profile_kernel
    if tools_arg == "all":
        from kernelbench.agent.tools import ALL_TOOLS
        return [t.name for t in ALL_TOOLS]
    return [t.strip() for t in tools_arg.split(",") if t.strip()]


@pydra.main(base=AgentConfig)
def main(config: AgentConfig):
    from kernelbench.utils import SERVER_PRESETS
    from kernelbench.dataset import construct_kernelbench_dataset

    # ---- Resolve presets ----
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", 8192)
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", 0.7)

    if isinstance(config.is_reasoning_model, str):
        config.is_reasoning_model = config.is_reasoning_model.lower() in ["true", "1", "yes"]
    if isinstance(config.include_hardware_info, str):
        config.include_hardware_info = config.include_hardware_info.lower() in ["true", "1", "yes"]
    if isinstance(config.save_trajectory, str):
        config.save_trajectory = config.save_trajectory.lower() in ["true", "1", "yes"]

    print(f"[run_agent] Config: {config}")

    # ---- GPU arch ----
    if config.gpu_arch:
        if not isinstance(config.gpu_arch, list):
            config.gpu_arch = [config.gpu_arch]
        set_gpu_arch(config.gpu_arch)

    # ---- Backend precision constraints ----
    if config.backend.lower() == "tilelang":
        config.precision = "fp16"
    if config.backend.lower() == "thunderkittens":
        config.precision = "bf16"

    # ---- Load dataset + problem ----
    dataset = construct_kernelbench_dataset(
        level=config.level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )
    problem = dataset.get_problem_by_id(config.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name
    print(f"[run_agent] Problem: level={config.level} id={config.problem_id} name={problem_name}")

    # ---- Inference function ----
    inference_fn = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True,
        is_reasoning_model=config.is_reasoning_model,
        reasoning_effort=config.reasoning_effort,
        budget_tokens=config.budget_tokens,
        server_address=config.server_address,
        server_port=config.server_port,
    )
    # Attach a name for trajectory metadata
    inference_fn.__name__ = config.model_name

    # ---- Tool selection ----
    tool_names = _resolve_tools(config.tools)
    tools = get_tools(tool_names)
    print(f"[run_agent] Tools enabled: {[t.name for t in tools]}")

    # ---- Device ----
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )

    # ---- Build cache dir (consistent with existing eval scripts) ----
    build_dir = os.path.join(
        config.runs_dir, config.run_name,
        f"level_{config.level}_problem_{config.problem_id}_cache",
    )
    os.makedirs(build_dir, exist_ok=True)

    # ---- Create and run agent ----
    agent = KernelAgent(
        problem_id=config.problem_id,
        level=config.level,
        problem_name=problem_name,
        ref_arch_src=ref_arch_src,
        inference_fn=inference_fn,
        run_name=config.run_name,
        tool_names=[t.name for t in tools],
        max_turns=config.max_turns,
        max_tool_calls=config.max_tool_calls,
        backend=config.backend,
        precision=config.precision,
        device=device,
        build_dir=build_dir,
        num_correct_trials=config.num_correct_trials,
        num_perf_trials=config.num_perf_trials,
        timing_method=config.timing_method,
        include_hardware_info=config.include_hardware_info,
        warn_turns_remaining=config.warn_turns_remaining,
        tool_format=config.tool_format,
        verbose=config.verbose,
    )

    print(f"[run_agent] Starting agent loop (max_turns={config.max_turns}, max_tool_calls={config.max_tool_calls})")
    trajectory = agent.run()

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print(f"[run_agent] Agent run complete.")
    print(f"  Outcome:         {trajectory.outcome}")
    print(f"  Total turns:     {trajectory.total_turns}")
    print(f"  Total tool calls:{trajectory.total_tool_calls}")
    if trajectory.final_result:
        r = trajectory.final_result
        print(f"  Compiled:        {r.compiled}")
        print(f"  Correct:         {r.correctness}")
        if r.runtime > 0:
            print(f"  Kernel runtime:  {r.runtime:.2f} μs")
    print("=" * 60)

    # ---- Save trajectory ----
    if config.save_trajectory:
        traj_path = os.path.join(
            config.runs_dir,
            config.run_name,
            f"level_{config.level}_problem_{config.problem_id}_trajectory.json",
        )
        trajectory.save(traj_path)
        print(f"[run_agent] Trajectory saved to: {traj_path}")


if __name__ == "__main__":
    main()
