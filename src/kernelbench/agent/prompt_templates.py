"""
Prompts for the KernelBench multi-turn agent.

Three public builders:
    build_system_prompt(max_turns, max_tool_calls, backend, tool_names) -> str
        The system prompt, passed as `instructions` to the Responses API.
        Stable across turns within a run.  When profiling / analysis tools
        are present in *tool_names*, an optimization-workflow section is
        appended that references only the tools actually available.

    build_problem_message(ref_arch_src, backend, precision) -> str
        The first user-role message: task statement, backend-specific output
        format, and the reference PyTorch source.

    build_turn_warning_message(turns_remaining, tool_calls_remaining) -> str
        Injected as a user-role message every turn once the warning threshold
        is crossed.

Tool descriptions live in `tools.py` as JSON-schema `description` fields and
are not duplicated here.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Backend display-name map (shared between system prompt and problem message)
# ---------------------------------------------------------------------------

_BACKEND_DISPLAY: dict[str, str] = {
    "cuda": "CUDA",
    "triton": "Triton",
    "tilelang": "TileLang",
    "cute": "CUTLASS/CuTe",
}


def _backend_display(backend: str) -> str:
    """Return a human-readable backend name. Raises for unsupported backends."""
    key = backend.lower()
    if key not in _BACKEND_DISPLAY:
        raise NotImplementedError(
            f"Backend '{backend}' is not supported by the agent prompts. "
            f"Supported backends: {sorted(_BACKEND_DISPLAY.keys())}."
        )
    return _BACKEND_DISPLAY[key]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert GPU kernel engineer. Your task is to write a high-performance \
custom {backend_display} kernel that replaces a PyTorch reference implementation, \
producing numerically identical results.

You work iteratively: call tools to compile, test, and profile your kernel, read \
the results, and refine.

## Session limits

- You have {max_turns} turns. One turn is one response from you, which may \
contain reasoning and zero or more tool calls.
- You have {max_tool_calls} tool calls total across all turns.
- Only `submit_kernel` records your final result. It ends the session.

## Correctness requirements

- Outputs must match the reference within tolerance: fp32 uses atol=rtol=1e-4; \
fp16/bf16 use atol=rtol=1e-2.
- Do not use try/except fallbacks to the reference, patch timing functions, or \
otherwise work around correctness or timing measurement — these are detected \
and cause evaluation failure.
"""


_PROFILING_TOOLS = {"profile_kernel", "disassemble_kernel", "ert_roofline"}


def _build_optimization_workflow(tool_names: set[str]) -> str:
    """Build an optimization-workflow prompt section that only mentions tools
    the agent actually has access to."""
    has_profile = "profile_kernel" in tool_names
    has_disasm = "disassemble_kernel" in tool_names
    has_ert = "ert_roofline" in tool_names
    has_gpu_specs = "get_gpu_specs" in tool_names

    # Step 1 — understand hardware
    hw_parts: list[str] = []
    if has_gpu_specs:
        hw_parts.append("`get_gpu_specs`")
    if has_ert:
        hw_parts.append("`ert_roofline`")
    if hw_parts:
        hw_step = (
            f"1. **Understand the hardware**: call {' and '.join(hw_parts)} "
            "to know your bandwidth and compute ceilings.\n"
        )
    else:
        hw_step = ""

    # Step 3 — profile after correctness
    profile_parts: list[str] = []
    if has_profile:
        profile_parts.append(
            "`profile_kernel` (roofline analysis — shows whether you are "
            "memory- or compute-bound, bandwidth/compute utilization)"
        )
    if has_disasm:
        profile_parts.append(
            "`disassemble_kernel` (SASS/PTX inspection — shows register "
            "pressure, spills, instruction mix, tensor-core usage)"
        )
    if profile_parts:
        tools_list = " and/or ".join(profile_parts)
        profile_step = (
            "3. **Profile before submitting**: once correct, call "
            f"{tools_list} to identify the bottleneck.\n"
        )
        optimize_step = (
            "4. **Optimize based on profiling data**: rewrite the kernel to "
            "address the bottleneck — better tiling, shared-memory staging, "
            "vectorized loads, operator fusion, etc. Then re-verify "
            "correctness with `run_correctness`.\n"
            "5. **Iterate steps 3–4** as budget allows. Each "
            "profile → optimize cycle should target a specific bottleneck.\n"
        )
        submit_step = (
            "6. **Submit only when you've exhausted your optimization ideas** "
            "or are running low on turns/tool calls.\n"
        )
    else:
        profile_step = ""
        optimize_step = ""
        submit_step = (
            "3. **Submit** when you are confident the kernel is correct and "
            "well-optimized.\n"
        )

    section = (
        "\n## Optimization workflow\n\n"
        "Correctness is step 1, not the finish line. Follow this loop:\n\n"
        f"{hw_step}"
        "2. **Write an initial kernel**, then `compile_kernel` → "
        "`run_correctness`.\n"
        f"{profile_step}"
        f"{optimize_step}"
        f"{submit_step}"
        "\nDo NOT call `submit_kernel` the moment you have a correct kernel "
        "— a correct but unoptimized kernel is a wasted submission."
    )
    return section


def build_system_prompt(
    *,
    max_turns: int,
    max_tool_calls: int,
    backend: str,
    tool_names: list[str] | None = None,
) -> str:
    """Build the agent's system prompt (the API's `instructions` parameter).

    When *tool_names* includes any profiling/analysis tools, an optimization-
    workflow section is appended that references only the tools actually
    available.
    """
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        backend_display=_backend_display(backend),
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
    )
    if tool_names is not None:
        names = set(tool_names)
        if names & _PROFILING_TOOLS:
            prompt += _build_optimization_workflow(names)
    return prompt


# ---------------------------------------------------------------------------
# Per-backend output-format blocks
# ---------------------------------------------------------------------------


def _output_format_cuda() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. The file is executed with `exec()` before we instantiate \
`ModelNew`, so any module-level setup runs first.

Use `torch.utils.cpp_extension.load_inline` to compile and bind CUDA source at \
module load time, then call the compiled extension from `ModelNew.forward`. Do \
not submit raw CUDA C or a standalone .cu file."""


def _output_format_triton() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Write the kernel as a function decorated with `@triton.jit` (or \
`@triton.autotune`) using `triton.language` (commonly aliased as `tl`). Launch \
the kernel from `ModelNew.forward` with an appropriate grid."""


def _output_format_tilelang() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Write the kernel as a `@T.prim_func` using `tilelang.language` \
(aliased as `T`), compile it with `tilelang.compile(..., target="cuda")`, and \
invoke the compiled kernel from `ModelNew.forward`. Note: TileLang requires \
fp16 or bf16 precision."""


def _output_format_cute() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Use the CUTLASS/CuTe Python bindings (`cutlass`, and the `cute::` \
namespace in any inlined C++) to build the kernel, and invoke it from \
`ModelNew.forward`."""


_OUTPUT_FORMAT_BUILDERS = {
    "cuda": _output_format_cuda,
    "triton": _output_format_triton,
    "tilelang": _output_format_tilelang,
    "cute": _output_format_cute,
}


def _output_format(backend: str) -> str:
    """Return the output-format block for the given backend, or raise."""
    key = backend.lower()
    builder = _OUTPUT_FORMAT_BUILDERS.get(key)
    if builder is None:
        raise NotImplementedError(
            f"No output-format prompt for backend '{backend}'. "
            f"Supported backends: {sorted(_OUTPUT_FORMAT_BUILDERS.keys())}."
        )
    return builder()


# ---------------------------------------------------------------------------
# Problem message (first user turn)
# ---------------------------------------------------------------------------

_PROBLEM_TEMPLATE = """\
## Task

Optimize the following PyTorch model by replacing its forward computation with \
a custom {backend_display} kernel. Your `ModelNew` class must:

1. Accept the same constructor arguments as `Model`.
2. Implement `forward()` with the same signature.
3. Produce numerically equivalent outputs at {precision} precision.
4. Be faster than the PyTorch reference.

{output_format}

## Reference implementation

```python
{ref_arch_src}
```
"""


def build_problem_message(
    *,
    ref_arch_src: str,
    backend: str,
    precision: str,
) -> str:
    """Build the first user-role message describing the problem."""
    return _PROBLEM_TEMPLATE.format(
        backend_display=_backend_display(backend),
        precision=precision,
        output_format=_output_format(backend),
        ref_arch_src=ref_arch_src.rstrip(),
    )


# ---------------------------------------------------------------------------
# Turn-limit warning
# ---------------------------------------------------------------------------


def build_turn_warning_message(
    turns_remaining: int,
    tool_calls_remaining: int,
    has_profiling_tools: bool = False,
) -> str:
    """Build the soft warning injected as a user message near the session cap."""
    turn_word = "turn" if turns_remaining == 1 else "turns"
    call_word = "tool call" if tool_calls_remaining == 1 else "tool calls"
    msg = (
        f"Session warning: {turns_remaining} {turn_word} and "
        f"{tool_calls_remaining} {call_word} remain. "
    )
    if has_profiling_tools:
        msg += (
            "If you haven't profiled yet, do so now and make one final "
            "optimization pass before submitting."
        )
    else:
        msg += "Submit your best kernel soon."
    return msg
