"""
Prompts for the KernelBench multi-turn agent.

Three public builders:
    build_system_prompt(max_turns, max_tool_calls, backend, tool_names) -> str
        The system prompt, passed as `instructions` to the Responses API.

    build_problem_message(ref_arch_src, backend, precision) -> str
        First user-role message: task statement, output format, reference src.

    build_turn_warning_message(turns_remaining, tool_calls_remaining) -> str
        Status note injected as a user-role message once budget is low.

Tool descriptions live in `tools.py` and are surfaced to the model as part of
the function-calling schema. Don't duplicate them here.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Backend display-name map
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


_PROFILING_TOOLS = {"profile_kernel", "disassemble_kernel", "ert_roofline"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer. Write a custom {backend_display} kernel \
that replaces a PyTorch reference, producing numerically equivalent output \
faster than the reference.

You have {max_turns} turns and {max_tool_calls} tool calls. Each turn is one \
response from you, optionally with tool calls. `submit_kernel` records your \
final result and ends the session — call it once.

## Correctness

Outputs must match the reference within tolerance: fp32 uses atol=rtol=1e-4; \
fp16/bf16 use atol=rtol=1e-2. Do not use try/except fallbacks to the \
reference, patch timing or RNG functions, or otherwise route around the \
evaluation — these are detected and fail the run.

## Tools

Use the tools the way you would run an iteration loop by hand: write a \
kernel, compile it, check correctness, change something, repeat, submit. \
There is no required order and no required minimum number of \
iterations.{analysis_clause} The function-calling schema lists every \
tool with a short description of what it does and returns.

Speedup vs the PyTorch reference and SOL (Speed of Light — the higher \
of DRAM and compute utilization, on a 0–1 scale) are the objectives. \
Other counters in tool output (occupancy, warp stalls, register count, \
instruction mix) are diagnostics — useful for figuring out *why* a \
kernel is slow, not goals in themselves.
"""


_ANALYSIS_NOTE_WITH_PROFILING = (
    " You also have profiling and disassembly tools (`profile_kernel`, "
    "`disassemble_kernel`, `ert_roofline`) that report runtime hardware "
    "counters — DRAM bandwidth, warp stalls, occupancy, register usage, "
    "instruction mix — and an empirical roofline benchmark; each "
    "profiling call takes a few seconds to a few minutes."
)


def build_system_prompt(
    *,
    max_turns: int,
    max_tool_calls: int,
    backend: str,
    tool_names: list[str] | None = None,
) -> str:
    """Build the agent's system prompt (the API's `instructions` parameter)."""
    has_profiling = bool(tool_names and (set(tool_names) & _PROFILING_TOOLS))
    return _SYSTEM_PROMPT.format(
        backend_display=_backend_display(backend),
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        analysis_clause=_ANALYSIS_NOTE_WITH_PROFILING if has_profiling else "",
    )


# ---------------------------------------------------------------------------
# Per-backend output-format blocks
# ---------------------------------------------------------------------------


def _output_format_cuda() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. The file is executed with `exec()` before we instantiate \
`ModelNew`, so any module-level setup runs first.

Use `torch.utils.cpp_extension.load_inline` to compile and bind CUDA source \
at module load time, then call the compiled extension from \
`ModelNew.forward`. Do not submit raw CUDA C or a standalone .cu file.

Do not write your own `PYBIND11_MODULE(...)` block in `cpp_sources`. \
`load_inline` auto-generates one from the `functions=[...]` argument; \
including your own causes a duplicate-symbol redefinition error at compile \
time. List the host-side wrapper function names in `functions=[...]` instead."""


def _output_format_triton() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Write the kernel as a function decorated with `@triton.jit` (or \
`@triton.autotune`) using `triton.language` (commonly aliased as `tl`). \
Launch the kernel from `ModelNew.forward` with an appropriate grid."""


def _output_format_tilelang() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Write the kernel as a `@T.prim_func` using `tilelang.language` \
(aliased as `T`), compile it with `tilelang.compile(..., target="cuda")`, \
and invoke the compiled kernel from `ModelNew.forward`. TileLang requires \
fp16 or bf16 precision."""


def _output_format_cute() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Use the CUTLASS/CuTe Python bindings (`cutlass`, and the \
`cute::` namespace in any inlined C++) to build the kernel, and invoke it \
from `ModelNew.forward`."""


_OUTPUT_FORMAT_BUILDERS = {
    "cuda": _output_format_cuda,
    "triton": _output_format_triton,
    "tilelang": _output_format_tilelang,
    "cute": _output_format_cute,
}


def _output_format(backend: str) -> str:
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

Replace the forward computation of the PyTorch model below with a custom \
{backend_display} kernel. Your `ModelNew` class must:

1. Accept the same constructor arguments as `Model`.
2. Implement `forward()` with the same signature.
3. Produce numerically equivalent outputs at {precision} precision.
4. Run faster than the PyTorch reference.

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
# Turn-budget status
# ---------------------------------------------------------------------------


def build_turn_warning_message(
    turns_remaining: int,
    tool_calls_remaining: int,
) -> str:
    """Neutral status note injected once the budget is low.

    Tells the model what's left and that submitting now is fine. Does not
    instruct it to call any specific tool — that pushed the agent into
    net-negative changes after a correct kernel was already in hand.
    """
    turn_word = "turn" if turns_remaining == 1 else "turns"
    call_word = "tool call" if tool_calls_remaining == 1 else "tool calls"
    head = (
        f"Status: {turns_remaining} {turn_word} and "
        f"{tool_calls_remaining} {call_word} remain."
    )
    if turns_remaining <= 1:
        tail = " Last turn — call `submit_kernel` with your best correct kernel."
    else:
        tail = (
            " If you have a correct kernel and no concrete next change you "
            "want to try, submit now."
        )
    return head + tail
