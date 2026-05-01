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
    has_static = "static_check" in tool_names
    has_any_analysis = has_profile or has_disasm

    # ── Step 1: Understand hardware ──
    hw_parts: list[str] = []
    if has_gpu_specs:
        hw_parts.append("`get_gpu_specs`")
    if has_ert:
        hw_parts.append("`ert_roofline`")
    if hw_parts:
        hw_step = (
            f"1. **Understand the hardware**: call {' and '.join(hw_parts)}. "
        )
        if has_gpu_specs and has_ert:
            hw_step += (
                "`get_gpu_specs` gives theoretical peaks; `ert_roofline` "
                "gives *measured* peaks per memory level (L1 / L2 / DRAM) "
                "and per precision (FP32 / FP16 / tensor-core). Use the "
                "empirical numbers to set realistic optimization targets.\n"
            )
        elif has_gpu_specs:
            hw_step += "Know your bandwidth and compute ceilings.\n"
        elif has_ert:
            hw_step += (
                "Measured peaks per memory level and precision give you "
                "realistic optimization ceilings.\n"
            )
    else:
        hw_step = ""

    # ── Step 2: Initial kernel ──
    write_step = (
        "2. **Write an initial kernel**, then `compile_kernel` → "
        "`run_correctness`.\n"
    )

    # ── Steps 3–6: Analysis / optimize / iterate ──
    if has_any_analysis:
        # Build the analysis step dynamically from available tools
        analysis_actions: list[str] = []
        if has_profile:
            analysis_actions.append(
                "`profile_kernel` — gives runtime hardware counters: DRAM "
                "bandwidth, warp stalls, coalescing, bank conflicts, "
                "occupancy limiters, pipe utilization, cache hit rates, "
                "eligible warps, and a roofline classification"
            )
        if has_disasm:
            analysis_actions.append(
                "`disassemble_kernel` — gives compiler output: per-kernel "
                "register count, shared/local memory, register spills, "
                "instruction mix (memory vs compute vs control vs "
                "tensor-core), and whether the compiler is generating the "
                "instructions you expect"
            )
        analysis_list = "\n   - ".join(analysis_actions)
        analysis_step = (
            f"3. **Analyse the kernel** — call your analysis tools:\n"
            f"   - {analysis_list}\n"
        )
        if has_profile and has_disasm:
            analysis_step += (
                "   Use BOTH together: `profile_kernel` tells you *what* is "
                "slow at runtime (e.g. warp stalls on global memory); "
                "`disassemble_kernel` tells you *why* at the instruction "
                "level (e.g. register spills forcing local memory traffic, "
                "or no tensor-core instructions where you expected them). "
                "Cross-reference them to form a precise diagnosis.\n"
            )
        elif has_profile:
            analysis_step += (
                "   Read EVERY section of the output. The warp stall "
                "breakdown and memory access quality metrics are the most "
                "actionable — they directly name the bottleneck.\n"
            )
        elif has_disasm:
            analysis_step += (
                "   Check register pressure, spills, and instruction mix. "
                "High register count or spills directly hurt occupancy.\n"
            )

        optimize_step = (
            "4. **Fix the #1 bottleneck**: make ONE targeted change that "
            "directly addresses what the analysis told you. Then "
            "`compile_kernel` → `run_correctness` to verify correctness "
            "is preserved.\n"
        )

        # Build the iterate step referencing available tools
        tool_shortlist = []
        if has_profile:
            tool_shortlist.append("`profile_kernel`")
        if has_disasm:
            tool_shortlist.append("`disassemble_kernel`")
        tools_for_iterate = " and/or ".join(tool_shortlist)
        iterate_step = (
            f"5. **Analyse AGAIN** with {tools_for_iterate}. Check deltas — "
            "did the targeted metric improve? A new bottleneck may now "
            "dominate. Repeat steps 3–5 until you run low on budget or "
            "metrics plateau.\n"
        )

        static_note = ""
        if has_static:
            static_note = (
                " Run `static_check` before submitting to catch "
                "reward-hacking patterns that cause evaluation failure."
            )

        submit_step = (
            "6. **Submit** only after 2–3 analyse → optimise cycles, or "
            "when you've exhausted improvement ideas.{static_note}\n"
        ).format(static_note=static_note)
    else:
        analysis_step = ""
        optimize_step = ""
        iterate_step = ""
        submit_step = (
            "3. **Submit** when you are confident the kernel is correct and "
            "well-optimized.\n"
        )

    section = (
        "\n## Optimization workflow\n\n"
        "Correctness is a prerequisite, not the goal. Your job is to "
        "iteratively analyse and optimise. Follow this loop:\n\n"
        f"{hw_step}"
        f"{write_step}"
        f"{analysis_step}"
        f"{optimize_step}"
        f"{iterate_step}"
        f"{submit_step}"
        "\nDo NOT call `submit_kernel` the moment you have a correct kernel "
        "— a correct but unoptimized kernel is a wasted submission. "
        "Do NOT submit without analysing at least once."
        "\n\n**Optimization objective: runtime, with SOL as the efficiency "
        "check.** The primary target is the lowest possible kernel runtime "
        "vs the PyTorch reference (i.e. highest speedup). But beating the "
        "reference is not always evidence of an efficient kernel: many "
        "reference implementations are unfused PyTorch glue and easy to "
        "outpace without using the GPU well. Use SOL = max(dram_utilization, "
        "compute_utilization)/100 ∈ [0, 1] (Yuan & Mascagni, "
        "arXiv:2603.19173) as the second-order signal that tells you whether "
        "your kernel still has headroom:\n"
        "- low SOL + already faster than the reference → you're winning by "
        "  exploiting a slow baseline. There is more room; keep optimising.\n"
        "- high SOL + faster than reference → you're near the hardware "
        "  ceiling; further wins likely require an algorithmic change, not a "
        "  micro-optimisation.\n"
        "- low SOL + slower than reference → straight under-utilisation; "
        "  attack the dominant bottleneck.\n"
        "When `profile_kernel` is available, read its DRAM%/compute%/warp-"
        "stall breakdown and target the dominant bottleneck explicitly. "
        "Prefer kernels that are fast *and* high-SOL over kernels that are "
        "fast but low-SOL — both metrics are reported."
    )

    section += _build_tool_synergy_guide(tool_names)

    return section


def _build_tool_synergy_guide(tool_names: set[str]) -> str:
    """Build the combined tool-interpretation guide, covering all analysis
    tools the agent has access to."""
    has_profile = "profile_kernel" in tool_names
    has_disasm = "disassemble_kernel" in tool_names
    has_ert = "ert_roofline" in tool_names

    sections: list[str] = ["\n\n## How to use your analysis tools"]

    # ── Tool overview: when to call what ──
    sections.append("""
### When to call each tool

| Tool | What it tells you | When to call it |
|---|---|---|""")
    if "get_gpu_specs" in tool_names:
        sections.append(
            "| `get_gpu_specs` | Theoretical peak BW, TFLOPS, SM count, "
            "shared mem / register limits | Once at start |"
        )
    if has_ert:
        sections.append(
            "| `ert_roofline` | *Measured* peak BW per memory level "
            "(L1/L2/DRAM) and TFLOPS per precision (FP32/FP16/TC). "
            "Ridge points. | Once at start — gives realistic ceilings |"
        )
    if has_profile:
        sections.append(
            "| `profile_kernel` | Runtime hardware counters: DRAM BW, "
            "warp stalls, coalescing, bank conflicts, occupancy, pipe "
            "utilization, cache hit rates, eligible warps, roofline "
            "classification, per-kernel breakdown, delta vs previous run | "
            "Every iteration — this is your primary feedback loop |"
        )
    if has_disasm:
        sections.append(
            "| `disassemble_kernel` | Compiler output: registers/thread, "
            "shared/local memory, spills, SASS instruction mix "
            "(memory/compute/control/tensor-core) | When you suspect "
            "register pressure, spills, or want to verify tensor-core "
            "codegen |"
        )

    # ── Cross-referencing tools ──
    if has_profile and has_disasm:
        sections.append("""
### Cross-referencing `profile_kernel` + `disassemble_kernel`

Profiling tells you *what* is slow; disassembly tells you *why*:

- **`profile_kernel` says low occupancy, limited by registers** → call \
`disassemble_kernel` to see exact register count per kernel and whether \
there are spills. If spilling, add `__launch_bounds__` or reduce live \
variables.
- **`profile_kernel` says Tensor pipe = 0% on a matmul-heavy kernel** → \
call `disassemble_kernel` to check if HMMA/IMMA instructions are present. \
If not, the compiler isn't generating tensor-core code — switch to \
half-precision inputs or use WMMA intrinsics explicitly.
- **`profile_kernel` says high compute instruction count** → call \
`disassemble_kernel` to see the instruction mix breakdown. A high ratio \
of control/predication instructions means branch divergence; a high ratio \
of memory instructions means the kernel is issuing too many loads/stores \
per compute op.""")

    if has_profile and has_ert:
        sections.append("""
### Cross-referencing `profile_kernel` + `ert_roofline`

`ert_roofline` gives you the empirical ceiling; `profile_kernel` gives you \
where you are relative to it:

- If `ert_roofline` measured DRAM BW = 2800 GB/s and `profile_kernel` shows \
DRAM BW = 1400 GB/s (50%), you have room to improve memory throughput.
- If your kernel is already at 90%+ of the measured peak, further memory \
optimization won't help — the kernel is as memory-efficient as physically \
possible. Shift focus to reducing total bytes moved (algorithmic change, \
operator fusion) or switching to compute-bound operation.""")

    # ── Profiling interpretation ──
    if has_profile:
        sections.append(_build_profiling_details())

    # ── Disassembly interpretation ──
    if has_disasm:
        sections.append(_build_disassembly_details())

    return "\n".join(sections)


def _build_profiling_details() -> str:
    """Interpretation guide for `profile_kernel` output."""
    return """
### Reading `profile_kernel` output

#### Bottleneck classification

- **MEMORY-BOUND**: speed limited by data movement. Reduce DRAM traffic, \
improve access patterns.
- **COMPUTE-BOUND**: speed limited by arithmetic. Reduce instruction count, \
use tensor cores.
- **LATENCY-BOUND** (low eligible warps + low BW% + low compute%): GPU is \
idle because nothing is ready to execute. Need more parallelism (larger grid, \
more ILP, prefetching).

#### Warp stall reasons → specific fixes

The top stall reason directly names the bottleneck:

| Stall | Root cause | Fix |
|---|---|---|
| `long_scoreboard` | Waiting for DRAM/L2 | Tile to shared memory, prefetch, coalesce, __ldg |
| `short_scoreboard` | Waiting for shared mem/L1 | Fix bank conflicts (pad +1), double-buffer, async copy |
| `mio_throttle` | Too many memory instructions | Batch loads, vectorized loads (float4), reduce address divergence |
| `barrier` | __syncthreads overhead | Reduce sync frequency, overlap compute with sync |
| `math_pipe_throttle` | Compute pipe saturated | Good utilization — try more ILP to overlap with memory |
| `lg_throttle` | Local/global queue full | Reduce outstanding memory requests per warp |
| `not_selected` | Scheduler busy | Usually benign — good latency hiding |

#### Memory access quality

- **Sectors/request > 1.0** = uncoalesced. 4.0 = 4x wasted BW. Fix: thread i \
must access consecutive addresses.
- **Smem bank conflicts > 0** = layout bug. Fix: pad arrays +1 in inner dim.
- **Cache effectiveness close to 1.0** = no caching benefit → tile for reuse.

#### Occupancy — check the Launch line for the limiter

- **regs/thread > 64**: use `__launch_bounds__`, reduce live variables.
- **smem/blk near SM limit**: smaller tiles or less shared memory.
- **block < 128**: too few threads per block.

#### Pipe utilization

- **Tensor=0% but compute-bound** → not using tensor cores. Switch to FP16/BF16 \
with WMMA or half-precision `torch.matmul`.
- **FMA high, Tensor low** → FP32 pipe bottleneck. Consider FP16 where precision \
allows.

#### Delta comparison

After optimising, `profile_kernel` shows deltas vs the previous run. \
GPU time ↓ = success. GPU time ↑ = regression — revert and try another approach. \
Check which bottleneck is now dominant.

#### Kernel breakdown

Top of output shows which CUDA kernel is hottest. If it's a library call \
(cuBLAS, cuDNN), you can't optimise it directly — fuse the surrounding ops."""


def _build_disassembly_details() -> str:
    """Interpretation guide for `disassemble_kernel` output."""
    return """
### Reading `disassemble_kernel` output

#### Per-kernel resource usage

Each kernel shows registers, shared memory, local memory, and spills. \
Key things to check:
- **Registers > 64/thread**: limits occupancy. Use `__launch_bounds__(maxThreads, \
minBlocks)` to cap the compiler's register allocation.
- **Spill stores/loads > 0**: the compiler ran out of registers and is spilling \
to slow local memory. This is a major perf bug. Reduce live variables, simplify \
expressions, or split the kernel.
- **Local memory > 0** (without spills): arrays declared in the kernel that \
didn't fit in registers. Consider using shared memory instead.

#### Instruction mix

The breakdown shows what fraction of instructions are memory, compute, control, \
tensor-core, etc:
- **High memory%**: kernel is dominated by load/store instructions. Fuse \
operations to increase compute per load.
- **High control%**: branch/predication overhead. Reduce conditionals, \
restructure for uniform control flow.
- **Tensor-core = 0**: if you expected HMMA/IMMA instructions, check that you're \
using half-precision inputs and appropriate matrix dimensions (multiples of 16).
- **Tensor-core > 0**: good — verify with `profile_kernel` that the tensor pipe \
utilization is high. If not, the tensor ops may be bottlenecked by data supply."""


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
not submit raw CUDA C or a standalone .cu file.

DO NOT write your own `PYBIND11_MODULE(...)` block in `cpp_sources`. \
`load_inline` auto-generates one from the `functions=[...]` argument; including \
your own causes a duplicate-symbol redefinition error at compile time. List the \
host-side wrapper function names in `functions=[...]` instead."""


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
        if turns_remaining >= 3:
            msg += (
                "If you haven't analysed your kernel yet, call "
                "`profile_kernel` (and `disassemble_kernel` if register "
                "pressure or instruction mix is unclear) NOW. Make a "
                "targeted optimisation based on the #1 bottleneck, then "
                "analyse again before submitting. Do not submit without at "
                "least one analyse → optimise cycle."
            )
        elif turns_remaining == 2:
            msg += (
                "LAST turn. Call `submit_kernel` now with your best kernel."
            )
    else:
        msg += "Submit your best kernel soon."
    return msg
