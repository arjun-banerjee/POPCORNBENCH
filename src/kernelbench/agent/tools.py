"""
Tool definitions and executors for the KernelBench agent.

Design principles
-----------------
- Each tool is a self-contained class with a clear input/output contract.
- Tools are synchronous. The interface is async-ready — wrap execute() calls
  in asyncio.to_thread if needed.
- input_schema follows JSON Schema (same format as MCP / OpenAI function
  calling), so wrapping any tool in an MCP server later is a thin transport
  layer only.
- ToolContext carries shared per-run state (problem source, device, backend,
  etc.) so tools don't need globals.

Tool catalogue
--------------
    compile_kernel    — try to compile; return compiler output
    run_correctness   — correctness trials only (no timing)
    profile_kernel    — nsight roofline profiling (opt-in, requires ncu)
    get_gpu_specs     — hardware specs for the current device
    static_check      — static reward-hack pattern detector
    submit_kernel     — full eval: correctness + timing (speedup not revealed)

Output format
-------------
Every tool's `output` string follows the rule:

    {ToolName} {PASSED|FAILED}: {one-line summary}
    {optional detail block}

Two tools are exceptions to the PASS/FAIL rule — they have no natural
success/failure framing:

    - get_gpu_specs : reference data only; starts with "GPU specs for {name}:"
    - static_check  : uses PASSED / PASSED (with warnings) / FAILED to
                      distinguish clean / warnings-only / strict-violation runs.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable

import torch


# When eval_kernel_against_ref returns None (e.g. torch JIT extension lock-file
# contention from oversubscribed agents on the same GPU), retry transparently
# instead of reporting failure to the model — it's our infra, not a bad kernel.
_LOCK_RETRY_ATTEMPTS = 5
_LOCK_RETRY_BASE_SLEEP_S = 0.5  # exponential backoff with jitter


def _per_kernel_build_dir(base_build_dir: str | None, kernel_code: str) -> str | None:
    """Return a content-keyed subdirectory of base_build_dir.

    Same kernel → same subdir → torch JIT cache hit (fast).
    Different kernel → different subdir → no shared lock files, no contention.

    Returns None if base_build_dir is None (caller falls back to default cache).
    """
    if not base_build_dir:
        return None
    digest = hashlib.sha1(kernel_code.encode("utf-8", errors="replace")).hexdigest()[:12]
    sub = os.path.join(base_build_dir, f"k_{digest}")
    os.makedirs(sub, exist_ok=True)
    return sub


def _retry_eval_on_lock(eval_fn: Callable[[], Any]) -> Any:
    """Call eval_fn(); if it returns None, retry with exponential backoff."""
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        result = eval_fn()
        if result is not None:
            return result
        if attempt == _LOCK_RETRY_ATTEMPTS - 1:
            return None
        sleep_s = _LOCK_RETRY_BASE_SLEEP_S * (2 ** attempt) + random.uniform(0, 0.25)
        time.sleep(sleep_s)
    return None

from kernelbench.eval import (
    KernelExecResult,
    eval_kernel_against_ref,
    load_custom_model,
    load_custom_model_with_tempfile,
    graceful_eval_cleanup,
    get_torch_dtype_from_string,
)
from kernelbench.kernel_static_checker import validate_kernel_static


# ---------------------------------------------------------------------------
# ToolContext — shared per-problem state passed into every tool
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Shared per-run state passed into every tool call."""

    ref_arch_src: str  # reference PyTorch model source
    backend: str  # "cuda" | "triton" | "tilelang" | "cute" | "hip"
    precision: str  # "fp32" | "fp16" | "bf16"
    device: torch.device  # GPU device to run on
    build_dir: str | None = None  # CUDA compile cache directory
    num_correct_trials: int = 5  # correctness trials in submit_kernel
    num_perf_trials: int = 100  # timing trials in submit_kernel
    timing_method: str = "cuda_event"
    verbose: bool = False

    # Mutable: last profile result for delta comparison across iterations
    _last_profile_summary: Any = field(default=None, init=False, repr=False)

    @property
    def torch_precision(self) -> torch.dtype:
        return get_torch_dtype_from_string(self.precision)


# ---------------------------------------------------------------------------
# ToolResult — output of one tool execution
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str  # LLM-readable text
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    # JSON Schema for the tool's input parameters.
    # Tools that take no arguments (get_gpu_specs) use an empty-properties schema.
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, ctx: ToolContext, **kwargs) -> ToolResult: ...

    def to_responses_schema(self) -> dict[str, Any]:
        """
        OpenAI Responses-API tool schema (flat shape):

            {"type": "function", "name": ..., "description": ..., "parameters": ...}

        This is the shape consumed by `client.responses.create(tools=[...])`.
        It differs from the Chat Completions schema, which nests fields under
        a "function" key.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

    def to_mcp_schema(self) -> dict[str, Any]:
        """MCP tool schema (different envelope, same schema content)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# Shared input-schema fragment for kernel_code
# ---------------------------------------------------------------------------

_KERNEL_CODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kernel_code": {
            "type": "string",
            "description": (
                "Full Python source of the ModelNew kernel file. Must be a "
                "complete, valid Python module — not raw CUDA C/C++."
            ),
        }
    },
    "required": ["kernel_code"],
}


# ---------------------------------------------------------------------------
# 1. CompileKernelTool
# ---------------------------------------------------------------------------


class CompileKernelTool(Tool):
    name = "compile_kernel"
    description = (
        "Compile the kernel without running it. Use this first after writing "
        "or editing a kernel to catch syntax, linker, and CUDA-compilation "
        "errors cheaply before spending GPU time on correctness. Returns "
        "compiler output on failure, or a success confirmation."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        stdout_buf = StringIO()
        context: dict = {}
        build_dir = _per_kernel_build_dir(ctx.build_dir, kernel_code)

        try:
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            torch.cuda.set_device(ctx.device)

            with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
                backend_lower = ctx.backend.lower()
                if backend_lower in ("triton", "tilelang", "cute"):
                    ModelNew, tmp = load_custom_model_with_tempfile(
                        kernel_code, entry_point="ModelNew"
                    )
                    graceful_eval_cleanup({}, ctx.device, tmp)
                else:
                    ModelNew = load_custom_model(kernel_code, context, build_dir)
                    graceful_eval_cleanup(context, ctx.device)

            if ModelNew is None:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        "compile_kernel FAILED: ModelNew class not found or "
                        "syntax error prevented execution.\n"
                        f"{stdout_buf.getvalue()}"
                    ),
                    metadata={"compiled": False, "error": "ModelNew not found"},
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                output="compile_kernel PASSED: kernel compiled without errors.",
                metadata={"compiled": True},
            )

        except Exception as e:
            captured = stdout_buf.getvalue()
            # Keep captured stdout/stderr (that's where nvcc/linker messages land);
            # drop the Python-level traceback — it's frames inside eval.py that
            # the model can't act on.
            detail = captured if captured.strip() else f"{type(e).__name__}: {e}"
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(f"compile_kernel FAILED: {type(e).__name__}.\n{detail}"),
                metadata={"compiled": False, "error": str(e)},
            )


# ---------------------------------------------------------------------------
# 2. RunCorrectnessTool
# ---------------------------------------------------------------------------


class RunCorrectnessTool(Tool):
    name = "run_correctness"
    description = (
        "Run the kernel against the reference for correctness only — no timing. "
        "Use this after compile_kernel succeeds, to verify your kernel produces "
        "numerically equivalent outputs. Returns per-trial pass/fail status and "
        "the nature of any numerical or runtime errors."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        build_dir = _per_kernel_build_dir(ctx.build_dir, kernel_code)
        result: KernelExecResult | None = _retry_eval_on_lock(lambda: eval_kernel_against_ref(
            original_model_src=ctx.ref_arch_src,
            custom_model_src=kernel_code,
            num_correct_trials=ctx.num_correct_trials,
            num_perf_trials=0,
            measure_performance=False,
            verbose=ctx.verbose,
            build_dir=build_dir,
            device=ctx.device,
            backend=ctx.backend,
            precision=ctx.torch_precision,
            check_for_excessive_speedup=False,
        ))

        if result is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    "run_correctness FAILED: persistent build/lock contention "
                    "after retries. Please try a different kernel."
                ),
                metadata={},
            )

        if not result.compiled:
            err = result.metadata.get("compilation_error", "unknown error")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(f"run_correctness FAILED: kernel did not compile.\n{err}"),
                metadata={"compiled": False, "correctness": False},
            )

        trials_str = result.metadata.get("correctness_trials", "?")

        if result.correctness:
            lines = [f"run_correctness PASSED: {trials_str} trials all matched the reference."]
            if result.numerical_precision:
                np_stats = result.numerical_precision
                lines.append(
                    f"Numerical precision: max_abs_err={np_stats.get('max_abs_error', 0):.2e}  "
                    f"mean_abs_err={np_stats.get('mean_abs_error', 0):.2e}  "
                    f"max_rel_err={np_stats.get('max_rel_error', 0):.2e}"
                )
            return ToolResult(
                tool_name=self.name,
                success=True,
                output="\n".join(lines),
                metadata={"compiled": True, "correctness": True, "numerical_precision": result.numerical_precision},
            )

        # Failure path: report the first failing trial and any numeric diffs.
        # We drop `runtime_error_traceback` here — it's frames inside eval.py,
        # not actionable by the model. The core error string lives under
        # `runtime_error` and is surfaced below. Full traceback is still in
        # metadata for human debugging.
        lines = [f"run_correctness FAILED: {trials_str} trials did not all match."]
        for key in ("correctness_issue", "runtime_error"):
            val = result.metadata.get(key)
            if val:
                lines.append(f"{key}: {val}")
        for key in ("max_difference", "avg_difference"):
            val = result.metadata.get(key)
            if val:
                lines.append(f"{key}: {val}")

        return ToolResult(
            tool_name=self.name,
            success=False,
            output="\n".join(lines),
            metadata={"compiled": True, "correctness": False},
        )


# ---------------------------------------------------------------------------
# 3. ProfileKernelTool
# ---------------------------------------------------------------------------


class ProfileKernelTool(Tool):
    name = "profile_kernel"
    description = (
        "Profile the kernel with NVIDIA Nsight Compute. Returns comprehensive "
        "diagnostics: DRAM bandwidth utilization, compute throughput "
        "(FP32/FP16/tensor-core), per-kernel breakdown, warp stall reasons, "
        "memory coalescing quality, shared-memory bank conflicts, L1/L2 hit "
        "rates, occupancy with limiting factors (registers, smem, block size), "
        "pipe utilization, branch divergence, eligible-warps analysis, and "
        "targeted data-driven optimization hints. Supports delta comparison "
        "across iterations. Use when you have a correct kernel and need to "
        "understand why it is slow."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    _WORKER_SCRIPT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "scripts", "_profile_worker.py",
    )

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        import json
        import subprocess
        import sys
        import tempfile

        from kernelbench.profile import NSIGHT_AVAILABLE, check_ncu_available
        from kernelbench.agent.nsight_parser import (
            ROOFLINE_METRICS,
            parse_nsight_metrics,
        )

        if not NSIGHT_AVAILABLE:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="profile_kernel FAILED: nsight-python package not installed.",
                metadata={"available": False},
            )
        if not check_ncu_available():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="profile_kernel FAILED: ncu not found in PATH.",
                metadata={"available": False},
            )

        request = {
            "custom_model_src": kernel_code,
            "ref_model_src": ctx.ref_arch_src,
            "metrics": ROOFLINE_METRICS,
            "num_trials": 1,
            "seed": 42,
            "device_index": ctx.device.index or 0,
            "backend": ctx.backend,
            "precision": ctx.precision,
            "build_dir": _per_kernel_build_dir(ctx.build_dir, kernel_code),
            "verbose": ctx.verbose,
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(request, tmp)
                req_path = tmp.name

            proc = subprocess.run(
                [sys.executable, self._WORKER_SCRIPT, req_path],
                capture_output=True,
                text=True,
                timeout=300,
            )

            os.unlink(req_path)

            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "").strip()[-500:]
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        f"profile_kernel FAILED: worker exited with code "
                        f"{proc.returncode}.\n{stderr_tail}"
                    ),
                    metadata={"error": stderr_tail},
                )

            raw_output = json.loads(proc.stdout.strip().splitlines()[-1])

            if "error" in raw_output:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=f"profile_kernel FAILED: {raw_output['error']}",
                    metadata={"error": raw_output["error"]},
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="profile_kernel FAILED: profiling timed out (300s).",
                metadata={"error": "timeout"},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"profile_kernel FAILED: {type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        # Separate kernel breakdown from ncu metric values
        kernel_breakdown = raw_output.pop("_kernel_breakdown", [])
        raw_metrics = raw_output

        device_name = torch.cuda.get_device_name(ctx.device)
        previous = ctx._last_profile_summary
        summary = parse_nsight_metrics(
            raw_metrics, device_name, kernel_breakdown=kernel_breakdown
        )

        # Store for delta comparison on next invocation
        ctx._last_profile_summary = summary

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"profile_kernel PASSED: profiling complete.\n"
                f"{summary.format_for_llm(previous=previous)}"
            ),
            metadata={
                "raw_metrics": {
                    k: v for k, v in raw_metrics.items() if v is not None
                },
                "bottleneck": summary.bottleneck,
                "dram_utilization_pct": summary.dram_utilization_pct,
                "dominant_pipe": summary.dominant_pipe,
                "dominant_utilization_pct": summary.dominant_utilization_pct,
                "occupancy_pct": summary.occupancy_pct,
                "top_stall": (
                    max(summary.warp_stalls.items(), key=lambda x: x[1])
                    if summary.warp_stalls
                    else None
                ),
            },
        )


# ---------------------------------------------------------------------------
# 4. GetGpuSpecsTool (exception to PASS/FAIL rule — reference data only)
# ---------------------------------------------------------------------------


class GetGpuSpecsTool(Tool):
    name = "get_gpu_specs"
    description = (
        "Return peak hardware specs for the GPU this kernel will run on "
        "(memory bandwidth, TFLOPS per precision, SM count, shared memory per "
        "SM, register file size, etc.). Use this once at the start to calibrate "
        "your optimization targets."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, ctx: ToolContext, **_) -> ToolResult:
        from kernelbench.prompts.hardware.gpu_specs import GPU_SPEC_INFO
        from kernelbench.agent.nsight_parser import _DEVICE_NAME_TO_SPEC_KEY

        device_name = torch.cuda.get_device_name(ctx.device)
        total_mem_gb = (
            torch.cuda.get_device_properties(ctx.device).total_memory / 1024**3
        )

        spec_key = None
        for substr, key in _DEVICE_NAME_TO_SPEC_KEY:
            if substr in device_name:
                spec_key = key
                break

        lines = [
            f"GPU specs for {device_name}:",
            f"  total memory (runtime): {total_mem_gb:.1f} GB",
        ]
        if spec_key and spec_key in GPU_SPEC_INFO:
            lines.append(f"  spec entry: {spec_key}")
            for k, v in GPU_SPEC_INFO[spec_key].items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append("  (no detailed spec entry for this GPU in gpu_specs.py)")

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
            metadata={"device_name": device_name, "spec_key": spec_key},
        )


# ---------------------------------------------------------------------------
# 5. StaticCheckTool
# ---------------------------------------------------------------------------


class StaticCheckTool(Tool):
    name = "static_check"
    description = (
        "Run a static-analysis pass that detects reward-hacking patterns "
        "(try/except fallbacks to the reference, timing-function patches, "
        "lazy-tensor tricks, threading injection, etc.). Use this before "
        "submit_kernel as a sanity check — flagged submissions cause "
        "evaluation to fail."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        valid, errors, warnings = validate_kernel_static(
            code=kernel_code,
            backend=ctx.backend,
            precision=ctx.precision,
        )

        if valid and not warnings:
            output = "static_check PASSED: no violations or warnings detected."
        elif valid:
            lines = ["static_check PASSED (with warnings):"]
            for w in warnings:
                lines.append(f"  WARNING: {w}")
            output = "\n".join(lines)
        else:
            lines = ["static_check FAILED: strict violations found."]
            for e in errors:
                lines.append(f"  ERROR: {e}")
            if warnings:
                lines.append("Advisory warnings:")
                for w in warnings:
                    lines.append(f"  WARNING: {w}")
            output = "\n".join(lines)

        return ToolResult(
            tool_name=self.name,
            success=valid,
            output=output,
            metadata={"valid": valid, "errors": errors, "warnings": warnings},
        )


# ---------------------------------------------------------------------------
# 6. SubmitKernelTool
# ---------------------------------------------------------------------------


class SubmitKernelTool(Tool):
    """
    Final submission: full correctness + timing evaluation.

    Anti-reward-hacking policy:
    - Reports the kernel's absolute runtime in μs.
    - Does NOT reveal the reference runtime or speedup ratio.
    """

    name = "submit_kernel"
    description = (
        "Submit the final kernel for full evaluation: correctness check AND "
        "timing measurement. This ends the session — only call it when you "
        "are confident the kernel is correct and optimized. Returns kernel "
        "runtime in microseconds. The reference runtime and speedup ratio "
        "are NOT revealed."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        build_dir = _per_kernel_build_dir(ctx.build_dir, kernel_code)
        result: KernelExecResult | None = _retry_eval_on_lock(lambda: eval_kernel_against_ref(
            original_model_src=ctx.ref_arch_src,
            custom_model_src=kernel_code,
            num_correct_trials=ctx.num_correct_trials,
            num_perf_trials=ctx.num_perf_trials,
            measure_performance=True,
            timing_method=ctx.timing_method,
            verbose=ctx.verbose,
            build_dir=build_dir,
            device=ctx.device,
            backend=ctx.backend,
            precision=ctx.torch_precision,
            check_for_excessive_speedup=True,
        ))

        if result is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    "submit_kernel FAILED: persistent build/lock contention "
                    "after retries. Please try a different kernel."
                ),
                metadata={},
            )

        if not result.compiled:
            err = result.metadata.get("compilation_error", "unknown compilation error")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(f"submit_kernel FAILED: kernel did not compile.\n{err}"),
                metadata=result.model_dump(),
            )

        if not result.correctness:
            trials_str = result.metadata.get("correctness_trials", "?")
            lines = [
                f"submit_kernel FAILED: correctness check did not pass ({trials_str} trials)."
            ]
            for key in ("correctness_issue", "runtime_error"):
                val = result.metadata.get(key)
                if val:
                    lines.append(f"{key}: {val}")
            for key in ("max_difference", "avg_difference"):
                val = result.metadata.get(key)
                if val:
                    lines.append(f"{key}: {val}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="\n".join(lines),
                metadata=result.model_dump(),
            )

        # Correctness passed — report runtime but NOT speedup.
        trials_str = result.metadata.get("correctness_trials", "?")
        lines = [f"submit_kernel PASSED: {trials_str} correctness trials all passed."]
        if result.runtime > 0:
            lines.append(f"Kernel runtime: {result.runtime:.2f} μs")
            stats = result.runtime_stats
            if stats:

                def _fmt(v: Any) -> str:
                    return f"{v:.2f}" if isinstance(v, (int, float)) else "?"

                lines.append(
                    f"Runtime stats: mean={_fmt(stats.get('mean'))}μs  "
                    f"median={_fmt(stats.get('median'))}μs  "
                    f"std={_fmt(stats.get('std'))}μs"
                )

        # ── Extended metrics summary ──
        if result.numerical_precision:
            np_stats = result.numerical_precision
            lines.append(
                f"Numerical precision: max_abs_err={np_stats.get('max_abs_error', '?'):.2e}  "
                f"mean_abs_err={np_stats.get('mean_abs_error', '?'):.2e}  "
                f"max_rel_err={np_stats.get('max_rel_error', '?'):.2e}"
            )

        if result.memory_stats and result.memory_stats.get("peak_memory_mb"):
            mem = result.memory_stats
            lines.append(
                f"Memory: {mem.get('peak_memory_mb', '?')} MB peak "
                f"(ref: {mem.get('ref_peak_memory_mb', '?')} MB, "
                f"ratio: {mem.get('memory_ratio', '?')}x)"
            )

        if result.kernel_launch_stats and result.kernel_launch_stats.get("num_kernels", -1) > 0:
            kl = result.kernel_launch_stats
            lines.append(
                f"Kernel launches: {kl.get('num_kernels', '?')} "
                f"(ref: {kl.get('ref_num_kernels', '?')}, "
                f"fusion ratio: {kl.get('fusion_ratio', '?')})"
            )

        if result.energy_stats and result.energy_stats.get("energy_per_run_mj", -1) > 0:
            en = result.energy_stats
            lines.append(
                f"Energy: {en.get('energy_per_run_mj', '?')} mJ/run "
                f"(ref: {en.get('ref_energy_per_run_mj', '?')} mJ, "
                f"ratio: {en.get('energy_ratio', '?')}x)"
            )

        if result.sol_stats and result.sol_stats.get("sol_score", -1) >= 0:
            sol = result.sol_stats
            lines.append(f"SOL score: {sol.get('sol_score', '?')}")

        if result.metadata.get("excessive_speedup"):
            lines.append(
                "Flagged for excessive speedup — this submission may have "
                "been rejected by automated review."
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
            metadata=result.model_dump(),
        )


# ---------------------------------------------------------------------------
# 7. DisassembleKernelTool
# ---------------------------------------------------------------------------


class DisassembleKernelTool(Tool):
    """
    Disassemble compiled CUDA binary to inspect SASS, PTX, register usage,
    and instruction mix via cuobjdump and nvdisasm.
    """

    name = "disassemble_kernel"
    description = (
        "Disassemble the compiled CUDA kernel to inspect its native GPU "
        "assembly (SASS), PTX intermediate representation, and per-kernel "
        "resource usage (registers, shared memory, spills). Use this when "
        "you have a correct kernel and want to understand the compiler's "
        "code generation — register pressure, instruction mix (memory vs "
        "compute vs control), tensor-core usage, and register spills. "
        "Requires cuobjdump and nvdisasm (shipped with CUDA Toolkit)."
    )
    input_schema = _KERNEL_CODE_SCHEMA

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        from kernelbench.sass import (
            check_cuobjdump_available,
            check_nvdisasm_available,
            disassemble_kernelbench_model,
        )
        from kernelbench.agent.sass_parser import parse_disassembly

        if not check_cuobjdump_available():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="disassemble_kernel FAILED: cuobjdump not found in PATH.",
                metadata={"available": False},
            )

        nvdisasm_ok = check_nvdisasm_available()

        try:
            disasm_result = disassemble_kernelbench_model(
                custom_model_src=kernel_code,
                device=ctx.device,
                backend=ctx.backend,
                precision=ctx.torch_precision,
                build_dir=_per_kernel_build_dir(ctx.build_dir, kernel_code),
                include_ptx=True,
                include_nvdisasm=nvdisasm_ok,
                include_life_range=nvdisasm_ok,
                verbose=ctx.verbose,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"disassemble_kernel FAILED: {type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        device_name = torch.cuda.get_device_name(ctx.device)
        summary = parse_disassembly(disasm_result, device_name)

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"disassemble_kernel PASSED: disassembly analysis complete.\n"
                f"{summary.format_for_llm()}"
            ),
            metadata={
                "max_registers": summary.max_registers,
                "has_register_spills": summary.has_register_spills,
                "has_tensor_core_ops": summary.has_tensor_core_ops,
                "instruction_mix": summary.instruction_mix,
            },
        )


# ---------------------------------------------------------------------------
# 8. ErtRooflineTool
# ---------------------------------------------------------------------------


class ErtRooflineTool(Tool):
    """
    Run empirical roofline micro-benchmarks to measure actual peak bandwidth
    and compute throughput for the current GPU.
    """

    name = "ert_roofline"
    description = (
        "Run the Empirical Roofline Tool — micro-benchmarks that measure "
        "the actual (not theoretical) peak memory bandwidth and compute "
        "throughput of the current GPU. Returns measured bandwidth at each "
        "memory hierarchy level (L1, L2, DRAM/HBM) and peak TFLOPS "
        "(FP32, FP16/tensor-core). Also computes the ridge point — the "
        "arithmetic intensity where the roofline transitions from memory- "
        "to compute-bound. Results are cached per GPU so subsequent calls "
        "are instant. No arguments needed."
    )
    input_schema = {"type": "object", "properties": {}, "required": []}

    def execute(self, ctx: ToolContext, **_) -> ToolResult:
        from kernelbench.ert import run_ert_benchmarks

        try:
            model = run_ert_benchmarks(
                device=ctx.device,
                use_cache=True,
                verbose=ctx.verbose,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"ert_roofline FAILED: {type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=(
                f"ert_roofline PASSED: empirical roofline benchmarks complete.\n"
                f"{model.format_for_llm()}"
            ),
            metadata=model.to_dict(),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Tools that require special hardware/software access and are excluded from
# the default tool set.
_OPT_IN_TOOLS = {"profile_kernel", "disassemble_kernel", "ert_roofline"}

ALL_TOOLS: list[Tool] = [
    CompileKernelTool(),
    RunCorrectnessTool(),
    ProfileKernelTool(),
    GetGpuSpecsTool(),
    StaticCheckTool(),
    SubmitKernelTool(),
    DisassembleKernelTool(),
    ErtRooflineTool(),
]

TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def get_tools(tool_names: list[str] | None = None) -> list[Tool]:
    """
    Return the list of Tool instances for the given names.

    - tool_names=None → default set (all tools except opt-in tools that
      require special hardware/software: profile_kernel, disassemble_kernel,
      ert_roofline).
    - submit_kernel is always included regardless of the list — without it
      the agent has no way to record a final evaluation result.
    """
    if tool_names is None:
        selected = [t for t in ALL_TOOLS if t.name not in _OPT_IN_TOOLS]
    else:
        wanted = set(tool_names)
        wanted.add("submit_kernel")
        selected = [t for t in ALL_TOOLS if t.name in wanted]
    return selected
