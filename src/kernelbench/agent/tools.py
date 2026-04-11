"""
Tool definitions and executors for the KernelBench agent.

Design principles:
- Each tool is a self-contained class with a clear input/output contract.
- Tools are synchronous; the interface is async-ready (just wrap in asyncio.to_thread).
- input_schema follows JSON Schema (same format as MCP / OpenAI function calling),
  so wrapping in an MCP server later is a thin transport layer only.
- ToolContext carries the shared state (problem source, device, backend, etc.)
  so tools don't need global state.

Tool catalogue (all opt-in per run):
  compile_kernel     — try to compile, return errors or success
  run_correctness    — correctness trials only (no timing)
  profile_kernel     — nsight roofline profiling (requires nsight + permissions)
  get_gpu_specs      — return hardware specs for the current device
  static_check       — detect reward-hacking patterns
  submit_kernel      — full eval: correctness + timing (no speedup revealed)
"""

from __future__ import annotations

import ast
import os
import re
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional, Tuple

import torch

from kernelbench.eval import (
    KernelExecResult,
    eval_kernel_against_ref,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
    _process_input_tensor,
    graceful_eval_cleanup,
    set_seed,
    get_torch_dtype_from_string,
)
from kernelbench.kernel_static_checker import validate_kernel_static


# ---------------------------------------------------------------------------
# CUDA C detection + wrapping helpers
# ---------------------------------------------------------------------------

# Patterns that reliably identify raw CUDA C/C++ code submitted as Python
_CUDA_C_PATTERNS = re.compile(
    r"^\s*(?:#include\s*[<\"]|__global__\s+void|extern\s+\"C\"|torch::Tensor\s+\w+\()",
    re.MULTILINE,
)

_TORCH_TENSOR_FUNC_RE = re.compile(
    r"torch::Tensor\s+(\w+)\s*\([^)]*\)\s*\{",
)


def _is_cuda_c(code: str) -> bool:
    """Return True if code looks like raw CUDA C/C++, not a Python file."""
    try:
        ast.parse(code)
        return False   # valid Python — not CUDA C
    except SyntaxError:
        pass
    # Not valid Python — check for CUDA C signatures
    return bool(_CUDA_C_PATTERNS.search(code))


def _cuda_c_error_message(cuda_code: str) -> str:
    """
    Build a clear error message + minimal Python wrapper skeleton when the
    model submits raw CUDA C++ instead of a Python file.

    The message shows the model's own CUDA code already placed inside a
    triple-quoted string so it only needs to add the load_inline call and
    ModelNew class.
    """
    # Try to extract the torch::Tensor function name for the cpp_src hint.
    func_names = _TORCH_TENSOR_FUNC_RE.findall(cuda_code)
    cpp_hint = ""
    call_hint = "ext.YOUR_FUNCTION_NAME(...)"
    if func_names:
        # Use first detected function
        fn = func_names[0]
        # Build a rough signature (args unknown; just show the name)
        cpp_hint = f'cpp_src = "torch::Tensor {fn}(/* args */);"'
        call_hint = f"_ext.{fn}(...)"

    # Indent the CUDA code for embedding in a triple-quoted string
    indented = "\n".join("    " + ln for ln in cuda_code.splitlines())

    # Pre-compute values used in f-strings to avoid backslash-in-expression
    # errors on Python < 3.12.
    cpp_src_line = cpp_hint or 'cpp_src = "torch::Tensor YOUR_FUNC(/* args */);"'
    funcs_line = repr(func_names) if func_names else "['YOUR_FUNC']"

    return (
        "ERROR: You submitted raw CUDA C/C++ code, not a Python file.\n\n"
        "The kernel_code parameter must be a COMPLETE Python file that uses "
        "torch.utils.cpp_extension.load_inline to compile your CUDA kernel. "
        "Your CUDA code must live inside a Python triple-quoted string.\n\n"
        "Wrap your code like this:\n\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.cpp_extension import load_inline\n\n"
        'cuda_src = """\n'
        f"{indented}\n"
        '"""\n\n'
        f"{cpp_src_line}\n\n"
        "_ext = load_inline(\n"
        '    name="custom_kernel",\n'
        "    cpp_sources=cpp_src,\n"
        "    cuda_sources=cuda_src,\n"
        f"    functions={funcs_line},\n"
        "    verbose=False,\n"
        ")\n\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self, ...):\n"
        "        super().__init__()\n"
        "        # copy parameters from Model.__init__ here\n\n"
        "    def forward(self, ...):\n"
        f"        return {call_hint}\n"
        "```\n\n"
        "Resubmit kernel_code as a complete Python file following this pattern."
    )


# ---------------------------------------------------------------------------
# ToolContext — shared per-problem state passed into every tool
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """
    Immutable context shared across all tool calls for one agent run.
    Passed by the agent to each tool's execute() call.
    """
    ref_arch_src: str              # reference PyTorch model source
    backend: str                   # "cuda", "triton", "tilelang", "cute", "hip"
    precision: str                 # "fp32", "fp16", "bf16"
    device: torch.device           # GPU device to run on
    build_dir: Optional[str] = None          # directory for CUDA compile cache
    num_correct_trials: int = 5    # correctness trials in submit_kernel
    num_perf_trials: int = 100     # timing trials in submit_kernel
    timing_method: str = "cuda_event"
    verbose: bool = False

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
    output: str                              # human/LLM-readable text
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool ABC
# ---------------------------------------------------------------------------

class Tool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    # JSON Schema for the tool's input parameters.
    # kernel_code is optional because some tools (get_gpu_specs) take no args.
    input_schema: Dict[str, Any]

    @abstractmethod
    def execute(self, ctx: ToolContext, **kwargs) -> ToolResult:
        ...

    def to_openai_schema(self) -> Dict[str, Any]:
        """OpenAI function-calling schema (also valid MCP tool schema)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_mcp_schema(self) -> Dict[str, Any]:
        """MCP tool schema (identical structure, different envelope)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# 1. CompileKernelTool
# ---------------------------------------------------------------------------

class CompileKernelTool(Tool):
    """
    Try to compile the kernel without running it.
    Returns compiler errors (or success) so the model can fix syntax/link errors
    before spending time on correctness or profiling.
    """

    name = "compile_kernel"
    description = (
        "Attempt to compile the kernel code. Returns compiler output including "
        "any errors or warnings. Does NOT run correctness checks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "kernel_code": {
                "type": "string",
                "description": "Full Python source of the ModelNew kernel to compile.",
            }
        },
        "required": ["kernel_code"],
    }

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        if _is_cuda_c(kernel_code):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=_cuda_c_error_message(kernel_code),
                metadata={"compiled": False, "error": "submitted raw CUDA C, not Python"},
            )

        stdout_buf = StringIO()
        context: dict = {}

        try:
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            torch.cuda.set_device(ctx.device)

            with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
                backend_lower = ctx.backend.lower()
                if backend_lower in ["triton", "tilelang", "cute"]:
                    ModelNew, tmp = load_custom_model_with_tempfile(kernel_code, entry_point="ModelNew")
                    # Clean up the temp file
                    graceful_eval_cleanup({}, ctx.device, tmp)
                else:
                    ModelNew = load_custom_model(kernel_code, context, ctx.build_dir)
                    graceful_eval_cleanup(context, ctx.device)

            if ModelNew is None:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    output=(
                        "Compilation FAILED: ModelNew class not found in generated code, "
                        "or syntax error prevented execution.\n"
                        f"Captured output:\n{stdout_buf.getvalue()}"
                    ),
                    metadata={"compiled": False, "error": "ModelNew not found"},
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                output="Compilation SUCCEEDED. The kernel compiled without errors.",
                metadata={"compiled": True},
            )

        except Exception as e:
            captured = stdout_buf.getvalue()
            tb = traceback.format_exc()
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=(
                    f"Compilation FAILED.\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error: {e}\n"
                    f"Captured output:\n{captured}\n"
                    f"Traceback:\n{tb}"
                ),
                metadata={"compiled": False, "error": str(e)},
            )


# ---------------------------------------------------------------------------
# 2. RunCorrectnessTool
# ---------------------------------------------------------------------------

class RunCorrectnessTool(Tool):
    """
    Run correctness trials only — no timing measurement.
    Shows which trials passed/failed and the nature of any errors.
    Useful for iterating on correctness before worrying about performance.
    """

    name = "run_correctness"
    description = (
        "Compile and run correctness checks (no timing). "
        "Returns pass/fail for each trial and any error messages."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "kernel_code": {
                "type": "string",
                "description": "Full Python source of the ModelNew kernel to check.",
            }
        },
        "required": ["kernel_code"],
    }

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        if _is_cuda_c(kernel_code):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=_cuda_c_error_message(kernel_code),
                metadata={"compiled": False, "error": "submitted raw CUDA C, not Python"},
            )

        result: Optional[KernelExecResult] = eval_kernel_against_ref(
            original_model_src=ctx.ref_arch_src,
            custom_model_src=kernel_code,
            num_correct_trials=ctx.num_correct_trials,
            num_perf_trials=0,
            measure_performance=False,
            verbose=ctx.verbose,
            build_dir=ctx.build_dir,
            device=ctx.device,
            backend=ctx.backend,
            precision=ctx.torch_precision,
            check_for_excessive_speedup=False,
        )

        if result is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Correctness check failed: lock file or transient error. Please retry.",
                metadata={},
            )

        lines = []
        if not result.compiled:
            err = result.metadata.get("compilation_error", "unknown error")
            lines.append(f"Correctness check FAILED: kernel did not compile.")
            lines.append(f"Compilation error: {err}")
        elif result.correctness:
            trials = result.metadata.get("correctness_trials", "?")
            lines.append(f"Correctness check PASSED: {trials} trials all passed.")
        else:
            trials = result.metadata.get("correctness_trials", "?")
            lines.append(f"Correctness check FAILED: {trials} trials.")
            for key in ("correctness_issue", "runtime_error", "runtime_error_traceback"):
                val = result.metadata.get(key)
                if val:
                    lines.append(f"{key}: {val}")
            for key in ("max_difference", "avg_difference"):
                val = result.metadata.get(key)
                if val:
                    lines.append(f"{key}: {val}")

        return ToolResult(
            tool_name=self.name,
            success=result.correctness,
            output="\n".join(lines),
            metadata={"compiled": result.compiled, "correctness": result.correctness},
        )


# ---------------------------------------------------------------------------
# 3. ProfileKernelTool
# ---------------------------------------------------------------------------

class ProfileKernelTool(Tool):
    """
    Profile the kernel with NVIDIA Nsight Compute (opt-in; requires ncu + permissions).
    Returns a roofline summary: memory bandwidth utilization, compute utilization,
    arithmetic intensity, and bottleneck classification.
    Does NOT reveal speedup vs. reference.
    """

    name = "profile_kernel"
    description = (
        "Profile the kernel with NVIDIA Nsight Compute. "
        "Returns memory bandwidth utilization, compute throughput, arithmetic intensity, "
        "and roofline bottleneck classification. "
        "Requires ncu to be available and hardware counter access permissions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "kernel_code": {
                "type": "string",
                "description": "Full Python source of the ModelNew kernel to profile.",
            }
        },
        "required": ["kernel_code"],
    }

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        from kernelbench.profile import (
            NSIGHT_AVAILABLE,
            check_ncu_available,
            profile_kernelbench_model_with_nsight,
        )
        from kernelbench.agent.nsight_parser import ROOFLINE_METRICS, parse_nsight_metrics

        if not NSIGHT_AVAILABLE:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Nsight profiling is not available: nsight-python package not installed.",
                metadata={"available": False},
            )
        if not check_ncu_available():
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Nsight profiling is not available: ncu not found in PATH.",
                metadata={"available": False},
            )

        try:
            raw_metrics = profile_kernelbench_model_with_nsight(
                custom_model_src=kernel_code,
                ref_model_src=ctx.ref_arch_src,
                metrics=ROOFLINE_METRICS,
                num_trials=1,
                seed=42,
                device=ctx.device,
                backend=ctx.backend,
                precision=ctx.torch_precision,
                build_dir=ctx.build_dir,
                verbose=ctx.verbose,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=f"Nsight profiling raised an exception: {type(e).__name__}: {e}",
                metadata={"error": str(e)},
            )

        device_name = torch.cuda.get_device_name(ctx.device)
        summary = parse_nsight_metrics(raw_metrics, device_name)

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=summary.format_for_llm(),
            metadata={
                "raw_metrics": {k: v for k, v in raw_metrics.items() if v is not None},
                "bottleneck": summary.bottleneck,
                "bw_utilization_pct": summary.bw_utilization_pct,
                "compute_utilization_pct": summary.compute_utilization_pct,
            },
        )


# ---------------------------------------------------------------------------
# 4. GetGpuSpecsTool
# ---------------------------------------------------------------------------

class GetGpuSpecsTool(Tool):
    """
    Return hardware specifications for the current GPU.
    Useful for the model to calibrate its optimization strategy
    (memory bandwidth limits, tensor core availability, SM count, etc.).
    """

    name = "get_gpu_specs"
    description = (
        "Return hardware specifications for the GPU being used for evaluation "
        "(memory bandwidth, TFLOPS peaks, SM count, shared memory, etc.)."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self, ctx: ToolContext, **_) -> ToolResult:
        from kernelbench.prompts.hardware.gpu_specs import GPU_SPEC_INFO
        from kernelbench.agent.nsight_parser import _DEVICE_NAME_TO_SPEC_KEY

        device_name = torch.cuda.get_device_name(ctx.device)
        total_mem_gb = torch.cuda.get_device_properties(ctx.device).total_memory / 1024**3

        # Find best matching spec key
        spec_key = None
        for substr, key in _DEVICE_NAME_TO_SPEC_KEY:
            if substr in device_name:
                spec_key = key
                break

        lines = [f"GPU: {device_name}"]
        lines.append(f"Total GPU memory (runtime): {total_mem_gb:.1f} GB")

        if spec_key and spec_key in GPU_SPEC_INFO:
            spec = GPU_SPEC_INFO[spec_key]
            lines.append(f"\nSpec entry: {spec_key}")
            for k, v in spec.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append("(No detailed spec entry found in gpu_specs.py for this GPU.)")

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
    """
    Run the static reward-hack checker on the kernel code.
    Returns any detected patterns (try-except fallback, timing monkey-patches, etc.)
    before wasting time on compilation or evaluation.
    """

    name = "static_check"
    description = (
        "Run static analysis on the kernel code to detect potential reward-hacking patterns "
        "(try-except fallbacks, timing patches, lazy tensors, etc.). "
        "Returns errors (strict violations) and warnings (advisory)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "kernel_code": {
                "type": "string",
                "description": "Full Python source of the ModelNew kernel to check.",
            }
        },
        "required": ["kernel_code"],
    }

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        valid, errors, warnings = validate_kernel_static(
            code=kernel_code,
            backend=ctx.backend,
            precision=ctx.precision,
        )

        lines = []
        if valid and not warnings:
            lines.append("Static check PASSED: no violations or warnings detected.")
        elif valid:
            lines.append("Static check PASSED (with warnings):")
            for w in warnings:
                lines.append(f"  WARNING: {w}")
        else:
            lines.append("Static check FAILED (strict violations found):")
            for e in errors:
                lines.append(f"  ERROR: {e}")
            if warnings:
                lines.append("Warnings (advisory):")
                for w in warnings:
                    lines.append(f"  WARNING: {w}")

        return ToolResult(
            tool_name=self.name,
            success=valid,
            output="\n".join(lines),
            metadata={"valid": valid, "errors": errors, "warnings": warnings},
        )


# ---------------------------------------------------------------------------
# 6. SubmitKernelTool
# ---------------------------------------------------------------------------

class SubmitKernelTool(Tool):
    """
    Final submission: runs full correctness check + timing measurement.

    Anti-reward-hacking policy:
    - Reports the kernel's absolute runtime in μs.
    - Does NOT reveal the reference runtime or speedup ratio.
    - Correctness result is always shown (needed for the agent to know if it succeeded).
    """

    name = "submit_kernel"
    description = (
        "Submit the final kernel for full evaluation: correctness check + timing. "
        "This is the only tool that measures performance. "
        "Reports kernel runtime (μs) but NOT the reference runtime or speedup ratio."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "kernel_code": {
                "type": "string",
                "description": "Full Python source of the final ModelNew kernel to evaluate.",
            }
        },
        "required": ["kernel_code"],
    }

    def execute(self, ctx: ToolContext, kernel_code: str, **_) -> ToolResult:
        if _is_cuda_c(kernel_code):
            return ToolResult(
                tool_name=self.name,
                success=False,
                output=_cuda_c_error_message(kernel_code),
                metadata={"compiled": False, "error": "submitted raw CUDA C, not Python"},
            )

        result: Optional[KernelExecResult] = eval_kernel_against_ref(
            original_model_src=ctx.ref_arch_src,
            custom_model_src=kernel_code,
            num_correct_trials=ctx.num_correct_trials,
            num_perf_trials=ctx.num_perf_trials,
            measure_performance=True,
            timing_method=ctx.timing_method,
            verbose=ctx.verbose,
            build_dir=ctx.build_dir,
            device=ctx.device,
            backend=ctx.backend,
            precision=ctx.torch_precision,
            check_for_excessive_speedup=True,
        )

        if result is None:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="Evaluation failed: lock file or transient error. Please retry.",
                metadata={},
            )

        lines = []
        if not result.compiled:
            err = result.metadata.get("compilation_error", "unknown compilation error")
            lines.append("Submission FAILED: kernel did not compile.")
            lines.append(f"Compilation error: {err}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                output="\n".join(lines),
                metadata=result.model_dump(),
            )

        if not result.correctness:
            trials = result.metadata.get("correctness_trials", "?")
            lines.append(f"Submission FAILED: correctness check did not pass ({trials} trials).")
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

        # Correctness passed — report runtime but NOT speedup
        trials = result.metadata.get("correctness_trials", "?")
        lines.append(f"Submission PASSED: correctness {trials} trials all passed.")
        if result.runtime > 0:
            lines.append(f"Kernel runtime: {result.runtime:.2f} μs")
            stats = result.runtime_stats
            if stats:
                def _fmt(v):
                    return f"{v:.2f}" if isinstance(v, (int, float)) else "?"
                lines.append(
                    f"Runtime stats: mean={_fmt(stats.get('mean'))}μs  "
                    f"median={_fmt(stats.get('median'))}μs  "
                    f"std={_fmt(stats.get('std'))}μs"
                )
        if result.metadata.get("excessive_speedup"):
            lines.append(
                "WARNING: Excessive speedup flag raised. "
                "Please verify your kernel is not reward-hacking."
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output="\n".join(lines),
            metadata=result.model_dump(),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TOOLS: List[Tool] = [
    CompileKernelTool(),
    RunCorrectnessTool(),
    ProfileKernelTool(),
    GetGpuSpecsTool(),
    StaticCheckTool(),
    SubmitKernelTool(),
]

TOOL_REGISTRY: Dict[str, Tool] = {t.name: t for t in ALL_TOOLS}


def get_tools(tool_names: Optional[List[str]] = None) -> List[Tool]:
    """
    Return the list of Tool instances for the given names.
    If tool_names is None, returns all tools except profile_kernel
    (which requires special permissions and is opt-in).

    submit_kernel is always included regardless of the list — without it
    the agent has no way to record a final evaluation result.
    """
    if tool_names is None:
        default = [t for t in ALL_TOOLS if t.name != "profile_kernel"]
        return default

    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            raise ValueError(
                f"Unknown tool '{name}'. Available tools: {list(TOOL_REGISTRY.keys())}"
            )
        tools.append(TOOL_REGISTRY[name])

    # Always include submit_kernel
    if "submit_kernel" not in tool_names:
        tools.append(TOOL_REGISTRY["submit_kernel"])

    return tools
