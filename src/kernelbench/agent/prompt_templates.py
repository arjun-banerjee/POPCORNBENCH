"""
System and turn prompts for the KernelBench multi-turn agent.

Tool call format dialects
--------------------------
Different models are trained on different XML tool-call conventions.
Pass `tool_format` to `build_system_prompt()` to show the model its own dialect.

  "canonical"  (default) — our <tool_call><tool_name>…</tool_name>… </tool_call>
                           Works with Claude, Gemini, GPT-4, open-source instruct models.

  "nemotron"   — Nemotron/Llama function-calling XML:
                   <function=tool_name>
                   <parameter=kernel_code>…</parameter>
                   </function>

  "auto"       — Show both formats. Useful when model dialect is unknown.

The parser in agent.py understands all dialects regardless of which is shown,
so format mismatch causes confusion but never data loss.
"""

from __future__ import annotations
from typing import List, Literal

ToolFormat = Literal["canonical", "nemotron", "auto"]


# ---------------------------------------------------------------------------
# System intro (shared across all formats)
# ---------------------------------------------------------------------------

_SYSTEM_INTRO = """\
You are an expert GPU kernel engineer. Your task is to write a high-performance \
custom CUDA/GPU kernel that replaces the PyTorch reference implementation provided, \
while producing numerically identical results.

You will work iteratively: you can use tools to compile, test, and profile your \
kernel before making a final submission. Think carefully about the hardware architecture \
and optimization opportunities before writing code.\
"""


# ---------------------------------------------------------------------------
# Tool section — per-format examples
# ---------------------------------------------------------------------------

_TOOL_HEADER_CANONICAL = """
## Available Tools

Call tools using this XML format anywhere in your response.

Tool with kernel code:
```
<tool_call>
<tool_name>TOOL_NAME</tool_name>
<kernel_code>
```python
# your full kernel code here
```
</kernel_code>
</tool_call>
```

Tool with no arguments (e.g. `get_gpu_specs`):
```
<tool_call>
<tool_name>get_gpu_specs</tool_name>
</tool_call>
```

You may make multiple tool calls in one response. They are executed in order.
**Only `submit_kernel` records the final evaluation result.**
"""

_TOOL_HEADER_NEMOTRON = """
## Available Tools

Call tools using the function-call format anywhere in your response.

Tool with kernel code:
```
<function=TOOL_NAME>
<parameter=kernel_code>
```python
# your full kernel code here
```
</parameter>
</function>
```

Tool with no arguments (e.g. `get_gpu_specs`):
```
<function=get_gpu_specs>
</function>
```

You may make multiple tool calls in one response. They are executed in order.
**Only `submit_kernel` records the final evaluation result.**
"""

_TOOL_HEADER_AUTO = """
## Available Tools

Call tools using **either** of these XML formats anywhere in your response.

Format A:
```
<tool_call>
<tool_name>TOOL_NAME</tool_name>
<kernel_code>```python
# kernel code
```</kernel_code>
</tool_call>
```

Format B (Nemotron/Llama style):
```
<function=TOOL_NAME>
<parameter=kernel_code>```python
# kernel code
```</parameter>
</function>
```

For tools with no arguments, omit the code block entirely.
You may make multiple tool calls in one response. They are executed in order.
**Only `submit_kernel` records the final evaluation result.**
"""

_TOOL_HEADERS: dict = {
    "canonical": _TOOL_HEADER_CANONICAL,
    "nemotron":  _TOOL_HEADER_NEMOTRON,
    "auto":      _TOOL_HEADER_AUTO,
}


# ---------------------------------------------------------------------------
# Per-tool descriptions (format-neutral)
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTIONS = {
    "compile_kernel": (
        "**compile_kernel** — Try to compile your kernel. Returns compiler errors or success. "
        "Use this first to catch syntax and linker errors cheaply."
    ),
    "run_correctness": (
        "**run_correctness** — Run correctness trials (no timing). Shows which trials passed "
        "and the nature of any numerical or runtime errors."
    ),
    "profile_kernel": (
        "**profile_kernel** — Profile your kernel with NVIDIA Nsight Compute. Returns memory "
        "bandwidth utilization, compute throughput, arithmetic intensity, and roofline "
        "bottleneck classification. Use this to understand *why* your kernel is slow."
    ),
    "get_gpu_specs": (
        "**get_gpu_specs** — Return peak specs for the current GPU (memory bandwidth, TFLOPS, "
        "shared memory, register file, etc.). Use this to calibrate your optimization strategy."
    ),
    "static_check": (
        "**static_check** — Run static analysis to detect reward-hacking patterns before "
        "submitting (try-except fallbacks, timing patches, lazy tensors, etc.)."
    ),
    "submit_kernel": (
        "**submit_kernel** — Final submission. Runs full correctness check + timing measurement. "
        "Reports your kernel's runtime in μs. This terminates the session — only call it "
        "when you are confident in your kernel."
    ),
}


# ---------------------------------------------------------------------------
# Workflow hints and session limits (shared)
# ---------------------------------------------------------------------------

_STRATEGY_HINTS = """
## Suggested Workflow

1. Call `get_gpu_specs` to understand the hardware constraints.
2. Study the reference code and plan your optimization strategy.
3. Write a kernel and call `compile_kernel` to check for errors.
4. Call `run_correctness` to verify numerical correctness.
5. Call `profile_kernel` (if available) to identify bottlenecks.
6. Iterate on the optimization based on profiling feedback.
7. Call `submit_kernel` with your best kernel.

## Optimization Principles

- **Memory-bound kernels**: focus on coalesced access, shared memory tiling, \
vectorized loads (float4), and reducing redundant global memory traffic.
- **Compute-bound kernels**: maximize tensor core utilization, instruction-level \
parallelism, and warp occupancy.
- Avoid patterns flagged by the static checker (try-except fallbacks, timing patches).
- All outputs must be numerically equivalent to the reference within the tolerance \
for the specified precision (fp32: atol=rtol=1e-4, fp16/bf16: atol=rtol=1e-2).
"""

_TURN_LIMIT_NOTICE = """
## Session Limits

- You have **{max_turns} turns** (one turn = one response from you).
- You may make up to **{max_tool_calls} tool calls total** across all turns.
- Use your budget wisely: compile early, profile once, and submit when ready.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_system_prompt(
    available_tools: List[str],
    max_turns: int,
    max_tool_calls: int,
    tool_format: ToolFormat = "canonical",
) -> str:
    """
    Build the system prompt for the agent.

    Args:
        available_tools: Tool names enabled for this run (controls which tools
                         are listed in the prompt).
        max_turns:       Turn cap for this run.
        max_tool_calls:  Total tool call cap.
        tool_format:     XML dialect to show in the prompt instructions.
                         "canonical" | "nemotron" | "auto"

    Returns:
        Full system prompt string.
    """
    if tool_format not in _TOOL_HEADERS:
        raise ValueError(
            f"Unknown tool_format '{tool_format}'. "
            f"Choose from: {list(_TOOL_HEADERS.keys())}"
        )

    parts = [
        _SYSTEM_INTRO,
        _TOOL_HEADERS[tool_format],
    ]

    # List only the tools enabled for this run
    tool_lines = ["### Tools available in this session:\n"]
    for tool_name in available_tools:
        desc = _TOOL_DESCRIPTIONS.get(tool_name)
        if desc:
            tool_lines.append(f"- {desc}")
    parts.append("\n".join(tool_lines))

    parts.append(_STRATEGY_HINTS)
    parts.append(
        _TURN_LIMIT_NOTICE.format(max_turns=max_turns, max_tool_calls=max_tool_calls)
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# First user message — the problem statement
# ---------------------------------------------------------------------------

_LOAD_INLINE_EXAMPLE = """\
## Required Output Format

You must submit a **complete Python file** containing a `ModelNew` class that
uses `torch.utils.cpp_extension.load_inline` to compile and call a custom CUDA
kernel. The file is executed with `exec()` — it must be valid Python from line 1.

Minimal skeleton (adapt to the actual problem):

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];  // replace with actual computation
}

torch::Tensor my_op(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    my_kernel<<<(n+255)/256, 256>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
\"\"\"

cpp_src = "torch::Tensor my_op(torch::Tensor x);"

_ext = load_inline(
    name="my_kernel",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["my_op"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, ...):  # same args as Model
        super().__init__()
        # copy any parameters from Model here

    def forward(self, x):
        return _ext.my_op(x)
```

**Do NOT submit raw CUDA C code** — the harness expects a Python file."""


def build_problem_message(
    ref_arch_src: str,
    backend: str,
    precision: str,
    hardware_info: str = "",
) -> str:
    hw_block = f"\n## Hardware\n{hardware_info}\n" if hardware_info else ""

    return f"""\
## Task

Optimize the following PyTorch model by replacing its forward computation with \
a custom {backend.upper()} kernel. Your `ModelNew` class must:

1. Accept the same constructor arguments as `Model`.
2. Implement a `forward()` method with the same signature.
3. Produce numerically equivalent outputs (precision: **{precision}**).
4. Be faster than the PyTorch reference implementation.
{hw_block}
{_LOAD_INLINE_EXAMPLE}

## Reference Implementation

```python
{ref_arch_src}
```

Begin by reasoning about the computation, then use the available tools to \
build and test your kernel iteratively. When you are satisfied, call `submit_kernel`.
"""


# ---------------------------------------------------------------------------
# Tool result wrapper
# ---------------------------------------------------------------------------

def build_tool_results_message(tool_results_text: str) -> str:
    return (
        "## Tool Results\n\n"
        + tool_results_text
        + "\n\nContinue optimizing or call `submit_kernel` when ready."
    )


# ---------------------------------------------------------------------------
# Turn limit warning
# ---------------------------------------------------------------------------

def build_turn_warning_message(turns_remaining: int, tool_calls_remaining: int) -> str:
    return (
        f"**Session warning**: {turns_remaining} turn(s) and "
        f"{tool_calls_remaining} tool call(s) remaining. "
        "Consider calling `submit_kernel` with your best kernel."
    )
