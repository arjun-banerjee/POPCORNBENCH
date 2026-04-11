"""
Nsight metrics → human-readable ProfileSummary for the agent.

Takes raw metric values from profile.profile_with_nsight() and converts them
into a structured ProfileSummary that the agent can read and reason about.

Key design:
- Rule-based parsing (deterministic, reproducible, no LLM call needed)
- Compares achieved metrics against theoretical peaks from gpu_specs.py
- Classifies roofline position (memory-bound vs compute-bound)
- Returns a formatted text string suitable for LLM consumption

Nsight metric names used (standard ncu metric IDs):
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second   — global load BW (bytes/s)
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second   — global store BW (bytes/s)
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum        — FP32 add ops
  smsp__sass_thread_inst_executed_op_fmul_pred_on.sum        — FP32 mul ops
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum        — FP32 fma ops (counts as 2 FLOPs)
  sm__warps_active.avg.pct_of_peak_sustained_active          — occupancy %
  l2_global_load_bytes                                       — L2 cache bytes served
  gpu__time_duration.sum                                     — total GPU time (ns)

Not all metrics may be available depending on the ncu version/permissions;
each field degrades gracefully to None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Peak specs lookup — parsed from gpu_specs.py string values
# ---------------------------------------------------------------------------

# Map substrings that appear in torch.cuda.get_device_name() to gpu_specs keys
_DEVICE_NAME_TO_SPEC_KEY = [
    ("H100", "H100"),
    ("A100", "A100"),       # catches "A100-SXM4-40GB" etc.
    ("L40S", "L40S"),
    ("L4",   "L4"),
    ("T4",   "T4"),
    ("A10G", "A10G"),
    ("MI300X", "MI300X"),
    ("MI325X", "MI325X"),
    ("MI350X", "MI350X"),
    ("MI355X", "MI355X"),
]


def _parse_first_float(s: str) -> Optional[float]:
    """Extract the first number from a string like '1555 GB/s' or '3.35 TB/s'."""
    m = re.search(r"[\d.]+", s)
    return float(m.group()) if m else None


def _get_peak_specs(device_name: str) -> Dict[str, Optional[float]]:
    """
    Return peak bandwidth (GB/s) and FP32 TFLOPS for the named GPU.
    Falls back to None for unknown GPUs.
    """
    from kernelbench.prompts.hardware.gpu_specs import GPU_SPEC_INFO

    spec_key = None
    for substr, key in _DEVICE_NAME_TO_SPEC_KEY:
        if substr in device_name:
            spec_key = key
            break

    if spec_key is None or spec_key not in GPU_SPEC_INFO:
        return {"peak_bw_gbs": None, "peak_fp32_tflops": None}

    spec = GPU_SPEC_INFO[spec_key]

    # Memory bandwidth: handle "3.35 TB/s" → convert to GB/s
    bw_raw = spec.get("Memory Bandwidth", "")
    bw_val = _parse_first_float(bw_raw)
    if bw_val and "TB" in bw_raw:
        bw_val *= 1000.0  # TB/s → GB/s

    # FP32 TFLOPS: use "FP32 TFLOPS" key; fall back to "Single-Precision TFLOPS"
    tflops_raw = spec.get("FP32 TFLOPS", spec.get("Single-Precision TFLOPS", ""))
    tflops_val = _parse_first_float(tflops_raw)

    return {"peak_bw_gbs": bw_val, "peak_fp32_tflops": tflops_val}


# ---------------------------------------------------------------------------
# ProfileSummary dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfileSummary:
    """Structured summary of one Nsight profiling run."""
    # --- Achieved metrics ---
    gpu_time_us: Optional[float] = None        # kernel wall time in microseconds
    achieved_bw_gbs: Optional[float] = None    # memory BW actually achieved (GB/s)
    achieved_tflops: Optional[float] = None    # FP32 compute throughput (TFLOPS)
    occupancy_pct: Optional[float] = None      # warp occupancy (0-100)

    # --- Theoretical peaks (from gpu_specs.py) ---
    peak_bw_gbs: Optional[float] = None
    peak_fp32_tflops: Optional[float] = None

    # --- Derived ---
    bw_utilization_pct: Optional[float] = None
    compute_utilization_pct: Optional[float] = None
    arithmetic_intensity: Optional[float] = None   # FLOPs / byte
    ridge_point: Optional[float] = None            # FLOPs / byte at roofline knee
    bottleneck: Optional[str] = None               # "memory-bound" | "compute-bound" | "unknown"

    def format_for_llm(self) -> str:
        """Return a concise, LLM-readable text block."""
        lines = ["=== Kernel Profile Summary ==="]

        if self.gpu_time_us is not None:
            lines.append(f"Kernel runtime:           {self.gpu_time_us:.2f} μs")

        if self.achieved_bw_gbs is not None:
            peak_str = f" / {self.peak_bw_gbs:.0f} GB/s peak" if self.peak_bw_gbs else ""
            pct_str = f"  ({self.bw_utilization_pct:.1f}% of peak)" if self.bw_utilization_pct is not None else ""
            lines.append(f"Memory bandwidth:         {self.achieved_bw_gbs:.1f} GB/s{peak_str}{pct_str}")

        if self.achieved_tflops is not None:
            peak_str = f" / {self.peak_fp32_tflops:.1f} TFLOPS peak" if self.peak_fp32_tflops else ""
            pct_str = f"  ({self.compute_utilization_pct:.1f}% of peak)" if self.compute_utilization_pct is not None else ""
            lines.append(f"Compute throughput:       {self.achieved_tflops:.3f} TFLOPS{peak_str}{pct_str}")

        if self.arithmetic_intensity is not None:
            lines.append(f"Arithmetic intensity:     {self.arithmetic_intensity:.2f} FLOPs/byte")

        if self.ridge_point is not None:
            lines.append(f"Roofline ridge point:     {self.ridge_point:.2f} FLOPs/byte")

        if self.occupancy_pct is not None:
            lines.append(f"Warp occupancy:           {self.occupancy_pct:.1f}%")

        if self.bottleneck:
            lines.append(f"Bottleneck classification: {self.bottleneck.upper()}")
            if self.bottleneck == "memory-bound":
                lines.append("  → Optimization hint: reduce global memory traffic (shared memory tiling,")
                lines.append("    coalesced access patterns, vectorized loads/stores).")
            elif self.bottleneck == "compute-bound":
                lines.append("  → Optimization hint: increase compute density (tensor cores, instruction")
                lines.append("    level parallelism, loop unrolling).")

        if len(lines) == 1:
            lines.append("No profiling metrics were available.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parser: raw nsight dict → ProfileSummary
# ---------------------------------------------------------------------------

# Standard ncu metric names we request in the profiling tool
ROOFLINE_METRICS = [
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
]


def parse_nsight_metrics(
    raw_metrics: Dict[str, Optional[float]],
    device_name: str,
) -> ProfileSummary:
    """
    Convert raw nsight metric values → ProfileSummary.

    Args:
        raw_metrics: Dict from profile_with_nsight() — metric_name → float | None
        device_name: torch.cuda.get_device_name() string for peak lookup

    Returns:
        ProfileSummary with all computable fields filled in.
    """
    peaks = _get_peak_specs(device_name)
    summary = ProfileSummary(
        peak_bw_gbs=peaks["peak_bw_gbs"],
        peak_fp32_tflops=peaks["peak_fp32_tflops"],
    )

    # --- GPU time ---
    gpu_time_ns = raw_metrics.get("gpu__time_duration.sum")
    if gpu_time_ns is not None:
        summary.gpu_time_us = gpu_time_ns / 1e3

    # --- Memory bandwidth ---
    ld_bps = raw_metrics.get("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second")
    st_bps = raw_metrics.get("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second")
    if ld_bps is not None or st_bps is not None:
        total_bps = (ld_bps or 0.0) + (st_bps or 0.0)
        summary.achieved_bw_gbs = total_bps / 1e9
        if summary.peak_bw_gbs:
            summary.bw_utilization_pct = min(100.0, 100.0 * summary.achieved_bw_gbs / summary.peak_bw_gbs)

    # --- FP32 compute throughput ---
    fadd = raw_metrics.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0) or 0.0
    fmul = raw_metrics.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0) or 0.0
    ffma = raw_metrics.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0) or 0.0
    total_flops = fadd + fmul + 2.0 * ffma  # FMA = 2 FLOPs

    if total_flops > 0 and gpu_time_ns and gpu_time_ns > 0:
        gpu_time_s = gpu_time_ns / 1e9
        summary.achieved_tflops = total_flops / gpu_time_s / 1e12
        if summary.peak_fp32_tflops:
            summary.compute_utilization_pct = min(
                100.0, 100.0 * summary.achieved_tflops / summary.peak_fp32_tflops
            )

    # --- Arithmetic intensity ---
    total_bytes = None
    if ld_bps is not None and st_bps is not None and gpu_time_ns and gpu_time_ns > 0:
        gpu_time_s = gpu_time_ns / 1e9
        total_bytes = (ld_bps + st_bps) * gpu_time_s
    if total_flops > 0 and total_bytes and total_bytes > 0:
        summary.arithmetic_intensity = total_flops / total_bytes

    # --- Roofline ridge point ---
    if summary.peak_bw_gbs and summary.peak_fp32_tflops:
        # Ridge point = peak_tflops / peak_bw (in FLOPs/byte)
        summary.ridge_point = (summary.peak_fp32_tflops * 1e12) / (summary.peak_bw_gbs * 1e9)

    # --- Occupancy ---
    occ = raw_metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active")
    if occ is not None:
        summary.occupancy_pct = occ

    # --- Bottleneck classification ---
    if summary.arithmetic_intensity is not None and summary.ridge_point is not None:
        summary.bottleneck = (
            "memory-bound" if summary.arithmetic_intensity < summary.ridge_point
            else "compute-bound"
        )
    elif summary.bw_utilization_pct is not None and summary.compute_utilization_pct is not None:
        # Fallback: whichever utilization is higher is the bottleneck
        summary.bottleneck = (
            "memory-bound" if summary.bw_utilization_pct > summary.compute_utilization_pct
            else "compute-bound"
        )
    else:
        summary.bottleneck = "unknown"

    return summary
