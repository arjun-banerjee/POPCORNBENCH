"""
Nsight metrics → dense, actionable ProfileSummary for the agent.

Takes raw metric values and per-kernel breakdown from the profiling pipeline
and converts them into a structured summary that maximizes diagnostic signal
per token for an LLM code-optimization agent.

Key design:
- Rule-based parsing (deterministic, reproducible, no LLM call needed)
- DRAM-based bandwidth (not L1TEX) for accurate roofline position
- Multi-precision FLOP counting (FP32 + FP16 + tensor-core)
- Per-kernel breakdown with top-N by duration
- Warp stall reasons for targeted optimization hints
- Memory coalescing, bank conflicts, cache hit rates
- Occupancy limiters (registers, shared mem, block size)
- Pipe utilization (FMA, tensor, ALU)
- Data-driven optimization hints derived from measured bottlenecks
- Delta comparison to previous profiling iteration

Nsight metric naming reference (standard ncu metric IDs):
  DRAM:    dram__bytes_{read,write}.sum{,.per_second}
  L1TEX:   l1tex__t_bytes_pipe_lsu_mem_global_op_{ld,st}.sum{,.per_second}
  FP32:    smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma}_pred_on.sum
  FP16:    smsp__sass_thread_inst_executed_op_{hadd,hmul,hfma}_pred_on.sum
  Tensor:  sm__inst_executed_pipe_tensor.sum,
           smsp__inst_executed_pipe_tensor_op_{hmma,imma}.sum
  Stalls:  smsp__average_warps_issue_stalled_<reason>_per_issue_active.pct
  Launch:  launch__{registers_per_thread,shared_mem_per_block_*,block_size,grid_size}
  Pipes:   sm__pipe_{fma,tensor,alu}_cycles_active.avg.pct_of_peak_sustained_active

Not all metrics may be available depending on ncu version/permissions/GPU arch;
every field degrades gracefully to None.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Peak specs lookup — parsed from gpu_specs.py string values
# ---------------------------------------------------------------------------

_DEVICE_NAME_TO_SPEC_KEY = [
    ("H100", "H100"),
    ("A100", "A100"),
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
    """Return peak bandwidth (GB/s), FP32/FP16/TF32 TFLOPS for the named GPU."""
    from kernelbench.prompts.hardware.gpu_specs import GPU_SPEC_INFO

    spec_key = None
    for substr, key in _DEVICE_NAME_TO_SPEC_KEY:
        if substr in device_name:
            spec_key = key
            break

    out: Dict[str, Optional[float]] = {
        "peak_bw_gbs": None,
        "peak_fp32_tflops": None,
        "peak_fp16_tflops": None,
        "peak_tensor_tflops": None,
    }
    if spec_key is None or spec_key not in GPU_SPEC_INFO:
        return out

    spec = GPU_SPEC_INFO[spec_key]

    bw_raw = spec.get("Memory Bandwidth", "")
    bw_val = _parse_first_float(bw_raw)
    if bw_val and "TB" in bw_raw:
        bw_val *= 1000.0
    out["peak_bw_gbs"] = bw_val

    fp32_raw = spec.get("FP32 TFLOPS", spec.get("Single-Precision TFLOPS", ""))
    out["peak_fp32_tflops"] = _parse_first_float(fp32_raw)

    fp16_raw = spec.get(
        "FP16 Tensor Core TFLOPS",
        spec.get("Mixed-Precision (FP16/FP32) TFLOPS", ""),
    )
    out["peak_fp16_tflops"] = _parse_first_float(fp16_raw)

    tf32_raw = spec.get("TF32 Tensor Core TFLOPS", "")
    out["peak_tensor_tflops"] = _parse_first_float(tf32_raw)

    return out


# ---------------------------------------------------------------------------
# Comprehensive ncu metric list
# ---------------------------------------------------------------------------

NCU_METRICS = [
    # ── Timing ──
    "gpu__time_duration.sum",

    # ── DRAM (HBM) Bandwidth — real off-chip traffic for roofline ──
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",

    # ── L1TEX Requested (cache-effectiveness gap vs DRAM) ──
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",

    # ── FP32 Compute ──
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",

    # ── FP16 Compute ──
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",

    # ── Tensor Core ──
    "sm__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_tensor_op_hmma.sum",
    "smsp__inst_executed_pipe_tensor_op_imma.sum",

    # ── Occupancy ──
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",

    # ── Launch Geometry / Occupancy Limiters ──
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "launch__block_size",
    "launch__grid_size",
    "launch__shared_mem_per_block_driver",

    # ── Memory Coalescing ──
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",

    # ── Shared Memory Bank Conflicts ──
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",

    # ── L1 / L2 Hit Rates ──
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",

    # ── Pipe Utilization ──
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",

    # ── Branch / Predication Efficiency ──
    "smsp__thread_inst_executed_per_inst_executed.ratio",

    # ── Eligible Warps ──
    "smsp__warps_eligible.avg.per_cycle_active",

    # ── Warp Stall Reasons ──
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_membar_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_drain_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_sleeping_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_tex_throttle_per_issue_active.pct",
]

FAST_NCU_METRICS = [
    # ── Timing ──
    "gpu__time_duration.sum",
    # ── DRAM bandwidth (the dominant memory-bound signal) ──
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum.per_second",
    # ── Occupancy ──
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # ── Launch geometry ──
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_allocated",
    "launch__block_size",
    "launch__grid_size",
    # ── Coalescing ──
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    # ── Pipe utilization (compute saturation signal) ──
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    # ── L2 hit rate (cache effectiveness) ──
    "lts__t_sector_hit_rate.pct",
    # ── Top three warp-stall reasons ──
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.pct",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.pct",
    # ── Eligible warps (latency-bound signal) ──
    "smsp__warps_eligible.avg.per_cycle_active",
]

# Backward-compatible alias used by tools.py / _profile_worker.py.
# Switched from the comprehensive 46-metric list to the trimmed 16-metric
# fast set — ncu re-launches the kernel for each metric section, so cutting
# the list ~3x cuts profile wall-clock ~3x and dramatically reduces the
# 300s timeout failure rate on L3/L4 problems.
ROOFLINE_METRICS = FAST_NCU_METRICS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(raw: dict, key: str, default: Any = None) -> Optional[float]:
    """Get metric value, returning *default* for missing / None / invalid."""
    v = raw.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _delta_str(
    cur: Optional[float],
    prev: Optional[float],
    unit: str,
    lower_better: bool = False,
) -> str:
    """Compact inline delta annotation like ' [↓12.3μs ok]'."""
    if cur is None or prev is None:
        return ""
    diff = cur - prev
    if abs(diff) < 0.01:
        return ""
    arrow = "\u2193" if diff < 0 else "\u2191"
    good = (diff < 0) == lower_better
    marker = "ok" if good else "REGRESSED"
    return f"  [{arrow}{abs(diff):.1f}{unit} {marker}]"


# ---------------------------------------------------------------------------
# ProfileSummary dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfileSummary:
    """Dense, actionable summary of one Nsight profiling run."""

    # ── Timing ──
    gpu_time_us: Optional[float] = None

    # ── DRAM Bandwidth (real off-chip) ──
    dram_read_gbs: Optional[float] = None
    dram_write_gbs: Optional[float] = None
    dram_total_gbs: Optional[float] = None
    dram_utilization_pct: Optional[float] = None

    # ── L1TEX Requested (cache-effectiveness gap) ──
    l1tex_requested_gbs: Optional[float] = None
    cache_effectiveness: Optional[float] = None

    # ── Compute: FP32 ──
    fp32_flops: Optional[float] = None
    fp32_tflops: Optional[float] = None
    fp32_utilization_pct: Optional[float] = None

    # ── Compute: FP16 ──
    fp16_flops: Optional[float] = None
    fp16_tflops: Optional[float] = None

    # ── Compute: Tensor Core ──
    tensor_ops: Optional[float] = None
    hmma_ops: Optional[float] = None
    imma_ops: Optional[float] = None

    # ── Dominant compute pipe ──
    dominant_pipe: Optional[str] = None
    dominant_tflops: Optional[float] = None
    dominant_utilization_pct: Optional[float] = None

    # ── Arithmetic intensity & roofline ──
    arithmetic_intensity: Optional[float] = None
    ridge_point: Optional[float] = None
    bottleneck: Optional[str] = None

    # ── Peaks ──
    peak_bw_gbs: Optional[float] = None
    peak_fp32_tflops: Optional[float] = None
    peak_fp16_tflops: Optional[float] = None
    peak_tensor_tflops: Optional[float] = None

    # ── Occupancy ──
    occupancy_pct: Optional[float] = None
    active_warps_per_cycle: Optional[float] = None

    # ── Occupancy Limiters / Launch Geometry ──
    registers_per_thread: Optional[float] = None
    shared_mem_per_block: Optional[float] = None
    block_size: Optional[float] = None
    grid_size: Optional[float] = None
    dynamic_smem_bytes: Optional[float] = None

    # ── Memory Coalescing ──
    ld_sectors_per_request: Optional[float] = None
    st_sectors_per_request: Optional[float] = None

    # ── Shared Memory Bank Conflicts ──
    smem_ld_bank_conflicts: Optional[float] = None
    smem_st_bank_conflicts: Optional[float] = None

    # ── Cache Hit Rates ──
    l1_hit_rate_pct: Optional[float] = None
    l2_hit_rate_pct: Optional[float] = None

    # ── Pipe Utilization ──
    pipe_fma_pct: Optional[float] = None
    pipe_tensor_pct: Optional[float] = None
    pipe_alu_pct: Optional[float] = None

    # ── Branch Efficiency ──
    threads_per_inst_ratio: Optional[float] = None

    # ── Eligible Warps ──
    eligible_warps_per_cycle: Optional[float] = None

    # ── Warp Stall Reasons ──
    warp_stalls: Dict[str, float] = field(default_factory=dict)

    # ── Per-kernel breakdown (from torch.profiler) ──
    kernel_breakdown: List[Dict[str, Any]] = field(default_factory=list)

    # ── Raw metrics for compact table output ──
    raw_metrics: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # format_for_llm — the single entry-point the tool reads
    # ------------------------------------------------------------------

    def format_for_llm(self, previous: Optional["ProfileSummary"] = None) -> str:
        """
        Dense, actionable text block optimised for LLM consumption.

        If *previous* is provided, key metrics are annotated with deltas
        so the agent can see what improved or regressed.
        """
        lines: list[str] = ["=== Kernel Profile Summary ==="]

        # ── Per-kernel breakdown ──
        if self.kernel_breakdown:
            lines.append("")
            lines.append("--- Kernel Breakdown (by GPU time) ---")
            total_us = sum(k.get("cuda_time_us", 0) for k in self.kernel_breakdown)
            for i, k in enumerate(self.kernel_breakdown[:5]):
                name = k.get("name", "?")
                t_us = k.get("cuda_time_us", 0)
                pct = 100.0 * t_us / total_us if total_us > 0 else 0
                calls = k.get("calls", 1)
                lines.append(
                    f"  #{i+1}: {name}  {t_us:.1f}\u03bcs ({pct:.0f}%)  calls={calls}"
                )
            if len(self.kernel_breakdown) > 5:
                lines.append(
                    f"  ... +{len(self.kernel_breakdown) - 5} more kernels"
                )

        # ── Timing ──
        lines.append("")
        lines.append("--- Timing ---")
        if self.gpu_time_us is not None:
            d = _delta_str(
                self.gpu_time_us,
                previous.gpu_time_us if previous else None,
                "\u03bcs",
                lower_better=True,
            )
            lines.append(f"  GPU time: {self.gpu_time_us:.2f} \u03bcs{d}")

        # ── DRAM Bandwidth ──
        lines.append("")
        lines.append("--- Memory Bandwidth ---")
        if self.dram_total_gbs is not None:
            peak = f"/{self.peak_bw_gbs:.0f}" if self.peak_bw_gbs else ""
            pct = (
                f" ({self.dram_utilization_pct:.1f}%)"
                if self.dram_utilization_pct is not None
                else ""
            )
            d = _delta_str(
                self.dram_utilization_pct,
                previous.dram_utilization_pct if previous else None,
                "%",
            )
            lines.append(f"  DRAM BW:   {self.dram_total_gbs:.1f}{peak} GB/s{pct}{d}")
            if self.dram_read_gbs is not None and self.dram_write_gbs is not None:
                lines.append(
                    f"    read={self.dram_read_gbs:.1f}  write={self.dram_write_gbs:.1f} GB/s"
                )
        if self.l1tex_requested_gbs is not None:
            lines.append(f"  L1TEX requested: {self.l1tex_requested_gbs:.1f} GB/s")
        if self.cache_effectiveness is not None:
            lines.append(
                f"  Cache effectiveness: {self.cache_effectiveness:.2f}x"
                f"  (DRAM/L1TEX; lower=better caching)"
            )

        # ── L1 / L2 Hit Rates ──
        hit_parts: list[str] = []
        if self.l1_hit_rate_pct is not None:
            hit_parts.append(f"L1={self.l1_hit_rate_pct:.1f}%")
        if self.l2_hit_rate_pct is not None:
            hit_parts.append(f"L2={self.l2_hit_rate_pct:.1f}%")
        if hit_parts:
            lines.append(f"  Hit rates: {'  '.join(hit_parts)}")

        # ── Compute Throughput ──
        lines.append("")
        lines.append("--- Compute Throughput ---")
        if self.fp32_tflops is not None:
            peak = f"/{self.peak_fp32_tflops:.1f}" if self.peak_fp32_tflops else ""
            pct = (
                f" ({self.fp32_utilization_pct:.1f}%)"
                if self.fp32_utilization_pct is not None
                else ""
            )
            lines.append(f"  FP32:     {self.fp32_tflops:.3f}{peak} TFLOPS{pct}")
        if self.fp16_tflops is not None and self.fp16_tflops > 0:
            peak = f"/{self.peak_fp16_tflops:.1f}" if self.peak_fp16_tflops else ""
            lines.append(f"  FP16:     {self.fp16_tflops:.3f}{peak} TFLOPS")
        if self.tensor_ops is not None and self.tensor_ops > 0:
            detail: list[str] = []
            if self.hmma_ops:
                detail.append(f"hmma={self.hmma_ops:.0f}")
            if self.imma_ops:
                detail.append(f"imma={self.imma_ops:.0f}")
            det = f"  ({', '.join(detail)})" if detail else ""
            lines.append(f"  Tensor:   {self.tensor_ops:.0f} ops{det}")
        if self.dominant_pipe:
            d = _delta_str(
                self.dominant_utilization_pct,
                previous.dominant_utilization_pct if previous else None,
                "%",
            )
            util = (
                f"  util={self.dominant_utilization_pct:.1f}%"
                if self.dominant_utilization_pct is not None
                else ""
            )
            lines.append(f"  Dominant pipe: {self.dominant_pipe}{util}{d}")

        # ── Pipe Utilization ──
        pipe_parts: list[str] = []
        if self.pipe_fma_pct is not None:
            pipe_parts.append(f"FMA={self.pipe_fma_pct:.1f}%")
        if self.pipe_tensor_pct is not None:
            pipe_parts.append(f"Tensor={self.pipe_tensor_pct:.1f}%")
        if self.pipe_alu_pct is not None:
            pipe_parts.append(f"ALU={self.pipe_alu_pct:.1f}%")
        if pipe_parts:
            lines.append(f"  Pipe util: {', '.join(pipe_parts)}")

        # ── Roofline ──
        lines.append("")
        lines.append("--- Roofline Analysis ---")
        if self.arithmetic_intensity is not None:
            lines.append(
                f"  Arithmetic intensity: {self.arithmetic_intensity:.2f} FLOPs/byte"
            )
        if self.ridge_point is not None:
            pipe_tag = self.dominant_pipe or "FP32"
            lines.append(
                f"  Ridge point ({pipe_tag}): {self.ridge_point:.2f} FLOPs/byte"
            )
        if self.bottleneck:
            lines.append(f"  Classification: {self.bottleneck.upper()}")

        # ── Occupancy & Limiters ──
        lines.append("")
        lines.append("--- Occupancy ---")
        if self.occupancy_pct is not None:
            d = _delta_str(
                self.occupancy_pct,
                previous.occupancy_pct if previous else None,
                "%",
            )
            lines.append(f"  Achieved: {self.occupancy_pct:.1f}%{d}")
        if self.active_warps_per_cycle is not None:
            lines.append(
                f"  Active warps/cycle: {self.active_warps_per_cycle:.1f}"
            )
        if self.eligible_warps_per_cycle is not None:
            lines.append(
                f"  Eligible warps/cycle: {self.eligible_warps_per_cycle:.2f}"
            )
            if self.eligible_warps_per_cycle < 1.0:
                lines.append(
                    "    WARNING: <1 eligible warp/cycle = latency-bound, "
                    "needs more parallelism"
                )
        limiters: list[str] = []
        if self.registers_per_thread is not None:
            limiters.append(f"regs/thread={self.registers_per_thread:.0f}")
        if self.shared_mem_per_block is not None:
            limiters.append(f"smem/blk={self.shared_mem_per_block:.0f}B")
        if self.dynamic_smem_bytes is not None and self.dynamic_smem_bytes > 0:
            limiters.append(f"dyn_smem={self.dynamic_smem_bytes:.0f}B")
        if self.block_size is not None:
            limiters.append(f"block={self.block_size:.0f}")
        if self.grid_size is not None:
            limiters.append(f"grid={self.grid_size:.0f}")
        if limiters:
            lines.append(f"  Launch: {', '.join(limiters)}")

        # ── Memory Access Quality ──
        lines.append("")
        lines.append("--- Memory Access Quality ---")
        if self.ld_sectors_per_request is not None:
            q = (
                "perfect"
                if self.ld_sectors_per_request <= 1.05
                else f"{self.ld_sectors_per_request:.1f}x wasted BW"
            )
            lines.append(
                f"  Global ld coalescing: {self.ld_sectors_per_request:.2f}"
                f" sectors/req ({q})"
            )
        if self.st_sectors_per_request is not None:
            q = (
                "perfect"
                if self.st_sectors_per_request <= 1.05
                else f"{self.st_sectors_per_request:.1f}x wasted BW"
            )
            lines.append(
                f"  Global st coalescing: {self.st_sectors_per_request:.2f}"
                f" sectors/req ({q})"
            )
        conflict_parts: list[str] = []
        if self.smem_ld_bank_conflicts is not None:
            conflict_parts.append(f"ld={self.smem_ld_bank_conflicts:.0f}")
        if self.smem_st_bank_conflicts is not None:
            conflict_parts.append(f"st={self.smem_st_bank_conflicts:.0f}")
        if conflict_parts:
            lines.append(
                f"  Smem bank conflicts: {', '.join(conflict_parts)}"
            )

        # ── Branch Efficiency ──
        if self.threads_per_inst_ratio is not None:
            eff = self.threads_per_inst_ratio / 32.0 * 100.0
            lines.append(
                f"  Thread divergence: {self.threads_per_inst_ratio:.1f}/32"
                f" threads/inst ({eff:.0f}% efficiency)"
            )

        # ── Warp Stall Reasons ──
        if self.warp_stalls:
            lines.append("")
            lines.append("--- Warp Stall Reasons (% of issue cycles) ---")
            sorted_stalls = sorted(
                self.warp_stalls.items(), key=lambda x: x[1], reverse=True
            )
            for reason, pct in sorted_stalls[:8]:
                if pct > 0.5:
                    bar = "#" * min(20, int(pct / 5))
                    lines.append(f"  {reason:30s} {pct:5.1f}%  {bar}")

        # ── Data-driven hints ──
        hints = self._generate_hints()
        if hints:
            lines.append("")
            lines.append("--- Optimization Hints ---")
            for h in hints:
                lines.append(f"  -> {h}")

        # ── Delta summary vs previous iteration ──
        if previous is not None:
            delta_lines = self._compute_deltas(previous)
            if delta_lines:
                lines.append("")
                lines.append("--- Delta vs Previous Iteration ---")
                for dl in delta_lines:
                    lines.append(f"  {dl}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Data-driven hint generation
    # ------------------------------------------------------------------

    def _generate_hints(self) -> list[str]:
        hints: list[str] = []

        # Warp-stall-based hints (the single most actionable diagnostic)
        if self.warp_stalls:
            top_stall = max(
                self.warp_stalls.items(), key=lambda x: x[1], default=None
            )
            if top_stall and top_stall[1] > 15:
                reason, pct = top_stall
                coal_note = ""
                if (
                    "long_scoreboard" in reason
                    and self.ld_sectors_per_request
                    and self.ld_sectors_per_request > 1.5
                ):
                    coal_note = (
                        f" + sectors/req={self.ld_sectors_per_request:.1f}"
                        " -> uncoalesced global loads, restructure access"
                        " pattern or use __ldg"
                    )
                _STALL_HINTS = {
                    "long_scoreboard": (
                        "global memory latency"
                        " (waiting for DRAM/L2).{coal} Try shared-memory"
                        " tiling, prefetching, or reducing working set."
                    ),
                    "short_scoreboard": (
                        "shared memory / L1 / MIO dependency."
                        " Check smem layout for bank conflicts,"
                        " double-buffer, or async copies."
                    ),
                    "mio_throttle": (
                        "memory-instruction throttle. Too many outstanding"
                        " memory ops; batch loads, reduce address divergence."
                    ),
                    "barrier": (
                        "__syncthreads barriers. Reduce sync frequency,"
                        " balance work across threads, or overlap compute"
                        " with sync."
                    ),
                    "lg_throttle": (
                        "local/global memory throttle. Too many outstanding"
                        " requests; reduce memory footprint per warp."
                    ),
                    "math_pipe_throttle": (
                        "math pipe saturated. Good compute utilization but"
                        " check ILP; interleave memory ops for latency"
                        " hiding."
                    ),
                    "not_selected": (
                        "scheduler has eligible warps but didn't select."
                        " Usually benign; indicates good latency hiding."
                    ),
                    "wait": (
                        "fixed-latency dependency (e.g. register scoreboard)."
                        " Increase ILP to hide latency."
                    ),
                    "tex_throttle": (
                        "texture pipe throttle. Reduce texture/surface"
                        " reads or switch to global loads."
                    ),
                }
                for stall_key, msg in _STALL_HINTS.items():
                    if stall_key in reason:
                        msg_filled = msg.replace("{coal}", coal_note)
                        hints.append(f"{pct:.0f}% stalls: {msg_filled}")
                        break

        # Coalescing
        if self.ld_sectors_per_request and self.ld_sectors_per_request > 2.0:
            hints.append(
                f"Global load coalescing={self.ld_sectors_per_request:.1f}x"
                f" (ideal=1.0) -> {self.ld_sectors_per_request:.0f}x wasted"
                " DRAM BW. Make per-warp accesses contiguous."
            )
        if self.st_sectors_per_request and self.st_sectors_per_request > 2.0:
            hints.append(
                f"Global store coalescing={self.st_sectors_per_request:.1f}x"
                " -> restructure store pattern."
            )

        # Bank conflicts
        total_conflicts = (self.smem_ld_bank_conflicts or 0) + (
            self.smem_st_bank_conflicts or 0
        )
        if total_conflicts > 0:
            hints.append(
                f"Shared-memory bank conflicts ({total_conflicts:.0f})."
                " Pad arrays (+1 column) or restructure layout."
            )

        # Low occupancy with specific limiter
        if self.occupancy_pct is not None and self.occupancy_pct < 25:
            parts: list[str] = []
            if self.registers_per_thread and self.registers_per_thread > 64:
                parts.append(
                    f"registers ({self.registers_per_thread:.0f}/thread)"
                )
            if self.shared_mem_per_block and self.shared_mem_per_block > 32768:
                parts.append(
                    f"shared memory ({self.shared_mem_per_block:.0f}B/block)"
                )
            if self.block_size and self.block_size < 128:
                parts.append(f"small block ({self.block_size:.0f})")
            limiter = f" limited by {', '.join(parts)}" if parts else ""
            hints.append(
                f"Low occupancy ({self.occupancy_pct:.0f}%){limiter}."
                " Consider __launch_bounds__ or reducing register pressure."
            )

        # Latency-bound diagnosis: low eligible warps + low BW + low compute
        if (
            self.eligible_warps_per_cycle is not None
            and self.eligible_warps_per_cycle < 1.0
            and (self.dram_utilization_pct is None or self.dram_utilization_pct < 50)
            and (
                self.dominant_utilization_pct is None
                or self.dominant_utilization_pct < 50
            )
        ):
            hints.append(
                "Low eligible warps + low BW% + low compute% = LATENCY-BOUND."
                " Need more parallelism (larger grid, more ILP, prefetching)."
            )

        # Cache effectiveness note
        if self.cache_effectiveness is not None and self.cache_effectiveness < 0.3:
            hints.append(
                f"Cache very effective ({self.cache_effectiveness:.2f}x"
                " DRAM/requested). Data well-reused; focus on compute."
            )

        # Bottleneck-level fallback hints
        if self.bottleneck == "memory-bound":
            if not any(
                "coalescing" in h.lower() or "global memory" in h.lower()
                for h in hints
            ):
                hints.append(
                    "Memory-bound: reduce global traffic via shared-memory"
                    " tiling, coalesced access, vectorized loads (float4)."
                )
        elif self.bottleneck == "compute-bound":
            if (
                self.pipe_tensor_pct is not None
                and self.pipe_tensor_pct < 5
                and (self.tensor_ops is None or self.tensor_ops == 0)
            ):
                hints.append(
                    "Compute-bound but NOT using tensor cores. If applicable,"
                    " use FP16/BF16 with torch.matmul or WMMA intrinsics."
                )

        # Thread divergence
        if self.threads_per_inst_ratio is not None and self.threads_per_inst_ratio < 28:
            eff = self.threads_per_inst_ratio / 32.0 * 100.0
            hints.append(
                f"Thread divergence ({eff:.0f}% lane efficiency)."
                " Reduce branching or reorganise data for uniform"
                " control flow."
            )

        return hints

    # ------------------------------------------------------------------
    # Delta computation vs previous iteration
    # ------------------------------------------------------------------

    def _compute_deltas(self, prev: "ProfileSummary") -> list[str]:
        deltas: list[str] = []
        _PAIRS: list[tuple[str, str, str, bool]] = [
            ("GPU time", "gpu_time_us", "\u03bcs", True),
            ("DRAM BW%", "dram_utilization_pct", "%", False),
            ("Compute%", "dominant_utilization_pct", "%", False),
            ("Occupancy", "occupancy_pct", "%", False),
            ("Eligible warps/cyc", "eligible_warps_per_cycle", "", False),
            ("Ld sectors/req", "ld_sectors_per_request", "", True),
            ("L1 hit%", "l1_hit_rate_pct", "%", False),
            ("L2 hit%", "l2_hit_rate_pct", "%", False),
        ]
        for label, attr, unit, lower_better in _PAIRS:
            cur = getattr(self, attr, None)
            prv = getattr(prev, attr, None)
            if cur is not None and prv is not None:
                diff = cur - prv
                if abs(diff) > 0.01:
                    arrow = "\u2193" if diff < 0 else "\u2191"
                    good = (diff < 0) == lower_better
                    marker = "ok" if good else "REGRESSED"
                    deltas.append(
                        f"{label}: {prv:.2f}\u2192{cur:.2f}{unit}"
                        f" ({arrow}{abs(diff):.2f}) {marker}"
                    )
        return deltas


# ---------------------------------------------------------------------------
# Parser: raw nsight dict + kernel breakdown → ProfileSummary
# ---------------------------------------------------------------------------

def parse_nsight_metrics(
    raw_metrics: Dict[str, Optional[float]],
    device_name: str,
    kernel_breakdown: Optional[List[Dict[str, Any]]] = None,
) -> ProfileSummary:
    """
    Convert raw nsight metric values → ProfileSummary.

    Args:
        raw_metrics: metric_name → float | None from profile_with_nsight()
        device_name: torch.cuda.get_device_name() for peak lookup
        kernel_breakdown: optional per-kernel timing from torch.profiler

    Returns:
        ProfileSummary with all computable fields filled in.
    """
    peaks = _get_peak_specs(device_name)
    summary = ProfileSummary(
        peak_bw_gbs=peaks["peak_bw_gbs"],
        peak_fp32_tflops=peaks["peak_fp32_tflops"],
        peak_fp16_tflops=peaks["peak_fp16_tflops"],
        peak_tensor_tflops=peaks["peak_tensor_tflops"],
    )
    summary.raw_metrics = {k: v for k, v in raw_metrics.items() if v is not None}
    if kernel_breakdown:
        summary.kernel_breakdown = kernel_breakdown

    # ── GPU time ──
    gpu_time_ns = _g(raw_metrics, "gpu__time_duration.sum")
    gpu_time_s: Optional[float] = None
    if gpu_time_ns is not None and gpu_time_ns > 0:
        summary.gpu_time_us = gpu_time_ns / 1e3
        gpu_time_s = gpu_time_ns / 1e9

    # ── DRAM Bandwidth (the REAL metric for roofline) ──
    dram_rd_bps = _g(raw_metrics, "dram__bytes_read.sum.per_second")
    dram_wr_bps = _g(raw_metrics, "dram__bytes_write.sum.per_second")
    if dram_rd_bps is not None or dram_wr_bps is not None:
        summary.dram_read_gbs = (dram_rd_bps or 0.0) / 1e9
        summary.dram_write_gbs = (dram_wr_bps or 0.0) / 1e9
        summary.dram_total_gbs = summary.dram_read_gbs + summary.dram_write_gbs
    else:
        dram_rd = _g(raw_metrics, "dram__bytes_read.sum")
        dram_wr = _g(raw_metrics, "dram__bytes_write.sum")
        if (dram_rd is not None or dram_wr is not None) and gpu_time_s:
            total_bytes = (dram_rd or 0.0) + (dram_wr or 0.0)
            summary.dram_total_gbs = total_bytes / gpu_time_s / 1e9
            summary.dram_read_gbs = (dram_rd or 0.0) / gpu_time_s / 1e9
            summary.dram_write_gbs = (dram_wr or 0.0) / gpu_time_s / 1e9

    if summary.dram_total_gbs is not None and summary.peak_bw_gbs:
        summary.dram_utilization_pct = min(
            100.0, 100.0 * summary.dram_total_gbs / summary.peak_bw_gbs
        )

    # ── L1TEX Requested (cache-effectiveness gap) ──
    l1_ld_bps = _g(
        raw_metrics, "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second"
    )
    l1_st_bps = _g(
        raw_metrics, "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second"
    )
    if l1_ld_bps is not None or l1_st_bps is not None:
        summary.l1tex_requested_gbs = (
            (l1_ld_bps or 0.0) + (l1_st_bps or 0.0)
        ) / 1e9
        if summary.dram_total_gbs and summary.l1tex_requested_gbs > 0:
            summary.cache_effectiveness = (
                summary.dram_total_gbs / summary.l1tex_requested_gbs
            )

    # ── FP32 Compute ──
    fadd = _g(raw_metrics, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0)
    fmul = _g(raw_metrics, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0)
    ffma = _g(raw_metrics, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
    summary.fp32_flops = fadd + fmul + 2.0 * ffma
    if summary.fp32_flops > 0 and gpu_time_s:
        summary.fp32_tflops = summary.fp32_flops / gpu_time_s / 1e12
        if summary.peak_fp32_tflops:
            summary.fp32_utilization_pct = min(
                100.0, 100.0 * summary.fp32_tflops / summary.peak_fp32_tflops
            )

    # ── FP16 Compute ──
    hadd = _g(raw_metrics, "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", 0)
    hmul = _g(raw_metrics, "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", 0)
    hfma = _g(raw_metrics, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", 0)
    fp16_flops = hadd + hmul + 2.0 * hfma
    if fp16_flops > 0:
        summary.fp16_flops = fp16_flops
        if gpu_time_s:
            summary.fp16_tflops = fp16_flops / gpu_time_s / 1e12

    # ── Tensor Core ──
    summary.tensor_ops = _g(raw_metrics, "sm__inst_executed_pipe_tensor.sum")
    summary.hmma_ops = _g(raw_metrics, "smsp__inst_executed_pipe_tensor_op_hmma.sum")
    summary.imma_ops = _g(raw_metrics, "smsp__inst_executed_pipe_tensor_op_imma.sum")

    # ── Pipe Utilization ──
    pipe_fma = _g(
        raw_metrics,
        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    )
    pipe_tensor = _g(
        raw_metrics,
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    )
    pipe_alu = _g(
        raw_metrics,
        "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    )
    summary.pipe_fma_pct = pipe_fma
    summary.pipe_tensor_pct = pipe_tensor
    summary.pipe_alu_pct = pipe_alu

    # ── Dominant compute pipe ──
    # Tensor core pipe takes precedence when it has meaningful utilisation
    if pipe_tensor and pipe_tensor > 5:
        summary.dominant_pipe = "tensor-core"
        summary.dominant_utilization_pct = pipe_tensor
    elif pipe_fma and pipe_fma > 5:
        summary.dominant_pipe = "FP32"
        summary.dominant_tflops = summary.fp32_tflops
        summary.dominant_utilization_pct = (
            summary.fp32_utilization_pct if summary.fp32_utilization_pct else pipe_fma
        )
    elif summary.fp16_tflops and summary.fp16_tflops > 0:
        summary.dominant_pipe = "FP16"
        summary.dominant_tflops = summary.fp16_tflops
        if summary.peak_fp16_tflops:
            summary.dominant_utilization_pct = min(
                100.0, 100.0 * summary.fp16_tflops / summary.peak_fp16_tflops
            )
    elif summary.fp32_tflops and summary.fp32_tflops > 0:
        summary.dominant_pipe = "FP32"
        summary.dominant_tflops = summary.fp32_tflops
        summary.dominant_utilization_pct = summary.fp32_utilization_pct

    # ── Arithmetic intensity (FLOPs / DRAM byte) ──
    dram_rd_bytes = _g(raw_metrics, "dram__bytes_read.sum")
    dram_wr_bytes = _g(raw_metrics, "dram__bytes_write.sum")
    total_dram_bytes: Optional[float] = None
    if dram_rd_bytes is not None or dram_wr_bytes is not None:
        total_dram_bytes = (dram_rd_bytes or 0.0) + (dram_wr_bytes or 0.0)
    elif summary.dram_total_gbs is not None and gpu_time_s:
        total_dram_bytes = summary.dram_total_gbs * 1e9 * gpu_time_s

    dominant_flops = max(summary.fp32_flops or 0, fp16_flops)
    if dominant_flops > 0 and total_dram_bytes and total_dram_bytes > 0:
        summary.arithmetic_intensity = dominant_flops / total_dram_bytes

    # ── Ridge point (using dominant pipe's peak) ──
    peak_for_ridge = summary.peak_fp32_tflops
    if summary.dominant_pipe == "tensor-core" and summary.peak_tensor_tflops:
        peak_for_ridge = summary.peak_tensor_tflops
    elif summary.dominant_pipe == "FP16" and summary.peak_fp16_tflops:
        peak_for_ridge = summary.peak_fp16_tflops

    if summary.peak_bw_gbs and peak_for_ridge:
        summary.ridge_point = (peak_for_ridge * 1e12) / (summary.peak_bw_gbs * 1e9)

    # ── Occupancy ──
    occ = _g(raw_metrics, "sm__warps_active.avg.pct_of_peak_sustained_active")
    if occ is not None:
        summary.occupancy_pct = occ
    summary.active_warps_per_cycle = _g(
        raw_metrics, "sm__warps_active.avg.per_cycle_active"
    )
    summary.eligible_warps_per_cycle = _g(
        raw_metrics, "smsp__warps_eligible.avg.per_cycle_active"
    )

    # ── Launch Geometry / Occupancy Limiters ──
    summary.registers_per_thread = _g(raw_metrics, "launch__registers_per_thread")
    summary.shared_mem_per_block = _g(
        raw_metrics, "launch__shared_mem_per_block_allocated"
    )
    summary.block_size = _g(raw_metrics, "launch__block_size")
    summary.grid_size = _g(raw_metrics, "launch__grid_size")
    summary.dynamic_smem_bytes = _g(
        raw_metrics, "launch__shared_mem_per_block_driver"
    )

    # ── Memory Coalescing ──
    summary.ld_sectors_per_request = _g(
        raw_metrics,
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    )
    summary.st_sectors_per_request = _g(
        raw_metrics,
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio",
    )

    # ── Shared Memory Bank Conflicts ──
    summary.smem_ld_bank_conflicts = _g(
        raw_metrics, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
    )
    summary.smem_st_bank_conflicts = _g(
        raw_metrics, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    )

    # ── Cache Hit Rates ──
    summary.l1_hit_rate_pct = _g(raw_metrics, "l1tex__t_sector_hit_rate.pct")
    summary.l2_hit_rate_pct = _g(raw_metrics, "lts__t_sector_hit_rate.pct")

    # ── Branch Efficiency ──
    summary.threads_per_inst_ratio = _g(
        raw_metrics, "smsp__thread_inst_executed_per_inst_executed.ratio"
    )

    # ── Warp Stall Reasons ──
    _STALL_PREFIX = "smsp__average_warps_issue_stalled_"
    _STALL_SUFFIX = "_per_issue_active.pct"
    _STALL_REASONS = [
        "long_scoreboard",
        "short_scoreboard",
        "wait",
        "mio_throttle",
        "barrier",
        "lg_throttle",
        "not_selected",
        "math_pipe_throttle",
        "membar",
        "drain",
        "sleeping",
        "tex_throttle",
    ]
    for reason in _STALL_REASONS:
        key = f"{_STALL_PREFIX}{reason}{_STALL_SUFFIX}"
        val = _g(raw_metrics, key)
        if val is not None and val > 0:
            summary.warp_stalls[reason] = val

    # ── Bottleneck classification ──
    if summary.arithmetic_intensity is not None and summary.ridge_point is not None:
        summary.bottleneck = (
            "memory-bound"
            if summary.arithmetic_intensity < summary.ridge_point
            else "compute-bound"
        )
    elif (
        summary.dram_utilization_pct is not None
        and summary.dominant_utilization_pct is not None
    ):
        summary.bottleneck = (
            "memory-bound"
            if summary.dram_utilization_pct > summary.dominant_utilization_pct
            else "compute-bound"
        )
    else:
        summary.bottleneck = "unknown"

    return summary
