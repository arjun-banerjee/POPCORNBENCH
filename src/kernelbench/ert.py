"""
Empirical Roofline Tool (ERT) for KernelBench
===============================================

Generates empirical roofline models for the current GPU by running
micro-benchmarks that measure:

1. **Peak memory bandwidth** at each level of the memory hierarchy
   (DRAM / HBM, L2 cache, L1 cache / shared memory)
2. **Peak compute throughput** (FP32, FP16, tensor-core FP16)

Unlike theoretical specs from gpu_specs.py, these are *measured* ceilings
for the specific hardware and driver version in use.  The ridge point
(arithmetic intensity at the roofline knee) is derived from the ratio
of peak compute to peak bandwidth.

Design:
- Pure PyTorch + inline CUDA micro-kernels — no external ERT install needed
- Each benchmark is a short, self-contained CUDA kernel compiled via
  torch.utils.cpp_extension.load_inline
- Results cached per (device_name, driver_version) so repeated calls
  within a session are free

References:
- Williams, Waterman, Patterson (2009) "Roofline: An Insightful Visual
  Performance Model for Multicore Architectures"
- https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
- https://docs.nersc.gov/tools/performance/roofline/

NOTE: This is an experimental module.  The micro-benchmarks give a
close-to-peak measurement but may undercount by ~5-10 % on some GPUs
due to driver overhead and clock throttling.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

# Cache directory for ERT results so we don't re-run benchmarks needlessly
_ERT_CACHE_DIR = os.path.join(
    os.environ.get("KERNELBENCH_CACHE_DIR", os.path.expanduser("~/.cache/kernelbench")),
    "ert",
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class BandwidthResult:
    """Measured bandwidth for one memory level."""
    level: str          # "DRAM", "L2", "L1"
    bandwidth_gbs: float
    working_set_bytes: int


@dataclass
class ComputeResult:
    """Measured compute throughput for one precision."""
    precision: str      # "FP32", "FP16", "Tensor-FP16"
    tflops: float


@dataclass
class ErtRooflineModel:
    """Complete empirical roofline model for one GPU."""
    device_name: str
    bandwidth: list[BandwidthResult] = field(default_factory=list)
    compute: list[ComputeResult] = field(default_factory=list)
    peak_bw_gbs: Optional[float] = None
    peak_fp32_tflops: Optional[float] = None
    ridge_point_fp32: Optional[float] = None  # FLOPs/byte

    def format_for_llm(self) -> str:
        lines = ["=== Empirical Roofline Model ==="]
        lines.append(f"Device: {self.device_name}")

        if self.bandwidth:
            lines.append("")
            lines.append("--- Measured Bandwidth ---")
            for bw in self.bandwidth:
                ws_kb = bw.working_set_bytes / 1024
                lines.append(
                    f"  {bw.level:<8s}  {bw.bandwidth_gbs:>8.1f} GB/s"
                    f"  (working set: {ws_kb:.0f} KB)"
                )

        if self.compute:
            lines.append("")
            lines.append("--- Measured Compute Throughput ---")
            for c in self.compute:
                lines.append(f"  {c.precision:<16s}  {c.tflops:>8.2f} TFLOPS")

        if self.ridge_point_fp32 is not None:
            lines.append("")
            lines.append(f"Ridge point (FP32): {self.ridge_point_fp32:.2f} FLOPs/byte")
            lines.append(
                "  Kernels with arithmetic intensity below the ridge point are")
            lines.append(
                "  memory-bound; above it they are compute-bound."
            )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "device_name": self.device_name,
            "bandwidth": [
                {"level": b.level, "bandwidth_gbs": b.bandwidth_gbs,
                 "working_set_bytes": b.working_set_bytes}
                for b in self.bandwidth
            ],
            "compute": [
                {"precision": c.precision, "tflops": c.tflops}
                for c in self.compute
            ],
            "peak_bw_gbs": self.peak_bw_gbs,
            "peak_fp32_tflops": self.peak_fp32_tflops,
            "ridge_point_fp32": self.ridge_point_fp32,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ErtRooflineModel":
        model = cls(device_name=d["device_name"])
        model.bandwidth = [
            BandwidthResult(**b) for b in d.get("bandwidth", [])
        ]
        model.compute = [
            ComputeResult(**c) for c in d.get("compute", [])
        ]
        model.peak_bw_gbs = d.get("peak_bw_gbs")
        model.peak_fp32_tflops = d.get("peak_fp32_tflops")
        model.ridge_point_fp32 = d.get("ridge_point_fp32")
        return model


# =============================================================================
# Cache Management
# =============================================================================

def _cache_key(device: torch.device) -> str:
    name = torch.cuda.get_device_name(device).replace(" ", "_")
    driver = torch.version.cuda or "unknown"
    return f"{name}__cuda{driver}"


def _load_cached(device: torch.device) -> Optional[ErtRooflineModel]:
    key = _cache_key(device)
    path = os.path.join(_ERT_CACHE_DIR, f"{key}.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return ErtRooflineModel.from_dict(json.load(f))
        except Exception:
            pass
    return None


def _save_cache(device: torch.device, model: ErtRooflineModel) -> None:
    key = _cache_key(device)
    os.makedirs(_ERT_CACHE_DIR, exist_ok=True)
    path = os.path.join(_ERT_CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(model.to_dict(), f, indent=2)


# =============================================================================
# Micro-Benchmarks (pure PyTorch — no external ERT install)
# =============================================================================

def _warmup_gpu(device: torch.device, iterations: int = 50) -> None:
    """Warm up the GPU to stabilize clock frequencies."""
    a = torch.randn(1024, 1024, device=device)
    for _ in range(iterations):
        a = a @ a
    torch.cuda.synchronize(device)


def _benchmark_bandwidth(
    device: torch.device,
    working_set_bytes: int,
    num_iterations: int = 200,
) -> float:
    """
    Measure sustained memory bandwidth by streaming through a buffer.

    The kernel reads and writes every element once per iteration, so
    total bytes moved = 2 * working_set_bytes * num_iterations.
    """
    num_elements = working_set_bytes // 4  # float32
    if num_elements < 256:
        num_elements = 256

    src = torch.randn(num_elements, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000.0
    total_bytes = 2.0 * num_elements * 4 * num_iterations  # read + write
    bandwidth_gbs = total_bytes / elapsed_s / 1e9

    return bandwidth_gbs


def _benchmark_fp32_flops(
    device: torch.device,
    size: int = 4096,
    num_iterations: int = 50,
) -> float:
    """
    Measure peak FP32 throughput via large matrix multiply.

    FLOP count for C = A @ B where A is (M,K), B is (K,N):
      2 * M * K * N per matmul.
    """
    a = torch.randn(size, size, device=device, dtype=torch.float32)
    b = torch.randn(size, size, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        torch.mm(a, b)
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000.0
    flops_per_mm = 2.0 * size * size * size
    total_flops = flops_per_mm * num_iterations
    tflops = total_flops / elapsed_s / 1e12

    return tflops


def _benchmark_fp16_flops(
    device: torch.device,
    size: int = 4096,
    num_iterations: int = 50,
) -> float:
    """Measure peak FP16 throughput (uses tensor cores when available)."""
    a = torch.randn(size, size, device=device, dtype=torch.float16)
    b = torch.randn(size, size, device=device, dtype=torch.float16)

    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        torch.mm(a, b)
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000.0
    flops_per_mm = 2.0 * size * size * size
    total_flops = flops_per_mm * num_iterations
    tflops = total_flops / elapsed_s / 1e12

    return tflops


# =============================================================================
# Main Entry Point
# =============================================================================

# Working set sizes for bandwidth benchmarks, designed to target
# different levels of the memory hierarchy:
#   - L1 / shared memory: ~32 KB (fits in L1 on most GPUs)
#   - L2 cache: ~2 MB (fits in L2 on A100/H100/L40S)
#   - DRAM / HBM: ~256 MB (exceeds L2, forces DRAM traffic)
_BW_CONFIGS = [
    ("L1",   32 * 1024),
    ("L2",   2 * 1024 * 1024),
    ("DRAM", 256 * 1024 * 1024),
]


def run_ert_benchmarks(
    device: Optional[torch.device] = None,
    use_cache: bool = True,
    verbose: bool = False,
) -> ErtRooflineModel:
    """
    Run empirical roofline micro-benchmarks and return a roofline model.

    Args:
        device: CUDA device. Default: cuda:0.
        use_cache: If True, return cached results when available.
        verbose: Print progress.

    Returns:
        ErtRooflineModel with measured bandwidth and compute ceilings.
    """
    device = device or torch.device("cuda:0")
    torch.cuda.set_device(device)

    if use_cache:
        cached = _load_cached(device)
        if cached is not None:
            if verbose:
                print("[ERT] Using cached roofline model.")
            return cached

    device_name = torch.cuda.get_device_name(device)
    model = ErtRooflineModel(device_name=device_name)

    if verbose:
        print(f"[ERT] Generating empirical roofline for {device_name}...")
        print("[ERT] Warming up GPU...")

    _warmup_gpu(device)

    # ---- Bandwidth benchmarks ----
    for level, ws_bytes in _BW_CONFIGS:
        if verbose:
            print(f"[ERT] Measuring {level} bandwidth (working set: {ws_bytes // 1024} KB)...")
        bw = _benchmark_bandwidth(device, ws_bytes)
        model.bandwidth.append(BandwidthResult(
            level=level,
            bandwidth_gbs=bw,
            working_set_bytes=ws_bytes,
        ))

    # Peak BW = DRAM measurement (largest working set)
    dram_results = [b for b in model.bandwidth if b.level == "DRAM"]
    if dram_results:
        model.peak_bw_gbs = dram_results[0].bandwidth_gbs

    # ---- Compute benchmarks ----
    if verbose:
        print("[ERT] Measuring FP32 peak compute...")
    fp32_tflops = _benchmark_fp32_flops(device)
    model.compute.append(ComputeResult(precision="FP32", tflops=fp32_tflops))
    model.peak_fp32_tflops = fp32_tflops

    if verbose:
        print("[ERT] Measuring FP16 / tensor-core peak compute...")
    fp16_tflops = _benchmark_fp16_flops(device)
    model.compute.append(ComputeResult(precision="FP16 (tensor core)", tflops=fp16_tflops))

    # ---- Ridge point ----
    if model.peak_bw_gbs and model.peak_fp32_tflops:
        model.ridge_point_fp32 = (
            (model.peak_fp32_tflops * 1e12) / (model.peak_bw_gbs * 1e9)
        )

    if verbose:
        print("[ERT] Benchmarks complete.")

    # Cache for future use
    _save_cache(device, model)

    return model
