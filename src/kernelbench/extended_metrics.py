"""
Extended evaluation metrics for KernelBench.

Provides measurement functions for:
  1. GPU Memory Efficiency
  2. Kernel Launch Count / Fusion Quality (via torch.profiler)
  3. SOL (Speed-of-Light) Score
  4. Energy Efficiency (via NVML)
  5. Roofline / Occupancy (via Nsight or heuristics)
"""

import os

import torch
import torch.nn as nn
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. GPU Memory Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def measure_memory(
    model: nn.Module,
    inputs: list,
    device: torch.device,
) -> dict:
    """
    Measure peak GPU memory allocated during a forward pass.
    Returns dict with peak_memory_bytes.
    """
    torch.cuda.synchronize(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.empty_cache()

    baseline_mem = torch.cuda.memory_allocated(device=device)

    with torch.no_grad():
        _ = model(*inputs)
    torch.cuda.synchronize(device=device)

    peak_mem = torch.cuda.max_memory_allocated(device=device)
    return {
        "peak_memory_bytes": peak_mem,
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "baseline_memory_bytes": baseline_mem,
    }


def compute_memory_stats(
    custom_mem: dict,
    ref_mem: dict,
) -> dict:
    """Combine custom and reference memory measurements into a summary."""
    custom_peak = custom_mem.get("peak_memory_bytes", 0)
    ref_peak = ref_mem.get("peak_memory_bytes", 0)
    ratio = custom_peak / ref_peak if ref_peak > 0 else float("inf")
    return {
        "peak_memory_bytes": custom_peak,
        "peak_memory_mb": custom_mem.get("peak_memory_mb", 0),
        "ref_peak_memory_bytes": ref_peak,
        "ref_peak_memory_mb": ref_mem.get("peak_memory_mb", 0),
        "memory_ratio": round(ratio, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Kernel Launch Count / Fusion Quality
# ─────────────────────────────────────────────────────────────────────────────

def measure_kernel_launches(
    model: nn.Module,
    inputs: list,
    device: torch.device,
) -> dict:
    """
    Use torch.profiler to count CUDA kernel launches.
    Returns dict with num_kernels and per-kernel breakdown.
    """
    try:
        import torch.autograd.profiler as _profiler

        with torch.no_grad():
            with _profiler.profile(use_cuda=True) as prof:
                _ = model(*inputs)
            torch.cuda.synchronize(device)

        events = prof.function_events
        cuda_events = []
        for e in events:
            if e.cuda_time_total > 0 and not e.key.startswith("cudaDevice"):
                cuda_events.append({
                    "name": e.key,
                    "cuda_time_us": e.cuda_time_total,
                    "calls": e.count,
                })
        cuda_events.sort(key=lambda x: x["cuda_time_us"], reverse=True)

        return {
            "num_kernels": len(cuda_events),
            "kernel_breakdown": cuda_events[:10],
            "total_cuda_time_us": sum(e["cuda_time_us"] for e in cuda_events),
        }
    except Exception as e:
        return {"num_kernels": -1, "error": str(e)}


def compute_kernel_launch_stats(
    custom_launches: dict,
    ref_launches: dict,
) -> dict:
    """Combine custom and reference kernel launch data."""
    custom_n = custom_launches.get("num_kernels", -1)
    ref_n = ref_launches.get("num_kernels", -1)
    fusion_ratio = ref_n / custom_n if custom_n > 0 else 0.0
    return {
        "num_kernels": custom_n,
        "ref_num_kernels": ref_n,
        "fusion_ratio": round(fusion_ratio, 4),
        "kernel_breakdown": custom_launches.get("kernel_breakdown", []),
        "total_cuda_time_us": custom_launches.get("total_cuda_time_us", 0),
        "ref_total_cuda_time_us": ref_launches.get("total_cuda_time_us", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. SOL (Speed-of-Light) Score — via Nsight hardware counters
# ─────────────────────────────────────────────────────────────────────────────

_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts", "_profile_worker.py",
)


def profile_kernel_with_nsight(
    custom_model_src: str,
    ref_model_src: str,
    device: torch.device,
    backend: str = "cuda",
    precision: str = "fp32",
    build_dir: Optional[str] = None,
    verbose: bool = False,
    timeout: int = 300,
) -> Optional[dict]:
    """
    Profile a kernel via Nsight Compute in a **subprocess** and return a
    parsed ProfileSummary as a plain dict.  Returns None if Nsight is
    not available or profiling fails.

    Nsight relaunches the application under ncu, so this MUST run in a
    separate process (scripts/_profile_worker.py) to avoid restarting
    the caller.
    """
    import json
    import subprocess
    import sys
    import tempfile as _tempfile

    try:
        from kernelbench.profile import NSIGHT_AVAILABLE, check_ncu_available
        if not NSIGHT_AVAILABLE or not check_ncu_available():
            return None

        from kernelbench.agent.nsight_parser import ROOFLINE_METRICS, parse_nsight_metrics

        request = {
            "custom_model_src": custom_model_src,
            "ref_model_src": ref_model_src,
            "metrics": ROOFLINE_METRICS,
            "num_trials": 1,
            "seed": 42,
            "device_index": device.index if device.index is not None else 0,
            "backend": backend,
            "precision": precision,
            "build_dir": build_dir,
            "verbose": verbose,
        }

        with _tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(request, tmp)
            req_path = tmp.name

        proc = subprocess.run(
            [sys.executable, _WORKER_SCRIPT, req_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        try:
            os.unlink(req_path)
        except OSError:
            pass

        if proc.returncode != 0:
            if verbose:
                stderr_tail = (proc.stderr or "").strip()[-500:]
                print(f"[Nsight] Worker exited with code {proc.returncode}: {stderr_tail}")
            return None

        raw_output = json.loads(proc.stdout.strip().splitlines()[-1])
        if "error" in raw_output:
            if verbose:
                print(f"[Nsight] Worker error: {raw_output['error']}")
            return None

        kernel_breakdown = raw_output.pop("_kernel_breakdown", [])
        device_name = torch.cuda.get_device_name(device)
        summary = parse_nsight_metrics(raw_output, device_name, kernel_breakdown=kernel_breakdown)

        return {
            "gpu_time_us": summary.gpu_time_us,
            "dram_total_gbs": summary.dram_total_gbs,
            "dram_read_gbs": summary.dram_read_gbs,
            "dram_write_gbs": summary.dram_write_gbs,
            "dram_utilization_pct": summary.dram_utilization_pct,
            "fp32_tflops": summary.fp32_tflops,
            "fp32_utilization_pct": summary.fp32_utilization_pct,
            "fp16_tflops": summary.fp16_tflops,
            "arithmetic_intensity": summary.arithmetic_intensity,
            "ridge_point": summary.ridge_point,
            "bottleneck": summary.bottleneck,
            "occupancy_pct": summary.occupancy_pct,
            "dominant_pipe": summary.dominant_pipe,
            "dominant_utilization_pct": summary.dominant_utilization_pct,
            "peak_bw_gbs": summary.peak_bw_gbs,
            "peak_fp32_tflops": summary.peak_fp32_tflops,
            "peak_fp16_tflops": summary.peak_fp16_tflops,
            "pipe_fma_pct": summary.pipe_fma_pct,
            "pipe_tensor_pct": summary.pipe_tensor_pct,
            "pipe_alu_pct": summary.pipe_alu_pct,
            "registers_per_thread": summary.registers_per_thread,
            "shared_mem_per_block": summary.shared_mem_per_block,
            "block_size": summary.block_size,
            "l1_hit_rate_pct": summary.l1_hit_rate_pct,
            "l2_hit_rate_pct": summary.l2_hit_rate_pct,
            "ld_sectors_per_request": summary.ld_sectors_per_request,
            "st_sectors_per_request": summary.st_sectors_per_request,
            "warp_stalls": dict(summary.warp_stalls) if summary.warp_stalls else {},
        }
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"[Nsight] Profiling timed out ({timeout}s)")
        return None
    except Exception as e:
        if verbose:
            print(f"[Nsight] Profiling failed: {e}")
        return None


def compute_sol_score_from_nsight(nsight_profile: dict) -> dict:
    """
    Compute SOL from real Nsight hardware counters.

    SOL = max(dram_utilization_pct, compute_utilization_pct) / 100.
    This measures how close the kernel gets to the hardware ceiling.
    """
    dram_util = nsight_profile.get("dram_utilization_pct")
    compute_util = nsight_profile.get("dominant_utilization_pct")

    if dram_util is None and compute_util is None:
        return {"sol_score": -1, "source": "nsight", "note": "utilization metrics unavailable"}

    # Clip each utilization to [0, 100]. Nsight occasionally reports values
    # slightly above 100% for `sm__throughput.avg.pct_of_peak_sustained_active`
    # due to elastic-pipeline measurement artifacts, which would otherwise
    # produce SOL > 1. The SOL convention (achieved / peak) is bounded in [0, 1].
    dram_pct = max(0.0, min(100.0, dram_util or 0.0))
    compute_pct = max(0.0, min(100.0, compute_util or 0.0))
    sol_score = max(dram_pct, compute_pct) / 100.0
    # Belt-and-suspenders clamp on the final ratio.
    sol_score = max(0.0, min(1.0, sol_score))

    return {
        "sol_score": round(sol_score, 4),
        "source": "nsight_hardware_counters",
        "dram_utilization_pct": round(dram_pct, 2),
        "compute_utilization_pct": round(compute_pct, 2),
        "bottleneck": nsight_profile.get("bottleneck", "unknown"),
        "arithmetic_intensity": nsight_profile.get("arithmetic_intensity"),
        "ridge_point": nsight_profile.get("ridge_point"),
        "dominant_pipe": nsight_profile.get("dominant_pipe"),
        "peak_bw_gbs": nsight_profile.get("peak_bw_gbs"),
        "peak_fp32_tflops": nsight_profile.get("peak_fp32_tflops"),
        "hardware": "see peak_bw_gbs/peak_fp32_tflops",
    }


def compute_sol_score_heuristic(
    runtime_ms: float,
    ref_runtime_ms: float,
    device: torch.device,
) -> dict:
    """Fallback SOL estimate when Nsight is not available."""
    if runtime_ms <= 0 or ref_runtime_ms <= 0:
        return {"sol_score": -1, "source": "heuristic", "note": "invalid runtimes"}

    speedup = ref_runtime_ms / runtime_ms
    sol_score = min(speedup / 10.0, 1.0)
    return {
        "sol_score": round(sol_score, 4),
        "source": "heuristic_from_speedup",
        "bottleneck": "unknown",
        "note": "Install nsight-python + ncu for hardware-counter SOL",
        "hardware": torch.cuda.get_device_name(device),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Energy Efficiency (NVML)
# ─────────────────────────────────────────────────────────────────────────────

_NVML_AVAILABLE = False
try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    pass


def measure_energy(
    model: nn.Module,
    inputs: list,
    device: torch.device,
    num_trials: int = 50,
) -> dict:
    """
    Measure energy consumption by polling NVML power during repeated kernel execution.
    Returns energy in millijoules and average power in watts.

    NVML counters update at ~100ms intervals, so we run many iterations
    to get a meaningful measurement.
    """
    if not _NVML_AVAILABLE:
        return {"energy_mj": -1, "avg_power_w": -1, "note": "pynvml not installed"}

    try:
        pynvml.nvmlInit()
        device_idx = device.index if device.index is not None else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        torch.cuda.synchronize(device=device)

        for _ in range(3):
            with torch.no_grad():
                _ = model(*inputs)
        torch.cuda.synchronize(device=device)

        energy_before = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        import time
        t_start = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_trials):
                _ = model(*inputs)
        torch.cuda.synchronize(device=device)

        t_end = time.perf_counter()
        energy_after = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

        energy_mj = energy_after - energy_before
        elapsed_s = t_end - t_start
        avg_power_w = (energy_mj / 1000.0) / elapsed_s if elapsed_s > 0 else 0
        energy_per_run_mj = energy_mj / num_trials if num_trials > 0 else 0

        pynvml.nvmlShutdown()

        return {
            "total_energy_mj": round(energy_mj, 2),
            "energy_per_run_mj": round(energy_per_run_mj, 4),
            "avg_power_w": round(avg_power_w, 2),
            "elapsed_s": round(elapsed_s, 4),
            "num_trials": num_trials,
        }
    except Exception as e:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return {"energy_mj": -1, "avg_power_w": -1, "error": str(e)}


def compute_energy_stats(
    custom_energy: dict,
    ref_energy: dict,
) -> dict:
    """Combine custom and reference energy measurements."""
    custom_e = custom_energy.get("energy_per_run_mj", -1)
    ref_e = ref_energy.get("energy_per_run_mj", -1)
    ratio = ref_e / custom_e if custom_e > 0 and ref_e > 0 else -1
    return {
        "energy_per_run_mj": custom_e,
        "ref_energy_per_run_mj": ref_e,
        "energy_ratio": round(ratio, 4) if ratio > 0 else -1,
        "avg_power_w": custom_energy.get("avg_power_w", -1),
        "ref_avg_power_w": ref_energy.get("avg_power_w", -1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Roofline / Occupancy — from Nsight hardware counters
# ─────────────────────────────────────────────────────────────────────────────

def compute_roofline_stats_from_nsight(nsight_profile: dict) -> dict:
    """
    Build roofline / occupancy stats from real Nsight hardware counters.

    Includes: achieved bandwidth, achieved FLOP/s, arithmetic intensity,
    ridge point, occupancy, pipe utilization, coalescing quality,
    cache hit rates, register pressure, and warp stall data.
    """
    return {
        "source": "nsight_hardware_counters",
        # Bandwidth
        "dram_bandwidth_gbs": nsight_profile.get("dram_total_gbs"),
        "dram_read_gbs": nsight_profile.get("dram_read_gbs"),
        "dram_write_gbs": nsight_profile.get("dram_write_gbs"),
        "dram_utilization_pct": nsight_profile.get("dram_utilization_pct"),
        # Compute throughput
        "fp32_tflops": nsight_profile.get("fp32_tflops"),
        "fp32_utilization_pct": nsight_profile.get("fp32_utilization_pct"),
        "fp16_tflops": nsight_profile.get("fp16_tflops"),
        # Roofline position
        "arithmetic_intensity": nsight_profile.get("arithmetic_intensity"),
        "ridge_point": nsight_profile.get("ridge_point"),
        "bottleneck": nsight_profile.get("bottleneck"),
        # Occupancy
        "occupancy_pct": nsight_profile.get("occupancy_pct"),
        # Pipe utilization
        "dominant_pipe": nsight_profile.get("dominant_pipe"),
        "dominant_utilization_pct": nsight_profile.get("dominant_utilization_pct"),
        "pipe_fma_pct": nsight_profile.get("pipe_fma_pct"),
        "pipe_tensor_pct": nsight_profile.get("pipe_tensor_pct"),
        "pipe_alu_pct": nsight_profile.get("pipe_alu_pct"),
        # Memory coalescing (4.0 = perfectly coalesced)
        "ld_sectors_per_request": nsight_profile.get("ld_sectors_per_request"),
        "st_sectors_per_request": nsight_profile.get("st_sectors_per_request"),
        # Cache
        "l1_hit_rate_pct": nsight_profile.get("l1_hit_rate_pct"),
        "l2_hit_rate_pct": nsight_profile.get("l2_hit_rate_pct"),
        # Launch geometry / occupancy limiters
        "registers_per_thread": nsight_profile.get("registers_per_thread"),
        "shared_mem_per_block": nsight_profile.get("shared_mem_per_block"),
        "block_size": nsight_profile.get("block_size"),
        # Peak hardware specs
        "peak_bw_gbs": nsight_profile.get("peak_bw_gbs"),
        "peak_fp32_tflops": nsight_profile.get("peak_fp32_tflops"),
        "peak_fp16_tflops": nsight_profile.get("peak_fp16_tflops"),
        # Warp stalls
        "warp_stalls": nsight_profile.get("warp_stalls", {}),
    }


def compute_roofline_stats_heuristic(
    runtime_ms: float,
    device: torch.device,
) -> dict:
    """Fallback roofline stats when Nsight is not available."""
    name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "source": "heuristic",
        "hardware": name,
        "sm_count": props.multi_processor_count,
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
        "note": "Install nsight-python + ncu for real roofline from hardware counters",
    }
