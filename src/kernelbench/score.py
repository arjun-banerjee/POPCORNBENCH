import numpy as np
from typing import Optional

def geometric_mean_speed_ratio_correct_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    prod = np.prod(speed_up)
    n_correct = np.sum(is_correct) # Count number of correct samples

    return prod ** (1 / n_correct) if n_correct > 0 else 0

def geometric_mean_speed_ratio_correct_and_faster_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    speed_up = np.array([x for x in speed_up if x > 1])
    prod = np.prod(speed_up)
    n_correct_and_faster = len(speed_up)

    return prod ** (1 / n_correct_and_faster) if n_correct_and_faster > 0 else 0

def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0


# ─────────────────────────────────────────────────────────────────────────────
# Extended Metric Scoring Functions
# ─────────────────────────────────────────────────────────────────────────────

def memory_efficiency_score(results: list[dict]) -> dict:
    """
    Compute aggregate memory efficiency from eval results.
    Each entry should have a 'memory_stats' dict with 'memory_ratio'.
    Returns geometric mean of memory ratios (< 1 means kernel uses less memory).
    """
    ratios = [r["memory_stats"]["memory_ratio"] for r in results
              if r.get("memory_stats", {}).get("memory_ratio") is not None
              and r["memory_stats"]["memory_ratio"] > 0]
    if not ratios:
        return {"geo_mean_memory_ratio": None, "n": 0}
    arr = np.array(ratios)
    geo_mean = float(np.exp(np.mean(np.log(arr))))
    return {
        "geo_mean_memory_ratio": round(geo_mean, 4),
        "mean_memory_ratio": round(float(np.mean(arr)), 4),
        "min_memory_ratio": round(float(np.min(arr)), 4),
        "max_memory_ratio": round(float(np.max(arr)), 4),
        "n": len(ratios),
    }


def energy_efficiency_score(results: list[dict]) -> dict:
    """
    Compute aggregate energy efficiency from eval results.
    Each entry should have an 'energy_stats' dict with 'energy_ratio'.
    ratio > 1 means the kernel is more energy-efficient than the reference.
    """
    ratios = [r["energy_stats"]["energy_ratio"] for r in results
              if r.get("energy_stats", {}).get("energy_ratio", -1) > 0]
    if not ratios:
        return {"geo_mean_energy_ratio": None, "n": 0}
    arr = np.array(ratios)
    geo_mean = float(np.exp(np.mean(np.log(arr))))
    return {
        "geo_mean_energy_ratio": round(geo_mean, 4),
        "mean_energy_ratio": round(float(np.mean(arr)), 4),
        "n": len(ratios),
    }


def fusion_quality_score(results: list[dict]) -> dict:
    """
    Compute aggregate kernel fusion quality from eval results.
    fusion_ratio > 1 means the custom kernel uses fewer launches than the reference.
    """
    ratios = [r["kernel_launch_stats"]["fusion_ratio"] for r in results
              if r.get("kernel_launch_stats", {}).get("fusion_ratio") is not None
              and r["kernel_launch_stats"]["fusion_ratio"] > 0]
    if not ratios:
        return {"mean_fusion_ratio": None, "n": 0}
    arr = np.array(ratios)
    return {
        "mean_fusion_ratio": round(float(np.mean(arr)), 4),
        "n": len(ratios),
    }


def numerical_precision_score(results: list[dict]) -> dict:
    """
    Compute aggregate numerical precision from eval results.
    Lower is better — shows average and worst-case errors across all problems.
    """
    max_abs_errors = [r["numerical_precision"]["max_abs_error"] for r in results
                      if r.get("numerical_precision", {}).get("max_abs_error") is not None]
    mean_abs_errors = [r["numerical_precision"]["mean_abs_error"] for r in results
                       if r.get("numerical_precision", {}).get("mean_abs_error") is not None]
    if not max_abs_errors:
        return {"avg_max_abs_error": None, "n": 0}
    return {
        "avg_max_abs_error": float(np.mean(max_abs_errors)),
        "worst_max_abs_error": float(np.max(max_abs_errors)),
        "avg_mean_abs_error": float(np.mean(mean_abs_errors)) if mean_abs_errors else None,
        "n": len(max_abs_errors),
    }


def sol_score_aggregate(results: list[dict]) -> dict:
    """
    Compute aggregate SOL (Speed-of-Light) score from eval results.
    SOL score in [0, 1] measures how close to hardware limits the kernel gets.
    """
    scores = [r["sol_stats"]["sol_score"] for r in results
              if r.get("sol_stats", {}).get("sol_score", -1) >= 0]
    if not scores:
        return {"mean_sol_score": None, "n": 0}
    arr = np.array(scores)
    return {
        "mean_sol_score": round(float(np.mean(arr)), 4),
        "median_sol_score": round(float(np.median(arr)), 4),
        "n": len(scores),
    }