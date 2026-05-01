"""
Helpers for hardware-translation workflows (e.g. Level 5 A100 `.cu` → H100 CUDA).

Used by generation scripts and batch eval to resolve source kernels, optional `.txt`
prompt context, and optional Python-wrapped reference kernels for timing baselines.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_repo_path(repo_top: str, p: str | None) -> str | None:
    if not p:
        return None
    p = str(p).strip()
    if not p or p.lower() == "none":
        return None
    return p if os.path.isabs(p) else os.path.join(repo_top, p)


def read_source_kernel_for_problem(src_dir: str, problem_name: str) -> str:
    """
    Load translation source for ``problem_name``.

    Prefers a file named exactly ``problem_name`` under ``src_dir`` (legacy layout:
    copied next to the `.py` benchmark file). Otherwise tries the same basename
    with `.cu` then `.cuh` (Level 5 layout: ``kernels/a100/01_task.cu`` paired with
    ``01_task.py``).
    """
    candidate = os.path.join(src_dir, problem_name)
    if os.path.isfile(candidate):
        with open(candidate, encoding="utf-8") as f:
            return f.read()

    stem = Path(problem_name).stem
    for ext in (".cu", ".cuh"):
        alt = os.path.join(src_dir, stem + ext)
        if os.path.isfile(alt):
            with open(alt, encoding="utf-8") as f:
                return f.read()

    raise FileNotFoundError(
        f"No source kernel for problem '{problem_name}' under {src_dir} "
        f"(tried {candidate!r}, {stem!r}.cu, {stem!r}.cuh)."
    )


def load_hardware_translation_auxiliary_txt(
    repo_top: str,
    *,
    auxiliary_txt_path: str | None = None,
    auxiliary_txt_dir: str | None = None,
    problem_name: str,
) -> str:
    """
    Concatenate optional global `.txt` (single file) with optional per-problem
    ``<stem>.txt`` from ``auxiliary_txt_dir``.
    """
    chunks: list[str] = []

    p_path = resolve_repo_path(repo_top, auxiliary_txt_path)
    if p_path and os.path.isfile(p_path):
        with open(p_path, encoding="utf-8") as f:
            chunks.append(f.read().strip())

    d_path = resolve_repo_path(repo_top, auxiliary_txt_dir)
    if d_path and os.path.isdir(d_path):
        stem = Path(problem_name).stem
        per = os.path.join(d_path, f"{stem}.txt")
        if os.path.isfile(per):
            with open(per, encoding="utf-8") as f:
                chunks.append(f.read().strip())

    return "\n\n".join(c for c in chunks if c).strip()


def resolve_benchmark_reference_kernel_src(
    repo_top: str,
    benchmark_dir: str | None,
    problem_name: str,
) -> str | None:
    """
    Load optional Python source defining ``ModelNew`` for the target GPU reference
    implementation (e.g. wrapping ``kernels/h100/*.cu``).

    Looks for ``<stem>.py`` then the full ``problem_name`` if it ends with `.py`.
    """
    d_path = resolve_repo_path(repo_top, benchmark_dir)
    if not d_path or not os.path.isdir(d_path):
        return None

    stem = Path(problem_name).stem
    for name in (f"{stem}.py", problem_name if problem_name.endswith(".py") else None):
        if not name:
            continue
        path = os.path.join(d_path, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return f.read()

    return None
