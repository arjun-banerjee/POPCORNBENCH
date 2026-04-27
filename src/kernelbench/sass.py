"""
SASS / PTX Disassembly Module for KernelBench
===============================================

Disassemble compiled CUDA binaries to inspect PTX and SASS (native GPU assembly)
using NVIDIA's cuobjdump and nvdisasm tools.

Key Features:
- Disassemble compiled PyTorch CUDA extensions (.so) to SASS and PTX
- Extract per-kernel resource usage (registers, shared/local memory, stack)
- Analyze register liveness via nvdisasm
- Support for disassembling KernelBench ModelNew kernels end-to-end

Requirements:
- NVIDIA cuobjdump (ships with CUDA Toolkit)
- NVIDIA nvdisasm  (ships with CUDA Toolkit)
- The compiled .so must contain embedded cubin/fatbin sections

References:
- https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
"""

from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from shutil import which
from typing import Optional

import torch


# =============================================================================
# Availability Checks
# =============================================================================

def check_cuobjdump_available() -> bool:
    return which("cuobjdump") is not None


def check_nvdisasm_available() -> bool:
    return which("nvdisasm") is not None


# =============================================================================
# Locating Compiled .so from a PyTorch Extension
# =============================================================================

def _find_extension_so(build_dir: Optional[str], extension_name: Optional[str] = None) -> Optional[str]:
    """
    Find the compiled .so file for a PyTorch inline CUDA extension.

    Search order:
    1. build_dir (explicit TORCH_EXTENSIONS_DIR override)
    2. Default torch extensions cache

    Returns the path to the .so, or None.
    """
    search_dirs = []
    if build_dir and os.path.isdir(build_dir):
        search_dirs.append(build_dir)

    default_ext_dir = os.environ.get(
        "TORCH_EXTENSIONS_DIR",
        os.path.join(tempfile.gettempdir(), "torch_extensions"),
    )
    if os.path.isdir(default_ext_dir):
        search_dirs.append(default_ext_dir)

    for base in search_dirs:
        pattern = os.path.join(base, "**", "*.so")
        candidates = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)
        if extension_name:
            for c in candidates:
                if extension_name in os.path.basename(c):
                    return c
        if candidates:
            return candidates[0]

    return None


# =============================================================================
# Raw CLI Wrappers
# =============================================================================

def _run_cmd(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def cuobjdump_sass(so_path: str) -> str:
    """Run ``cuobjdump -sass <so_path>`` and return stdout."""
    r = _run_cmd(["cuobjdump", "-sass", so_path])
    if r.returncode != 0:
        raise RuntimeError(f"cuobjdump -sass failed (rc={r.returncode}): {r.stderr}")
    return r.stdout


def cuobjdump_ptx(so_path: str) -> str:
    """Run ``cuobjdump -ptx <so_path>`` and return stdout."""
    r = _run_cmd(["cuobjdump", "-ptx", so_path])
    if r.returncode != 0:
        raise RuntimeError(f"cuobjdump -ptx failed (rc={r.returncode}): {r.stderr}")
    return r.stdout


def cuobjdump_res_usage(so_path: str) -> str:
    """Run ``cuobjdump -res-usage <so_path>`` for per-kernel register/memory usage."""
    r = _run_cmd(["cuobjdump", "-res-usage", so_path])
    if r.returncode != 0:
        raise RuntimeError(f"cuobjdump -res-usage failed (rc={r.returncode}): {r.stderr}")
    return r.stdout


def cuobjdump_extract_elf(so_path: str, output_dir: str) -> list[str]:
    """
    Extract embedded cubin ELF sections from a .so into individual files.

    Returns list of extracted .cubin file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    r = _run_cmd(["cuobjdump", "-xelf", "all", so_path], timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"cuobjdump -xelf failed (rc={r.returncode}): {r.stderr}")

    cubins = sorted(glob.glob(os.path.join(output_dir, "*.cubin")))
    if not cubins:
        cwd_cubins = sorted(glob.glob("*.cubin"))
        for cb in cwd_cubins:
            dest = os.path.join(output_dir, os.path.basename(cb))
            os.rename(cb, dest)
            cubins.append(dest)

    return cubins


def nvdisasm_sass(cubin_path: str) -> str:
    """Run ``nvdisasm <cubin>`` and return the SASS disassembly."""
    r = _run_cmd(["nvdisasm", cubin_path])
    if r.returncode != 0:
        raise RuntimeError(f"nvdisasm failed (rc={r.returncode}): {r.stderr}")
    return r.stdout


def nvdisasm_life_range(cubin_path: str, mode: str = "count") -> str:
    """
    Run ``nvdisasm -plr -lrm <mode> <cubin>`` for register liveness analysis.

    Args:
        mode: "count" (number of live registers per instruction),
              "wide" (one column per register, spaced), or
              "narrow" (compact one-char columns).
    """
    r = _run_cmd(["nvdisasm", "-plr", "-lrm", mode, cubin_path])
    if r.returncode != 0:
        raise RuntimeError(f"nvdisasm -plr failed (rc={r.returncode}): {r.stderr}")
    return r.stdout


# =============================================================================
# High-Level Disassembly API
# =============================================================================

@dataclass
class DisassemblyResult:
    """All disassembly artifacts for one compiled CUDA extension."""
    so_path: str
    sass: Optional[str] = None
    ptx: Optional[str] = None
    res_usage: Optional[str] = None
    nvdisasm_sass: Optional[str] = None
    register_life_range: Optional[str] = None
    errors: list[str] = field(default_factory=list)


def disassemble_so(
    so_path: str,
    include_ptx: bool = True,
    include_nvdisasm: bool = True,
    include_life_range: bool = True,
) -> DisassemblyResult:
    """
    Disassemble a compiled .so file and collect all available artifacts.

    Gracefully degrades: each section is attempted independently and failures
    are collected in ``result.errors`` rather than raised.
    """
    result = DisassemblyResult(so_path=so_path)

    # cuobjdump -sass
    try:
        result.sass = cuobjdump_sass(so_path)
    except Exception as e:
        result.errors.append(f"cuobjdump -sass: {e}")

    # cuobjdump -ptx
    if include_ptx:
        try:
            result.ptx = cuobjdump_ptx(so_path)
        except Exception as e:
            result.errors.append(f"cuobjdump -ptx: {e}")

    # cuobjdump -res-usage
    try:
        result.res_usage = cuobjdump_res_usage(so_path)
    except Exception as e:
        result.errors.append(f"cuobjdump -res-usage: {e}")

    # nvdisasm (requires extracting .cubin first)
    if include_nvdisasm or include_life_range:
        try:
            with tempfile.TemporaryDirectory(prefix="kb_nvdisasm_") as tmpdir:
                cubins = cuobjdump_extract_elf(so_path, tmpdir)
                if cubins:
                    cubin = cubins[0]
                    if include_nvdisasm:
                        try:
                            result.nvdisasm_sass = nvdisasm_sass(cubin)
                        except Exception as e:
                            result.errors.append(f"nvdisasm: {e}")
                    if include_life_range:
                        try:
                            result.register_life_range = nvdisasm_life_range(cubin, mode="count")
                        except Exception as e:
                            result.errors.append(f"nvdisasm -plr: {e}")
                else:
                    result.errors.append("cuobjdump -xelf: no .cubin sections found")
        except Exception as e:
            result.errors.append(f"cubin extraction: {e}")

    return result


# =============================================================================
# KernelBench Model Disassembly
# =============================================================================

def disassemble_kernelbench_model(
    custom_model_src: str,
    ref_model_src: Optional[str] = None,
    device: Optional[torch.device] = None,
    backend: str = "cuda",
    precision: torch.dtype = torch.float32,
    build_dir: Optional[str] = None,
    include_ptx: bool = True,
    include_nvdisasm: bool = True,
    include_life_range: bool = True,
    verbose: bool = False,
) -> DisassemblyResult:
    """
    Compile a KernelBench model and disassemble the resulting binary.

    This handles the full lifecycle:
    1. Compile the custom model via load_custom_model / load_inline
    2. Locate the resulting .so file
    3. Run cuobjdump and nvdisasm on it
    4. Clean up

    Args:
        custom_model_src: Python source code with ModelNew.
        ref_model_src: Optional reference model source (for get_init_inputs).
        device: CUDA device. Default: cuda:0.
        backend: Compilation backend.
        precision: torch.dtype.
        build_dir: Explicit build directory for compiled extensions.
        include_ptx: Also dump PTX intermediate representation.
        include_nvdisasm: Also run nvdisasm on extracted .cubin.
        include_life_range: Also run nvdisasm register liveness analysis.
        verbose: Print progress.

    Returns:
        DisassemblyResult with all collected artifacts.
    """
    from kernelbench.eval import (
        load_custom_model,
        load_custom_model_with_tempfile,
        graceful_eval_cleanup,
    )

    device = device or torch.device("cuda:0")
    torch.cuda.set_device(device)

    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    context: dict = {}
    tempfile_handle = None

    use_temp_build = build_dir is None
    if use_temp_build:
        build_dir = tempfile.mkdtemp(prefix="kb_sass_build_")
        os.environ["TORCH_EXTENSIONS_DIR"] = build_dir

    try:
        if verbose:
            print("[SASS] Compiling custom model...")

        backend_lower = backend.lower()
        if backend_lower in ("triton", "tilelang", "cute"):
            ModelNew, tempfile_handle = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            ModelNew = load_custom_model(custom_model_src, context, build_dir)

        torch.cuda.synchronize(device=device)

        if verbose:
            print("[SASS] Compilation complete, locating .so...")

        so_path = _find_extension_so(build_dir)
        if so_path is None:
            return DisassemblyResult(
                so_path="<not found>",
                errors=["Could not locate compiled .so file after compilation."],
            )

        if verbose:
            print(f"[SASS] Found .so at: {so_path}")
            print("[SASS] Running disassembly...")

        result = disassemble_so(
            so_path,
            include_ptx=include_ptx,
            include_nvdisasm=include_nvdisasm,
            include_life_range=include_life_range,
        )

        if verbose:
            print("[SASS] Disassembly complete.")

        return result

    finally:
        graceful_eval_cleanup(context, device, tempfile_handle)
        if use_temp_build:
            os.environ.pop("TORCH_EXTENSIONS_DIR", None)
