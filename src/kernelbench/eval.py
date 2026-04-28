"""
Helpers for Evaluations
"""

import hashlib
import importlib
import json
import linecache
import os, subprocess
import random
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Union, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from pydantic import BaseModel

from . import timing, dataset, extended_metrics

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_error_name(e: Exception) -> str:
    """
    Get the error name, for logging purposes
    """
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def fetch_ref_arch_from_problem_id(
    problem_id: int, dataset: "BaseDataset", with_name=False
) -> Union[str, tuple[str, str]]:
    """
    Fetches the reference architecture for a given problem_id from the dataset.
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem = dataset.get_problem_by_id(problem_id)
    ref_arch = problem.code

    if not with_name:
        return ref_arch
    else:
        # Use problem.name as fallback when path is None (e.g., for HuggingFace datasets)
        name = problem.path if problem.path is not None else problem.name
        return (name, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    kb_dataset = dataset.construct_kernelbench_dataset(level)
    return fetch_ref_arch_from_problem_id(problem_id, kb_dataset, with_name)


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)


def get_torch_dtype_from_string(precision: str) -> torch.dtype:
    """
    Get the torch dtype for specific precision
    """
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else:  # future, FP8, FP4, etc. support?
        raise ValueError(f"Invalid precision not supported: {precision}")


def get_tolerance_for_precision(precision: str | torch.dtype) -> float:
    """
    Get the tolerance from a string representing the percision.
    These tolerances are inspired by torchbench (PyTorch Benchmarking Suite):
    Reference:
    https://github.com/pytorch/benchmark/blob/cfd835c35d04513ced9a59bd074eeb21dc8187d7/torchbenchmark/util/env_check.py#L519
    """
    if isinstance(precision, str):
        precision = get_torch_dtype_from_string(precision)

    PRECISION_TOLERANCES = {
        # By default for fp32, 1e-4 is used according to torchbench.
        torch.float32: 1e-4,
        # torchbench states for bf16 and fp16, use 1e-3 as tolerance and 1e-2 if it's too strict.
        # @todo: Let user configure own tolerance as an option
        torch.float16: 1e-2,
        torch.bfloat16: 1e-2,
    }
    assert precision in PRECISION_TOLERANCES, (
        f"Invalid precision not supported: {precision}"
    )
    return PRECISION_TOLERANCES[precision]


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    # Execution
    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}  # NOTE: to include warning if any

    # Timing
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance

    # new: added ref time either through fetching prev runs or through execution
    # could do eager for level 1 and compile for level 2 and 3
    ref_runtime: float = (
        -1.0
    )  # in us, only recorded if we decide to measure performance
    ref_runtime_stats: dict = {}  # only recorded if we decide to measure performance

    # Translation mode: timing of the source-DSL implementation that the
    # candidate was translated from. Only populated when source_kernel_src and
    # source_backend are passed to eval_kernel_against_ref.
    source_runtime: float = -1.0
    source_runtime_stats: dict = {}
    source_backend: Optional[str] = None
    speedup_vs_source: float = -1.0

    # ── Extended metrics (populated when measure_performance=True) ──
    # GPU Memory Efficiency
    memory_stats: dict = {}  # peak_memory_bytes, ref_peak_memory_bytes, memory_ratio

    # Continuous Numerical Precision (populated even for correct kernels)
    numerical_precision: dict = {}  # max_abs_error, mean_abs_error, max_rel_error, mean_rel_error

    # Kernel Launch Count / Fusion Quality
    kernel_launch_stats: dict = {}  # num_kernels, ref_num_kernels, fusion_ratio, kernel_breakdown

    # SOL (Speed-of-Light) Score
    sol_stats: dict = {}  # sol_score, arithmetic_intensity, achieved_bandwidth_gbps, achieved_gflops, bottleneck

    # Energy Efficiency
    energy_stats: dict = {}  # energy_mj, ref_energy_mj, energy_ratio, avg_power_w

    # Roofline / Occupancy
    roofline_stats: dict = {}  # roofline_efficiency, occupancy_pct, memory_throughput_pct, compute_throughput_pct


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(model_custom_src, entry_point="ModelNew"):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.

    Returns both a Model class and the temporary file. The temporary file must be
    deleted manually be the caller.

    This is a hack that is needed for triton code as compile / exec do not play well
    with the @triton.jit decorator.
    """

    # Create a temporary named file with a .py extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        # Write the code string into the file
        tmp_file.write(model_custom_src)
        # Capture the path to the file
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return ModelNew, temp_file


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def build_compile_cache_legacy(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str, verbose: bool = False, build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            f"import os\nos.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(
        ["python", tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)

    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode("utf-8"))
        print("[CPU Precompile] stderr: \n", stderr.decode("utf-8"))

    return returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def _process_input_tensor(input, device, backend="cuda", precision=torch.float32):
    """
    Helper function to move tensors to the correct device and apply backend-specific dtype casting.

    Args:
        input: Input tensor or non-tensor value
        device: Target CUDA device
        backend: Backend type (e.g., 'cuda', 'triton', 'cute')
        precision: torch.dtype
    Returns:
        Processed tensor on correct device with correct dtype, or original value if not a tensor
    """

    # sometimes things like init inputs are floats (like in the case of labels / targets, classification losses, etc.)
    if not isinstance(input, torch.Tensor):
        return input

    # cast to the desired percision dtype for activations
    input_tensor = input.to(dtype=precision)

    # Default for all other backends and float types
    return input_tensor.to(device=device)


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    measure_performance: bool = False,
    timing_method: str = "cuda_event",  # see timing.py
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),  # have to run on GPU
    backend: str = "cuda",  # can be 'cuda', 'triton', 'tilelang', or 'cute'
    precision: torch.dtype = torch.float32,
    # Translation mode: when both are provided, also compile + time the source
    # kernel and report speedup_vs_source on the result. The source kernel must
    # implement ModelNew using the same Model interface as the PyTorch reference.
    source_kernel_src: Optional[str] = None,
    source_backend: Optional[str] = None,
    # Guard against potential reward hacking [optional but ongoing enhancement]
    check_for_excessive_speedup: bool = True,
    excessive_speedup_threshold: float = 10,  # flag if the kernel is more than <excessive_speedup_threshold>x faster than the reference
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    NOTE: we are thinking about refactor this to be more modularized
    and we can add more checks as our other ongiong PRs are working on

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    backend: str, one of 'cuda', 'triton', 'tilelang', or 'cute'
    precision: torch.dtype for computation (note: tilelang only supports fp16)
    timing_method: str, method to time kernel, see timing.py for more details

    ONGOING EFFORT to refactor and modularize this, and adding more tests for eval.
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"

    # Backend-GPU vendor validation
    from .utils import get_gpu_vendor

    vendor = get_gpu_vendor(device)
    backend_lower = backend.lower()
    # HIP is AMD-only
    if backend_lower == "hip" and vendor != "amd":
        raise ValueError(f"HIP backend requires AMD GPU, got {vendor}")
    # cuda/cute/thunderkittens are NVIDIA-only (triton/tilelang/helion/numba work on both or NVIDIA)
    if backend_lower in ["cuda", "cute", "thunderkittens"] and vendor == "amd":
        raise ValueError(f"{backend} backend requires NVIDIA GPU, got AMD")
    # NKI targets AWS Trainium/Inferentia — cannot run on standard CUDA/AMD GPUs
    if backend_lower == "nki":
        import warnings

        warnings.warn(
            "NKI backend targets AWS Trainium/Inferentia. Compilation check only on non-Neuron hardware."
        )

    if backend_lower == "tilelang":
        assert precision == torch.float16 or precision == torch.bfloat16, (
            "TileLang only supports fp16 or bfloat16"
        )

    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)

    # Backends that use tempfile approach and need CUDA_VISIBLE_DEVICES
    # TileLang, Triton, and CuTe all use tempfile for proper module loading
    # Helion, NKI, Pallas, Numba, and Mojo also need tempfile for JIT decorators / imports
    uses_tempfile = backend.lower() in [
        "triton",
        "tilelang",
        "cute",
        "helion",
        "nki",
        "pallas",
        "numba",
        "mojo",
    ]

    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    if uses_tempfile:
        # need to set env var for triton/cute code to guarantee no wrong device shenanigans
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert device.type == "cuda", (
                "CUDA is not availible on device, cannot run Eval"
            )
            device_num = device.index
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        # NVIDIA uses CUDA_VISIBLE_DEVICES, AMD uses HIP_VISIBLE_DEVICES
        if vendor == "amd":
            os.environ["HIP_VISIBLE_DEVICES"] = str(device_num)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()

    # Convert inputs to appropriate dtypes for GPU computation
    init_inputs = [
        _process_input_tensor(x, device, backend, precision) for x in init_inputs
    ]

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")

    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        tempfile = None
        # add hash for later to distinguish between multi-turn kernels

        backend_lower = backend.lower()
        tempfile_backends = [
            "triton",
            "tilelang",
            "cute",
            "helion",
            "nki",
            "pallas",
            "numba",
            "mojo",
        ]
        if backend_lower in tempfile_backends:
            # Use tempfile approach for DSLs that require proper module import
            # for JIT decorators / special imports to work
            ModelNew, tempfile = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            # Default CUDA backend
            ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device, tempfile)
            return None
        else:
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = e
            graceful_eval_cleanup(context, device, tempfile)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # Check if ModelNew was successfully loaded (load_custom_model returns None on syntax errors)
    if ModelNew is None:
        print(
            "Failed to load custom model: Syntax error or ModelNew not found in generated code. Record as compilation failure."
        )
        metadata["compilation_error_name"] = "SyntaxError"
        metadata["compilation_error"] = (
            "Syntax error in custom generated code or ModelNew not found"
        )
        graceful_eval_cleanup(context, device, tempfile)
        return KernelExecResult(compiled=False, metadata=metadata)  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            original_model = original_model.to(device=device, dtype=precision)
            custom_model = custom_model.to(device=device, dtype=precision)
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device, tempfile)
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            backend=backend,
            precision=precision,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # ── Collect continuous numerical precision metrics from correctness trials ──
    if kernel_exec_result and kernel_exec_result.correctness:
        accum = kernel_exec_result.metadata.get("_precision_metrics_accum")
        if accum:
            kernel_exec_result.numerical_precision = {
                "max_abs_error": accum.get("max_abs_error", 0.0),
                "mean_abs_error": accum.get("mean_abs_error", 0.0),
                "max_rel_error": accum.get("max_rel_error", 0.0),
                "mean_rel_error": accum.get("mean_rel_error", 0.0),
            }
        kernel_exec_result.metadata.pop("_precision_metrics_accum", None)

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [
                    _process_input_tensor(x, device, backend, precision) for x in inputs
                ]

                model_new = custom_model.to(device=device, dtype=precision)
                torch.cuda.synchronize(device=device)

                timing_fn = timing.get_timing_function(timing_method)
                elapsed_times = timing_fn(
                    model_new,
                    inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = timing.get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats

                # ── Extended Metric 1: GPU Memory Efficiency ──
                try:
                    if verbose:
                        print("[Eval] Measuring GPU Memory Efficiency")
                    custom_mem = extended_metrics.measure_memory(
                        model_new, inputs, device
                    )
                    ref_mem = extended_metrics.measure_memory(
                        original_model, inputs, device
                    )
                    kernel_exec_result.memory_stats = (
                        extended_metrics.compute_memory_stats(custom_mem, ref_mem)
                    )
                except Exception as e:
                    if verbose:
                        print(f"[Eval] Memory measurement failed: {e}")
                    kernel_exec_result.memory_stats = {"error": str(e)}

                # ── Extended Metric 2: Kernel Launch Count / Fusion ──
                try:
                    if verbose:
                        print("[Eval] Measuring Kernel Launch Count / Fusion")
                    custom_launches = extended_metrics.measure_kernel_launches(
                        model_new, inputs, device
                    )
                    ref_launches = extended_metrics.measure_kernel_launches(
                        original_model, inputs, device
                    )
                    kernel_exec_result.kernel_launch_stats = (
                        extended_metrics.compute_kernel_launch_stats(
                            custom_launches, ref_launches
                        )
                    )
                except Exception as e:
                    if verbose:
                        print(f"[Eval] Kernel launch measurement failed: {e}")
                    kernel_exec_result.kernel_launch_stats = {"error": str(e)}

                # ── Extended Metric 3: Energy Efficiency ──
                try:
                    if verbose:
                        print("[Eval] Measuring Energy Efficiency")
                    custom_energy = extended_metrics.measure_energy(
                        model_new, inputs, device, num_trials=50
                    )
                    ref_energy = extended_metrics.measure_energy(
                        original_model, inputs, device, num_trials=50
                    )
                    kernel_exec_result.energy_stats = (
                        extended_metrics.compute_energy_stats(custom_energy, ref_energy)
                    )
                except Exception as e:
                    if verbose:
                        print(f"[Eval] Energy measurement failed: {e}")
                    kernel_exec_result.energy_stats = {"error": str(e)}

        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    ###############################################################
    # Excessive speedup check + reference timing + SOL/roofline
    ##############################################################

    if measure_performance and check_for_excessive_speedup:
        if verbose:
            print("[Eval] Additional checks to flag excessive speedup")

        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]

        model_new = custom_model.to(device=device, dtype=precision)
        torch.cuda.synchronize(device=device)

        timing_fn = timing.get_timing_function(timing_method)
        reference_elapsed_times = timing_fn(
            original_model,
            inputs,
            num_trials=num_perf_trials,
            verbose=verbose,
            device=device,
        )
        reference_runtime_stats = timing.get_timing_stats(
            reference_elapsed_times, device=device
        )
        kernel_exec_result.ref_runtime = reference_runtime_stats["mean"]
        kernel_exec_result.ref_runtime_stats = reference_runtime_stats

        effective_speedup = kernel_exec_result.ref_runtime / kernel_exec_result.runtime

        if verbose:
            print(
                f"[Eval] Effective Speedup is {effective_speedup:.2f}x using timing method {timing_method}"
            )

        if effective_speedup > excessive_speedup_threshold:
            kernel_exec_result.metadata["excessive_speedup"] = True
            print(
                f"[WARNING] Excessive speedup {effective_speedup:.2f}x over {excessive_speedup_threshold}x threshold detected"
            )
            print(
                f"[WARNING] Double check your kernel carefully to ensure it is not reward hacking."
            )

        # ── Extended Metrics 4 & 5: SOL Score + Roofline (Nsight with fallback) ──
        nsight_profile = None
        try:
            if verbose:
                print(
                    "[Eval] Attempting Nsight profiling for SOL / Roofline (subprocess)"
                )
            precision_str = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
                precision, "fp32"
            )
            nsight_profile = extended_metrics.profile_kernel_with_nsight(
                custom_model_src=custom_model_src,
                ref_model_src=original_model_src,
                device=device,
                backend=backend,
                precision=precision_str,
                build_dir=build_dir,
                verbose=verbose,
            )
            if nsight_profile and verbose:
                print(
                    f"[Eval] Nsight profiling succeeded: occupancy={nsight_profile.get('occupancy_pct')}%, "
                    f"DRAM util={nsight_profile.get('dram_utilization_pct')}%"
                )
        except Exception as e:
            if verbose:
                print(
                    f"[Eval] Nsight profiling failed (will use heuristic fallback): {e}"
                )

        if nsight_profile:
            kernel_exec_result.sol_stats = (
                extended_metrics.compute_sol_score_from_nsight(nsight_profile)
            )
            kernel_exec_result.roofline_stats = (
                extended_metrics.compute_roofline_stats_from_nsight(nsight_profile)
            )
        else:
            if verbose:
                print("[Eval] Using heuristic fallback for SOL / Roofline")
            kernel_exec_result.sol_stats = extended_metrics.compute_sol_score_heuristic(
                runtime_ms=kernel_exec_result.runtime,
                ref_runtime_ms=kernel_exec_result.ref_runtime,
                device=device,
            )
            kernel_exec_result.roofline_stats = (
                extended_metrics.compute_roofline_stats_heuristic(
                    runtime_ms=kernel_exec_result.runtime,
                    device=device,
                )
            )

    # Translation mode: also time the source-DSL implementation so the result
    # carries speedup_vs_source alongside the existing speedup_vs_pytorch.
    # Skipped if the candidate failed correctness or wasn't timed.
    if (
        measure_performance
        and source_kernel_src is not None
        and source_backend is not None
        and kernel_exec_result is not None
        and kernel_exec_result.correctness
        and kernel_exec_result.runtime > 0
    ):
        source_tempfile = None
        try:
            source_backend_lower = source_backend.lower()
            source_uses_tempfile = source_backend_lower in [
                "triton",
                "tilelang",
                "cute",
                "helion",
                "nki",
                "pallas",
                "numba",
                "mojo",
            ]
            if source_uses_tempfile:
                SourceModel, source_tempfile = load_custom_model_with_tempfile(
                    source_kernel_src, entry_point="ModelNew"
                )
            elif source_backend_lower == "pytorch":
                # The "pytorch source" is the reference Model itself; time the
                # already-loaded original_model rather than recompiling.
                SourceModel = None
            else:
                SourceModel = load_custom_model(source_kernel_src, {}, build_dir)

            if SourceModel is None and source_backend_lower != "pytorch":
                raise RuntimeError(
                    "Source kernel failed to load (ModelNew not found or syntax error)"
                )

            with torch.no_grad():
                set_seed(seed_num)
                if source_backend_lower == "pytorch":
                    source_model = original_model
                else:
                    source_model = SourceModel(*init_inputs)
                    source_model = source_model.to(device=device, dtype=precision)
                torch.cuda.synchronize(device=device)

            set_seed(seed_num)
            source_inputs = get_inputs()
            source_inputs = [
                _process_input_tensor(x, device, source_backend_lower, precision)
                for x in source_inputs
            ]

            timing_fn = timing.get_timing_function(timing_method)
            source_elapsed = timing_fn(
                source_model,
                source_inputs,
                num_trials=num_perf_trials,
                verbose=verbose,
                device=device,
            )
            source_stats = timing.get_timing_stats(source_elapsed, device=device)
            kernel_exec_result.source_backend = source_backend_lower
            kernel_exec_result.source_runtime = source_stats["mean"]
            kernel_exec_result.source_runtime_stats = source_stats
            if kernel_exec_result.runtime > 0:
                kernel_exec_result.speedup_vs_source = (
                    source_stats["mean"] / kernel_exec_result.runtime
                )
            if verbose:
                print(
                    f"[Eval] Source ({source_backend_lower}) runtime: "
                    f"{source_stats['mean']:.3f}, speedup_vs_source: "
                    f"{kernel_exec_result.speedup_vs_source:.2f}x"
                )
        except Exception as e:
            if verbose:
                print(f"[Eval] Failed to time source kernel ({source_backend}): {e}")
            kernel_exec_result.metadata["source_kernel_error"] = str(e)
            kernel_exec_result.metadata["source_kernel_error_name"] = get_error_name(e)
        finally:
            if source_tempfile is not None:
                try:
                    source_tempfile.close()
                    os.remove(source_tempfile.name)
                except OSError:
                    pass

    graceful_eval_cleanup(context, device, tempfile)
    return kernel_exec_result


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def _compare_outputs(output, output_new, tolerance):
    """
    Compare model outputs that may be single tensors or tuples/lists of tensors.
    Returns (match: bool, mismatch_details: dict).
    Precision metrics are always populated in details under 'precision_metrics'.
    """
    if isinstance(output, (tuple, list)):
        if not isinstance(output_new, (tuple, list)):
            return False, {
                "error": f"Output type mismatch: expected {type(output).__name__}, got {type(output_new).__name__}"
            }
        if len(output) != len(output_new):
            return False, {
                "error": f"Output count mismatch: expected {len(output)} elements, got {len(output_new)}"
            }
        all_match = True
        details = {}
        combined_precision = {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
            "num_elements": 0,
        }
        for i, (o, o_new) in enumerate(zip(output, output_new)):
            match_i, details_i = _compare_outputs(o, o_new, tolerance)
            if not match_i:
                all_match = False
                details[f"element_{i}"] = {
                    k: v for k, v in details_i.items() if k != "precision_metrics"
                }
            pm = details_i.get("precision_metrics")
            if pm:
                combined_precision["max_abs_error"] = max(
                    combined_precision["max_abs_error"], pm.get("max_abs_error", 0.0)
                )
                combined_precision["max_rel_error"] = max(
                    combined_precision["max_rel_error"], pm.get("max_rel_error", 0.0)
                )
                n_old = combined_precision["num_elements"]
                n_new = pm.get("num_elements", 0)
                if n_old + n_new > 0:
                    combined_precision["mean_abs_error"] = (
                        combined_precision["mean_abs_error"] * n_old
                        + pm.get("mean_abs_error", 0.0) * n_new
                    ) / (n_old + n_new)
                    combined_precision["mean_rel_error"] = (
                        combined_precision["mean_rel_error"] * n_old
                        + pm.get("mean_rel_error", 0.0) * n_new
                    ) / (n_old + n_new)
                combined_precision["num_elements"] = n_old + n_new
        details["precision_metrics"] = combined_precision
        return all_match, details

    if isinstance(output, torch.Tensor):
        if not isinstance(output_new, torch.Tensor):
            return False, {
                "error": f"Type mismatch: expected Tensor, got {type(output_new).__name__}"
            }
        if output.shape != output_new.shape:
            return False, {
                "shape_mismatch": f"Expected {output.shape}, got {output_new.shape}"
            }
        if output.dtype == torch.bool or output_new.dtype == torch.bool:
            if torch.equal(output, output_new):
                return True, {
                    "precision_metrics": {
                        "max_abs_error": 0.0,
                        "mean_abs_error": 0.0,
                        "max_rel_error": 0.0,
                        "mean_rel_error": 0.0,
                        "num_elements": output.numel(),
                    }
                }
            diff_count = (output != output_new).sum().item()
            return False, {
                "bool_mismatch_count": diff_count,
                "precision_metrics": {
                    "max_abs_error": 1.0,
                    "mean_abs_error": diff_count / max(output.numel(), 1),
                    "max_rel_error": 1.0,
                    "mean_rel_error": diff_count / max(output.numel(), 1),
                    "num_elements": output.numel(),
                },
            }
        out_f = output.float()
        out_new_f = output_new.float()
        abs_diff = torch.abs(out_f - out_new_f)
        max_diff = abs_diff.max().item()
        avg_diff = abs_diff.mean().item()
        denom = torch.abs(out_f).clamp(min=1e-12)
        rel_diff = abs_diff / denom
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()
        precision_metrics = {
            "max_abs_error": max_diff,
            "mean_abs_error": avg_diff,
            "max_rel_error": max_rel,
            "mean_rel_error": mean_rel,
            "num_elements": output.numel(),
        }
        if torch.allclose(out_f, out_new_f, atol=tolerance, rtol=tolerance):
            return True, {"precision_metrics": precision_metrics}
        return False, {
            "max_difference": f"{max_diff:.6f}",
            "avg_difference": f"{avg_diff:.6f}",
            "precision_metrics": precision_metrics,
        }

    # Scalar or other non-tensor type
    if output == output_new:
        return True, {
            "precision_metrics": {
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "max_rel_error": 0.0,
                "mean_rel_error": 0.0,
                "num_elements": 1,
            }
        }
    return False, {
        "value_mismatch": f"Expected {output}, got {output_new}",
        "precision_metrics": {
            "max_abs_error": float("inf"),
            "mean_abs_error": float("inf"),
            "max_rel_error": float("inf"),
            "mean_rel_error": float("inf"),
            "num_elements": 1,
        },
    }


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Optional[torch.device] = None,
    backend: str = "cuda",
    precision: torch.dtype = torch.float32,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    backend: backend type for handling dtype conversions
    precision: torch.dtype
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            # Convert inputs to appropriate dtypes for GPU computation
            inputs = [
                _process_input_tensor(x, device, backend, precision) for x in inputs
            ]

            set_seed(trial_seed)

            model = original_model_instance.to(device=device, dtype=precision)

            set_seed(trial_seed)

            model_new = new_model_instance.to(device=device, dtype=precision)

            # Re-seed so both forwards see the same RNG stream
            # (critical for stochastic kernels like MCMC, sampling, etc.)
            forward_seed = (trial_seed + 1) % (2**32)
            set_seed(forward_seed)
            output = model(*inputs)
            torch.cuda.synchronize(device=device)

            try:
                set_seed(forward_seed)
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)

                tolerance = get_tolerance_for_precision(precision)
                match, mismatch_details = _compare_outputs(
                    output, output_new, tolerance
                )

                pm = mismatch_details.get("precision_metrics")
                if pm:
                    existing_pm = metadata.get("_precision_metrics_accum")
                    if existing_pm is None:
                        metadata["_precision_metrics_accum"] = dict(pm)
                    else:
                        existing_pm["max_abs_error"] = max(
                            existing_pm.get("max_abs_error", 0.0),
                            pm.get("max_abs_error", 0.0),
                        )
                        existing_pm["max_rel_error"] = max(
                            existing_pm.get("max_rel_error", 0.0),
                            pm.get("max_rel_error", 0.0),
                        )
                        n_old = existing_pm.get("num_elements", 0)
                        n_new = pm.get("num_elements", 0)
                        if n_old + n_new > 0:
                            existing_pm["mean_abs_error"] = (
                                existing_pm.get("mean_abs_error", 0.0) * n_old
                                + pm.get("mean_abs_error", 0.0) * n_new
                            ) / (n_old + n_new)
                            existing_pm["mean_rel_error"] = (
                                existing_pm.get("mean_rel_error", 0.0) * n_old
                                + pm.get("mean_rel_error", 0.0) * n_new
                            ) / (n_old + n_new)
                        existing_pm["num_elements"] = n_old + n_new

                if match:
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")
                else:
                    if "shape_mismatch" in str(mismatch_details):
                        metadata = register_and_format_exception(
                            "correctness_issue",
                            f"Output shape mismatch: {mismatch_details}",
                            metadata,
                        )
                        metadata["correctness_issue_name"] = "correctness_issue"
                        if verbose:
                            print(
                                f"[FAIL] trial {trial}: Output shape mismatch: {mismatch_details}"
                            )
                        return KernelExecResult(
                            compiled=True, correctness=False, metadata=metadata
                        )

                    for key, val in mismatch_details.items():
                        if isinstance(val, dict):
                            for sub_key, sub_val in val.items():
                                metadata.setdefault(sub_key, []).append(str(sub_val))
                        else:
                            metadata.setdefault(key, []).append(str(val))
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output mismatch: {mismatch_details}"
                        )

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")
                print("\n[Full Traceback]:")
                traceback.print_exc()
                print("\n")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                metadata["runtime_error_name"] = get_error_name(e)
                # Also store the full traceback in metadata for debugging
                metadata["runtime_error_traceback"] = traceback.format_exc()
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata


def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


# if __name__ == "__main__":
# fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# Note: fetch_baseline_time is available in kernelbench.timing module
