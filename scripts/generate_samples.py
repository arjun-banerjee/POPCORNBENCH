import json
import os, sys
from dataclasses import dataclass

import pydra
import torch

from pydra import Config, REQUIRED

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref
from kernelbench.prompt_constructor_toml import (
    get_prompt_for_backend,
    get_custom_prompt,
    get_hardware_translation_prompt,
    get_translation_prompt,
)
from kernelbench.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    extract_last_code,
    maybe_multithread,
    set_gpu_arch,
)
from kernelbench.kernel_static_checker import validate_kernel_static

"""
Batch Generate Samples for Particular Level

Assume 1 sample per problem here
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


class GenerationConfig(Config):
    def __init__(self):

        self.dataset_src = REQUIRED  # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED

        # subset of problems to generate, otherwise generate on all problems in the level
        self.subset = (
            None,
            None,
        )  # (start_id, end_id), both inclusive - logical 1-indexed IDs

        self.run_name = REQUIRED  # name of the run (used under runs_dir unless run_dir is set)

        # If set, kernels are written here directly (overrides joining runs_dir + run_name).
        self.run_dir = None

        # num of thread pool to call inference server in parallel
        self.num_workers = 64
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = None
        self.model_name = None
        self.max_tokens = None
        self.temperature = 0.0
        # Local server override (for server_type=local, e.g. vLLM on localhost:8000)
        self.server_address = None
        self.server_port = None

        # Reasoning model specific parameters
        self.is_reasoning_model = False  # set to True for o1, o3, Gemini 2.5 thinking, etc.
        self.reasoning_effort = "low"  # for o1/o3: "low", "medium", "high"
        self.budget_tokens = 0  # for Claude extended thinking mode

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        self.verbose = False
        self.store_type = "local"  # TODO: add Database Integration

        # Number of samples to generate per problem for pass@k analysis
        self.num_samples = 1  # Default to 1 sample per problem

        # log=true enables both log_prompt and log_generated_kernel (like generate_and_eval_single_sample)
        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False  # save full LLM reply before code extraction

        self.backend = "cuda"

        self.precision = "fp32"
        self.prompt_option = "one_shot"  # zero_shot, one_shot, few_shot, translation
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.custom_prompt_key = None

        # Translation mode: source DSL identifier (e.g., "cuda", "triton",
        # "pytorch"). When set, source kernels are loaded from
        # KernelBench/level{L}/_translation_sources/{source_backend}/{problem_filename}
        # (override with source_kernel_dir). For source_backend="pytorch" the
        # PyTorch reference itself is used.
        self.source_backend = None
        self.source_kernel_dir = None  # optional override

        # Hardware translation mode: re-optimize kernels from one GPU arch to
        # another (same DSL). Set prompt_option=hardware_translation and provide
        # source_hardware_gpu_name (the GPU the source kernels were tuned for)
        # plus hardware_gpu_name (the target GPU). Source kernels are loaded from
        # source_kernel_dir or _translation_sources/{backend}/.
        self.source_hardware_gpu_name = None

        self.check_kernel = True  # [experimental] optional static checker catching potential hacking patterns

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int  # logically indexed
    sample_id: int


def _resolve_source_kernel_src(
    config: GenerationConfig, problem, ref_arch_src: str
) -> str:
    """Look up the source-DSL implementation for this problem when in
    translation mode. Returns the file contents as a string."""
    source_backend = str(config.source_backend).lower()
    if source_backend == "pytorch":
        return ref_arch_src

    # Determine the source-kernel directory
    if config.source_kernel_dir:
        src_dir = config.source_kernel_dir
        if not os.path.isabs(src_dir):
            src_dir = os.path.join(REPO_TOP_DIR, src_dir)
    else:
        src_dir = os.path.join(
            REPO_TOP_DIR,
            "KernelBench",
            f"level{config.level}",
            "_translation_sources",
            source_backend,
        )
    candidate = os.path.join(src_dir, problem.name)
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"No source kernel for problem '{problem.name}' under {src_dir}. "
            "Run scripts/build_translation_dataset.py to populate this directory."
        )
    with open(candidate, "r") as f:
        return f.read()


def generate_sample_single(
    work: WorkArgs,
    config: GenerationConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
) -> bool:
    # 1. Fetch Problem - unified interface
    problem = dataset.get_problem_by_id(work.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name

    if config.custom_prompt_key:
        custom_prompt = get_custom_prompt(
            config.custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    elif config.prompt_option == "translation":
        if not config.source_backend:
            raise ValueError(
                "prompt_option=translation requires source_backend (e.g., 'cuda', 'triton', 'pytorch')."
            )
        source_kernel_src = _resolve_source_kernel_src(config, problem, ref_arch_src)
        custom_prompt = get_translation_prompt(
            ref_arch_src=ref_arch_src,
            source_kernel_src=source_kernel_src,
            source_backend=str(config.source_backend).lower(),
            target_backend=config.backend,
            option="translation",
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    elif config.prompt_option == "hardware_translation":
        if not config.source_hardware_gpu_name:
            raise ValueError(
                "prompt_option=hardware_translation requires source_hardware_gpu_name "
                "(the GPU the source kernels were optimized for, e.g. 'H100')."
            )
        if not config.hardware_gpu_name:
            raise ValueError(
                "prompt_option=hardware_translation requires hardware_gpu_name "
                "(the target GPU to re-optimize for, e.g. 'A100')."
            )
        if not config.source_backend:
            config.source_backend = config.backend
        source_kernel_src = _resolve_source_kernel_src(config, problem, ref_arch_src)
        custom_prompt = get_hardware_translation_prompt(
            ref_arch_src=ref_arch_src,
            source_kernel_src=source_kernel_src,
            backend=config.backend,
            source_gpu_name=config.source_hardware_gpu_name,
            target_gpu_name=config.hardware_gpu_name,
            precision=config.precision,
        )
    else:
        custom_prompt = get_prompt_for_backend(
            ref_arch_src,
            config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    if config.log_prompt:
        prompt_path = os.path.join(
            run_dir,
            f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt",
        )
        with open(prompt_path, "w") as f:
            f.write(custom_prompt)

    # Query server with constructed prompt
    raw_text = inference_server(custom_prompt)
    if config.log_generated_kernel:
        raw_path = os.path.join(
            run_dir,
            f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_raw.txt",
        )
        with open(raw_path, "w") as f:
            f.write(raw_text if raw_text is not None else "")

    custom_kernel = extract_first_code(raw_text, ["python", "cpp"])
    if custom_kernel is None and raw_text is not None:
        custom_kernel = extract_last_code(raw_text, ["python", "cpp"])
    assert custom_kernel is not None, "Custom CUDA code generation failed"

    # Optional: we provide a static code checker for kernel code using regex matching
    # NOTE: by no means, is this checker complete, but it might could help catch some potential hacks and issues
    if config.check_kernel:
        static_check_status, error, warnings = validate_kernel_static(custom_kernel,
            backend=config.backend,
            precision=config.precision, 
            # uses the default set of forbidden and warning patterns, 
            # you could adapt the patterns to your own setting (degree of banning cuda stream, allowing some torch ops)
        )
        assert static_check_status, f"Static check failed for sample {work.sample_id} for problem {work.problem_id}: {problem_name}. Error: {error}. Warnings: {warnings}"
        if warnings:
            print(f"Static check warnings for sample {work.sample_id} for problem {work.problem_id}: {problem_name}. Warnings: {warnings}")

    if config.verbose:
        print(
            f"Generated sample {work.sample_id} for problem {work.problem_id}: {problem_name}"
        )

    # Store to local file
    kernel_path = os.path.join(
        run_dir,
        f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py",
    )
    with open(kernel_path, "w") as f:
        f.write(custom_kernel)

    return True


def generate_sample_launcher(
    work: WorkArgs,
    config: GenerationConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
):
    try:
        return generate_sample_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating sample {work.problem_id} {work.sample_id}: {e}")
        return None


def check_kernel_exists(
    run_dir: str, level: int, problem_id: int, sample_id: int
) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )
    return os.path.exists(kernel_path)


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    from kernelbench.utils import SERVER_PRESETS
    
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", "None")
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", "None")
    
    # Convert string boolean to actual boolean for reasoning model flag
    if isinstance(config.is_reasoning_model, str):
        config.is_reasoning_model = config.is_reasoning_model.lower() in ['true', '1', 'yes']

    if isinstance(config.log, str):
        config.log = config.log.lower() in ("true", "1", "yes")
    if config.log:
        config.log_prompt = True
        config.log_generated_kernel = True

    if isinstance(config.log_prompt, str):
        config.log_prompt = config.log_prompt.lower() in ("true", "1", "yes")
    if isinstance(config.log_generated_kernel, str):
        config.log_generated_kernel = config.log_generated_kernel.lower() in (
            "true",
            "1",
            "yes",
        )

    run_dir_override = getattr(config, "run_dir", None)
    if isinstance(run_dir_override, str):
        trimmed = run_dir_override.strip()
        if trimmed.lower() in ("", "none"):
            run_dir_override = None
        else:
            run_dir_override = trimmed
    else:
        run_dir_override = None
    
    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        if trimmed.lower() in {"", "none"}:
            custom_prompt_key = None
        else:
            custom_prompt_key = trimmed
    config.custom_prompt_key = custom_prompt_key

    include_hardware = config.include_hardware_info
    if isinstance(include_hardware, str):
        include_hardware = include_hardware.lower() in ["true", "1", "yes"]
    config.include_hardware_info = include_hardware

    supported_backends = {
        "cuda", "triton", "cute", "tilelang", "thunderkittens",
        "helion", "hip", "nki", "pallas", "numba", "mojo",
    }
    backend = config.backend.lower()
    if backend not in supported_backends:
        raise ValueError(
            f"Unsupported backend: {config.backend}. Must be one of {sorted(supported_backends)}."
        )
    config.backend = backend
    if backend == "tilelang":
        config.precision = "fp16"
    if backend == "thunderkittens":
        config.precision = "bf16"

    config.prompt_option = str(config.prompt_option).lower()
    valid_prompt_options = {"zero_shot", "one_shot", "few_shot", "translation", "hardware_translation"}
    if not config.custom_prompt_key:
        if config.prompt_option not in valid_prompt_options:
            raise ValueError(
                f"Invalid prompt_option '{config.prompt_option}'. Must be one of {sorted(valid_prompt_options)}."
            )
        if include_hardware and not config.hardware_gpu_name:
            raise ValueError(
                "include_hardware_info is True but hardware_gpu_name is not provided."
            )

    print(f"Starting Batch Generation with config: {config}")

    # Dataset Configurations - Unified loading
    dataset = construct_kernelbench_dataset(
        level=config.level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )

    all_problem_ids = dataset.get_problem_ids()

    if config.subset == (None, None):
        problem_ids_to_run = all_problem_ids
    else:
        start, end = config.subset
        problem_ids_to_run = [pid for pid in all_problem_ids if start <= pid <= end]
        if not problem_ids_to_run:
            print(f"Warning: No problems found in subset range {config.subset}")

    print(
        f"Generating {config.num_samples} sample(s) each for level {config.level} problems: {problem_ids_to_run}"
    )

    # set up run directory
    if run_dir_override:
        run_dir = os.path.abspath(os.path.expanduser(run_dir_override))
    else:
        run_dir = os.path.join(config.runs_dir, config.run_name)
    run_exists = os.path.exists(run_dir)
    if run_exists:
        print(f"\n⚠️  WARNING: Run directory already exists: {run_dir}")
        print(
            "   Existing kernels will be skipped. Use a different run_name, a new run_dir, "
            "or remove the directory for a fresh run.\n"
        )
    os.makedirs(run_dir, exist_ok=True)
    pydra.save_yaml(config.to_dict(), os.path.join(run_dir, "generation_config.yaml"))

    assert (
        config.store_type == "local"
    ), "supporting local file-system based storage for now"  # database integreation coming soon, need to migrate from CUDA Monkeys code

    problems_to_run = []
    total_problems = 0
    already_completed = 0
    for problem_id in problem_ids_to_run:
        for sample_id in range(config.num_samples):
            total_problems += 1
            if not check_kernel_exists(run_dir, config.level, problem_id, sample_id):
                problems_to_run.append(
                    WorkArgs(problem_id=int(problem_id), sample_id=sample_id)
                )
            else:
                already_completed += 1
    
    if already_completed > 0:
        print(f"📁 Found {already_completed}/{total_problems} kernels already generated. Generating remaining {len(problems_to_run)} kernels.")

    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        is_reasoning_model=config.is_reasoning_model,
        reasoning_effort=config.reasoning_effort,
        budget_tokens=config.budget_tokens,
        server_address=config.server_address,
        server_port=config.server_port,
    )

    # Launch workers
    generation_results = maybe_multithread(
        generate_sample_launcher,
        problems_to_run,
        config.num_workers,
        time_interval=config.api_query_interval,
        # extra args
        config=config,
        dataset=dataset,
        inference_server=inference_server,
        run_dir=run_dir,
    )

    num_generated_samples = len(generation_results)
    num_attempted = len(problems_to_run)
    num_failed_problems = num_attempted - num_generated_samples
    
    if num_attempted == 0:
        print(f"\n✅ All {total_problems} kernels already exist in {run_dir}")
        print(f"   Use a different run_name if you want to generate fresh samples.\n")
    else:
        print(
            f"\nGenerated {num_generated_samples} samples for total {num_attempted} problems, Please retry for the {num_failed_problems} failed problems."
        )


if __name__ == "__main__":
    main()
