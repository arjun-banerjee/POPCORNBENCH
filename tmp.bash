# 1) Install Python 3.12 for uv
uv python install 3.12
uv venv --python 3.12 /home/adalal542/.venvs/vllm312
/home/adalal542/.venvs/vllm312/bin/python -m ensurepip --upgrade
CUDA_HOME=/usr/local/cuda /home/adalal542/.venvs/vllm312/bin/python -m pip install --upgrade pip setuptools wheel
CUDA_HOME=/usr/local/cuda /home/adalal542/.venvs/vllm312/bin/python -m pip install "vllm>=0.19.0"

source /home/adalal542/.venvs/vllm312/bin/activate
export CUDA_HOME=/usr/local/cuda
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --attention-backend FLASH_ATTN

export SGLANG_API_KEY=dummy
# After changing PyTorch CUDA build, clear stale JIT extensions: rm -rf ~/.cache/torch_extensions
# Match gpu_arch to hardware (H100 -> Hopper). Default script value is Ada (wrong for H100).
uv run python scripts/generate_and_eval_single_sample.py \
  dataset_src=huggingface \
  level=1 \
  problem_id=1 \
  gpu_arch=Hopper \
  server_type=local \
  server_address=localhost \
  server_port=8000 \
  model_name=Qwen/Qwen2.5-Coder-32B-Instruct \
  prompt_option=few_shot \
  temperature=0.2 \
  log=true log_generated_kernel=true


uv run python scripts/generate_and_eval_single_sample.py \
  dataset_src=huggingface \
  level=1 \
  problem_id=1 \
  gpu_arch=Hopper \
  server_type=local \
  server_address=localhost \
  server_port=8000 \
  model_name=Qwen/Qwen3-Coder-Next \
  prompt_option=one_shot \
  temperature=0.2 \
  log=true log_generated_kernel=true

# dataset_src could be "local" or "huggingface"
# add .verbose_logging for more visibility

uv run python scripts/generate_samples.py \
  run_name=test_hf_level_1 \
  run_dir=/scratch/adalal542/my_kernel_runs/qwen3-coder-30b-a3b-instruct-fp8/test_hf_level_1 \
  dataset_src=huggingface \
  level=1 \
  num_workers=50 \
  server_type=local \
  server_address=localhost \
  server_port=8000 \
  model_name=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  temperature=0
  log=true \
  log_generated_kernel=true

uv run python scripts/eval_from_generations.py \
  run_name=test_hf_level_1 \
  dataset_src=local \
  level=1 \
  num_gpu_devices=8 \
  timeout=300
