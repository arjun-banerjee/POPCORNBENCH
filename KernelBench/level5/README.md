# Distributed / multi-GPU reference problems

These modules mirror common **NCCL-backed** patterns exposed through `torch.distributed`: all-reduce (data parallel sync), broadcast, all-gather (tensor-parallel reunion), reduce-scatter, all-to-all, and a two-rank pipeline using send/recv plus broadcast. They are **not** wired into the default KernelBench level loaders or Hugging Face dataset; they live here as an extension set you can point to with `ref_origin=local` (paths under this directory) or import in your own harness.

Shared helpers live in `src/kernelbench/distributed_collectives.py` so `import kernelbench...` works when the project is installed (`uv sync`).

## Run with multiple processes

NCCL expects one process per GPU in the usual setup. From the repository root:

```bash
torchrun --standalone --nproc_per_node=2 uv run python scripts/run_and_check.py \
  ref_origin=local \
  ref_arch_src_path=distributed/06_pipeline_stage_p2p.py \
  kernel_src_path=<path-to-your-kernel.py> \
  eval_mode=local
```

Use `nproc_per_node` equal to the world size each problem expects (see `06_pipeline_stage_p2p.py`, which uses **exactly two** ranks for the P2P path). For CPU-only smoke tests, the code falls back to the `gloo` backend when CUDA is unavailable.

## Single-process behavior

If `WORLD_SIZE` is unset or 1, each file implements a **numerically equivalent** PyTorch path so imports and single-GPU sanity checks stay valid. Shapes and dtypes follow the in-file globals.

## Suggested mapping to collectives

| File | Collective | Typical use |
|------|------------|-------------|
| `01_all_reduce_data_parallel.py` | `all_reduce` | Gradient / moment sync in DP |
| `02_broadcast_parameter_shard.py` | `broadcast` | Tensor from rank 0 to all ranks |
| `03_all_gather_tensor_parallel.py` | `all_gather` | Reassemble sharded activations |
| `04_reduce_scatter_grad_shard.py` | `reduce_scatter` | Reduce then shard across ranks |
| `05_all_to_all_permutation.py` | `all_to_all_single` | Split tensors across ranks (symmetric exchange) |
| `06_pipeline_stage_p2p.py` | `send` / `recv` / `broadcast` | Minimal pipeline between two ranks |

These are **reference PyTorch** problems, not hand-written CUDA. A model that replaces them would typically call the same collectives from a custom kernel launch or framework integration.
