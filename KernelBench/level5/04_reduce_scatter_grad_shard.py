"""
Reference: each rank contributes the same per-rank tensor into every reduce_scatter slot;
output is world_size * x (then scaled back for a stable reference).

Demonstrates NCCL reduce_scatter-style reduction into per-rank buffers.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from kernelbench.distributed_collectives import (
    default_device,
    get_world_size,
    is_distributed_run,
    maybe_init_process_group,
)

d_model = 32


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_init_process_group()
        ws = get_world_size()
        if not is_distributed_run() or ws == 1:
            return x

        # Each rank supplies the same x into every slot; slot k output is sum_p x = ws * x.
        input_list = [x.clone() for _ in range(ws)]
        out = torch.empty_like(x)
        dist.reduce_scatter(out, input_list, op=dist.ReduceOp.SUM)
        return out / ws


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(3)
    return [torch.randn(4, d_model, device=dev, generator=g)]


def get_init_inputs():
    return []
