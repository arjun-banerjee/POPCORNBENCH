"""
Reference: broadcast tensor from rank 0 (NCCL broadcast), then apply a linear map.

Multi-GPU: rank != 0 starts from zeros; after broadcast all ranks match rank 0's shard.
Single-GPU: equivalent to linear(x).
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from kernelbench.distributed_collectives import (
    default_device,
    get_rank,
    is_distributed_run,
    maybe_init_process_group,
)

features = 128


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(features, features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_init_process_group()
        if is_distributed_run() and get_rank() != 0:
            x = torch.zeros_like(x)
        if is_distributed_run():
            dist.broadcast(x, src=0)
        return self.lin(x)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    return [torch.randn(16, features, device=dev, generator=g)]


def get_init_inputs():
    return []
