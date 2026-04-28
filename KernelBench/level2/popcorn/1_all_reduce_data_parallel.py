"""
Reference: data-parallel style gradient / statistic sync (NCCL all_reduce).

Multi-GPU: SUM all_reduce on the forward tensor. Single-GPU: identity (sum over 1 rank).
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

hidden = 256


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_init_process_group()
        y = self.lin(x)
        if not is_distributed_run():
            return y
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        ws = get_world_size()
        return y / ws


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(0)
    return [torch.randn(32, hidden, device=dev, generator=g)]


def get_init_inputs():
    return []
