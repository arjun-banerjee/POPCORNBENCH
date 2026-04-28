"""
Reference: shard the last dimension per rank, then all_gather to reconstruct (tensor-parallel style).

Multi-GPU: each rank keeps one column block; output is concat along last dim (matches full input).
Single-GPU: returns x unchanged.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from kernelbench.distributed_collectives import (
    default_device,
    get_rank,
    get_world_size,
    is_distributed_run,
    maybe_init_process_group,
)

dim = 64


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_init_process_group()
        ws = get_world_size()
        rank = get_rank()
        if not is_distributed_run() or ws == 1:
            return x * self.scale

        chunks = x.chunk(ws, dim=-1)
        local = chunks[rank].contiguous() * self.scale
        gathered = [torch.empty_like(local) for _ in range(ws)]
        dist.all_gather(gathered, local)
        return torch.cat(gathered, dim=-1)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(2)
    return [torch.randn(8, dim, device=dev, generator=g)]


def get_init_inputs():
    return []
