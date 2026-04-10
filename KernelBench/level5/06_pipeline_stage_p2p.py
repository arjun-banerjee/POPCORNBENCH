"""
Reference: two-stage pipeline with send/recv between rank 0 and rank 1, then broadcast of the final
activation (NCCL point-to-point + broadcast).

Distributed run expects ``WORLD_SIZE == 2``. Other world sizes fall back to a fused sequential forward
so imports and single-GPU eval still work.
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

in_features = 64
hidden_features = 96
out_features = 48


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_features, bias=True)
        self.lin2 = nn.Linear(hidden_features, out_features, bias=True)

    def _sequential(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(torch.relu(self.lin1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_init_process_group()
        ws = get_world_size()
        if not is_distributed_run() or ws != 2:
            return self._sequential(x)

        rank = get_rank()
        if rank == 0:
            h = torch.relu(self.lin1(x))
            dist.send(h.contiguous(), dst=1)
            out = torch.empty(x.size(0), out_features, device=x.device, dtype=x.dtype)
            dist.broadcast(out, src=1)
            return out
        h = torch.empty(x.size(0), hidden_features, device=x.device, dtype=x.dtype)
        dist.recv(h, src=0)
        out = self.lin2(h)
        dist.broadcast(out, src=1)
        return out


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(5)
    return [torch.randn(12, in_features, device=dev, generator=g)]


def get_init_inputs():
    return []
