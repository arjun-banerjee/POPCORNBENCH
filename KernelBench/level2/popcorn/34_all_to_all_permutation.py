"""
Reference: all_to_all_single with equal splits (NCCL all_to_all family).

With identical inputs on every rank, a symmetric equal-split exchange is an identity on the
concatenated layout; we multiply by a learnable scalar so the module is non-trivial.
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

# Must be divisible by world_size when running multi-GPU.
flat_dim = 64


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maybe_init_process_group()
        ws = get_world_size()
        b, d = x.shape
        assert d == flat_dim, f"expected last dim {flat_dim}, got {d}"
        if not is_distributed_run() or ws == 1:
            return x * self.scale

        chunk = d // ws
        assert chunk * ws == d, "flat_dim must be divisible by world_size"
        flat = x.reshape(b * d)
        out_flat = torch.empty_like(flat)
        sizes = [chunk * b] * ws
        dist.all_to_all_single(out_flat, flat, output_split_sizes=sizes, input_split_sizes=sizes)
        return out_flat.reshape(b, d) * self.scale


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(4)
    return [torch.randn(2, flat_dim, device=dev, generator=g)]


def get_init_inputs():
    return []
