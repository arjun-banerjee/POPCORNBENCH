"""
Virtual ranks: input packs R per-rank tensors along dim 0. Output is the mean over ranks (B, D).

No distributed runtime required; the reference always reduces along dim 0.
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

R = 4
B = 16
D = 48


class Model(nn.Module):
    """x shape (R, B, D) -> mean over R -> (B, D)."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=0)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(7)
    return [torch.randn(R, B, D, device=dev, generator=g)]


def get_init_inputs():
    return []
