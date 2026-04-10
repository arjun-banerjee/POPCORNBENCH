"""
Order-sensitive virtual collective: out = sum_r GELU(x_r), same shape as one rank slice (B, D).

Note: this is not equal to GELU(sum_r x_r).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelbench.distributed_collectives import default_device

R = 4
B = 14
D = 40


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x).sum(dim=0)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(11)
    return [torch.randn(R, B, D, device=dev, generator=g)]


def get_init_inputs():
    return []
