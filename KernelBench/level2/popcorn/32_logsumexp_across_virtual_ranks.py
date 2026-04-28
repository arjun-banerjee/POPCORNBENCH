"""
Stable log-sum-exp over virtual ranks: out = logsumexp(x, dim=0), shape (B, D).
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

R = 4
B = 10
D = 32


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x, dim=0)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(12)
    return [torch.randn(R, B, D, device=dev, generator=g)]


def get_init_inputs():
    return []
