"""
Fixed permutation along the feature axis: y[b, j] = x[b, perm[j]].
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

B = 10
N = 36

_g = torch.Generator()
_g.manual_seed(10)
PERM = torch.randperm(N, generator=_g)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, PERM.to(device=x.device)]


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(10)
    return [torch.randn(B, N, device=dev, generator=g)]


def get_init_inputs():
    return []
