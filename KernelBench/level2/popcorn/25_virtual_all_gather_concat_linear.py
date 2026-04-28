"""
Virtual all-gather on the last dim: stack R tensors of shape (B, H) into (B, R*H), then Linear.
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

R = 4
B = 12
H = 32
K = 56


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(R * H, K, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, b, h = x.shape
        assert r == R and h == H
        parts = [x[i] for i in range(R)]
        gathered = torch.cat(parts, dim=-1)
        return self.proj(gathered)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(9)
    return [torch.randn(R, B, H, device=dev, generator=g)]


def get_init_inputs():
    return []
