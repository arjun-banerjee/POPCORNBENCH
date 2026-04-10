"""
R int32 bitmasks per batch element: reduce with bitwise OR across virtual ranks, then popcount (32-bit).

Output shape (B,) float32: number of set bits in the unsigned lower 32 bits of the OR result.
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

R = 4
B = 24


def _popcount_u32(v: torch.Tensor) -> torch.Tensor:
    """v: int32 or int64 tensor (B,), return float32 popcount in 0..32."""
    b = v.shape[0]
    u = v.to(torch.int64) & 0xFFFFFFFF
    c = torch.zeros(b, dtype=torch.int64, device=v.device)
    for k in range(32):
        c = c + ((u >> k) & 1)
    return c.to(torch.float32)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (R, B) int32
        acc = x[0].to(torch.int64)
        for r in range(1, R):
            acc = acc | x[r].to(torch.int64)
        acc = acc & 0xFFFFFFFF
        return _popcount_u32(acc.to(torch.int32))


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(15)
    ints = torch.randint(-(2**31), 2**31 - 1, (R, B), device=dev, generator=g)
    return [ints.to(torch.int32)]


def get_init_inputs():
    return []
