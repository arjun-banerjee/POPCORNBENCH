"""
Over virtual ranks R, for each (B, C) position take the maximum value; on ties choose the smallest rank index.

Output (B, C, 2): [:,:,0] = best value, [:,:,1] = winning rank as float32.
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

R = 4
B = 8
C = 28


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (R, B, C)
        best_val = x[0].clone()
        best_idx = torch.zeros(B, C, dtype=torch.long, device=x.device)
        for r in range(1, R):
            xv = x[r]
            improved = xv > best_val
            tied = (xv == best_val) & ~improved
            best_idx = torch.where(
                improved,
                torch.full_like(best_idx, r),
                torch.where(
                    tied,
                    torch.minimum(best_idx, torch.full_like(best_idx, r)),
                    best_idx,
                ),
            )
            best_val = torch.where(improved, xv, best_val)
        stacked = torch.stack([best_val, best_idx.to(best_val.dtype)], dim=-1)
        return stacked


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(16)
    return [torch.randn(R, B, C, device=dev, generator=g)]


def get_init_inputs():
    return []
