"""
Unequal virtual shard lengths: x is (R, S_MAX) with a fixed boolean mask marking valid cells.
Output (S_MAX,) is sum_r x[r, j] * mask[r, j] (masked sum along the virtual rank dimension).

Padding positions are masked out so they do not contribute.
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

R = 4
S_MAX = 20

# mask[r, j] == True iff rank r contributes a real value at column j
_MASK_DATA = [
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "shard_mask",
            torch.tensor(_MASK_DATA, dtype=torch.bool),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.shard_mask.to(dtype=x.dtype, device=x.device)
        return (x * m).sum(dim=0)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(8)
    return [torch.randn(R, S_MAX, device=dev, generator=g)]


def get_init_inputs():
    return []
