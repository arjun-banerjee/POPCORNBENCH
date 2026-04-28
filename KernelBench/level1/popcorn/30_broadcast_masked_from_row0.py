"""
Simulated broadcast of row 0 into masked columns for all rows: where mask[j] is True,
every row matches x[0, j]; elsewhere keep x[b, j].
"""

import torch
import torch.nn as nn

from kernelbench.distributed_collectives import default_device

B = 18
D = 44

# Fixed mask (not all True, not all False)
_MASK = torch.tensor(
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    dtype=torch.bool,
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("row_mask", _MASK.clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.row_mask.to(device=x.device).view(1, D).expand(B, D)
        row0 = x[0:1].expand(B, D)
        return torch.where(m, row0, x)


def get_inputs():
    dev = default_device()
    g = torch.Generator(device=dev)
    g.manual_seed(13)
    return [torch.randn(B, D, device=dev, generator=g)]


def get_init_inputs():
    return []
