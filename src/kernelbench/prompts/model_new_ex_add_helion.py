import torch
import torch.nn as nn
import helion
import helion.language as hl


@helion.kernel
def add_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for tile_idx in hl.tile(a.size()):
        out[tile_idx] = a[tile_idx] + b[tile_idx]
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return add_kernel(a, b)
