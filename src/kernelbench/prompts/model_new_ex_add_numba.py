import torch
import torch.nn as nn
from numba import cuda
import numpy as np


@cuda.jit
def add_kernel(a, b, out, n):
    """Numba CUDA kernel for element-wise addition."""
    idx = cuda.grid(1)
    if idx < n:
        out[idx] = a[idx] + b[idx]


def numba_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch the Numba CUDA element-wise add kernel."""
    n = a.numel()
    out = torch.empty_like(a)

    # Use cuda array interface for zero-copy interop
    a_flat = a.contiguous().view(-1)
    b_flat = b.contiguous().view(-1)
    out_flat = out.view(-1)

    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block

    add_kernel[blocks, threads_per_block](
        cuda.as_cuda_array(a_flat),
        cuda.as_cuda_array(b_flat),
        cuda.as_cuda_array(out_flat),
        n,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return numba_add(a, b)
