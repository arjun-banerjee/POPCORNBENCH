import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ModelNew is PyTorch-only. run_and_check with check_kernel=True and backend=cuda
# requires real CUDA (__global__ + load_inline/cpp_extension) in the kernel file—use
# check_kernel=False until you add a custom kernel implementation.

class Model(nn.Module):
    """
    Fused GroupNorm + SiLU (Swish), a common post-conv / residual-branch pattern in
    ResNet-family backbones (SiLU instead of ReLU) and in spatial blocks of diffusion
    transformers (DiT) where features are (N, C, H, W).

    Reference pipeline for a single fused kernel: normalize channels with
    `GroupNorm`, then apply `SiLU` elementwise (`x * sigmoid(x)`).
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        assert num_channels % num_groups == 0
        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) feature map
        Returns:
            Same shape as x
        """
        return F.silu(self.norm(x))


class ModelNew(nn.Module):
    """
    KernelBench candidate entry point. Replace with a fused CUDA/Triton kernel;
    default delegates to `Model` for harness smoke tests (same file as ref + kernel).
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self._impl = Model(num_channels, num_groups, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


batch_size = 8
num_channels = 256
num_groups = 32
height = 24
width = 24


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [num_channels, num_groups]
