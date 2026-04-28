import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).

class Model(nn.Module):
    """
    Fused 1D convolution + GroupNorm for temporal sequences (N, C, T).

    Typical in time-series heads, audio / motor trajectories, and sequence models
    that use strided or padded Conv1d followed by per-channel normalization without
    collapsing the time dimension (GroupNorm over `out_channels` on (N, C_out, T')).

    Reference fusion target: `Conv1d` → `GroupNorm` (no activation), matching a
    common stem or residual branch in temporal backbones.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        assert out_channels % num_groups == 0
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=out_channels,
            eps=eps,
            affine=affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, T) temporal feature map
        Returns:
            (N, C_out, T') with T' determined by conv stride / padding / kernel
        """
        return self.norm(self.conv(x))


class ModelNew(nn.Module):
    """
    KernelBench candidate entry point. Replace with a fused CUDA/Triton kernel;
    default delegates to `Model` for harness smoke tests.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        self._impl = Model(
            in_channels,
            out_channels,
            num_groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            eps=eps,
            affine=affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


batch_size = 16
in_channels = 128
out_channels = 256
num_groups = 32
seq_len = 1024
kernel_size = 3
stride = 1
padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        num_groups,
        kernel_size,
        stride,
        padding,
    ]
