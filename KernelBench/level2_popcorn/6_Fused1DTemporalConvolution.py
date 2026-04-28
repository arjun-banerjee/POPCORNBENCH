import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    **Fused depthwise–pointwise 1D temporal convolution** on ``(B, C, T)`` sequences.

    Fuses a **temporal depthwise** ``k``-tap conv (per-channel along time) with
    **SiLU** and a **pointwise** ``1×1`` mix—standard in temporal encoders for
    proprio, audio, or token-wise time series in robotics / VLA backbones.

    Padding uses dilated effective radius so length is unchanged for odd
    ``kernel_size``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "odd kernel_size keeps T unchanged with symmetric padding"
        assert dilation >= 1
        self.padding = ((kernel_size - 1) * dilation) // 2
        self.dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) temporal features.
        Returns:
            (B, C_out, T)
        """
        x = self.dw(x)
        x = F.silu(x)
        return self.pw(x)


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self._impl = Model(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


batch_size = 16
in_channels = 128
out_channels = 128
kernel_size = 5
dilation = 2
seq_len = 2048


def get_inputs():
    x = torch.randn(batch_size, in_channels, seq_len)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dilation]
