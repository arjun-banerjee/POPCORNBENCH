import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise-separable 1D convolution: a depthwise Conv1d (each channel independently)
    followed by a pointwise Conv1d (1x1, mixes channels). Used in Enformer and genomic sequence
    models for efficient spatial+channel factorized processing.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the pointwise convolution.
        kernel_size (int): Kernel size of the depthwise convolution.
        stride (int, optional): Stride of the depthwise convolution. Defaults to 1.
        padding (int, optional): Padding for the depthwise convolution. Defaults to 0.
        bias (bool, optional): If True, adds a learnable bias. Defaults to False.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

batch_size = 32
in_channels = 256
out_channels = 512
kernel_size = 9
seq_len = 4096
stride = 1
padding = 4

def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_len)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]