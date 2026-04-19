import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise 1D convolution where each input channel is convolved with its own filter.
    Fundamental primitive in Hyena, Mamba, and genomic sequence models (Enformer, HyenaDNA).

    Args:
        channels (int): Number of input/output channels.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If True, adds a learnable bias. Defaults to False.
    """
    def __init__(self, channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d = nn.Conv1d(channels, channels, kernel_size, stride=stride, padding=padding, groups=channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d(x)

batch_size = 32
channels = 512
kernel_size = 3
seq_len = 8192
stride = 1
padding = 1

def get_inputs():
    return [torch.randn(batch_size, channels, seq_len)]

def get_init_inputs():
    return [channels, kernel_size, stride, padding]