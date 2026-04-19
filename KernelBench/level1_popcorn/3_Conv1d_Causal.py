import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a causal 1D convolution: left-padded so each output only depends on current and past inputs.
    Used in autoregressive sequence models (Mamba, Hyena, WaveNet, TCN) for causal sequence processing.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        dilation (int, optional): Dilation rate. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias. Defaults to False.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (self.causal_padding, 0))
        return self.conv1d(x)

batch_size = 32
in_channels = 256
out_channels = 256
kernel_size = 4
dilation = 1
seq_len = 8192

def get_inputs():
    return [torch.randn(batch_size, in_channels, seq_len)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dilation]