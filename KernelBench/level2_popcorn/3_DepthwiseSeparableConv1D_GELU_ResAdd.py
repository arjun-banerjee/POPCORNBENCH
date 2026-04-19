import torch
import torch.nn as nn

class Model(nn.Module):
    """
    1D depthwise separable convolution block: pointwise Conv1d (channel mixing), depthwise Conv1d (spatial),
    GELU activation, and residual addition. Used in genomic foundation models (Enformer) for efficient
    sequence processing of DNA/RNA.
    """
    def __init__(self, channels, expanded_channels, kernel_size):
        super(Model, self).__init__()
        self.pointwise = nn.Conv1d(channels, expanded_channels, 1)
        self.depthwise = nn.Conv1d(expanded_channels, expanded_channels, kernel_size,
                                   padding=kernel_size // 2, groups=expanded_channels)
        self.activation = nn.GELU()
        self.project_back = nn.Conv1d(expanded_channels, channels, 1)

    def forward(self, x):
        residual = x
        x = self.pointwise(x)
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.project_back(x)
        x = x + residual
        return x

batch_size = 32
channels = 256
expanded_channels = 512
seq_len = 4096
kernel_size = 9

def get_inputs():
    return [torch.randn(batch_size, channels, seq_len)]

def get_init_inputs():
    return [channels, expanded_channels, kernel_size]