import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 1D convolution via FFT for long learnable filters. Computes circular convolution by
    zero-padding to avoid wraparound, performing FFT on both input and filter, multiplying in
    frequency domain, and inverse FFT. This is the core computational primitive in Hyena, H3,
    S4, and other SSM-based architectures for subquadratic sequence mixing.

    Args:
        channels (int): Number of independent channels (each has its own filter).
        seq_len (int): Length of input sequences.
    """
    def __init__(self, channels: int, seq_len: int):
        super(Model, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.fft_size = 2 * seq_len
        self.filter = nn.Parameter(torch.randn(channels, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        x_padded = torch.nn.functional.pad(x, (0, self.seq_len))
        k_padded = torch.nn.functional.pad(self.filter, (0, self.seq_len))
        x_f = torch.fft.rfft(x_padded, dim=-1)
        k_f = torch.fft.rfft(k_padded, dim=-1)
        out_f = x_f * k_f.unsqueeze(0)
        out = torch.fft.irfft(out_f, n=self.fft_size, dim=-1)
        return out[..., :self.seq_len]

batch_size = 16
channels = 256
seq_len = 8192

def get_inputs():
    return [torch.randn(batch_size, channels, seq_len)]

def get_init_inputs():
    return [channels, seq_len]