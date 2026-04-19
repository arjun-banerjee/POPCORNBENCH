import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Full Hyena operator (order 2): projects input into value (v), query-like (q), and key-like (k)
    branches, applies short depthwise convolutions to q and k, generates a long convolution filter
    via a small MLP, convolves v with the filter via FFT, then applies two rounds of elementwise
    gating. This is the core sequence mixing operator in StripedHyena, HyenaDNA, and Evo
    for subquadratic long-range dependency modeling.

    Args:
        d_model (int): Model dimension.
        seq_len (int): Maximum sequence length.
        order (int): Hyena recurrence order (number of gating rounds, typically 2).
        short_filter_size (int): Kernel size for short depthwise convolutions.
    """
    def __init__(self, d_model: int, seq_len: int, order: int, short_filter_size: int):
        super(Model, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.order = order
        self.fft_size = 2 * seq_len

        self.in_proj = nn.Linear(d_model, d_model * (order + 1), bias=True)

        self.short_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, short_filter_size,
                      padding=short_filter_size // 2, groups=d_model)
            for _ in range(order)
        ])

        self.filter_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
        )
        t = torch.linspace(0, 1, seq_len).unsqueeze(-1)
        self.register_buffer('filter_positions', t)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        projected = self.in_proj(x)
        chunks = projected.chunk(self.order + 1, dim=-1)
        v = chunks[0].transpose(1, 2)
        gates = [c.transpose(1, 2) for c in chunks[1:]]

        for i in range(self.order):
            gates[i] = self.short_convs[i](gates[i])

        h = self.filter_mlp(self.filter_positions).transpose(0, 1)

        v_padded = F.pad(v, (0, self.seq_len))
        h_padded = F.pad(h, (0, self.seq_len))
        v_f = torch.fft.rfft(v_padded, dim=-1)
        h_f = torch.fft.rfft(h_padded, dim=-1)
        y_f = v_f * h_f.unsqueeze(0)
        y = torch.fft.irfft(y_f, n=self.fft_size, dim=-1)[..., :L]

        for gate in gates:
            y = y * gate

        y = y.transpose(1, 2)
        y = self.out_proj(y)
        return y

batch_size = 16
d_model = 256
seq_len = 8192
order = 2
short_filter_size = 3

def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]

def get_init_inputs():
    return [d_model, seq_len, order, short_filter_size]