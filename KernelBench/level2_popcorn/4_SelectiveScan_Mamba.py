import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Selective scan (Mamba's S6 layer): projects input to derive input-dependent SSM parameters
    (delta, B, C), discretizes a learned state matrix A, runs the linear recurrence, and
    applies a skip connection via D. The fusion opportunity is keeping all intermediates
    (discretized parameters, hidden states) in SRAM rather than materializing them to global memory.

    Args:
        d_model (int): Input feature dimension.
        d_state (int): Hidden state dimension per channel (N in Mamba, typically 16).
    """
    def __init__(self, d_model: int, d_state: int):
        super(Model, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)
        self.proj_delta = nn.Linear(d_model, d_model, bias=True)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state

        delta = torch.nn.functional.softplus(self.proj_delta(x))
        B_t = self.proj_B(x)
        C_t = self.proj_C(x)

        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = delta.unsqueeze(-1) * B_t.unsqueeze(2)
        x_db = x.unsqueeze(-1) * dB

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + x_db[:, t]
            y_t = (h * C_t[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

batch_size = 16
d_model = 256
d_state = 16
seq_len = 2048

def get_inputs():
    return [torch.randn(batch_size, seq_len, d_model)]

def get_init_inputs():
    return [d_model, d_state]