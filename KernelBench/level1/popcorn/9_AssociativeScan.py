import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a parallel associative scan (linear recurrence) over a 1D sequence.
    Given coefficient a and input b at each timestep, computes:
        h_0 = b_0
        h_t = a_t * h_{t-1} + b_t  for t > 0
    This is the core parallelizable primitive underlying Mamba's selective scan and
    other SSM architectures. Sequential in naive PyTorch but can be parallelized
    via the associative scan algorithm in O(L log L) span.

    Args:
        d_model (int): Number of independent recurrence channels.
    """
    def __init__(self, d_model: int):
        super(Model, self).__init__()
        self.d_model = d_model

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        B, L, D = a.shape
        h = torch.zeros(B, D, device=a.device, dtype=a.dtype)
        outputs = []
        for t in range(L):
            h = a[:, t] * h + b[:, t]
            outputs.append(h)
        return torch.stack(outputs, dim=1)

batch_size = 32
d_model = 256
seq_len = 4096

def get_inputs():
    a = torch.sigmoid(torch.randn(batch_size, seq_len, d_model))
    b = torch.randn(batch_size, seq_len, d_model)
    return [a, b]

def get_init_inputs():
    return [d_model]