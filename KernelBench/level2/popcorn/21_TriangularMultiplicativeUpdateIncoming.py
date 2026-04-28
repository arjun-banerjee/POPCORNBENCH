import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Triangular multiplicative update – incoming edges.
    Mirror of the outgoing variant: updates pair (i,j) by aggregating over
    the shared index k via edges (k,i) and (k,j).
    """

    def __init__(self, pair_dim, hidden_dim):
        super().__init__()
        self.layer_norm_in = nn.LayerNorm(pair_dim)
        self.proj_a = nn.Linear(pair_dim, hidden_dim)
        self.gate_a = nn.Linear(pair_dim, hidden_dim)
        self.proj_b = nn.Linear(pair_dim, hidden_dim)
        self.gate_b = nn.Linear(pair_dim, hidden_dim)
        self.layer_norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, pair_dim)
        self.out_gate = nn.Linear(pair_dim, pair_dim)

    def forward(self, pair_repr: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, N, N, pair_dim)
            pair_mask: (B, N, N)
        Returns:
            updated pair representation (B, N, N, pair_dim)
        """
        mask = pair_mask.unsqueeze(-1)
        x = self.layer_norm_in(pair_repr)

        a = self.proj_a(x) * torch.sigmoid(self.gate_a(x)) * mask  # (B, N, N, H)
        b = self.proj_b(x) * torch.sigmoid(self.gate_b(x)) * mask

        # Incoming: aggregate over k with a[k,i] * b[k,j]
        out = torch.einsum("bkih,bkjh->bijh", a, b)

        out = self.layer_norm_out(out)
        out = self.out_proj(out)
        out = torch.sigmoid(self.out_gate(pair_repr)) * out
        return out


pair_dim = 64
hidden_dim = 32
seq_len = 32
batch_size = 2


def get_inputs():
    pair_repr = torch.randn(batch_size, seq_len, seq_len, pair_dim)
    pair_mask = torch.ones(batch_size, seq_len, seq_len)
    return [pair_repr, pair_mask]


def get_init_inputs():
    return [pair_dim, hidden_dim]
