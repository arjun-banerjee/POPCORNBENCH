import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Outer-product mean: converts MSA representation to a pair
    representation update.  Used in AlphaFold2 to communicate information
    from the MSA stack to the pair stack.  For each pair (i,j), the
    outer product of projected MSA features at positions i and j is
    averaged over the MSA depth dimension.
    """

    def __init__(self, msa_dim, pair_dim, hidden_dim=32):
        super().__init__()
        self.layer_norm = nn.LayerNorm(msa_dim)
        self.proj_a = nn.Linear(msa_dim, hidden_dim)
        self.proj_b = nn.Linear(msa_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * hidden_dim, pair_dim)

    def forward(self, msa_repr: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa_repr: (B, S, N, msa_dim) – MSA representation
            msa_mask: (B, S, N)          – 1 for valid rows/positions
        Returns:
            pair update (B, N, N, pair_dim)
        """
        x = self.layer_norm(msa_repr)
        a = self.proj_a(x)  # (B, S, N, H)
        b = self.proj_b(x)

        mask = msa_mask.unsqueeze(-1)  # (B, S, N, 1)
        a = a * mask
        b = b * mask

        outer = torch.einsum("bsih,bsjk->bijhk", a, b)  # (B, N, N, H, H)
        denom = msa_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).clamp(min=1)
        outer = outer / denom  # mean over S

        B, N1, N2, H1, H2 = outer.shape
        outer = outer.reshape(B, N1, N2, H1 * H2)
        return self.out_proj(outer)


msa_dim = 64
pair_dim = 64
hidden_dim = 32
seq_len = 32
msa_depth = 8
batch_size = 2


def get_inputs():
    msa_repr = torch.randn(batch_size, msa_depth, seq_len, msa_dim)
    msa_mask = torch.ones(batch_size, msa_depth, seq_len)
    return [msa_repr, msa_mask]


def get_init_inputs():
    return [msa_dim, pair_dim, hidden_dim]
