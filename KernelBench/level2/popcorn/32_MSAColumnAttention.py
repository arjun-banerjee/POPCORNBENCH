import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Column-wise gated self-attention over MSA representations.
    For each residue position, attention is computed across the MSA depth
    (sequence) dimension.  This lets the model share information between
    different aligned sequences at the same position.
    """

    def __init__(self, msa_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = msa_dim // num_heads
        assert msa_dim % num_heads == 0

        self.layer_norm = nn.LayerNorm(msa_dim)
        self.query = nn.Linear(msa_dim, msa_dim, bias=False)
        self.key = nn.Linear(msa_dim, msa_dim, bias=False)
        self.value = nn.Linear(msa_dim, msa_dim, bias=False)
        self.gate = nn.Linear(msa_dim, msa_dim)
        self.out_proj = nn.Linear(msa_dim, msa_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, msa_repr: torch.Tensor, msa_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            msa_repr: (B, S, N, msa_dim)
            msa_mask: (B, S, N)
        Returns:
            updated MSA (B, S, N, msa_dim)
        """
        B, S, N, D = msa_repr.shape
        h = self.num_heads
        d = self.head_dim

        # Transpose so column (residue position) is the batch dim for attention
        x = msa_repr.permute(0, 2, 1, 3)  # (B, N, S, D)
        x = self.layer_norm(x)

        q = self.query(x).view(B, N, S, h, d).permute(0, 1, 3, 2, 4)
        k = self.key(x).view(B, N, S, h, d).permute(0, 1, 3, 2, 4)
        v = self.value(x).view(B, N, S, h, d).permute(0, 1, 3, 2, 4)
        g = torch.sigmoid(self.gate(x)).view(B, N, S, h, d).permute(0, 1, 3, 2, 4)

        attn = torch.einsum("bnhid,bnhjd->bnhij", q, k) / math.sqrt(d)

        col_mask = msa_mask.permute(0, 2, 1)  # (B, N, S)
        mask = col_mask.view(B, N, 1, 1, S).expand_as(attn)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bnhij,bnhjd->bnhid", attn, v)
        out = g * out
        out = out.permute(0, 1, 3, 2, 4).reshape(B, N, S, D)
        out = self.out_proj(out)
        return out.permute(0, 2, 1, 3)  # back to (B, S, N, D)


msa_dim = 64
num_heads = 4
seq_len = 32
msa_depth = 8
batch_size = 2


def get_inputs():
    msa_repr = torch.randn(batch_size, msa_depth, seq_len, msa_dim)
    msa_mask = torch.ones(batch_size, msa_depth, seq_len)
    return [msa_repr, msa_mask]


def get_init_inputs():
    return [msa_dim, num_heads]
