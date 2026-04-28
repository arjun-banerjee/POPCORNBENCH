import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Row-wise gated self-attention over Multiple Sequence Alignment (MSA)
    representations.  Each row of the MSA is an independent sequence; for
    each row, standard multi-head attention is performed across residue
    positions, with an additive bias from the pair representation.
    """

    def __init__(self, msa_dim, pair_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = msa_dim // num_heads
        assert msa_dim % num_heads == 0

        self.layer_norm = nn.LayerNorm(msa_dim)
        self.query = nn.Linear(msa_dim, msa_dim, bias=False)
        self.key = nn.Linear(msa_dim, msa_dim, bias=False)
        self.value = nn.Linear(msa_dim, msa_dim, bias=False)
        self.gate = nn.Linear(msa_dim, msa_dim)
        self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.out_proj = nn.Linear(msa_dim, msa_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        msa_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            msa_repr:  (B, S, N, msa_dim)
            pair_repr: (B, N, N, pair_dim)
            msa_mask:  (B, S, N)
        Returns:
            updated MSA (B, S, N, msa_dim)
        """
        B, S, N, D = msa_repr.shape
        h = self.num_heads
        d = self.head_dim

        x = self.layer_norm(msa_repr)
        q = self.query(x).view(B, S, N, h, d).permute(0, 1, 3, 2, 4)
        k = self.key(x).view(B, S, N, h, d).permute(0, 1, 3, 2, 4)
        v = self.value(x).view(B, S, N, h, d).permute(0, 1, 3, 2, 4)
        g = torch.sigmoid(self.gate(x)).view(B, S, N, h, d).permute(0, 1, 3, 2, 4)

        # Pair bias: shared across MSA rows
        bias = self.pair_bias(pair_repr).permute(0, 3, 1, 2)  # (B, h, N, N)

        attn = torch.einsum("bshid,bshjd->bshij", q, k) / math.sqrt(d)
        attn = attn + bias.unsqueeze(1)

        mask = msa_mask.view(B, S, 1, 1, N).expand_as(attn)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bshij,bshjd->bshid", attn, v)
        out = g * out
        out = out.permute(0, 1, 3, 2, 4).reshape(B, S, N, D)
        return self.out_proj(out)


msa_dim = 64
pair_dim = 64
num_heads = 4
seq_len = 32
msa_depth = 8
batch_size = 2


def get_inputs():
    msa_repr = torch.randn(batch_size, msa_depth, seq_len, msa_dim)
    pair_repr = torch.randn(batch_size, seq_len, seq_len, pair_dim)
    msa_mask = torch.ones(batch_size, msa_depth, seq_len)
    return [msa_repr, pair_repr, msa_mask]


def get_init_inputs():
    return [msa_dim, pair_dim, num_heads]
