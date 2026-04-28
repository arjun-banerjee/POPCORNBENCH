import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Triangular self-attention over pair representations, as used in
    AlphaFold2's Evoformer stack.  Starting-node variant: for each
    row i of the pair matrix, standard multi-head self-attention is
    performed across the column dimension j, with an additive pair
    bias.  This lets positions (i,j) and (i,k) exchange information
    through their shared starting node i.
    """

    def __init__(self, pair_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        assert pair_dim % num_heads == 0

        self.layer_norm = nn.LayerNorm(pair_dim)
        self.query = nn.Linear(pair_dim, pair_dim, bias=False)
        self.key = nn.Linear(pair_dim, pair_dim, bias=False)
        self.value = nn.Linear(pair_dim, pair_dim, bias=False)
        self.gate = nn.Linear(pair_dim, pair_dim)
        self.bias_proj = nn.Linear(pair_dim, num_heads, bias=False)
        self.out_proj = nn.Linear(pair_dim, pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_repr: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, N, N, pair_dim)
            pair_mask: (B, N, N)
        Returns:
            updated pair representation (B, N, N, pair_dim)
        """
        B, N, _, D = pair_repr.shape
        h = self.num_heads
        d = self.head_dim

        x = self.layer_norm(pair_repr)
        g = torch.sigmoid(self.gate(x))

        # Pair bias from the pair representation itself
        bias = self.bias_proj(x).permute(0, 3, 1, 2)  # (B, h, N, N)

        # Reshape to (B*N, N, D) — attention across columns for each row
        x_flat = x.reshape(B * N, N, D)
        q = self.query(x_flat).view(B * N, N, h, d).transpose(1, 2)  # (B*N, h, N, d)
        k = self.key(x_flat).view(B * N, N, h, d).transpose(1, 2)
        v = self.value(x_flat).view(B * N, N, h, d).transpose(1, 2)

        attn = (q @ k.transpose(-1, -2)) / math.sqrt(d)  # (B*N, h, N, N)
        attn = attn.view(B, N, h, N, N)

        # Add pair bias: bias (B, h, N, N) → (B, 1, h, N, N) broadcast across rows
        attn = attn + bias.unsqueeze(1)

        # Apply mask: pair_mask (B, N, N) → mask over the key dimension
        mask = pair_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N, N) – last dim is keys
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.view(B * N, h, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, N, D)
        out = g * out
        return self.out_proj(out)


pair_dim = 64
num_heads = 4
seq_len = 32
batch_size = 2


def get_inputs():
    pair_repr = torch.randn(batch_size, seq_len, seq_len, pair_dim)
    pair_mask = torch.ones(batch_size, seq_len, seq_len)
    return [pair_repr, pair_mask]


def get_init_inputs():
    return [pair_dim, num_heads]
