import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSARowSelfAttention(nn.Module):
    def __init__(self, msa_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = msa_dim // num_heads
        self.norm = nn.LayerNorm(msa_dim)
        self.q = nn.Linear(msa_dim, msa_dim, bias=False)
        self.k = nn.Linear(msa_dim, msa_dim, bias=False)
        self.v = nn.Linear(msa_dim, msa_dim, bias=False)
        self.out = nn.Linear(msa_dim, msa_dim)

    def forward(self, msa):
        B, S, N, D = msa.shape
        h, d = self.num_heads, self.head_dim
        x = self.norm(msa)
        q = self.q(x).view(B, S, N, h, d).permute(0, 1, 3, 2, 4)
        k = self.k(x).view(B, S, N, h, d).permute(0, 1, 3, 2, 4)
        v = self.v(x).view(B, S, N, h, d).permute(0, 1, 3, 2, 4)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2, 4).reshape(B, S, N, D)
        return self.out(out)


class OuterProductMeanSmall(nn.Module):
    def __init__(self, msa_dim, pair_dim, hidden=32):
        super().__init__()
        self.norm = nn.LayerNorm(msa_dim)
        self.a = nn.Linear(msa_dim, hidden)
        self.b = nn.Linear(msa_dim, hidden)
        self.out = nn.Linear(hidden * hidden, pair_dim)

    def forward(self, msa):
        x = self.norm(msa)
        a = self.a(x)
        b = self.b(x)
        outer = torch.einsum("bsih,bsjk->bijhk", a, b)
        S = msa.shape[1]
        outer = outer / max(S, 1)
        B, N1, N2, H1, H2 = outer.shape
        return self.out(outer.reshape(B, N1, N2, H1 * H2))


class TriMulOut(nn.Module):
    def __init__(self, pair_dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.pa = nn.Linear(pair_dim, hidden)
        self.pb = nn.Linear(pair_dim, hidden)
        self.ga = nn.Linear(pair_dim, hidden)
        self.gb = nn.Linear(pair_dim, hidden)
        self.norm_out = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, pair_dim)
        self.gate_out = nn.Linear(pair_dim, pair_dim)

    def forward(self, pair):
        x = self.norm(pair)
        a = self.pa(x) * torch.sigmoid(self.ga(x))
        b = self.pb(x) * torch.sigmoid(self.gb(x))
        o = torch.einsum("bikh,bjkh->bijh", a, b)
        o = self.out(self.norm_out(o))
        return torch.sigmoid(self.gate_out(pair)) * o


class Model(nn.Module):
    """
    Full Evoformer block from AlphaFold2 combining MSA row self-attention,
    outer-product mean update, and triangular multiplicative update of the
    pair representation.  A single block of the iterative refinement stack.
    """

    def __init__(self, msa_dim, pair_dim, num_heads, tri_hidden):
        super().__init__()
        self.msa_attn = MSARowSelfAttention(msa_dim, num_heads)
        self.opm = OuterProductMeanSmall(msa_dim, pair_dim)
        self.tri_mul = TriMulOut(pair_dim, tri_hidden)

    def forward(
        self, msa_repr: torch.Tensor, pair_repr: torch.Tensor
    ) -> tuple:
        """
        Args:
            msa_repr:  (B, S, N, msa_dim)
            pair_repr: (B, N, N, pair_dim)
        Returns:
            (updated_msa, updated_pair)
        """
        msa_repr = msa_repr + self.msa_attn(msa_repr)
        pair_repr = pair_repr + self.opm(msa_repr)
        pair_repr = pair_repr + self.tri_mul(pair_repr)
        return msa_repr, pair_repr


msa_dim = 64
pair_dim = 64
num_heads = 4
tri_hidden = 32
seq_len = 32
msa_depth = 8
batch_size = 2


def get_inputs():
    msa = torch.randn(batch_size, msa_depth, seq_len, msa_dim)
    pair = torch.randn(batch_size, seq_len, seq_len, pair_dim)
    return [msa, pair]


def get_init_inputs():
    return [msa_dim, pair_dim, num_heads, tri_hidden]
