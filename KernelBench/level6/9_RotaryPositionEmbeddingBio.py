import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE) applied to protein residue sequences.
    Used in ESM-2 and similar protein language models.  Encodes absolute
    position via rotation of query/key pairs in 2-D subspaces, preserving
    relative-position information in dot-product attention.
    """

    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def _build_cos_sin(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """
        Args:
            q: (B, H, N, D)  – queries
            k: (B, H, N, D)  – keys
        Returns:
            (q_rot, k_rot) each (B, H, N, D) with RoPE applied
        """
        N = q.shape[2]
        cos, sin = self._build_cos_sin(N, q.device)  # (N, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


dim = 64
num_heads = 8
seq_len = 256
batch_size = 4


def get_inputs():
    q = torch.randn(batch_size, num_heads, seq_len, dim)
    k = torch.randn(batch_size, num_heads, seq_len, dim)
    return [q, k]


def get_init_inputs():
    return [dim]
