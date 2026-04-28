import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).

class Model(nn.Module):
    """
    Single-layer multi-head cross-attention with RoPE on Q and K, attention
    dropout, and output projection—intended as a fusion target for robotics
    stacks (e.g. proprio / action queries attending to vision or memory).

    Pipeline (conceptually one fused kernel): linear Q from query stream,
    linear K/V from key–value stream → reshape heads → apply RoPE separately
    along query positions (T_q) and key positions (T_k) → scaled dot-product,
    softmax, dropout on weights → aggregate values → output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires an even head dimension"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE along the sequence dimension (last-but-one after head). x: (B, H, L, d)."""
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().view(1, 1, seq_len, self.head_dim)
        sin = emb.sin().view(1, 1, seq_len, self.head_dim)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x_query: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_query: (B, T_q, dim) — e.g. robot or decoder tokens
            x_kv:    (B, T_k, dim) — e.g. encoder / memory tokens
        Returns:
            (B, T_q, dim)
        """
        B, T_q, _ = x_query.shape
        _, T_k, _ = x_kv.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(x_query).view(B, T_q, h, d).transpose(1, 2)
        k = self.k_proj(x_kv).view(B, T_k, h, d).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, T_k, h, d).transpose(1, 2)

        q = self._rope(q, T_q)
        k = self._rope(k, T_k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.dim)
        return self.out_proj(out)


class ModelNew(nn.Module):
    """
    KernelBench candidate entry point (`eval_kernel_against_ref` / `run_and_check`).
    Replace this class with a fused CUDA/Triton (etc.) implementation; the default
    delegates to `Model` so you can smoke-test the harness with the same file for
    both `ref_arch_src_path` and `kernel_src_path`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self._impl = Model(dim, num_heads, dropout, rope_base)

    def forward(self, x_query: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        return self._impl(x_query, x_kv)


batch_size = 4
seq_len_q = 64
seq_len_kv = 128
dim = 256
num_heads = 8
# Use 0.0 so `run_and_check` / multi-trial correctness matches: the harness does not
# re-seed RNG between the reference forward and `ModelNew` forward, so attention
# dropout > 0 would make outputs differ even for identical math.
dropout_p = 0.0


def get_inputs():
    x_query = torch.randn(batch_size, seq_len_q, dim)
    x_kv = torch.randn(batch_size, seq_len_kv, dim)
    return [x_query, x_kv]


def get_init_inputs():
    return [dim, num_heads, dropout_p]
