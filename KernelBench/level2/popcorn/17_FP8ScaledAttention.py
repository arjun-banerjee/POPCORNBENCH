import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Scaled low-precision attention with explicit dequantization semantics.

    The inputs are quantized tensors plus per-head scales. This mirrors the data
    path of FP8-style scaled matmul attention even though the reference computes
    everything through explicit dequantization in PyTorch.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q_q: torch.Tensor,
        k_q: torch.Tensor,
        v_q: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> torch.Tensor:
        q = q_q.float() * q_scale.view(1, 1, -1, 1)
        k = k_q.float() * k_scale.view(1, 1, -1, 1)
        v = v_q.float() * v_scale.view(1, 1, -1, 1)

        batch_size, seq_len, num_heads, head_dim = q.shape
        out = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(head_dim)

        for b in range(batch_size):
            for h in range(num_heads):
                for t in range(seq_len):
                    q_t = q[b, t, h]
                    k_hist = k[b, : t + 1, h]
                    v_hist = v[b, : t + 1, h]
                    scores = (k_hist * q_t.unsqueeze(0)).sum(dim=-1) * scale
                    attn = torch.softmax(scores, dim=0)
                    out[b, t, h] = (attn.unsqueeze(-1) * v_hist).sum(dim=0)

        return out


batch_size = 2
seq_len = 72
num_heads = 4
head_dim = 32


def _quantize_per_head(x: torch.Tensor):
    scale = x.abs().amax(dim=(0, 1, 3)).clamp(min=1e-4) / 127.0
    q = torch.clamp(torch.round(x / scale.view(1, 1, -1, 1)), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def get_inputs():
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    q_q, q_scale = _quantize_per_head(q)
    k_q, k_scale = _quantize_per_head(k)
    v_q, v_scale = _quantize_per_head(v)
    return [q_q, k_q, v_q, q_scale, k_scale, v_scale]


def get_init_inputs():
    return []
