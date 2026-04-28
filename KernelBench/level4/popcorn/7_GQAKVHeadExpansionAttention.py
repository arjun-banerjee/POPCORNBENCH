import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Causal grouped-query attention with KV-head expansion.

    Query heads outnumber KV heads. Keys and values are expanded by repeating
    each KV head across its assigned query-head group before causal attention.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[2]
        group_size = num_q_heads // num_kv_heads

        expanded_k = k.repeat_interleave(group_size, dim=2)
        expanded_v = v.repeat_interleave(group_size, dim=2)

        out = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(head_dim)

        for b in range(batch_size):
            for h in range(num_q_heads):
                for t in range(seq_len):
                    q_t = q[b, t, h]
                    k_hist = expanded_k[b, : t + 1, h]
                    v_hist = expanded_v[b, : t + 1, h]
                    scores = (k_hist * q_t.unsqueeze(0)).sum(dim=-1) * scale
                    attn = torch.softmax(scores, dim=0)
                    out[b, t, h] = (attn.unsqueeze(-1) * v_hist).sum(dim=0)

        return out


batch_size = 2
seq_len = 80
num_q_heads = 8
num_kv_heads = 2
head_dim = 32


def get_inputs():
    q = torch.randn(batch_size, seq_len, num_q_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, dtype=torch.float32)
    return [q, k, v]


def get_init_inputs():
    return []
