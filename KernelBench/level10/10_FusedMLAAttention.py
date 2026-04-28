import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fused MLA-style attention from compressed KV latents.

    The compressed latent representation is expanded into keys and values on the
    fly inside the attention computation rather than materializing a KV cache
    first. This isolates the main compute pattern behind fused MLA attention.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        w_up_k: torch.Tensor,
        w_up_v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        rank = kv_latent.shape[-1]
        out = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(head_dim)

        for b in range(batch_size):
            for h in range(num_heads):
                for t in range(seq_len):
                    logits = []
                    values = []
                    for s in range(t + 1):
                        latent = kv_latent[b, s]
                        k = torch.matmul(latent, w_up_k[:, h])
                        v = torch.matmul(latent, w_up_v[:, h])
                        logits.append((q[b, t, h] * k).sum() * scale)
                        values.append(v)
                    attn = torch.softmax(torch.stack(logits, dim=0), dim=0)
                    value_stack = torch.stack(values, dim=0)
                    out[b, t, h] = (attn.unsqueeze(-1) * value_stack).sum(dim=0)

        return out


batch_size = 2
seq_len = 32
num_heads = 4
head_dim = 16
rank = 24


def get_inputs():
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    kv_latent = torch.randn(batch_size, seq_len, rank, dtype=torch.float32)
    w_up_k = torch.randn(rank, num_heads, head_dim, dtype=torch.float32)
    w_up_v = torch.randn(rank, num_heads, head_dim, dtype=torch.float32)
    return [q, kv_latent, w_up_k, w_up_v]


def get_init_inputs():
    return []
