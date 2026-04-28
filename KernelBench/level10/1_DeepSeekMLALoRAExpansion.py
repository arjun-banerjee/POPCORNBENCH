import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Low-rank expansion of compressed KV activations into multi-head K/V tensors.

    This isolates the MLA-style compressed-cache pattern: a hidden activation is
    projected into a low-rank latent space and then expanded back into separate
    key and value tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden: torch.Tensor,
        w_down: torch.Tensor,
        w_up_k: torch.Tensor,
        w_up_v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, model_dim = hidden.shape
        rank = w_down.shape[1]
        kv_heads = 4
        head_dim = w_up_k.shape[1] // kv_heads

        latent = hidden.reshape(batch_size * seq_len, model_dim) @ w_down
        k = latent @ w_up_k
        v = latent @ w_up_v
        k = k.view(batch_size, seq_len, kv_heads, head_dim)
        v = v.view(batch_size, seq_len, kv_heads, head_dim)
        return torch.stack((k, v), dim=2)


batch_size = 4
seq_len = 96
model_dim = 256
rank = 48
kv_heads = 4
head_dim = 32


def get_inputs():
    hidden = torch.randn(batch_size, seq_len, model_dim, dtype=torch.float32)
    w_down = torch.randn(model_dim, rank, dtype=torch.float32)
    w_up_k = torch.randn(rank, kv_heads * head_dim, dtype=torch.float32)
    w_up_v = torch.randn(rank, kv_heads * head_dim, dtype=torch.float32)
    return [hidden, w_down, w_up_k, w_up_v]


def get_init_inputs():
    return []
