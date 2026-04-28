import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Apply rotary position embedding to incoming keys and update the KV cache.

    This isolates a common decode-time kernel: rotate the fresh key vectors for
    the current positions and store both keys and values into the running cache.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        updated_k = cache_k.clone()
        updated_v = cache_v.clone()
        rot_dim = k_new.shape[-1]

        for b in range(k_new.shape[0]):
            for t in range(k_new.shape[1]):
                pos = int(positions[b, t].item())
                c = cos[pos]
                s = sin[pos]
                x = k_new[b, t]
                x_even = x[..., 0::2]
                x_odd = x[..., 1::2]
                rot_even = x_even * c - x_odd * s
                rot_odd = x_even * s + x_odd * c
                rotated = torch.empty_like(x)
                rotated[..., 0::2] = rot_even
                rotated[..., 1::2] = rot_odd
                updated_k[b, pos] = rotated
                updated_v[b, pos] = v_new[b, t]

        return torch.stack((updated_k, updated_v), dim=0)


batch_size = 2
cache_len = 128
update_len = 16
num_heads = 4
head_dim = 32


def _rope_tables(cache_len: int, half_dim: int):
    positions = torch.arange(cache_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.pow(10000.0, -torch.arange(half_dim, dtype=torch.float32) / half_dim).unsqueeze(0)
    angles = positions * freqs
    return torch.cos(angles), torch.sin(angles)


def get_inputs():
    k_new = torch.randn(batch_size, update_len, num_heads, head_dim, dtype=torch.float32)
    v_new = torch.randn(batch_size, update_len, num_heads, head_dim, dtype=torch.float32)
    cos, sin = _rope_tables(cache_len, head_dim // 2)
    cache_k = torch.randn(batch_size, cache_len, num_heads, head_dim, dtype=torch.float32)
    cache_v = torch.randn(batch_size, cache_len, num_heads, head_dim, dtype=torch.float32)
    base = torch.arange(update_len, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1)
    positions = (base + torch.tensor([[7], [31]], dtype=torch.int32)) % cache_len
    return [k_new, v_new, cos, sin, cache_k, cache_v, positions]


def get_init_inputs():
    return []
