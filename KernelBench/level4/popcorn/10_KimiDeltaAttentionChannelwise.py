import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Channel-wise gated delta attention recurrence.

    The recurrent state decays independently for each output channel, which
    makes the update closer to channel-gated delta attention variants used in
    fast long-context baselines.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = q.shape
        state = torch.zeros(batch_size, num_heads, head_dim, head_dim, dtype=q.dtype, device=q.device)
        outputs = []

        for t in range(seq_len):
            gate_t = gate[:, t].unsqueeze(-2)
            kv_outer = k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
            state = gate_t * state + kv_outer
            y_t = (q[:, t].unsqueeze(-1) * state).sum(dim=-2)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


batch_size = 2
seq_len = 128
num_heads = 4
head_dim = 24


def get_inputs():
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    gate = torch.sigmoid(torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32))
    return [q, k, v, gate]


def get_init_inputs():
    return []
