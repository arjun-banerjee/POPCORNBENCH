import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Weighted expert-output combine back into token-major order.

    Expert outputs are scattered back to token positions and scaled by the
    routing weight associated with each expert contribution.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        expert_hidden: torch.Tensor,
        token_idx: torch.Tensor,
        gates: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        out = torch.zeros(num_tokens, expert_hidden.shape[1], dtype=expert_hidden.dtype, device=expert_hidden.device)
        for row in range(expert_hidden.shape[0]):
            token = int(token_idx[row].item())
            out[token] += gates[row] * expert_hidden[row]
        return out


num_tokens = 1536
hidden_dim = 128
fanout = 2


def get_inputs():
    expert_hidden = torch.randn(num_tokens * fanout, hidden_dim, dtype=torch.float32)
    token_idx = torch.arange(num_tokens, dtype=torch.int32).repeat_interleave(fanout)
    gates = torch.softmax(torch.randn(num_tokens, fanout, dtype=torch.float32), dim=-1).reshape(-1)
    return [expert_hidden, token_idx, gates, num_tokens]


def get_init_inputs():
    return []
