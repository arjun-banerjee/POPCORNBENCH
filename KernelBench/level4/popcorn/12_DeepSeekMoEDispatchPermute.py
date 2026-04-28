import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Pack tokens into expert-major order for MoE execution.

    Each token is assigned to one expert and one slot within that expert's
    buffer. The kernel writes token activations into the packed dispatch layout.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        token_hidden: torch.Tensor,
        expert_idx: torch.Tensor,
        slot_idx: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        num_rows = int(expert_offsets[-1].item())
        out = torch.zeros(num_rows, token_hidden.shape[1], dtype=token_hidden.dtype, device=token_hidden.device)
        for token in range(token_hidden.shape[0]):
            expert = int(expert_idx[token].item())
            row = int(expert_offsets[expert].item() + slot_idx[token].item())
            out[row] = token_hidden[token]
        return out


num_tokens = 2048
hidden_dim = 128
num_experts = 16


def get_inputs():
    token_hidden = torch.randn(num_tokens, hidden_dim, dtype=torch.float32)
    expert_idx = torch.arange(num_tokens, dtype=torch.int32) % num_experts
    counts = torch.bincount(expert_idx.to(torch.int64), minlength=num_experts).to(torch.int32)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32)
    expert_offsets[1:] = torch.cumsum(counts, dim=0)
    slot_cursor = torch.zeros(num_experts, dtype=torch.int32)
    slot_idx = torch.empty(num_tokens, dtype=torch.int32)
    for token in range(num_tokens):
        expert = int(expert_idx[token].item())
        slot_idx[token] = slot_cursor[expert]
        slot_cursor[expert] += 1
    return [token_hidden, expert_idx, slot_idx, expert_offsets]


def get_init_inputs():
    return []
