import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Grounded top-2 MoE router.

    Router logits are biased by token-expert similarity against learned
    grounding embeddings, then normalized over the selected top-2 experts.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        token_hidden: torch.Tensor,
        router_logits: torch.Tensor,
        expert_ground: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        grounded = router_logits + alpha * (token_hidden @ expert_ground.t())
        top_vals, top_idx = torch.topk(grounded, k=2, dim=-1)
        top_weights = torch.softmax(top_vals, dim=-1)
        return torch.stack((top_idx.to(torch.float32), top_weights), dim=-1)


num_tokens = 4096
hidden_dim = 128
num_experts = 16
alpha = 0.35


def get_inputs():
    token_hidden = torch.randn(num_tokens, hidden_dim, dtype=torch.float32)
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32)
    expert_ground = torch.randn(num_experts, hidden_dim, dtype=torch.float32)
    return [token_hidden, router_logits, expert_ground, alpha]


def get_init_inputs():
    return []
