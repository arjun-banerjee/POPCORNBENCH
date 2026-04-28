import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    Graph-Fused VLA (GF-VLA) style **scene graph relation update** in one forward.

    Objects are nodes with vision-language aligned embeddings; directed pairs
    (i → j) carry pairwise conditioning (geometry, language grounding, or fused
    VLA features). This reference fuses:

      LayerNorm(nodes) → multi-head attention over objects with **pairwise
      relation MLP bias** on edges → residual write-back.

    Intended as a fusion target for navigation / manipulation stacks that maintain
    explicit scene graphs and update them from multimodal observations.
    """

    def __init__(self, dim: int, pair_dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.rel_mlp = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.SiLU(),
            nn.Linear(pair_dim, num_heads),
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        nodes: torch.Tensor,
        pair_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            nodes: (B, N, D) object / region embeddings.
            pair_feats: (B, N, N, P) pairwise VLA or geometric conditioning.
            edge_mask: (B, N, N) float in {0, 1}; 1 keeps attention over column j.
        Returns:
            (B, N, D) updated node embeddings (residual around attention block).
        """
        B, N, D = nodes.shape
        _, N2, N3, P = pair_feats.shape
        assert N == N2 == N3 and P == self.pair_dim
        h = self.norm(nodes)

        q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        rel_bias = self.rel_mlp(pair_feats).permute(0, 3, 1, 2)
        logits = logits + rel_bias

        logits = logits.masked_fill(edge_mask.unsqueeze(1) < 0.5, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return nodes + self.out_proj(ctx)


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, dim: int, pair_dim: int, num_heads: int):
        super().__init__()
        self._impl = Model(dim, pair_dim, num_heads)

    def forward(
        self,
        nodes: torch.Tensor,
        pair_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._impl(nodes, pair_feats, edge_mask)


batch_size = 4
num_objects = 24
dim = 96
pair_dim = 16
num_heads = 4


def get_inputs():
    nodes = torch.randn(batch_size, num_objects, dim)
    pair_feats = torch.randn(batch_size, num_objects, num_objects, pair_dim)
    edge_mask = torch.ones(batch_size, num_objects, num_objects)
    return [nodes, pair_feats, edge_mask]


def get_init_inputs():
    return [dim, pair_dim, num_heads]
