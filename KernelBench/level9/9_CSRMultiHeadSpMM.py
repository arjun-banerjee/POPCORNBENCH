import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Multi-head sparse aggregation with per-edge attention weights.

    Each edge carries one weight per head, and source node features are arranged
    as `(num_nodes, num_heads, head_dim)`. The output aggregates weighted
    source features into `(num_nodes, num_heads, head_dim)`.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        edge_weight: torch.Tensor,
        node_feat: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = row_ptr.numel() - 1
        num_heads = node_feat.shape[1]
        head_dim = node_feat.shape[2]
        out = torch.zeros(num_nodes, num_heads, head_dim, dtype=node_feat.dtype, device=node_feat.device)

        for dst in range(num_nodes):
            start = int(row_ptr[dst].item())
            end = int(row_ptr[dst + 1].item())
            if end > start:
                src = col_idx[start:end].long()
                weights = edge_weight[start:end].unsqueeze(-1)
                out[dst] = (node_feat[src] * weights).sum(dim=0)

        return out


num_nodes = 320
avg_degree = 14
num_heads = 8
head_dim = 32


def get_inputs():
    degree = torch.full((num_nodes,), avg_degree, dtype=torch.int32)
    degree = torch.clamp(degree + ((torch.arange(num_nodes, dtype=torch.int32) % 5) - 2), min=1)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(degree, dim=0)
    num_edges = int(row_ptr[-1].item())
    col_idx = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int32)
    edge_weight = torch.randn(num_edges, num_heads, dtype=torch.float32)
    node_feat = torch.randn(num_nodes, num_heads, head_dim, dtype=torch.float32)
    return [row_ptr, col_idx, edge_weight, node_feat]


def get_init_inputs():
    return []
