import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fused sparse attention-value aggregation in CSR format.

    For each destination row, compute a softmax over edge logits and then use
    the normalized weights to aggregate source node values. This is a common
    fused graph-attention pattern.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        edge_scores: torch.Tensor,
        node_value: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = row_ptr.numel() - 1
        feat_dim = node_value.shape[1]
        out = torch.zeros(num_nodes, feat_dim, dtype=node_value.dtype, device=node_value.device)

        for dst in range(num_nodes):
            start = int(row_ptr[dst].item())
            end = int(row_ptr[dst + 1].item())
            if end > start:
                weights = torch.softmax(edge_scores[start:end], dim=0)
                src = col_idx[start:end].long()
                out[dst] = (node_value[src] * weights.unsqueeze(-1)).sum(dim=0)

        return out


num_nodes = 384
avg_degree = 18
feat_dim = 64


def get_inputs():
    degree = torch.full((num_nodes,), avg_degree, dtype=torch.int32)
    degree = torch.clamp(degree + ((torch.arange(num_nodes, dtype=torch.int32) % 7) - 3), min=1)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(degree, dim=0)
    num_edges = int(row_ptr[-1].item())
    col_idx = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int32)
    edge_scores = torch.randn(num_edges, dtype=torch.float32)
    node_value = torch.randn(num_nodes, feat_dim, dtype=torch.float32)
    return [row_ptr, col_idx, edge_scores, node_value]


def get_init_inputs():
    return []
