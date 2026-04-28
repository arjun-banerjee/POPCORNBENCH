import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Multi-head row-wise softmax over CSR edge scores.

    This extends scalar graph edge softmax to multiple attention heads while
    preserving sparse CSR layout. Each row is normalized independently for each
    head.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        row_ptr: torch.Tensor,
        edge_scores: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = row_ptr.numel() - 1
        out = torch.empty_like(edge_scores)

        for node in range(num_nodes):
            start = int(row_ptr[node].item())
            end = int(row_ptr[node + 1].item())
            if end > start:
                out[start:end] = torch.softmax(edge_scores[start:end], dim=0)

        return out


num_nodes = 384
avg_degree = 18
num_heads = 8


def _make_row_ptr():
    degree = torch.full((num_nodes,), avg_degree, dtype=torch.int32)
    degree = torch.clamp(degree + ((torch.arange(num_nodes, dtype=torch.int32) % 5) - 2), min=1)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(degree, dim=0)
    return row_ptr


def get_inputs():
    row_ptr = _make_row_ptr()
    num_edges = int(row_ptr[-1].item())
    edge_scores = torch.randn(num_edges, num_heads, dtype=torch.float32)
    return [row_ptr, edge_scores]


def get_init_inputs():
    return []
