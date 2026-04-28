import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Row-wise softmax over edge scores stored in CSR format.

    This is a core primitive in graph attention networks and sparse attention on
    irregular neighborhoods. Each node owns a contiguous edge segment in
    `edge_scores[row_ptr[i]:row_ptr[i + 1]]`, and the kernel normalizes scores
    independently within each segment.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        row_ptr: torch.Tensor,
        edge_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            row_ptr:      (num_nodes + 1,) CSR row offsets
            edge_scores:  (num_edges,) unnormalized attention logits
        Returns:
            (num_edges,) softmax-normalized edge weights per source node
        """
        num_nodes = row_ptr.numel() - 1
        out = torch.empty_like(edge_scores)

        for node in range(num_nodes):
            start = int(row_ptr[node].item())
            end = int(row_ptr[node + 1].item())
            if end > start:
                out[start:end] = torch.softmax(edge_scores[start:end], dim=0)

        return out


num_nodes = 512
avg_degree = 24
batchless_num_edges = num_nodes * avg_degree


def _make_row_ptr(num_nodes: int, avg_degree: int) -> torch.Tensor:
    base = torch.full((num_nodes,), avg_degree, dtype=torch.int32)
    degree_jitter = (torch.arange(num_nodes, dtype=torch.int32) % 7) - 3
    degrees = torch.clamp(base + degree_jitter, min=1)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(degrees, dim=0)
    return row_ptr


def get_inputs():
    row_ptr = _make_row_ptr(num_nodes, avg_degree)
    num_edges = int(row_ptr[-1].item())
    edge_scores = torch.randn(num_edges, dtype=torch.float32)
    return [row_ptr, edge_scores]


def get_init_inputs():
    return []
