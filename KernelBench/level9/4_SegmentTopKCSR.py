import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Per-row top-k selection over CSR edge segments.

    Returns the top-k values and their original edge indices for each node's
    neighborhood. Useful for graph sparsification, hard attention, and
    candidate-pruning workloads.
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(
        self,
        row_ptr: torch.Tensor,
        edge_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = row_ptr.numel() - 1
        topk_vals = torch.full((num_nodes, self.k), float("-inf"), dtype=edge_scores.dtype, device=edge_scores.device)
        topk_idx = torch.full((num_nodes, self.k), -1, dtype=torch.int64, device=edge_scores.device)

        for node in range(num_nodes):
            start = int(row_ptr[node].item())
            end = int(row_ptr[node + 1].item())
            if end <= start:
                continue
            segment = edge_scores[start:end]
            curr_k = min(self.k, end - start)
            vals, idx = torch.topk(segment, k=curr_k, dim=0)
            topk_vals[node, :curr_k] = vals
            topk_idx[node, :curr_k] = idx.long() + start

        return topk_vals, topk_idx


num_nodes = 320
avg_degree = 12
k = 4


def _make_row_ptr():
    degree = torch.full((num_nodes,), avg_degree, dtype=torch.int32)
    degree = torch.clamp(degree + ((torch.arange(num_nodes, dtype=torch.int32) % 7) - 3), min=1)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(degree, dim=0)
    return row_ptr


def get_inputs():
    row_ptr = _make_row_ptr()
    num_edges = int(row_ptr[-1].item())
    edge_scores = torch.randn(num_edges, dtype=torch.float32)
    return [row_ptr, edge_scores]


def get_init_inputs():
    return [k]
