import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Per-destination max aggregation over neighbor features in CSR format.

    This pattern appears in max-pooling GNNs and neighborhood feature
    extraction where reductions are not sums but feature-wise maxima.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        row_ptr: torch.Tensor,
        col_idx: torch.Tensor,
        node_feat: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = row_ptr.numel() - 1
        feat_dim = node_feat.shape[1]
        out = torch.full((num_nodes, feat_dim), float("-inf"), dtype=node_feat.dtype, device=node_feat.device)

        for dst in range(num_nodes):
            start = int(row_ptr[dst].item())
            end = int(row_ptr[dst + 1].item())
            if end > start:
                out[dst] = node_feat[col_idx[start:end].long()].amax(dim=0)

        return out


num_nodes = 640
avg_degree = 16
feat_dim = 80


def get_inputs():
    degree = torch.full((num_nodes,), avg_degree, dtype=torch.int32)
    degree = torch.clamp(degree + ((torch.arange(num_nodes, dtype=torch.int32) % 7) - 3), min=1)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int32)
    row_ptr[1:] = torch.cumsum(degree, dim=0)
    num_edges = int(row_ptr[-1].item())
    col_idx = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int32)
    node_feat = torch.randn(num_nodes, feat_dim, dtype=torch.float32)
    return [row_ptr, col_idx, node_feat]


def get_init_inputs():
    return []
