import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Scatter-add edge features into destination-node accumulators.

    This is a canonical graph primitive for COO-formatted workloads where edge
    messages are already materialized and must be reduced into node states.
    """

    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes

    def forward(
        self,
        dst_idx: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> torch.Tensor:
        feat_dim = edge_feat.shape[1]
        out = torch.zeros(self.num_nodes, feat_dim, dtype=edge_feat.dtype, device=edge_feat.device)
        out.index_add_(0, dst_idx.long(), edge_feat)
        return out


num_nodes = 768
num_edges = 16384
feat_dim = 64


def get_inputs():
    dst_idx = torch.randint(0, num_nodes, (num_edges,), dtype=torch.int32)
    edge_feat = torch.randn(num_edges, feat_dim, dtype=torch.float32)
    return [dst_idx, edge_feat]


def get_init_inputs():
    return [num_nodes]
