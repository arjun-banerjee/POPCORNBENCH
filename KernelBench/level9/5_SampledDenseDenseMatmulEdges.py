import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Edge-only sampled dense-dense matmul.

    For each graph edge (src, dst), compute a dot product between source and
    destination node embeddings. This pattern appears in link prediction,
    sampled attention, and graph contrastive learning.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        src_feat: torch.Tensor,
        dst_feat: torch.Tensor,
    ) -> torch.Tensor:
        src = src_feat[src_idx.long()]
        dst = dst_feat[dst_idx.long()]
        return (src * dst).sum(dim=-1)


num_src_nodes = 1024
num_dst_nodes = 768
num_edges = 16384
feat_dim = 128


def get_inputs():
    src_idx = torch.randint(0, num_src_nodes, (num_edges,), dtype=torch.int32)
    dst_idx = torch.randint(0, num_dst_nodes, (num_edges,), dtype=torch.int32)
    src_feat = torch.randn(num_src_nodes, feat_dim, dtype=torch.float32)
    dst_feat = torch.randn(num_dst_nodes, feat_dim, dtype=torch.float32)
    return [src_idx, dst_idx, src_feat, dst_feat]


def get_init_inputs():
    return []
