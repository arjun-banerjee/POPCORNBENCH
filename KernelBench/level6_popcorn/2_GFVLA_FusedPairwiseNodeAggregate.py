import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    GF-VLA style **fused pairwise → node** scene-graph readout.

    For every ordered pair (i, j), fuse ``[node_i || node_j || pair_feat_ij]`` with
    an MLP, mask invalid graph edges, then **max-pool over outgoing neighbors j**
    to produce an updated per-object signal (object-centric scene graph refresh).

    Complements ``7_GFVLA_FusedSceneGraphUpdate`` (attention) with an MLP + reduce
    fusion pattern common for relation classifiers and affordance heads.
    """

    def __init__(self, dim: int, pair_dim: int, hidden: int):
        super().__init__()
        self.dim = dim
        self.pair_dim = pair_dim
        self.fuse = nn.Sequential(
            nn.Linear(2 * dim + pair_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(
        self,
        nodes: torch.Tensor,
        pair_feats: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            nodes: (B, N, D)
            pair_feats: (B, N, N, P)
            edge_mask: (B, N, N) float {0,1}
        Returns:
            (B, N, D) residual ``nodes + LN( max_j fuse(...) )``
        """
        B, N, D = nodes.shape
        assert pair_feats.shape == (B, N, N, self.pair_dim)

        hi = nodes.unsqueeze(2).expand(-1, -1, N, -1)
        hj = nodes.unsqueeze(1).expand(-1, N, -1, -1)
        fused = self.fuse(torch.cat([hi, hj, pair_feats], dim=-1))

        neg = torch.finfo(fused.dtype).min
        fused = fused.masked_fill(edge_mask.unsqueeze(-1) < 0.5, neg)
        pooled, _ = fused.max(dim=2)
        return nodes + self.out_norm(pooled)


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, dim: int, pair_dim: int, hidden: int):
        super().__init__()
        self._impl = Model(dim, pair_dim, hidden)

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
hidden = 192


def get_inputs():
    nodes = torch.randn(batch_size, num_objects, dim)
    pair_feats = torch.randn(batch_size, num_objects, num_objects, pair_dim)
    edge_mask = torch.ones(batch_size, num_objects, num_objects)
    return [nodes, pair_feats, edge_mask]


def get_init_inputs():
    return [dim, pair_dim, hidden]
