import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    Fused radius (ball) neighborhood query over a 3D point map with per-query feature
    aggregation—common in navigation / manipulation perception (PointNet++-style set
    abstraction, local maps around end-effector or goal queries on LiDAR / depth).

    Single forward fusion target:
      1) pairwise Euclidean distances between each query center and every point,
      2) membership in the closed ball of radius ``r``,
      3) channel-wise **max** pooling of point features over in-ball neighbors.

    Points with no neighbors inside the radius contribute a **zero** feature vector
    for that query (instead of ``-inf`` max).
    """

    def __init__(self, in_channels: int, radius: float):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("radius_sq", torch.tensor(float(radius) ** 2))

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, P, 3) point coordinates (e.g. world or sensor frame).
            queries: (B, Q, 3) ball centers (e.g. navigation waypoints, grasp samples).
            features: (B, P, C) per-point descriptors (geometry or lifted RGB-D).
        Returns:
            (B, Q, C) max-pooled features over neighbors with ||p - q||_2 <= r.
        """
        B, P, _ = points.shape
        _, Q, _ = queries.shape
        _, P2, C = features.shape
        assert P == P2 and C == self.in_channels

        # Squared distances (B, Q, P)
        dist_sq = torch.cdist(queries, points, p=2.0) ** 2
        in_ball = dist_sq <= self.radius_sq

        # (B, 1, P, C) broadcasts with (B, Q, P) for masked max over P
        feats = features.unsqueeze(1)
        neg_large = torch.finfo(features.dtype).min
        masked = torch.where(
            in_ball.unsqueeze(-1),
            feats,
            torch.full_like(feats, neg_large),
        )
        pooled, _ = masked.max(dim=2)

        empty = in_ball.sum(dim=2) == 0
        if empty.any():
            pooled = pooled.masked_fill(empty.unsqueeze(-1), 0.0)

        return pooled


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, in_channels: int, radius: float):
        super().__init__()
        self._impl = Model(in_channels, radius)

    def forward(
        self,
        points: torch.Tensor,
        queries: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self._impl(points, queries, features)


batch_size = 4
num_points = 512
num_queries = 64
in_channels = 32
radius = 0.12


def get_inputs():
    points = torch.randn(batch_size, num_points, 3)
    queries = torch.randn(batch_size, num_queries, 3)
    features = torch.randn(batch_size, num_points, in_channels)
    return [points, queries, features]


def get_init_inputs():
    return [in_channels, radius]
