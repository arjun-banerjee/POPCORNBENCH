import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    Farthest Point Sampling (FPS) on 3D point coordinates (B, N, 3).

    Used in navigation / manipulation perception (PointNet++, PVCNN, voxelization
    pipelines) to subsample a dense cloud while spreading support across the shape.

    Greedy construction (reference fusion target): fix the first index to **0** for
    determinism, then repeatedly choose the point whose **minimum** squared distance
    to the set of already chosen points is **largest** (classical FPS).
    """

    def __init__(self, num_samples: int):
        super().__init__()
        assert num_samples >= 1
        self.num_samples = num_samples

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) point coordinates.
        Returns:
            indices: (B, M) int64, each row is M chosen indices in [0, N).
        """
        B, N, _ = points.shape
        M = self.num_samples
        assert M <= N, "FPS cannot sample more points than exist"
        device = points.device
        dtype = points.dtype

        batch_idx = torch.arange(B, device=device)
        indices = torch.empty(B, M, dtype=torch.long, device=device)

        # min squared distance from each point to the current chosen set (large init)
        min_dist_sq = torch.full((B, N), float("inf"), device=device, dtype=dtype)
        # First center: point index 0 (deterministic)
        farthest = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(M):
            indices[:, i] = farthest
            center = points[batch_idx, farthest]
            dist_sq = ((points - center.unsqueeze(1)) ** 2).sum(dim=-1)
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
            # Exclude the just-chosen index from being selected again
            min_dist_sq[batch_idx, farthest] = -1.0
            farthest = min_dist_sq.argmax(dim=-1)

        return indices


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, num_samples: int):
        super().__init__()
        self._impl = Model(num_samples)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self._impl(points)


batch_size = 4
num_points = 1024
num_samples = 128


def get_inputs():
    points = torch.randn(batch_size, num_points, 3)
    return [points]


def get_init_inputs():
    return [num_samples]
