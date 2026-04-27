import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Compute all-pairs Euclidean distance matrix from a 3-D point cloud.
    A core primitive in molecular dynamics simulations, docking, and
    distance-geometry-based structure prediction.
    """

    def __init__(self):
        super().__init__()

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) – 3-D coordinates of N atoms / residues
        Returns:
            dist: (B, N, N) – pairwise Euclidean distance matrix
        """
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        return dist


num_atoms = 512
batch_size = 8


def get_inputs():
    return [torch.randn(batch_size, num_atoms, 3)]


def get_init_inputs():
    return []
