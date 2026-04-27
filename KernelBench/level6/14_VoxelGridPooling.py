import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Voxel-grid pooling for 3-D molecular point clouds.  Discretizes
    continuous 3-D coordinates into a regular grid and pools per-atom
    features into each voxel.  Used in 3-D-CNN approaches for molecular
    property prediction and ligand docking (e.g., PointVoxel-CNN,
    3D-Infomax).
    """

    def __init__(self, feat_dim, grid_size=32, voxel_size=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.linear = nn.Linear(feat_dim, feat_dim)

    def forward(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coords:   (B, N, 3) – atom coordinates
            features: (B, N, feat_dim) – per-atom features
        Returns:
            voxel_grid: (B, feat_dim, G, G, G) – pooled voxel features
        """
        B, N, _ = coords.shape
        G = self.grid_size

        features = self.linear(features)

        # Center coordinates and discretize
        center = coords.mean(dim=1, keepdim=True)
        shifted = (coords - center) / self.voxel_size + G / 2.0
        indices = shifted.long().clamp(0, G - 1)  # (B, N, 3)

        voxel_grid = coords.new_zeros(B, self.feat_dim, G, G, G)
        ix = indices[:, :, 0]
        iy = indices[:, :, 1]
        iz = indices[:, :, 2]

        batch_idx = torch.arange(B, device=coords.device).unsqueeze(1).expand(-1, N)
        for d in range(self.feat_dim):
            feat_d = features[:, :, d]
            voxel_grid[:, d].index_put_(
                (batch_idx.reshape(-1), ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)),
                feat_d.reshape(-1),
                accumulate=True,
            )

        return voxel_grid


feat_dim = 32
grid_size = 32
voxel_size = 1.0
num_atoms = 256
batch_size = 4


def get_inputs():
    coords = torch.randn(batch_size, num_atoms, 3) * 5.0
    features = torch.randn(batch_size, num_atoms, feat_dim)
    return [coords, features]


def get_init_inputs():
    return [feat_dim, grid_size, voxel_size]
