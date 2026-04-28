import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Radial Basis Function (RBF) expansion of interatomic distances.
    Standard featurization in molecular graph neural networks (SchNet,
    DimeNet, GemNet) that converts scalar distances into a feature vector
    of Gaussian-basis evaluations.
    """

    def __init__(self, num_rbf=64, cutoff=5.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        offsets = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("offsets", offsets)
        coeff = -0.5 / (offsets[1] - offsets[0]).item() ** 2
        self.coeff = coeff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: (B, N, N) – pairwise distance matrix (or (B, E) edge list)
        Returns:
            rbf_features: (*distances.shape, num_rbf) – expanded features
        """
        # Cosine cutoff envelope
        d_scaled = distances.unsqueeze(-1)  # (..., 1)
        envelope = 0.5 * (torch.cos(math.pi * d_scaled / self.cutoff) + 1.0)
        envelope = envelope * (d_scaled < self.cutoff).float()

        rbf = torch.exp(self.coeff * (d_scaled - self.offsets) ** 2)
        return rbf * envelope


num_atoms = 256
num_rbf = 64
cutoff = 5.0
batch_size = 8


def get_inputs():
    coords = torch.randn(batch_size, num_atoms, 3)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    distances = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
    return [distances]


def get_init_inputs():
    return [num_rbf, cutoff]
