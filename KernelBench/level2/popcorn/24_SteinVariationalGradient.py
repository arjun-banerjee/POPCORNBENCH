import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Stein Variational Gradient Descent (SVGD) update step.
    Maintains a set of particles approximating a posterior and updates
    them via a kernel-smoothed gradient that balances fitting the target
    with repulsion between particles.  The RBF kernel bandwidth is set
    via the median heuristic.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Target: N(0, I) for benchmarking
        self.register_buffer("target_mean", torch.zeros(dim))

    def _score_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient of log p(x) for N(0, I)."""
        return -x

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            particles: (N, dim) – current particle positions
        Returns:
            phi: (N, dim) – SVGD update direction
        """
        N = particles.shape[0]
        score = self._score_fn(particles)  # (N, dim)

        # Pairwise squared distances
        diff = particles.unsqueeze(0) - particles.unsqueeze(1)  # (N, N, dim)
        dist_sq = (diff ** 2).sum(-1)  # (N, N)

        # Median heuristic for bandwidth
        median_sq = torch.median(dist_sq.view(-1)).clamp(min=1e-5)
        h = median_sq / max(torch.log(torch.tensor(float(N))).item(), 1.0)

        K = torch.exp(-dist_sq / (2.0 * h))  # (N, N)
        grad_K = -diff / h * K.unsqueeze(-1)  # (N, N, dim)

        # SVGD update: phi(x_i) = (1/N) sum_j [ K(x_j, x_i) * score(x_j) + grad_K(x_j, x_i) ]
        phi = (K @ score + grad_K.sum(dim=0)) / N
        return phi


dim = 32
num_particles = 1024


def get_inputs():
    return [torch.randn(num_particles, dim)]


def get_init_inputs():
    return [dim]
