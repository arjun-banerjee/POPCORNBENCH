import torch
import torch.nn as nn


class Model(nn.Module):
    """
    One full sweep of Gibbs sampling for a multivariate Gaussian with
    known precision matrix.  Each coordinate is sampled from its full
    conditional distribution while holding all other coordinates fixed.
    Benchmarks the sequential-scan pattern common in Bayesian models
    with tractable conditionals (e.g., topic models, Boltzmann machines).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Precision matrix: diagonally dominant for well-conditioning
        L = torch.randn(dim, dim) * 0.1
        precision = L @ L.t() + torch.eye(dim) * 2.0
        self.register_buffer("precision", precision)
        self.register_buffer("mu", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, dim) – current state
        Returns:
            x_new: (B, dim) – state after one full Gibbs sweep
        """
        x = x.clone()
        for i in range(self.dim):
            prec_ii = self.precision[i, i]
            residual = x @ self.precision[i] - x[:, i] * prec_ii
            cond_mean = self.mu[i] - residual / prec_ii
            cond_var = 1.0 / prec_ii
            x[:, i] = cond_mean + torch.sqrt(cond_var) * torch.randn(x.shape[0], device=x.device)
        return x


dim = 32
batch_size = 4096


def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
