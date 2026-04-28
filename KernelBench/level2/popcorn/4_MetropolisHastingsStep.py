import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    Batched Metropolis-Hastings MCMC step.  Given a current state and a
    log-probability function (here a Gaussian target for benchmarking),
    proposes a new state via a Gaussian random walk, computes the
    acceptance ratio, and stochastically accepts or rejects.
    """

    def __init__(self, dim, step_size=0.1):
        super().__init__()
        self.dim = dim
        self.step_size = step_size
        # Target: N(mu, sigma^2 I)
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))

    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (((x - self.mu) / self.sigma) ** 2).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, dim) – current chain states
        Returns:
            (x_new, accepted):
                x_new:    (B, dim) – states after accept/reject
                accepted: (B,)    – boolean mask of accepted proposals
        """
        proposal = x + self.step_size * torch.randn_like(x)
        log_alpha = self._log_prob(proposal) - self._log_prob(x)
        log_u = torch.log(torch.rand(x.shape[0], device=x.device).clamp(min=1e-10))
        accepted = log_u < log_alpha
        x_new = torch.where(accepted.unsqueeze(-1), proposal, x)
        return x_new, accepted


dim = 64
batch_size = 4096


def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
