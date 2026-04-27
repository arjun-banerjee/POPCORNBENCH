import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Self-normalized importance sampling estimate.
    Given samples from a proposal distribution q, computes importance
    weights w_i = p(x_i)/q(x_i), normalizes them, and returns both
    the weighted estimate of E_p[f(x)] and the effective sample size.

    Target p is a mixture of Gaussians; proposal q is a single Gaussian.
    """

    def __init__(self, dim, num_components=3):
        super().__init__()
        self.dim = dim
        self.num_components = num_components
        # Target mixture components
        self.register_buffer("mix_weights", torch.ones(num_components) / num_components)
        mus = torch.randn(num_components, dim) * 2.0
        self.register_buffer("mix_mus", mus)
        self.register_buffer("mix_stds", torch.ones(num_components, dim))
        # Proposal
        self.register_buffer("q_mu", torch.zeros(dim))
        self.register_buffer("q_std", torch.ones(dim) * 3.0)

    def _log_p(self, x):
        """Log density of target mixture."""
        diff = x.unsqueeze(1) - self.mix_mus.unsqueeze(0)  # (B, K, D)
        log_comp = -0.5 * (diff / self.mix_stds.unsqueeze(0)) ** 2 - self.mix_stds.log().unsqueeze(0)
        log_comp = log_comp.sum(-1)  # (B, K)
        return torch.logsumexp(log_comp + self.mix_weights.log().unsqueeze(0), dim=-1)

    def _log_q(self, x):
        """Log density of proposal."""
        return (-0.5 * ((x - self.q_mu) / self.q_std) ** 2 - self.q_std.log()).sum(-1)

    def forward(self, samples: torch.Tensor) -> tuple:
        """
        Args:
            samples: (N, dim) – samples drawn from q
        Returns:
            (weighted_mean, ess):
                weighted_mean: (dim,) – importance-weighted mean estimate of E_p[x]
                ess: scalar – effective sample size
        """
        log_w = self._log_p(samples) - self._log_q(samples)
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        w = torch.exp(log_w)

        weighted_mean = (w.unsqueeze(-1) * samples).sum(dim=0)
        ess = 1.0 / (w ** 2).sum()
        return weighted_mean, ess


dim = 16
num_samples = 4096


def get_inputs():
    q_std = 3.0
    return [torch.randn(num_samples, dim) * q_std]


def get_init_inputs():
    return [dim]
