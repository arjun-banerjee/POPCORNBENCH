import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Hamiltonian Monte Carlo (HMC) proposal using leapfrog integration.
    Simulates Hamiltonian dynamics on a batched set of particles to
    produce proposals with high acceptance probability.  The target
    distribution is a standard Gaussian for benchmarking.
    """

    def __init__(self, dim, num_leapfrog_steps=10, step_size=0.1):
        super().__init__()
        self.dim = dim
        self.num_leapfrog_steps = num_leapfrog_steps
        self.step_size = step_size
        self.register_buffer("mu", torch.zeros(dim))

    def _grad_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient of log p(x) for N(0, I)."""
        return -(x - self.mu)

    def _log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -0.5 * (x ** 2).sum(dim=-1)

    def forward(self, q: torch.Tensor) -> tuple:
        """
        Args:
            q: (B, dim) – current positions
        Returns:
            (q_new, accepted):
                q_new:    (B, dim)
                accepted: (B,)
        """
        eps = self.step_size
        p = torch.randn_like(q)
        current_p = p.clone()
        current_q = q.clone()

        # Half step for momentum
        p = p + 0.5 * eps * self._grad_log_prob(q)

        for _ in range(self.num_leapfrog_steps - 1):
            q = q + eps * p
            p = p + eps * self._grad_log_prob(q)

        q = q + eps * p
        p = p + 0.5 * eps * self._grad_log_prob(q)
        p = -p  # negate for reversibility

        current_H = -self._log_prob(current_q) + 0.5 * (current_p ** 2).sum(-1)
        proposed_H = -self._log_prob(q) + 0.5 * (p ** 2).sum(-1)

        log_alpha = current_H - proposed_H
        log_u = torch.log(torch.rand(q.shape[0], device=q.device).clamp(min=1e-10))
        accepted = log_u < log_alpha
        q_new = torch.where(accepted.unsqueeze(-1), q, current_q)
        return q_new, accepted


dim = 64
batch_size = 2048


def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim]
