import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    One iteration of Expectation-Maximization for a Gaussian Mixture Model.
    E-step: compute soft assignments (responsibilities).
    M-step: update means, covariances, and mixing weights.
    The dominant cost is the batched evaluation of K Gaussian densities
    over N data points.
    """

    def __init__(self, dim, num_components):
        super().__init__()
        self.dim = dim
        self.K = num_components
        self.register_buffer("mus", torch.randn(num_components, dim))
        self.register_buffer("covs", torch.eye(dim).unsqueeze(0).expand(num_components, -1, -1).clone())
        self.register_buffer("log_pi", torch.zeros(num_components))

    def _log_gaussian(self, x, mu, cov):
        """Log N(x | mu, cov) for batched x."""
        D = mu.shape[-1]
        diff = x - mu.unsqueeze(0)  # (N, D)
        L = torch.linalg.cholesky(cov)
        solve = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
        maha = (solve ** 2).sum(-1)
        log_det = 2.0 * L.diagonal().log().sum()
        return -0.5 * (D * math.log(2 * math.pi) + log_det + maha)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (N, dim) – data points
        Returns:
            (new_mus, new_covs, new_log_pi, log_lik):
                new_mus:    (K, dim)
                new_covs:   (K, dim, dim)
                new_log_pi: (K,)
                log_lik:    scalar – total log-likelihood
        """
        N, D = x.shape
        K = self.K
        log_pi = self.log_pi - torch.logsumexp(self.log_pi, dim=0)

        # E-step
        log_resp = torch.stack([
            self._log_gaussian(x, self.mus[k], self.covs[k]) + log_pi[k]
            for k in range(K)
        ], dim=-1)  # (N, K)
        log_lik = torch.logsumexp(log_resp, dim=-1).sum()
        log_resp = log_resp - torch.logsumexp(log_resp, dim=-1, keepdim=True)
        resp = torch.exp(log_resp)  # (N, K)

        # M-step
        N_k = resp.sum(dim=0).clamp(min=1e-8)  # (K,)
        new_mus = (resp.t() @ x) / N_k.unsqueeze(-1)
        new_covs = torch.zeros_like(self.covs)
        for k in range(K):
            diff = x - new_mus[k].unsqueeze(0)
            weighted = diff * resp[:, k:k+1]
            new_covs[k] = (weighted.t() @ diff) / N_k[k] + 1e-4 * torch.eye(D, device=x.device)

        new_log_pi = N_k.log() - math.log(N)

        return new_mus, new_covs, new_log_pi, log_lik


dim = 8
num_components = 5
num_data = 2048


def get_inputs():
    return [torch.randn(num_data, dim)]


def get_init_inputs():
    return [dim, num_components]
