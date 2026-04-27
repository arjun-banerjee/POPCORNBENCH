import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Reparameterization trick for differentiable sampling from a diagonal
    Gaussian.  Separates stochasticity from parameters so that gradients
    flow through mu and log_var.  Also demonstrates a "mixture of
    Gaussians" reparameterization with learnable component weights.
    """

    def __init__(self, latent_dim, num_components=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        if num_components > 1:
            self.logits = nn.Parameter(torch.zeros(num_components))

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> tuple:
        """
        Args:
            mu:      (B, K, D) or (B, D) – means
            log_var: (B, K, D) or (B, D) – log-variances
        Returns:
            (z, log_q_z):
                z:       (B, D) – sampled latents
                log_q_z: (B,)   – log-density of sample under q
        """
        std = torch.exp(0.5 * log_var)

        if self.num_components > 1 and mu.dim() == 3:
            weights = torch.softmax(self.logits, dim=0)  # (K,)
            # Sample component
            comp = torch.multinomial(weights.expand(mu.shape[0], -1), 1).squeeze(-1)
            batch_idx = torch.arange(mu.shape[0], device=mu.device)
            mu_k = mu[batch_idx, comp]
            std_k = std[batch_idx, comp]
            z = mu_k + std_k * torch.randn_like(mu_k)
            # log q(z) under mixture
            log_probs = -0.5 * ((z.unsqueeze(1) - mu) / std) ** 2 - log_var * 0.5 - 0.5 * torch.log(torch.tensor(2 * 3.14159265))
            log_probs = log_probs.sum(-1) + weights.log().unsqueeze(0)
            log_q_z = torch.logsumexp(log_probs, dim=1)
        else:
            if mu.dim() == 3:
                mu = mu[:, 0]
                std = std[:, 0]
                log_var = log_var[:, 0]
            z = mu + std * torch.randn_like(mu)
            log_q_z = (-0.5 * ((z - mu) / std) ** 2 - 0.5 * log_var - 0.5 * torch.log(torch.tensor(2 * 3.14159265))).sum(-1)

        return z, log_q_z


latent_dim = 64
num_components = 4
batch_size = 256


def get_inputs():
    mu = torch.randn(batch_size, num_components, latent_dim)
    log_var = torch.randn(batch_size, num_components, latent_dim) * 0.5
    return [mu, log_var]


def get_init_inputs():
    return [latent_dim, num_components]
