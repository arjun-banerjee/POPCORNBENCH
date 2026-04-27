import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Evidence Lower Bound (ELBO) computation for a Variational Autoencoder.
    Combines a reconstruction term (binary cross-entropy) with the
    KL divergence between the approximate posterior q(z|x) = N(mu, sigma^2)
    and the prior p(z) = N(0, I).  The core loss function in VAE training.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, input_dim) – input data in [0, 1]
        Returns:
            (elbo, recon_loss, kl_div):
                elbo:       (B,)
                recon_loss: (B,)
                kl_div:     (B,)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_recon = self.decoder(z)
        recon = F.binary_cross_entropy(x_recon, x, reduction="none").sum(-1)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        elbo = -(recon + kl)
        return elbo, recon, kl


input_dim = 784
hidden_dim = 256
latent_dim = 32
batch_size = 128


def get_inputs():
    return [torch.rand(batch_size, input_dim)]


def get_init_inputs():
    return [input_dim, hidden_dim, latent_dim]
