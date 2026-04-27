import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) with full encoder–decoder
    forward pass and loss computation.  The encoder conditions on both
    input x and label y; the decoder conditions on z and y.  Returns
    reconstruction loss + KL divergence.
    """

    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder: q(z | x, y)
        self.enc = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: p(x | z, y)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, input_dim)  – input data
            y: (B, label_dim)  – conditioning label (one-hot)
        Returns:
            (loss, recon_loss, kl_loss):
                loss:       scalar – total CVAE loss
                recon_loss: scalar
                kl_loss:    scalar
        """
        h = self.enc(torch.cat([x, y], dim=-1))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        x_recon = self.dec(torch.cat([z, y], dim=-1))
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.shape[0]
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / x.shape[0]
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss


input_dim = 256
label_dim = 10
hidden_dim = 128
latent_dim = 32
batch_size = 128


def get_inputs():
    x = torch.randn(batch_size, input_dim)
    y = torch.zeros(batch_size, label_dim)
    y.scatter_(1, torch.randint(0, label_dim, (batch_size, 1)), 1.0)
    return [x, y]


def get_init_inputs():
    return [input_dim, label_dim, hidden_dim, latent_dim]
