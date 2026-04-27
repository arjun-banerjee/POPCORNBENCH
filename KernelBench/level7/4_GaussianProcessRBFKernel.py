import torch
import torch.nn as nn


class Model(nn.Module):
    """
    RBF (Squared Exponential) covariance matrix computation for Gaussian
    Processes.  Given two sets of input points, computes the full
    kernel matrix K[i,j] = sigma^2 * exp(-||x_i - x_j||^2 / (2 * l^2)).
    The dominant cost in GP inference and a natural target for fusion.
    """

    def __init__(self, input_dim, lengthscale=1.0, variance=1.0):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.tensor(lengthscale).log())
        self.log_variance = nn.Parameter(torch.tensor(variance).log())

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X1: (B, N, D) – first set of inputs
            X2: (B, M, D) – second set of inputs
        Returns:
            K: (B, N, M) – covariance matrix
        """
        l2 = self.log_lengthscale.exp() ** 2
        var = self.log_variance.exp()

        # Squared distances via expansion
        X1_sq = (X1 ** 2).sum(-1, keepdim=True)  # (B, N, 1)
        X2_sq = (X2 ** 2).sum(-1, keepdim=True)  # (B, M, 1)
        dist_sq = X1_sq - 2.0 * X1 @ X2.transpose(-1, -2) + X2_sq.transpose(-1, -2)
        dist_sq = dist_sq.clamp(min=0.0)

        return var * torch.exp(-0.5 * dist_sq / l2)


input_dim = 8
num_train = 512
num_test = 128
batch_size = 16


def get_inputs():
    X1 = torch.randn(batch_size, num_train, input_dim)
    X2 = torch.randn(batch_size, num_test, input_dim)
    return [X1, X2]


def get_init_inputs():
    return [input_dim]
