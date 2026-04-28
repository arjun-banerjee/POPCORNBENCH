import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Bayesian linear regression with a conjugate Gaussian prior.
    Given features X and targets y, computes the posterior distribution
    over weights: p(w | X, y) = N(mu_post, Sigma_post) in closed form
    via the matrix inversion lemma / Cholesky decomposition.
    """

    def __init__(self, input_dim, prior_precision=1.0, noise_precision=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision

    def forward(
        self, X: torch.Tensor, y: torch.Tensor, X_test: torch.Tensor
    ) -> tuple:
        """
        Args:
            X:      (B, N, D) – training features
            y:      (B, N, 1) – training targets
            X_test: (B, M, D) – test features
        Returns:
            (pred_mean, pred_var):
                pred_mean: (B, M, 1)
                pred_var:  (B, M)
        """
        D = X.shape[-1]
        eye = torch.eye(D, device=X.device, dtype=X.dtype).unsqueeze(0)

        # Posterior precision: Lambda_post = alpha * I + beta * X^T X
        Lambda_post = self.prior_precision * eye + self.noise_precision * (X.transpose(-1, -2) @ X)
        L = torch.linalg.cholesky(Lambda_post)

        # Posterior mean: mu_post = Lambda_post^{-1} * beta * X^T y
        rhs = self.noise_precision * (X.transpose(-1, -2) @ y)
        mu_post = torch.cholesky_solve(rhs, L)

        # Predictive distribution
        pred_mean = X_test @ mu_post
        Sigma_post = torch.cholesky_inverse(L)
        pred_var = (X_test @ Sigma_post * X_test).sum(-1) + 1.0 / self.noise_precision

        return pred_mean, pred_var


input_dim = 16
num_train = 256
num_test = 64
batch_size = 16


def get_inputs():
    X = torch.randn(batch_size, num_train, input_dim)
    y = torch.randn(batch_size, num_train, 1)
    X_test = torch.randn(batch_size, num_test, input_dim)
    return [X, y, X_test]


def get_init_inputs():
    return [input_dim]
