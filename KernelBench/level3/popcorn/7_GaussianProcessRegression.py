import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Gaussian Process posterior predictive distribution via Cholesky
    decomposition.  Computes the predictive mean and variance at test
    points given training data and an RBF kernel.  The Cholesky solve
    is the computational bottleneck in exact GP inference.
    """

    def __init__(self, input_dim, noise=0.01, lengthscale=1.0, variance=1.0):
        super().__init__()
        self.noise = noise
        self.log_lengthscale = nn.Parameter(torch.tensor(lengthscale).log())
        self.log_variance = nn.Parameter(torch.tensor(variance).log())

    def _rbf_kernel(self, X1, X2):
        l2 = self.log_lengthscale.exp() ** 2
        var = self.log_variance.exp()
        dist_sq = ((X1.unsqueeze(2) - X2.unsqueeze(1)) ** 2).sum(-1)
        return var * torch.exp(-0.5 * dist_sq / l2)

    def forward(
        self, X_train: torch.Tensor, Y_train: torch.Tensor, X_test: torch.Tensor
    ) -> tuple:
        """
        Args:
            X_train: (B, N, D)
            Y_train: (B, N, 1)
            X_test:  (B, M, D)
        Returns:
            (mean, var):
                mean: (B, M, 1) – predictive mean
                var:  (B, M)    – predictive variance
        """
        K_nn = self._rbf_kernel(X_train, X_train)
        K_nn = K_nn + self.noise * torch.eye(K_nn.shape[-1], device=K_nn.device).unsqueeze(0)
        K_nm = self._rbf_kernel(X_train, X_test)
        K_mm = self._rbf_kernel(X_test, X_test)

        L = torch.linalg.cholesky(K_nn)
        alpha = torch.cholesky_solve(Y_train, L)

        mean = K_nm.transpose(-1, -2) @ alpha

        v = torch.linalg.solve_triangular(L, K_nm, upper=False)
        var = K_mm.diagonal(dim1=-2, dim2=-1) - (v ** 2).sum(dim=-2)
        var = var.clamp(min=1e-6)

        return mean, var


input_dim = 4
num_train = 128
num_test = 32
batch_size = 8


def get_inputs():
    X_train = torch.randn(batch_size, num_train, input_dim)
    Y_train = torch.randn(batch_size, num_train, 1)
    X_test = torch.randn(batch_size, num_test, input_dim)
    return [X_train, Y_train, X_test]


def get_init_inputs():
    return [input_dim]
