import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Stack of planar normalizing flow layers.  Each layer applies the
    invertible transform f(z) = z + u * tanh(w^T z + b), with a
    log-determinant-Jacobian correction.  Used to transform a simple
    base distribution (e.g. standard Gaussian) into a complex posterior.
    """

    def __init__(self, dim, num_flows):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows
        self.w = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(num_flows)])
        self.u = nn.ParameterList([nn.Parameter(torch.randn(dim)) for _ in range(num_flows)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_flows)])

    def _constrain_u(self, w, u):
        """Ensure invertibility: w^T u >= -1."""
        wtu = (w * u).sum()
        m = -1 + F.softplus(wtu)
        u_hat = u + (m - wtu) * w / (w ** 2).sum()
        return u_hat

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Args:
            z: (B, dim) – samples from base distribution
        Returns:
            (z_out, sum_log_det):
                z_out:       (B, dim) – transformed samples
                sum_log_det: (B,)     – total log |det df/dz|
        """
        sum_log_det = torch.zeros(z.shape[0], device=z.device)

        for i in range(self.num_flows):
            w = self.w[i]
            u = self._constrain_u(w, self.u[i])
            b = self.b[i]

            linear = z @ w + b  # (B,)
            h = torch.tanh(linear)
            h_prime = 1.0 - h ** 2

            z = z + u.unsqueeze(0) * h.unsqueeze(-1)
            psi = h_prime.unsqueeze(-1) * w.unsqueeze(0)  # (B, dim)
            log_det = torch.log(torch.abs(1.0 + psi @ u) + 1e-8)
            sum_log_det = sum_log_det + log_det

        return z, sum_log_det


dim = 32
num_flows = 8
batch_size = 1024


def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim, num_flows]
