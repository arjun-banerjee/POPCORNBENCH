import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    **Fused probability-flow ODE integration** (flow-matching style) in one ``forward``.

    Models the ODE ``dx/dt = v_θ(x, t)`` on ``t ∈ [0, 1]`` with a learned velocity
    field. A **single fused pass** unrolls **fixed-step Euler** integration for
    ``num_steps`` substeps—typical for **real-time trajectory / action generation**
    where latency is dominated by repeated ``v_θ`` evaluations.

    This is the continuous-time analogue of a rectified-flow / conditional-flow
    sampler: one ``forward`` returns an approximate **x(1)** from **x(0)** without a
    Python loop across separate module calls (fusion target for the integrator +
    field).
    """

    def __init__(self, dim: int, num_steps: int):
        super().__init__()
        assert dim >= 1 and num_steps >= 1
        self.dim = dim
        self.num_steps = num_steps
        self.register_buffer("dt", torch.tensor(1.0 / float(num_steps)))

        self.velocity = nn.Sequential(
            nn.Linear(dim + 1, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: (B, dim) initial state at ``t = 0`` (e.g. noise or anchor trajectory).
        Returns:
            (B, dim) approximate state at ``t = 1`` after fused Euler integration.
        """
        B, d = x0.shape
        assert d == self.dim
        x = x0
        dt = self.dt.to(dtype=x0.dtype, device=x0.device)
        for k in range(self.num_steps):
            t = (float(k) + 0.5) / float(self.num_steps)
            t_b = torch.full((B, 1), t, device=x0.device, dtype=x0.dtype)
            inp = torch.cat([x, t_b], dim=-1)
            v = self.velocity(inp)
            x = x + dt * v
        return x


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, dim: int, num_steps: int):
        super().__init__()
        self._impl = Model(dim, num_steps)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        return self._impl(x0)


batch_size = 8
dim = 64
num_steps = 32


def get_inputs():
    return [torch.randn(batch_size, dim)]


def get_init_inputs():
    return [dim, num_steps]
