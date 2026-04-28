import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


def _linear_beta_schedule(num_timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps)


class Model(nn.Module):
    """
    Fused **multi-step** diffusion denoising (VP-DDPM-style) in a single ``forward``.

    Instead of one scheduler step per Python call, this reference unrolls
    ``num_fused_steps`` consecutive **deterministic** reverse updates (posterior mean,
    no sampler noise)—a typical fusion target for robotics / world-model diffusion
    heads where latency matters.

    Each inner step: predict ``ε_θ(x, t)`` with a small MLP, then apply the standard
    DDPM mean update from discrete time ``t`` to ``t-1``. Batch elements may start
    from different ``t_start``; inputs must satisfy ``t_start >= num_fused_steps`` so
    indices stay valid.
    """

    def __init__(self, dim: int, num_fused_steps: int, total_timesteps: int = 1000):
        super().__init__()
        assert num_fused_steps >= 1
        assert total_timesteps >= num_fused_steps + 1
        self.dim = dim
        self.num_fused_steps = num_fused_steps
        self.total_timesteps = total_timesteps

        self.eps_net = nn.Sequential(
            nn.Linear(dim + 1, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        betas = _linear_beta_schedule(total_timesteps, 1e-4, 0.02)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1-alpha_bar_t)) * eps)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "coef_eps",
            betas / torch.sqrt(1.0 - alphas_cumprod),
        )

    def forward(self, x: torch.Tensor, t_start: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, dim) noisy latent at discrete time ``t_start`` (same semantics as DDPM).
            t_start: (B,) int64 timesteps in ``[num_fused_steps, total_timesteps - 1]``.
        Returns:
            (B, dim) latent after ``num_fused_steps`` fused denoise steps toward 0.
        """
        B, d = x.shape
        assert d == self.dim
        assert t_start.shape == (B,)
        # Harness paths (measure_ref_program_time / eval) cast activations to fp32;
        # discrete timesteps must survive as whole-valued floats → long indices.
        t_start = t_start.long()

        x_cur = x
        for k in range(self.num_fused_steps):
            t_idx = t_start - k
            # ``get_inputs`` samples t_start >= num_fused_steps so t_idx >= 1 always.
            t_norm = t_idx.float() / float(self.total_timesteps)
            eps = self.eps_net(torch.cat([x_cur, t_norm.unsqueeze(-1)], dim=-1))

            c1 = self.sqrt_recip_alphas[t_idx].view(B, 1)
            c2 = self.coef_eps[t_idx].view(B, 1)
            x_cur = c1 * (x_cur - c2 * eps)

        return x_cur


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, dim: int, num_fused_steps: int, total_timesteps: int = 1000):
        super().__init__()
        self._impl = Model(dim, num_fused_steps, total_timesteps=total_timesteps)

    def forward(self, x: torch.Tensor, t_start: torch.Tensor) -> torch.Tensor:
        return self._impl(x, t_start)


batch_size = 8
dim = 64
num_fused_steps = 16
total_timesteps = 1000


def get_inputs():
    x = torch.randn(batch_size, dim)
    low = num_fused_steps
    high = total_timesteps
    t_start = torch.randint(low, high, (batch_size,), dtype=torch.long)
    return [x, t_start]


def get_init_inputs():
    return [dim, num_fused_steps, total_timesteps]
