import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Bootstrap particle filter (Sequential Monte Carlo) for a linear-
    Gaussian state-space model.  Implements the predict–weight–resample
    cycle for one time step.  The state transition is x_{t} = A x_{t-1} + w
    and the observation model is y_t = C x_t + v.
    """

    def __init__(self, state_dim, obs_dim, num_particles):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.num_particles = num_particles
        self.register_buffer("A", 0.9 * torch.eye(state_dim))
        self.register_buffer("C", torch.randn(obs_dim, state_dim) * 0.5)
        self.process_noise = 0.1
        self.obs_noise = 0.5

    def forward(
        self,
        particles: torch.Tensor,
        log_weights: torch.Tensor,
        observation: torch.Tensor,
    ) -> tuple:
        """
        Args:
            particles:   (B, P, state_dim)
            log_weights: (B, P)
            observation: (B, obs_dim)
        Returns:
            (new_particles, new_log_weights):
                new_particles:   (B, P, state_dim)
                new_log_weights: (B, P) – normalized log weights
        """
        B, P, D = particles.shape

        # Predict: propagate through dynamics
        pred = (particles @ self.A.t()) + self.process_noise * torch.randn_like(particles)

        # Weight: compute log p(y_t | x_t)
        obs_pred = pred @ self.C.t()  # (B, P, obs_dim)
        diff = observation.unsqueeze(1) - obs_pred
        log_lik = -0.5 * (diff ** 2).sum(-1) / (self.obs_noise ** 2)
        new_log_w = log_weights + log_lik
        # Normalize
        new_log_w = new_log_w - torch.logsumexp(new_log_w, dim=1, keepdim=True)

        # Systematic resample
        weights = torch.exp(new_log_w)
        cumsum = torch.cumsum(weights, dim=1)
        u = (torch.arange(P, device=particles.device, dtype=particles.dtype) + torch.rand(B, 1, device=particles.device)) / P
        indices = torch.searchsorted(cumsum, u)
        indices = indices.clamp(0, P - 1)

        batch_idx = torch.arange(B, device=particles.device).unsqueeze(1).expand(-1, P)
        resampled = pred[batch_idx, indices]
        uniform_log_w = torch.full_like(new_log_w, -torch.log(torch.tensor(float(P))))

        return resampled, uniform_log_w


state_dim = 4
obs_dim = 2
num_particles = 1024
batch_size = 16


def get_inputs():
    particles = torch.randn(batch_size, num_particles, state_dim)
    log_weights = torch.full((batch_size, num_particles), -torch.log(torch.tensor(float(num_particles))))
    observation = torch.randn(batch_size, obs_dim)
    return [particles, log_weights, observation]


def get_init_inputs():
    return [state_dim, obs_dim, num_particles]
