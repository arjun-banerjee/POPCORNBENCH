import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Log-likelihood under the Dirichlet-Multinomial (Polya) distribution.
    Given count vectors n and concentration parameters alpha, computes
    log p(n | alpha) using the log-gamma formulation.  Core computation
    in Bayesian topic models and mixed-membership models.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.log_alpha = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, counts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            counts: (B, V) – integer count vectors (as float for differentiability)
        Returns:
            log_lik: (B,) – log p(counts | alpha) for each sample
        """
        alpha = torch.exp(self.log_alpha).unsqueeze(0)  # (1, V)
        N = counts.sum(-1)  # (B,)
        alpha_sum = alpha.sum(-1)  # (1,)

        # log p(n | alpha) = log Gamma(alpha_sum) - log Gamma(N + alpha_sum)
        #                   + sum_v [ log Gamma(n_v + alpha_v) - log Gamma(alpha_v) ]
        log_lik = (
            torch.lgamma(alpha_sum)
            - torch.lgamma(N + alpha_sum)
            + (torch.lgamma(counts + alpha) - torch.lgamma(alpha)).sum(-1)
        )
        return log_lik


vocab_size = 256
batch_size = 512


def get_inputs():
    counts = torch.randint(0, 10, (batch_size, vocab_size)).float()
    return [counts]


def get_init_inputs():
    return [vocab_size]
