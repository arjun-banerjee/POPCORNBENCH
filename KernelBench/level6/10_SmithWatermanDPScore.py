import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Differentiable Smith-Waterman-style local sequence alignment scoring.
    Uses a soft-max relaxation of the DP recurrence so gradients can flow
    through the alignment matrix.  
    """
    def __init__(self, gap_open=-1.0, temperature=1.0):
        super().__init__()
        self.gap_open = gap_open
        self.temperature = temperature

    def forward(self, score_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_matrix: (B, M, N) – per-position similarity scores
        Returns:
            alignment_score: (B,) – soft local-alignment score (log-sum-exp over DP cells)
        """
        B, M, N = score_matrix.shape
        neg_inf = torch.tensor(float("-inf"), device=score_matrix.device, dtype=score_matrix.dtype)

        # H[i][j] = max(0, H[i-1][j-1]+S[i][j], H[i-1][j]+gap, H[i][j-1]+gap)
        H = score_matrix.new_full((B, M + 1, N + 1), 0.0)

        for i in range(1, M + 1):
            for j in range(1, N + 1):
                match = H[:, i - 1, j - 1] + score_matrix[:, i - 1, j - 1]
                delete = H[:, i - 1, j] + self.gap_open
                insert = H[:, i, j - 1] + self.gap_open
                zero = torch.zeros(B, device=score_matrix.device)
                stack = torch.stack([zero, match, delete, insert], dim=-1)
                H[:, i, j] = self.temperature * torch.logsumexp(stack / self.temperature, dim=-1)

        return H.view(B, -1).max(dim=-1).values


seq_m = 32
seq_n = 32
batch_size = 16


def get_inputs():
    return [torch.randn(batch_size, seq_m, seq_n)]


def get_init_inputs():
    return []
