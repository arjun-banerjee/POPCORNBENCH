import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Contact map prediction head.  Takes residue-pair embeddings and
    predicts a symmetric binary contact probability matrix (residue i
    within 8 Å of residue j).  Uses a small MLP on the symmetrised pair
    representation followed by a sigmoid.  Common final stage in protein
    structure prediction pipelines.
    """

    def __init__(self, pair_dim, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(pair_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_repr: (B, N, N, pair_dim)
        Returns:
            contact_probs: (B, N, N) – symmetric contact probability
        """
        x = self.layer_norm(pair_repr)
        # Symmetrise
        x = (x + x.permute(0, 2, 1, 3)) / 2.0
        logits = self.mlp(x).squeeze(-1)  # (B, N, N)
        return torch.sigmoid(logits)


pair_dim = 64
hidden_dim = 32
seq_len = 64
batch_size = 4


def get_inputs():
    return [torch.randn(batch_size, seq_len, seq_len, pair_dim)]


def get_init_inputs():
    return [pair_dim, hidden_dim]
