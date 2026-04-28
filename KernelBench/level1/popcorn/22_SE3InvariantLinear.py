import torch
import torch.nn as nn


class Model(nn.Module):
    """
    SE(3)-invariant linear layer.  Maps node features that are augmented
    with 3-D coordinate information through a linear transform that
    respects rotational and translational invariance.  The layer operates
    on (scalar_features, vector_features) pairs where scalar features
    transform trivially and vector features transform as 3-D vectors.
    Used in equivariant GNNs (EGNN, PaiNN, TFN).
    """

    def __init__(self, scalar_in, scalar_out, vector_in, vector_out):
        super().__init__()
        self.scalar_linear = nn.Linear(scalar_in + vector_in, scalar_out)
        self.vector_weight = nn.Parameter(torch.randn(vector_out, vector_in))
        nn.init.xavier_uniform_(self.vector_weight)
        self.vector_gate = nn.Linear(scalar_in + vector_in, vector_out)

    def forward(
        self, scalar_feat: torch.Tensor, vector_feat: torch.Tensor
    ) -> tuple:
        """
        Args:
            scalar_feat: (B, N, scalar_in) – invariant node features
            vector_feat: (B, N, vector_in, 3) – equivariant vector features
        Returns:
            (scalar_out, vector_out):
                scalar_out: (B, N, scalar_out)
                vector_out: (B, N, vector_out, 3)
        """
        # Invariant scalars from vector norms
        vec_norm = vector_feat.norm(dim=-1)  # (B, N, vector_in)
        combined = torch.cat([scalar_feat, vec_norm], dim=-1)

        s_out = self.scalar_linear(combined)

        # Equivariant vector transform: linear combination of input vectors
        v_out = torch.einsum("oi,bnid->bnod", self.vector_weight, vector_feat)
        # Gating by scalar-derived signal
        gate = torch.sigmoid(self.vector_gate(combined)).unsqueeze(-1)  # (B, N, vector_out, 1)
        v_out = gate * v_out

        return s_out, v_out


scalar_in = 64
scalar_out = 64
vector_in = 16
vector_out = 16
num_nodes = 128
batch_size = 8


def get_inputs():
    s = torch.randn(batch_size, num_nodes, scalar_in)
    v = torch.randn(batch_size, num_nodes, vector_in, 3)
    return [s, v]


def get_init_inputs():
    return [scalar_in, scalar_out, vector_in, vector_out]
