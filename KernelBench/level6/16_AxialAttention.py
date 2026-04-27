import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Axial attention for 2-D feature maps.  Factorises full 2-D attention
    into sequential row-wise and column-wise attention passes, reducing
    complexity from O(N^2 * M^2) to O(N*M*(N+M)).  Used in bioimage
    analysis (cell segmentation, cryo-EM micrograph processing) and
    axial-attention protein language models.
    """

    def __init__(self, dim, num_heads, height, width, dropout=0.0):
        super().__init__()
        self.height = height
        self.width = width
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.row_norm = nn.LayerNorm(dim)
        self.row_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.row_out = nn.Linear(dim, dim)

        self.col_norm = nn.LayerNorm(dim)
        self.col_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.col_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def _attn(self, x, qkv_proj, out_proj, norm):
        """Standard multi-head self-attention over last-but-one dim."""
        B_outer, L, D = x.shape
        h, d = self.num_heads, self.head_dim
        x = norm(x)
        qkv = qkv_proj(x).view(B_outer, L, 3, h, d)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_outer, L, D)
        return out_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D) – 2-D feature map
        Returns:
            (B, H, W, D) – updated feature map
        """
        B, H, W, D = x.shape

        # Row-wise attention: treat each row as a sequence
        x_row = x.reshape(B * H, W, D)
        x = x + self._attn(x_row, self.row_qkv, self.row_out, self.row_norm).reshape(B, H, W, D)

        # Column-wise attention: treat each column as a sequence
        x_col = x.permute(0, 2, 1, 3).reshape(B * W, H, D)
        x = x + self._attn(x_col, self.col_qkv, self.col_out, self.col_norm).reshape(B, W, H, D).permute(0, 2, 1, 3)

        return x


dim = 64
num_heads = 4
height = 32
width = 32
batch_size = 4


def get_inputs():
    return [torch.randn(batch_size, height, width, dim)]


def get_init_inputs():
    return [dim, num_heads, height, width]
