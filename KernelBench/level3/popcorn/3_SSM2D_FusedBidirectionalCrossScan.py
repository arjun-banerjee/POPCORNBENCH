import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    Fused **2D bidirectional cross-scan** block for robotics **SSM-hybrid** backbones
    (e.g. Vision Mamba / VMamba-style encoders on overhead maps, costmaps, or
    multi-sensor tiles).

    One forward fuses four fixed scan permutations of ``(H, W)`` into 1D sequences,
    applies a **shared depthwise Conv1d** (lightweight stand-in for a selective SSM
    along the scan), maps each result back to the grid, **sums** the four
    directional contributions, and adds a **residual**.

    Scan order (same spirit as 2D cross-scan / S2D):
      LR — row-major left→right, rows top→bottom  
      RL — row-major right→left per row  
      TB — column-wise top→bottom, columns left→right  
      BT — flip vertical then same as TB  
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        assert channels >= 1
        assert kernel_size % 2 == 1, "use odd kernel_size for same-length conv1d"
        self.channels = channels
        pad = kernel_size // 2
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=pad,
            groups=channels,
            bias=True,
        )

    def _apply_scan_conv_unscan(self, x: torch.Tensor) -> torch.Tensor:
        """Shared depthwise conv along flattened 2D, then map back to (B,C,H,W)."""
        B, C, H, W = x.shape
        seq = x.flatten(2)
        y = self.dw_conv(seq)
        return y.view(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features (map / image / fused tile).
        Returns:
            (B, C, H, W) after fused four-way cross-scan + residual.
        """
        B, C, H, W = x.shape
        assert C == self.channels

        acc = torch.zeros_like(x)

        # LR: row-major
        acc = acc + self._apply_scan_conv_unscan(x)

        # RL: flip W, scan, unflip
        xr = x.flip(dims=(-1,))
        acc = acc + self._apply_scan_conv_unscan(xr).flip(dims=(-1,))

        # TB: columns as sequences (permute to W,H, flatten)
        xt = x.permute(0, 1, 3, 2).contiguous()
        Bt, Ct, Wt, Ht = xt.shape
        seq_t = xt.flatten(2)
        yt = self.dw_conv(seq_t).view(Bt, Ct, Wt, Ht).permute(0, 1, 3, 2).contiguous()
        acc = acc + yt

        # BT: flip H then same TB path
        xb = x.flip(dims=(-2,))
        xbt = xb.permute(0, 1, 3, 2).contiguous()
        Bb, Cb, Wb, Hb = xbt.shape
        seq_b = xbt.flatten(2)
        yb = self.dw_conv(seq_b).view(Bb, Cb, Wb, Hb).permute(0, 1, 3, 2).contiguous().flip(
            dims=(-2,)
        )
        acc = acc + yb

        return x + acc


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self._impl = Model(channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


batch_size = 4
channels = 64
height = 32
width = 32
kernel_size = 3


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [channels, kernel_size]
