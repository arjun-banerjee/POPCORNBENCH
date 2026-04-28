import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


def _fake_quant_symmetric_int8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric int8 fake-quant: round(x/scale) clamped to [-127, 127], times scale."""
    q = torch.clamp(torch.round(x / scale), -127.0, 127.0)
    return q * scale


class Model(nn.Module):
    """
    **QuantVLA selective layout fusion** (reference target for a fused CUDA/Triton kernel).

    Vision–language–action stacks often use **token-wise mixed precision**: some
    timesteps or modalities tolerate aggressive quantization while others stay in
    full precision. This module fuses in **one forward**:

      1. **Layout routing** — ``sigmoid(layout_logits)`` yields a per-token mix
         coefficient in ``(0, 1)`` (quant vs FP residual).
      2. **Quantitative path** — global symmetric **int8-style fake quantization**
         (scale from the tensor max, ``/ 127``).
      3. **Selective blend** — ``α · x_quant + (1 - α) · x``.
      4. **Head** — linear projection on the mixed activations.

    This matches a “quantitative selective layout” block: the layout logits (from a
    router, modality tags, or calibration) choose how strongly each position uses the
    quantized layout vs the identity layout.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, layout_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) VLA token activations (fused modalities).
            layout_logits: (B, T) unbounded logits; larger → more mass on quantized path.
        Returns:
            (B, T, D) updated activations.
        """
        B, T, D = x.shape
        assert D == self.dim
        assert layout_logits.shape == (B, T)

        alpha = torch.sigmoid(layout_logits).unsqueeze(-1)
        with torch.no_grad():
            amax = x.detach().abs().max().clamp(min=1e-8)
        scale = amax / 127.0
        x_q = _fake_quant_symmetric_int8(x, scale)
        mixed = alpha * x_q + (1.0 - alpha) * x
        return self.out_proj(mixed)


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, dim: int):
        super().__init__()
        self._impl = Model(dim)

    def forward(self, x: torch.Tensor, layout_logits: torch.Tensor) -> torch.Tensor:
        return self._impl(x, layout_logits)


batch_size = 8
seq_len = 64
dim = 128


def get_inputs():
    x = torch.randn(batch_size, seq_len, dim)
    layout_logits = torch.randn(batch_size, seq_len)
    return [x, layout_logits]


def get_init_inputs():
    return [dim]
