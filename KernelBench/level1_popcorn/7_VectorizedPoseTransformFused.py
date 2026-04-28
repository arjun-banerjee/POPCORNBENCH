import torch
import torch.nn as nn
import torch.nn.functional as F

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


def _quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Unit quaternion ``q = (w, x, y, z)`` (scalar-first) to rotation matrix ``R``,
    shape ``(B, 3, 3)``, with ``p' = p @ R.T`` for row vectors ``p`` (``N×3``).
    """
    q = F.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    row0 = torch.stack((m00, m01, m02), dim=-1)
    row1 = torch.stack((m10, m11, m12), dim=-1)
    row2 = torch.stack((m20, m21, m22), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


class Model(nn.Module):
    """
    **Fused vectorized pose transform**: rotation from unit quaternion ``q`` and
    translation ``t`` applied to a batch of 3D points.

    One forward fuses **quaternion → ``3×3``** (Hamilton, scalar-first ``w,x,y,z``)
    with **batched matmul** and **broadcast translation**—typical for LiDAR / camera
    extrinsics, hand-eye calibration, and world ↔ body frame changes in VLA stacks.

    ``points' = points @ R.T + t`` with ``points`` row-wise ``(B, N, 3)``.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        points: torch.Tensor,
        quat: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3) points in the **source** frame.
            quat: (B, 4) unit quaternion ``(w, x, y, z)`` mapping source → target rotation.
            trans: (B, 3) translation in the **target** frame (same convention as ``R``).
        Returns:
            (B, N, 3) points in the **target** frame.
        """
        B, N, _ = points.shape
        assert quat.shape == (B, 4) and trans.shape == (B, 3)

        R = _quaternion_to_rotation_matrix(quat)
        return torch.matmul(points, R.transpose(-1, -2)) + trans.unsqueeze(1)


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self):
        super().__init__()
        self._impl = Model()

    def forward(
        self,
        points: torch.Tensor,
        quat: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        return self._impl(points, quat, trans)


batch_size = 8
num_points = 512


def get_inputs():
    points = torch.randn(batch_size, num_points, 3)
    quat = torch.randn(batch_size, 4)
    trans = torch.randn(batch_size, 3)
    return [points, quat, trans]


def get_init_inputs():
    return []
