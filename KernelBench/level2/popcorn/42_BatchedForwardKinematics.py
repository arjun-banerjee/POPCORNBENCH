import math

import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda
# until the candidate file contains a real CUDA kernel (see kernel_static_checker).


class Model(nn.Module):
    """
    **Batched forward kinematics (FK)** for a **planar serial** revolute chain.

    Fixed link lengths ``L_j`` (buffer), joint angles ``θ_j`` (batch ``B``). For each
    link, the heading is ``Σ_{k≤j} θ_k``; joint positions are the cumulative sum of
    ``L_j [cos(heading), sin(heading)]``—fully **vectorized** over ``B`` and ``J``.

    Returns positions of the **end of each link** in the base frame ``(B, J, 2)``
    (last row is the end-effector in the plane). Typical fusion target for
    batched geometry in calibration, whole-body control, and VLA motion heads.
    """

    def __init__(self, num_joints: int, link_scale: float = 1.0):
        super().__init__()
        assert num_joints >= 1
        self.num_joints = num_joints
        lengths = torch.ones(num_joints, dtype=torch.float32) * float(link_scale)
        self.register_buffer("link_lengths", lengths)

    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_angles: (B, J) radians, one revolute per link.
        Returns:
            positions: (B, J, 2) world-frame (x, y) at each link end; ``[..., -1, :]`` is EE.
        """
        B, J = joint_angles.shape
        assert J == self.num_joints

        heading = torch.cumsum(joint_angles, dim=-1)
        c = torch.cos(heading)
        s = torch.sin(heading)
        direction = torch.stack((c, s), dim=-1)
        displacements = direction * self.link_lengths.view(1, J, 1)
        return torch.cumsum(displacements, dim=1)


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, num_joints: int, link_scale: float = 1.0):
        super().__init__()
        self._impl = Model(num_joints, link_scale=link_scale)

    def forward(self, joint_angles: torch.Tensor) -> torch.Tensor:
        return self._impl(joint_angles)


batch_size = 32
num_joints = 7
link_scale = 0.15


def get_inputs():
    return [torch.empty(batch_size, num_joints).uniform_(-math.pi, math.pi)]


def get_init_inputs():
    return [num_joints, link_scale]
