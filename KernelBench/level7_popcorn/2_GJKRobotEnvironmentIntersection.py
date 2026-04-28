"""
Robot–environment convex collision in **xy** (ground plane).

For an **axis-aligned robot footprint** vs. a **single convex obstacle triangle**, a
**separating-axis test (SAT)** gives the same **intersection predicate** as **GJK** in
2D, with a compact fully batched reference. A fused device kernel often implements
**GJK (+ EPA for penetration depth)** for arbitrary 3D polyhedra; this problem is
the canonical 2D footprint special case.
"""

import torch
import torch.nn as nn

# Default ModelNew is PyTorch-only. run_and_check: use check_kernel=False with backend=cuda


def _rect_corners_xy(robot_min: torch.Tensor, robot_max: torch.Tensor) -> torch.Tensor:
    """(B, 4, 2) corners of axis-aligned rectangles."""
    x0, x1 = robot_min[:, 0], robot_max[:, 0]
    y0, y1 = robot_min[:, 1], robot_max[:, 1]
    return torch.stack(
        (
            torch.stack((x0, y0), dim=-1),
            torch.stack((x1, y0), dim=-1),
            torch.stack((x1, y1), dim=-1),
            torch.stack((x0, y1), dim=-1),
        ),
        dim=1,
    )


def _project_min_max(points: torch.Tensor, axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """points (B, K, 2), axis (B, 2) or (2,) broadcast → (B,) min and max projections."""
    if axis.dim() == 1:
        axis = axis.view(1, 2).expand(points.shape[0], -1)
    proj = (points * axis.unsqueeze(1)).sum(dim=-1)
    return proj.min(dim=-1).values, proj.max(dim=-1).values


def _batched_sat_rect_triangle(
    robot_min: torch.Tensor,
    robot_max: torch.Tensor,
    env_triangle: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Separating-axis test: AABB vs triangle in 2D, fully vectorized over batch.
    Returns (B,) float 1.0 if overlapping else 0.0.
    """
    B = robot_min.shape[0]
    corners = _rect_corners_xy(robot_min, robot_max)
    hit = torch.ones(B, device=robot_min.device, dtype=robot_min.dtype)

    # World axes (rectangle is AABB)
    for ax in (
        torch.tensor([1.0, 0.0], device=robot_min.device, dtype=robot_min.dtype),
        torch.tensor([0.0, 1.0], device=robot_min.device, dtype=robot_min.dtype),
    ):
        r0, r1 = _project_min_max(corners, ax)
        t0, t1 = _project_min_max(env_triangle, ax)
        separated = (r1 < t0 - eps) | (t1 < r0 - eps)
        hit = hit * (~separated).to(hit.dtype)

    # Triangle edge normals (per batch)
    for i in range(3):
        e = env_triangle[:, (i + 1) % 3] - env_triangle[:, i]
        n = torch.stack((-e[:, 1], e[:, 0]), dim=-1)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
        r0, r1 = _project_min_max(corners, n)
        t0, t1 = _project_min_max(env_triangle, n)
        separated = (r1 < t0 - eps) | (t1 < r0 - eps)
        hit = hit * (~separated).to(hit.dtype)

    return hit


class Model(nn.Module):
    """
    **Batched robot footprint vs. environment triangle** intersection in **xy**.

    Robot: axis-aligned box ``[robot_min, robot_max]`` per batch element.
    Environment: one **triangle** ``(B, 3, 2)``.

    Reference: **vectorized SAT** (5 axis tests). Fused-kernel target: same
    predicate via **GJK/EPA** or specialized SAT SIMD for planning loops.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        robot_min: torch.Tensor,
        robot_max: torch.Tensor,
        env_triangle: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            robot_min: (B, 2), robot_max: (B, 2) with max >= min componentwise.
            env_triangle: (B, 3, 2) obstacle vertices in order (any consistent winding).
        Returns:
            (B, 1) float in ``{0.0, 1.0}`` — **1** if the sets intersect.
        """
        B = robot_min.shape[0]
        assert robot_max.shape == (B, 2) and env_triangle.shape == (B, 3, 2)
        h = _batched_sat_rect_triangle(robot_min, robot_max, env_triangle, self.eps)
        return h.unsqueeze(-1)


class ModelNew(nn.Module):
    """KernelBench candidate entry point; default delegates to ``Model``."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._impl = Model(eps=eps)

    def forward(
        self,
        robot_min: torch.Tensor,
        robot_max: torch.Tensor,
        env_triangle: torch.Tensor,
    ) -> torch.Tensor:
        return self._impl(robot_min, robot_max, env_triangle)


batch_size = 16


def get_inputs():
    robot_min = torch.empty(batch_size, 2).uniform_(-1.0, 0.5)
    robot_max = robot_min + torch.empty(batch_size, 2).uniform_(0.2, 0.8)
    env_triangle = torch.empty(batch_size, 3, 2).uniform_(-0.8, 0.8)
    return [robot_min, robot_max, env_triangle]


def get_init_inputs():
    return []
