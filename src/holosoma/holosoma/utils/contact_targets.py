"""Relative hand<->object proximity targets for the "C-D lite" reward term.

Pure ops (torch-only): no simulator dependency, testable in CI. xyzw quaternion
convention (``w_last=True``), consistent with the MotionLoader.
"""
from __future__ import annotations

import torch

from holosoma.utils.rotations import quat_apply, quat_inverse


def relative_position_in_object_frame(
    point_w: torch.Tensor, object_pos_w: torch.Tensor, object_quat_w: torch.Tensor
) -> torch.Tensor:
    """Position of a point expressed in the object's LOCAL frame.

    ``point_w`` (..., 3) world; ``object_pos_w`` (..., 3) world;
    ``object_quat_w`` (..., 4) xyzw. Returns (..., 3) =
    ``R(object_quat)^-1 . (point_w - object_pos_w)``. Invariant to world placement
    (env origins): it only depends on the point<->object relative pose.
    """
    diff = point_w - object_pos_w
    return quat_apply(quat_inverse(object_quat_w, w_last=True), diff, w_last=True)


def beta_from_distance(distance: torch.Tensor, beta_scale: float) -> torch.Tensor:
    """Proximity weight beta: ``exp(-clamp(distance, 0) / beta_scale)``.

    In (0, 1], = 1 at contact (d=0), -> 0 far away. ``distance`` (...) in m,
    ``beta_scale`` (m) > 0.
    """
    return torch.exp(-distance.clamp_min(0.0) / beta_scale)


def beta_weighted_position_reward(
    rel_cur: torch.Tensor, rel_ref: torch.Tensor, beta: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Bounded reward ``exp(-err/sigma^2)``, beta-weighted mean over the hands.

    ``rel_cur``/``rel_ref`` (E, H, 3); ``beta`` (E, H); ``sigma`` (m). Returns (E,)
    in (0, 1]. In free space (beta -> 0): err -> 0 => reward -> 1 (neutral, never
    penalizing).
    """
    d2 = torch.sum(torch.square(rel_cur - rel_ref), dim=-1)          # (E, H)
    w = beta / beta.sum(dim=-1, keepdim=True).clamp_min(1e-6)        # (E, H)
    err = (w * d2).sum(dim=-1)                                       # (E,)
    return torch.exp(-err / sigma**2)
