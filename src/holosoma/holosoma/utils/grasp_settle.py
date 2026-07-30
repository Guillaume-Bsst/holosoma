"""Grasp-consistent object initialisation & settling helpers for object-interaction WBT.

Background
----------
When a whole-body-tracking episode is reset in the *middle* of a manipulation clip
(the robot is carrying the object), the standard Reference-State-Initialisation path
teleports the robot and the object independently and adds independent per-actor noise.
That destroys the hand<->object relative transform, so at the first physics step the
object is either penetrating the hands (and gets ejected) or floating free (and drops),
and the episode dies immediately on the object-tracking termination.

These helpers make those contact resets stable by preserving / restoring the reference
grasp transform (object pose expressed in the hand frame). They are intentionally pure
torch with no IsaacSim dependency so they can be unit-tested standalone and shared by
both the training reset/step path (``MotionCommand``) and the offline probe harness.

Conventions
-----------
All quaternions are xyzw (``w_last=True``), matching the rest of the WBT stack.
"""

from __future__ import annotations

import torch

from holosoma.utils.rotations import quat_apply, quat_inverse, quat_mul, quat_rotate_inverse


def grasp_relative_transform(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    obj_pos: torch.Tensor,
    obj_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express the object pose in the anchor (hand) frame.

    Args:
        anchor_pos:  (..., 3) anchor world position.
        anchor_quat: (..., 4) anchor world orientation (xyzw).
        obj_pos:     (..., 3) object world position.
        obj_quat:    (..., 4) object world orientation (xyzw).

    Returns:
        rel_pos:  (..., 3) object position in the anchor frame.
        rel_quat: (..., 4) object orientation in the anchor frame (xyzw).
    """
    rel_pos = quat_rotate_inverse(anchor_quat, obj_pos - anchor_pos, w_last=True)
    rel_quat = quat_mul(quat_inverse(anchor_quat, w_last=True), obj_quat, w_last=True)
    return rel_pos, rel_quat


def apply_grasp_transform(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    rel_pos: torch.Tensor,
    rel_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the object world pose from an anchor pose and a stored grasp transform.

    Inverse of :func:`grasp_relative_transform`: feeding the same ``anchor_pos``/``anchor_quat``
    round-trips back to the original object pose. Feeding a *different* anchor pose rigidly
    carries the object with the anchor at the stored relative transform (a kinematic weld).
    """
    obj_pos = anchor_pos + quat_apply(anchor_quat, rel_pos, w_last=True)
    obj_quat = quat_mul(anchor_quat, rel_quat, w_last=True)
    return obj_pos, obj_quat


def select_grasp_anchor(anchor_pos: torch.Tensor, object_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick, per env, the nearest candidate anchor body to the object.

    Args:
        anchor_pos: (N, A, 3) candidate anchor world positions (e.g. the two wrists).
        object_pos: (N, 3) object world position.

    Returns:
        anchor_idx:  (N,) long index into ``A`` of the nearest anchor.
        anchor_dist: (N,) distance from the nearest anchor to the object (metres).
    """
    dist = torch.norm(anchor_pos - object_pos.unsqueeze(1), dim=-1)  # (N, A)
    anchor_dist, anchor_idx = dist.min(dim=1)
    return anchor_idx, anchor_dist


def anneal_prob(step: int, prob_start: float, prob_end: float, anneal_steps: int) -> float:
    """Linearly anneal a probability from ``prob_start`` to ``prob_end`` over ``anneal_steps`` steps.

    Clamps outside the ramp: returns ``prob_start`` at/before step 0 and ``prob_end`` at/after
    ``anneal_steps``. ``anneal_steps <= 0`` returns ``prob_end`` immediately (no ramp).
    """
    if anneal_steps <= 0:
        return float(prob_end)
    alpha = min(max(step / float(anneal_steps), 0.0), 1.0)
    return float(prob_start + (prob_end - prob_start) * alpha)


def gather_anchor(
    anchor_pos: torch.Tensor, anchor_quat: torch.Tensor, anchor_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the chosen anchor's pose per env.

    Args:
        anchor_pos:  (N, A, 3) candidate anchor positions.
        anchor_quat: (N, A, 4) candidate anchor orientations (xyzw).
        anchor_idx:  (N,) index into ``A`` selected per env.

    Returns:
        (N, 3) position and (N, 4) orientation of the selected anchor.
    """
    ar = torch.arange(anchor_idx.shape[0], device=anchor_idx.device)
    return anchor_pos[ar, anchor_idx], anchor_quat[ar, anchor_idx]
