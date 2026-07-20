"""Batched GPU (torch) geometry for an axis-aligned box, in the box's own local frame.

Two primitives, both fully vectorized (no python loop over the batch):
  - ``box_nearest_and_signed_distance``: analytical box SDF (Quilez) + the actual nearest surface
    point ("witness"), used to compute the CURRENT hand<->box contact live during training, on GPU,
    the same quantities HoloV2's retargeting pipeline (``targets/interaction/fields.py``) computes
    offline from real point clouds (distance/witness -- see gvhmr-fp-pipeline/contact_from_retarget.py).
  - ``box_surface_geodesic_distance``: distance BETWEEN two points constrained to the box surface,
    walking the surface rather than cutting through the box -- same-face is exact (flat 2D), adjacent
    faces are exact (unfold about the shared edge), opposite faces fall back to a penalized straight
    3D distance (a single point rarely jumps between opposite faces frame-to-frame, so an exact
    through-a-third-face path isn't worth the extra branching for a reward signal).
"""

from __future__ import annotations

import torch

_OPPOSITE_FACE_PENALTY = 1.3  # inflate the crude fallback so it never reads as "closer" than a true path


def _safe_sign(x: torch.Tensor) -> torch.Tensor:
    """Like ``torch.sign`` but never 0 -- a coordinate sitting exactly on the box's symmetry plane
    (e.g. a query at the box centre) still needs SOME face to push to / assign to."""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def box_nearest_and_signed_distance(
    p_local: torch.Tensor, half_extents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance + nearest surface point, box-local frame.

    Args:
        p_local: (..., 3) query point(s) in the box's local frame.
        half_extents: (3,) box half-extents (same box for every query -- rigid, constant geometry).

    Returns:
        signed_dist: (...,) negative when ``p_local`` is inside the box.
        nearest_local: (..., 3) closest point ON the box surface, box-local frame.
    """
    q = p_local.abs() - half_extents
    outside_dist = torch.clamp(q, min=0.0).norm(dim=-1)
    inside_dist = torch.clamp(q.max(dim=-1).values, max=0.0)
    signed_dist = outside_dist + inside_dist

    outside_mask = q.max(dim=-1).values > 0.0  # any axis past the face -> the clamp formula applies
    nearest_outside = torch.clamp(p_local, -half_extents, half_extents)

    # Fully inside: push the axis with the LEAST room (nearest face) out to that face, sign-preserved.
    slack = half_extents - p_local.abs()  # >= 0 everywhere when inside
    min_axis = slack.argmin(dim=-1)
    flat_shape = p_local.shape[:-1]
    idx = torch.arange(p_local.numel() // 3, device=p_local.device)
    p_flat = p_local.reshape(-1, 3)
    axis_flat = min_axis.reshape(-1)
    nearest_inside_flat = p_flat.clone()
    nearest_inside_flat[idx, axis_flat] = _safe_sign(p_flat[idx, axis_flat]) * half_extents[axis_flat]
    nearest_inside = nearest_inside_flat.reshape(*flat_shape, 3)

    nearest_local = torch.where(outside_mask.unsqueeze(-1), nearest_outside, nearest_inside)
    return signed_dist, nearest_local


def _face_axis_and_sign(p_local: torch.Tensor, half_extents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Which face a surface point lies on: axis in {0,1,2} (least slack to the boundary) + its sign."""
    slack = half_extents - p_local.abs()
    axis = slack.argmin(dim=-1)
    flat_shape = p_local.shape[:-1]
    idx = torch.arange(p_local.numel() // 3, device=p_local.device)
    p_flat = p_local.reshape(-1, 3)
    axis_flat = axis.reshape(-1)
    sign_flat = _safe_sign(p_flat[idx, axis_flat])
    return axis, sign_flat.reshape(flat_shape)


def box_surface_geodesic_distance(
    p1_local: torch.Tensor, p2_local: torch.Tensor, half_extents: torch.Tensor
) -> torch.Tensor:
    """Approximate geodesic (surface-walking) distance between two points ON the box surface.

    Args:
        p1_local, p2_local: (..., 3) surface points, box-local frame (e.g. one from the retargeting
            reference witness, the other from ``box_nearest_and_signed_distance`` on the live sim).
        half_extents: (3,) box half-extents.

    Returns:
        (...,) geodesic distance estimate (metres). Exact for same-face and adjacent-face pairs;
        penalized straight-line fallback for opposite faces (see module docstring).
    """
    axis1, sign1 = _face_axis_and_sign(p1_local, half_extents)
    axis2, sign2 = _face_axis_and_sign(p2_local, half_extents)

    same_axis = axis1 == axis2
    same_face = same_axis & (sign1 == sign2)
    opposite_face = same_axis & ~same_face

    euclid = (p1_local - p2_local).norm(dim=-1)

    # Adjacent-face unfolding: axis3 = the axis shared by neither face (0+1+2=3, so 3-a1-a2 is exact
    # whenever a1 != a2). Guard axis3 into a valid index even where the mask doesn't apply (same_axis
    # rows), since torch still evaluates the gather everywhere before `where` selects it out.
    axis3 = (3 - axis1 - axis2).clamp(0, 2)
    h = half_extents

    def _gather(t: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        return torch.gather(t, -1, axis.unsqueeze(-1)).squeeze(-1)

    p1_a2 = _gather(p1_local, axis2)
    p2_a1 = _gather(p2_local, axis1)
    p1_a3 = _gather(p1_local, axis3)
    p2_a3 = _gather(p2_local, axis3)
    h_a2 = h[axis2]
    h_a1 = h[axis1]

    u1 = (p1_a2 - sign2 * h_a2).abs()
    u2 = (p2_a1 - sign1 * h_a1).abs()
    adjacent_dist = torch.sqrt((u1 + u2) ** 2 + (p1_a3 - p2_a3) ** 2)

    dist = torch.where(same_face, euclid, adjacent_dist)
    dist = torch.where(opposite_face, euclid * _OPPOSITE_FACE_PENALTY, dist)
    return dist
