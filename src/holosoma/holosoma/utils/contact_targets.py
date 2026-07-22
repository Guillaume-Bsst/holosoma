"""Cibles de proximité relative main↔objet pour le terme de reward « C-D lite ».

Ops pures (torch-only) : aucune dépendance simulateur, testables en CI. Convention
quaternion xyzw (``w_last=True``), cohérente avec le MotionLoader.
"""
from __future__ import annotations

import torch

from holosoma.utils.rotations import quat_apply, quat_inverse


def relative_position_in_object_frame(
    point_w: torch.Tensor, object_pos_w: torch.Tensor, object_quat_w: torch.Tensor
) -> torch.Tensor:
    """Position d'un point exprimée dans le repère LOCAL de l'objet.

    ``point_w`` (..., 3) monde ; ``object_pos_w`` (..., 3) monde ;
    ``object_quat_w`` (..., 4) xyzw. Retourne (..., 3) =
    ``R(object_quat)^-1 · (point_w − object_pos_w)``. Invariant au placement monde
    (env origins) : ne dépend que du relatif point↔objet.
    """
    diff = point_w - object_pos_w
    return quat_apply(quat_inverse(object_quat_w, w_last=True), diff, w_last=True)


def beta_from_distance(distance: torch.Tensor, beta_scale: float) -> torch.Tensor:
    """Poids β de proximité : ``exp(−clamp(distance, 0) / beta_scale)``.

    ∈ (0, 1], = 1 au contact (d=0), → 0 au loin. ``distance`` (...) en m,
    ``beta_scale`` (m) > 0.
    """
    return torch.exp(-distance.clamp_min(0.0) / beta_scale)


def beta_weighted_position_reward(
    rel_cur: torch.Tensor, rel_ref: torch.Tensor, beta: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Reward bornée ``exp(−err/σ²)``, moyenne β-pondérée sur les mains.

    ``rel_cur``/``rel_ref`` (E, H, 3) ; ``beta`` (E, H) ; ``sigma`` (m). Retourne (E,)
    ∈ (0, 1]. En espace libre (β→0) : err→0 ⇒ reward→1 (neutre, jamais pénalisant).
    """
    d2 = torch.sum(torch.square(rel_cur - rel_ref), dim=-1)          # (E, H)
    w = beta / beta.sum(dim=-1, keepdim=True).clamp_min(1e-6)        # (E, H)
    err = (w * d2).sum(dim=-1)                                       # (E,)
    return torch.exp(-err / sigma**2)
