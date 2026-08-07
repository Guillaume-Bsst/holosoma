"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def _reduce_over_contact_hands(per_hand: torch.Tensor, contact_lr: torch.Tensor) -> torch.Tensor:
    """Average a per-hand reward (N, 2) over the hands the REFERENCE has in contact.

    0 when neither hand is in contact -- same convention as the single-anchor contact rewards
    (see object_flat_contact_quality_exp: paying off-contact frames builds a "die before contact"
    local optimum). Averaging rather than summing keeps the term's scale identical to the
    single-anchor version, so the configured weights carry over unchanged whether the clip is
    bimanual or not.
    """
    active = contact_lr.float()
    n_active = active.sum(dim=-1)
    return torch.where(
        n_active > 0, (per_hand * active).sum(dim=-1) / n_active.clamp_min(1.0), torch.zeros_like(n_active)
    )


def _multiscale_exp(
    error: torch.Tensor, sigma: float | None, sigmas: tuple[float, ...] | None, sigma_weights: tuple[float, ...] | None
) -> torch.Tensor:
    """``exp(-error/sigma^2)``, or a weighted average of several such kernels.

    A single Gaussian cannot be both wide enough to give gradient from far away and narrow enough to
    keep paying near the target: the wide one is nearly flat exactly where precision matters, the
    narrow one underflows to zero before the policy can find the basin. The historical workaround
    was to configure the SAME function twice at two sigmas and let the weighted sum do the blending
    -- which works, but spends two full term weights (2.0 for the box position, 1.5 for the flat
    contact) on one physical quantity, and silently makes those quantities outweigh whole-body
    tracking.

    Averaging the kernels inside one term gives an identical gradient profile bounded to 0..1, so
    the term costs one weight instead of two. ``sigma_weights`` preserves the emphasis the split
    version encoded in its two weights (e.g. 2:1 for fine:coarse).

    ``sigma`` alone keeps the original single-scale behaviour for every preset that has not moved.
    """
    if sigmas is None:
        assert sigma is not None, "one of sigma / sigmas must be given"
        return torch.exp(-error / sigma**2)
    w = sigma_weights if sigma_weights is not None else (1.0,) * len(sigmas)
    assert len(w) == len(sigmas), f"sigma_weights {w} does not match sigmas {sigmas}"
    total = sum(w)
    out = torch.zeros_like(error)
    for s, wi in zip(sigmas, w):
        out = out + wi * torch.exp(-error / s**2)
    return out / total


def _bimanual_available(mc: MotionCommand) -> bool:
    """True when per-hand contact and both anchors exist, so a term can grade each hand separately."""
    return (
        mc._anchor_body_indexes is not None
        and mc._anchor_body_indexes.numel() == 2
        and getattr(mc.motion, "has_dyn_contact", False)
    )


# ================================================================================================
# Achievable-reward gates
# ================================================================================================
#
# Several reward terms are structurally 0 on frames where the reference has no contact to grade --
# a deliberate convention (see object_flat_contact_quality_exp: paying off-contact frames builds a
# "die before contact" local optimum). The consequence is that the reward BUDGET is not constant
# over a clip: on femto14_box36 the hands touch the box on 90 of 327 frames, so 3.5 of the 12.5
# positive weight is unreachable on 72% of the clip, while the 3.0 of box-POSE terms is fully paid
# on the 128 approach frames for not touching a box that moves 1.8 cm on its own.
#
# That makes the scalar reward incomparable across phases: 9.0 during the approach and 9.0 during
# the carry are not the same achievement, and nothing in the logs said so -- a rising reward curve
# can be the policy improving, or merely episodes reaching into richer parts of the clip.
#
# Rather than change the terms (the off-contact 0 is well motivated), each term declares a gate and
# the manager sums the ceiling it could have reached, logging reward/achievable next to the reward.
# A gate returns a per-env float mask: 1.0 where the term can pay, 0.0 where it is structurally 0.

def gate_always(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Default: the term can always pay (all the pose/velocity tracking terms)."""
    return torch.ones(env.num_envs, device=env.device)


def gate_object_contact(env: WholeBodyTrackingManager) -> torch.Tensor:
    """1 where the REFERENCE has at least one hand on the box.

    Uses ``ref_hand_contact_lr`` -- the same source the gated terms themselves reduce over -- so the
    gate cannot drift from the terms it describes.
    """
    mc = _get_motion_command_and_assert_type(env)
    if not mc.motion.has_object:
        return torch.zeros(env.num_envs, device=env.device)
    return mc.ref_hand_contact_lr().any(dim=-1).float()


def gate_support_contact(env: WholeBodyTrackingManager) -> torch.Tensor:
    """1 where the REFERENCE brings a hand to the table."""
    mc = _get_motion_command_and_assert_type(env)
    if not getattr(mc.motion, "has_support_contact", False):
        return torch.zeros(env.num_envs, device=env.device)
    return mc.motion.support_ref_contact[mc.time_steps].float()


def gate_dyn_sidecar(env: WholeBodyTrackingManager) -> torch.Tensor:
    """1 when the clip carries the stage-05 contact fields (the feet terms are 0 without them)."""
    mc = _get_motion_command_and_assert_type(env)
    return torch.full(
        (env.num_envs,), float(getattr(mc.motion, "has_dyn_contact", False)), device=env.device
    )


def gate_feet_double_support(env: WholeBodyTrackingManager) -> torch.Tensor:
    """1 where the REFERENCE has BOTH feet loaded -- the only frames FeetLoadShare grades."""
    mc = _get_motion_command_and_assert_type(env)
    if not getattr(mc.motion, "has_dyn_contact", False):
        return torch.zeros(env.num_envs, device=env.device)
    return mc.dyn_foot_contact_lr.all(dim=-1).float()


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(
    env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95, normalize: bool = False
) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    With ``normalize=False`` (default, unchanged) this returns a raw sum of RADIANS past the soft
    limit, over all DOF. That number is not comparable between joints -- 0.1 rad past the limit on a
    0.5 rad-range wrist is nearly the whole remaining travel, on a 5 rad-range shoulder it is a
    detail -- and it carries a physical unit while every other reward term is dimensionless. It is
    also the reason this term historically needed a weight of -10, two orders of magnitude away from
    every other term, which makes the aggregate reward budget unreadable.

    With ``normalize=True`` each violation is divided by the margin between the soft and the hard
    limit, ``0.5 * range * (1 - soft_dof_pos_limit)``. One unit then means exactly "this joint has
    consumed all of its soft-limit margin and is at its hard limit", identically for every joint, so
    the term reads as a count of saturated DOF and a weight of -1 is meaningful on its own.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit
        normalize: Express violations as a fraction of the soft->hard margin instead of radians.

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    if normalize:
        # clamp_min guards a degenerate URDF where soft_dof_pos_limit == 1.0 leaves no margin.
        margin = (0.5 * r * (1.0 - soft_dof_pos_limit)).clamp_min(1e-6)
        out_of_limits = out_of_limits / margin
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(
    env: WholeBodyTrackingManager,
    sigma: float | None = None,
    sigmas: tuple[float, ...] | None = None,
    sigma_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Box position tracking. Pass ``sigmas`` for the multi-scale form (see ``_multiscale_exp``)."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return _multiscale_exp(error, sigma, sigmas, sigma_weights)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def object_grasp_relative_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    """Dense grasp signal: track the hand<->object RELATIVE position during contact frames.

    The global object rewards above only constrain the object's world pose; while carrying, what
    actually matters is that the object stays where the grasp expects it, expressed in the hand
    frame. On frames where the REFERENCE is in contact -- measured PER HAND by the stage-05 physics
    solve when the clip carries it, else ground-truth from the retargeting pipeline's own
    point-cloud interaction fields, else the runtime nearest-anchor distance threshold (see
    MotionCommand.ref_hand_contact_lr) -- this returns ``exp(-||rel_sim - rel_ref||^2 / sigma^2)``,
    averaged over the hands actually in contact; 0 on free frames.
    """
    from holosoma.utils.grasp_settle import gather_anchor, grasp_relative_transform

    mc = _get_motion_command_and_assert_type(env)
    if mc._anchor_body_indexes is None:
        return torch.ones(env.num_envs, device=env.device)

    def _rel_reward(a_pos_ref, a_quat_ref, a_pos_sim, a_quat_sim) -> torch.Tensor:
        rel_ref, _ = grasp_relative_transform(a_pos_ref, a_quat_ref, mc.object_pos_w, mc.object_quat_w)
        rel_sim, _ = grasp_relative_transform(
            a_pos_sim, a_quat_sim, mc.simulator_object_pos_w, mc.simulator_object_quat_w
        )
        return torch.exp(-torch.sum(torch.square(rel_sim - rel_ref), dim=-1) / sigma**2)

    # Bimanual when the clip carries measured per-hand contact: the box is genuinely held by BOTH
    # hands on most carry frames, and a single-anchor term grades only the nearest one -- leaving
    # the other hand free to be anywhere.
    if _bimanual_available(mc):
        per_hand = torch.stack(
            [
                _rel_reward(
                    mc.anchor_pos_w[:, k],
                    mc.anchor_quat_w[:, k],
                    mc.robot_anchor_pos_w[:, k],
                    mc.robot_anchor_quat_w[:, k],
                )
                for k in range(2)
            ],
            dim=-1,
        )
        return _reduce_over_contact_hands(per_hand, mc.ref_hand_contact_lr())

    anchor_idx, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    a_pos_ref, a_quat_ref = gather_anchor(mc.anchor_pos_w, mc.anchor_quat_w, anchor_idx)
    a_pos_sim, a_quat_sim = gather_anchor(mc.robot_anchor_pos_w, mc.robot_anchor_quat_w, anchor_idx)
    reward = _rel_reward(a_pos_ref, a_quat_ref, a_pos_sim, a_quat_sim)
    # 0 (not 1.0) off-contact -- see object_flat_contact_quality_exp: paying these contact rewards
    # off-contact rewards the easy pre-contact phase and builds a "die before contact" local optimum.
    return torch.where(ref_contact, reward, torch.zeros_like(reward))


def object_surface_contact_error_exp(
    env: WholeBodyTrackingManager, sigma_geodesic: float, sigma_dist: float
) -> torch.Tensor:
    """WHERE on the object surface + how deep the current contact is, vs the retargeting reference.

    ``object_grasp_relative_error_exp`` tracks a single rigid relative pose (hand -> object); it can't
    tell "gripping the right spot on the box" from "gripping the wrong spot but at the right overall
    offset". This term is the live (GPU, every step) counterpart of HoloV2's own contact channels
    (``distance``/``witness`` -- see gvhmr-fp-pipeline/contact_from_retarget.py, which bakes the
    REFERENCE witness/distance into the motion NPZ): it computes the CURRENT sim hand's nearest point
    on the box surface (``box_geometry.box_nearest_and_signed_distance``) and compares it to the
    reference witness via the box's surface geodesic (not a straight line through the box), plus the
    plain signed-distance gap for contact depth/pressure. Neutral (1) outside reference-contact
    frames or when the loaded motion doesn't carry a reference witness.
    """
    from holosoma.utils.box_geometry import box_nearest_and_signed_distance, box_surface_geodesic_distance
    from holosoma.utils.grasp_settle import gather_anchor
    from holosoma.utils.rotations import quat_rotate_inverse

    mc = _get_motion_command_and_assert_type(env)
    if mc._anchor_body_indexes is None or not mc.motion.has_gt_witness:
        return torch.ones(env.num_envs, device=env.device)

    anchor_idx, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    w_ref = mc.motion.object_ref_witness_local[mc.time_steps]  # (N, 3) box-local
    d_ref = mc.motion.object_ref_contact_dist[mc.time_steps]  # (N,)

    a_pos_sim, _ = gather_anchor(mc.robot_anchor_pos_w, mc.robot_anchor_quat_w, anchor_idx)
    box_pos, box_quat = mc.simulator_object_pos_w, mc.simulator_object_quat_w
    p_local = quat_rotate_inverse(box_quat, a_pos_sim - box_pos, w_last=True)

    half_extents = torch.tensor(mc.grasp_settle_cfg.box_half_extents, device=env.device, dtype=p_local.dtype)
    d_current, w_current = box_nearest_and_signed_distance(p_local, half_extents)

    geo = box_surface_geodesic_distance(w_ref, w_current, half_extents)
    reward_geo = torch.exp(-torch.square(geo) / sigma_geodesic**2)
    reward_dist = torch.exp(-torch.square(d_ref - d_current) / sigma_dist**2)

    reward = reward_geo * reward_dist
    # 0 (not 1.0) off-contact -- see object_flat_contact_quality_exp (attractor removal).
    return torch.where(ref_contact, reward, torch.zeros_like(reward))


def object_flat_contact_quality_exp(
    env: WholeBodyTrackingManager,
    sigma: float | None = None,
    sigmas: tuple[float, ...] | None = None,
    sigma_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """Contact QUALITY reward: reward the contact hand's flat-face keypoints to be flush on the box.

    Independent of the reference witness (unlike object_surface_contact_error_exp, which matches the
    -- oblique, marginal -- reference contact). On reference-contact frames, take K fixed keypoints on
    the contact hand's flat face (grasp_settle_cfg.flat_contact_offsets, in the wrist frame), map them
    to the box-local frame and reward ALL K being at the box surface via exp(-mean(signed_dist^2)/
    sigma^2). K>=3 coplanar points at distance 0 == a flat patch pressed against the box face, which
    resists the rotational escape a single contact point cannot (the 155deg tumble). Pairs with the
    physicality curriculum: once the box is physical the policy must present this patch to hold it.
    Neutral (1) off contact frames / without an object.
    """
    from holosoma.utils.box_geometry import box_nearest_and_signed_distance
    from holosoma.utils.grasp_settle import gather_anchor
    from holosoma.utils.rotations import quat_apply, quat_rotate_inverse

    mc = _get_motion_command_and_assert_type(env)
    if mc._anchor_body_indexes is None:
        return torch.ones(env.num_envs, device=env.device)

    offsets = torch.tensor(
        mc.grasp_settle_cfg.flat_contact_offsets, device=env.device, dtype=torch.float32
    )  # (K,3)
    n, k = env.num_envs, offsets.shape[0]
    # chiral hands (rubber): the right palm is the y-mirror of the left -> per-anchor offsets.
    # None (half-sphere) = same offsets for both anchors, unchanged behaviour.
    offsets_r_cfg = mc.grasp_settle_cfg.flat_contact_offsets_right
    offsets_r = (
        torch.tensor(offsets_r_cfg, device=env.device, dtype=torch.float32) if offsets_r_cfg is not None else offsets
    )
    half = torch.tensor(mc.grasp_settle_cfg.box_half_extents, device=env.device, dtype=torch.float32)

    def _flush_reward(a_pos: torch.Tensor, a_quat: torch.Tensor, off_env: torch.Tensor) -> torch.Tensor:
        # world keypoints: a_pos + R(a_quat) @ offset, per keypoint
        a_quat_k = a_quat.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
        off_k = off_env.reshape(n * k, 3)
        pts_w = a_pos.unsqueeze(1) + quat_apply(a_quat_k, off_k, w_last=True).reshape(n, k, 3)  # (N,K,3)

        box_pos = mc.simulator_object_pos_w.unsqueeze(1)  # (N,1,3)
        box_quat_k = mc.simulator_object_quat_w.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
        pts_local = quat_rotate_inverse(box_quat_k, (pts_w - box_pos).reshape(n * k, 3), w_last=True).reshape(
            n, k, 3
        )
        signed_dist, _ = box_nearest_and_signed_distance(pts_local, half)  # (N,K)
        return _multiscale_exp(torch.mean(torch.square(signed_dist), dim=-1), sigma, sigmas, sigma_weights)

    # Bimanual: grade the flat-patch quality of EACH hand the reference loads. A one-hand version
    # leaves the second hand ungraded on the ~35% of frames where the reference presses both, which
    # is exactly the two-sided pinch that holds a 0.36 m box.
    if _bimanual_available(mc):
        per_hand = torch.stack(
            [
                _flush_reward(
                    mc.robot_anchor_pos_w[:, side],
                    mc.robot_anchor_quat_w[:, side],
                    (offsets if side == 0 else offsets_r).unsqueeze(0).expand(n, k, 3),
                )
                for side in range(2)
            ],
            dim=-1,
        )
        return _reduce_over_contact_hands(per_hand, mc.ref_hand_contact_lr())

    anchor_idx, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    a_pos, a_quat = gather_anchor(mc.robot_anchor_pos_w, mc.robot_anchor_quat_w, anchor_idx)  # (N,3),(N,4)
    if offsets_r_cfg is not None:
        both = torch.stack([offsets, offsets_r], dim=0)  # (2,K,3)
        off_env = both[anchor_idx.clamp(0, 1)]  # (N,K,3) per-env selection by contact anchor
    else:
        off_env = offsets.unsqueeze(0).expand(n, k, 3)
    reward = _flush_reward(a_pos, a_quat, off_env)
    # 0 (not 1.0) off-contact: this is an ADDITIVE reward (weight 1.0), so returning 1.0 whenever the
    # reference isn't in contact pays a constant survival bonus for the (easy) pre-contact phase, which
    # competes with the (harder) carry phase -> a "die before contact" local optimum the policy gets
    # stuck in with high run-to-run variance. A contact-QUALITY bonus should simply be 0 when there is
    # no contact to grade.
    return torch.where(ref_contact, reward, torch.zeros_like(reward))


def support_surface_contact_error_exp(
    env: WholeBodyTrackingManager, sigma_geodesic: float, sigma_dist: float
) -> torch.Tensor:
    """robot<->TABLE : pendant SYMETRIQUE de ``object_surface_contact_error_exp`` mais pour l'objet
    STATIQUE support (la table). Sur les frames ou la REFERENCE approche la table (main proche,
    ``support_ref_contact`` bake par add_support_contact.py), recompense la main courante a etre au
    bon endroit de la SURFACE de la table (witness de reference ``support_ref_witness_local`` +
    profondeur ``support_ref_contact_dist``), via le meme SDF-boite (demi-tailles du mesh table
    ``support_half_extents``) et la geodesique de surface.

    But (demande utilisateur) : le robot SAIT ou est la table, s'en approche/s'y place correctement
    et ne fonce pas dedans -- au lieu de la traiter comme du sol. Neutre (0) hors des frames de
    reference-contact ou si le clip ne porte pas de table (attracteur off-contact retire, cf.
    object_flat_contact_quality_exp). La table etant statique et plantee a sa pose de clip, sa pose
    monde courante == pose de reference (mc.support_pos_w/quat_w).
    """
    from holosoma.utils.box_geometry import box_nearest_and_signed_distance, box_surface_geodesic_distance
    from holosoma.utils.grasp_settle import gather_anchor
    from holosoma.utils.rotations import quat_rotate_inverse

    mc = _get_motion_command_and_assert_type(env)
    if mc._anchor_body_indexes is None or not getattr(mc.motion, "has_support_contact", False):
        return torch.zeros(env.num_envs, device=env.device)

    ts = mc.time_steps
    ref_contact = mc.motion.support_ref_contact[ts]  # (N,)
    anchor_idx = mc.motion.support_ref_anchor_idx[ts]  # (N,) 0=left,1=right
    w_ref = mc.motion.support_ref_witness_local[ts]  # (N,3) table-local
    d_ref = mc.motion.support_ref_contact_dist[ts]  # (N,)

    a_pos_sim, _ = gather_anchor(mc.robot_anchor_pos_w, mc.robot_anchor_quat_w, anchor_idx)
    tab_pos, tab_quat = mc.support_pos_w, mc.support_quat_w  # statique planté
    p_local = quat_rotate_inverse(tab_quat, a_pos_sim - tab_pos, w_last=True)

    half = mc.motion.support_half_extents.to(p_local.dtype)
    d_current, w_current = box_nearest_and_signed_distance(p_local, half)

    geo = box_surface_geodesic_distance(w_ref, w_current, half)
    reward = torch.exp(-torch.square(geo) / sigma_geodesic**2) * torch.exp(
        -torch.square(d_ref - d_current) / sigma_dist**2
    )
    return torch.where(ref_contact, reward, torch.zeros_like(reward))


# ================================================================================================
# Stage-05 dynamics rewards (SPIDER physics sidecar -- see MotionLoader's dyn_* fields)
# ================================================================================================


def torque_envelope_penalty(
    env: WholeBodyTrackingManager, margin: float = 1.5, action_term_name: str = "joint_control"
) -> torch.Tensor:
    """Penalise commanded torque that exceeds what the MOTION demands, per joint.

    The stage-05 physics solve knows how much torque this motion needs at every frame
    (``dyn_tau``), so "the policy is fighting itself / bracing / shaking" becomes measurable instead
    of having to be guessed at through an action-rate proxy. This is deliberately ONE-SIDED::

        penalty = sum_j relu(|tau_cmd_j| - margin * |tau_ref_j|)^2 / tau_limit_j^2

    Under-torque is never penalised. That matters because ``tau_ref`` was solved with SPIDER's stock
    kp=500 actuators, which can track the reference through contact transitions far more sharply
    than the real gains can: at those instants ``tau_ref`` spikes to values a real-gain controller
    simply cannot reach, and a two-sided (tracking) version of this term would punish the policy for
    failing to do the impossible. Asking only that it not spend MORE than the motion needs, with a
    ``margin`` of headroom on top, is the part that transfers.

    Normalised by the actuator limit per joint, otherwise the knee (139 N.m) would drown out the
    wrist (5 N.m) and the term would only ever grade the legs. Returns 0 for clips without a
    sidecar, so the same reward config runs on un-enriched motions.
    """
    mc = _get_motion_command_and_assert_type(env)
    if not getattr(mc.motion, "has_dyn_tau", False):
        return torch.zeros(env.num_envs, device=env.device)

    action_term = env.action_manager.get_term(action_term_name)
    # Mean |tau| over the decimation sub-steps of this control step: the reference is one value per
    # control step, so comparing it to a single sub-step's torque would grade sampling phase.
    tau_cmd = action_term.torques_substep.abs().mean(dim=1)  # (E, num_dof)
    tau_ref = mc.dyn_tau.abs()  # (E, num_dof), same DOF order

    excess = torch.relu(tau_cmd - margin * tau_ref) / env.torque_limits
    return torch.sum(torch.square(excess), dim=-1)


class _FeetTermBase(RewardTermBase):
    """Shared foot-body resolution for the stage-05 feet terms.

    ``dyn_foot_contact_lr`` is ordered [left, right] (fixed by merge_dynamics.py), so the two foot
    bodies must be resolved in that same order -- hence explicit names rather than a regex, whose
    match order would be an accident of the body list.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        names = cfg.params.get("foot_body_names", ("left_ankle_roll_link", "right_ankle_roll_link"))
        body_names = self.env.simulator.body_names  # type: ignore[attr-defined]
        for n in names:
            assert n in body_names, f"foot body '{n}' not in robot bodies: {body_names}"
        self.foot_indexes = torch.tensor(
            [body_names.index(n) for n in names], dtype=torch.long, device=self.env.device
        )

    def _ref_stance(self, env: WholeBodyTrackingManager) -> torch.Tensor | None:
        mc = env.command_manager.get_state("motion_command")
        if not isinstance(mc, MotionCommand) or not getattr(mc.motion, "has_dyn_contact", False):
            return None
        return mc.dyn_foot_contact_lr  # (E, 2) bool, [left, right]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


class FeetContactSchedule(_FeetTermBase):
    """Reward matching the reference's per-foot stance/swing schedule.

    Nothing in the existing reward set says WHEN each foot should be on the ground: the body-position
    and velocity tracking terms constrain where the feet are, which leaves the policy free to invent
    its own support sequence (shuffling, double-support-through-everything) as long as the links end
    up roughly in place. The physics solve knows exactly which foot carries load at each frame, so
    this grades the contact schedule directly::

        reward = mean over both feet of 1{sim foot loaded == reference foot loaded}

    Both error directions are graded, and they are different faults: a foot down that should be
    swinging is a shuffle, a foot up that should be planted is a loss of support. 0..1 per step,
    0 for clips without the sidecar.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        ref_stance = self._ref_stance(env)
        if ref_stance is None:
            return torch.zeros(env.num_envs, device=env.device)
        # Same history-max convention as UndesiredContacts: a foot that touched at any point during
        # the control step counts as loaded, which is what the reference flag means too.
        forces = self.env.simulator.contact_forces_history[:, :, self.foot_indexes]  # (E, H, 2, 3)
        sim_stance = torch.max(torch.norm(forces, dim=-1), dim=1)[0] > self.threshold  # (E, 2)
        return (sim_stance == ref_stance).float().mean(dim=-1)


class FeetSlipOnRefStance(_FeetTermBase):
    """Penalise horizontal foot velocity while the REFERENCE has that foot planted.

    Slip is only a fault during stance -- a swinging foot is supposed to move fast, and a penalty
    that cannot tell the two apart either taxes the swing or has to be weighted so low it stops
    mattering. Gating on the reference's own stance flag (rather than the sim's measured contact)
    keeps the signal well defined even when the policy has lost the contact it should have: a foot
    that should be planted and is instead sliding through the air still gets penalised.

    Returns ``sum over feet of stance * ||v_xy||^2`` (0 for clips without the sidecar). Vertical
    velocity is excluded: lifting off early is a schedule error, which FeetContactSchedule grades.
    """

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        ref_stance = self._ref_stance(env)
        if ref_stance is None:
            return torch.zeros(env.num_envs, device=env.device)
        v_xy = self.env.simulator._rigid_body_vel[:, self.foot_indexes, :2]  # (E, 2, 2)
        return torch.sum(ref_stance.float() * torch.sum(torch.square(v_xy), dim=-1), dim=-1)


class FeetLoadShare(_FeetTermBase):
    """Reward matching the reference's left/right load DISTRIBUTION during double support.

    ``FeetContactSchedule`` grades only the binary stance flags, which carry almost no information
    on a carrying clip: the reference is in double support on 257 of 327 frames (79%) of
    ``femto14_box36``, so four frames out of five have both flags True and that term sits pinned at
    1.0 whatever the policy does with its weight. What separates a carry from a shuffle over those
    frames is HOW the load is split between the feet -- the weight shift that places the CoM -- and
    the stage-05 solve measures exactly that (``dyn_foot_grf_lr``)::

        f = |F_left| / (|F_left| + |F_right|)     reward = exp(-(f_sim - f_ref)^2 / sigma^2)

    Grading the SHARE rather than the absolute force is deliberate. ``dyn_foot_grf_lr`` peaks at
    2427 N on a single foot in this clip (~7x body weight) at contact transitions -- the same
    kp=500 solver artefact that makes ``dyn_tau`` unusable as an absolute target (cf.
    ``torque_envelope_penalty``). A ratio is immune to it: any magnitude error common to both feet
    cancels and only the distribution survives. It is also dimensionless and bounded, so the same
    sigma carries across robots with no force normalisation.

    Both sides use the norm of the net contact force vector, which is what ``merge_dynamics.py``
    bakes into the reference (``norm`` of the summed foot<->floor contact forces), not just its
    vertical component.

    Returns 0 (neutral) unless the reference has BOTH feet loaded and both totals are non-degenerate:

    - single support / flight: the share is trivially 0 or 1 and says nothing about weight
      distribution -- ``FeetContactSchedule`` already grades those frames;
    - the default-pose segments prepended/appended to the clip pad the stance flags with True but
      the GRF with ZERO (see the ``pad_foot`` / ``pad_grf`` pair in ``command/terms/wbt.py``), so
      the reference share is genuinely undefined there and must not be read -- the ``min_force``
      gate on ``tot_ref`` is what catches those frames, not the stance flags;
    - robot airborne in sim: 0 is correct rather than a penalty, and since 0 is below any achievable
      share score, breaking contact can never pay -- no "lift off to dodge the term" optimum.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.sigma = cfg.params.get("sigma", 0.25)
        self.min_force = cfg.params.get("min_force", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        ref_stance = self._ref_stance(env)
        if ref_stance is None:
            return torch.zeros(env.num_envs, device=env.device)
        mc = _get_motion_command_and_assert_type(env)
        grf_ref = mc.dyn_foot_grf_lr  # (E, 2), N, [left, right]
        # Most recent net force, NOT contact_forces_history: only the first
        # ``effective_history_length`` slots of that buffer are refreshed per control step
        # (isaacsim.py:894-900), the rest still hold the previous step's values. FeetContactSchedule
        # survives reading it because a stale entry under a max() only ever reasserts a contact that
        # just existed; a load RATIO reduced over the same window would be silently blended with the
        # previous step's weight distribution.
        f_sim = torch.norm(env.simulator.contact_forces[:, self.foot_indexes], dim=-1)  # (E, 2)

        tot_ref = grf_ref.sum(dim=-1)
        tot_sim = f_sim.sum(dim=-1)
        valid = ref_stance.all(dim=-1) & (tot_ref > self.min_force) & (tot_sim > self.min_force)

        share_ref = grf_ref[:, 0] / tot_ref.clamp_min(self.min_force)
        share_sim = f_sim[:, 0] / tot_sim.clamp_min(self.min_force)
        rew = torch.exp(-torch.square(share_sim - share_ref) / self.sigma**2)
        return torch.where(valid, rew, torch.zeros_like(rew))


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    """Penalise contacts the REFERENCE motion does not have at this frame.

    A contact is "undesired" only if the retargeted reference doesn't put that body against
    something at that instant -- a static body-name blocklist can't express that. Carrying a 0.36 m
    box, the reference itself presses the forearm/wrist against it; with a fixed regex (which only
    ever exempted feet, ankles and ``wrist_yaw``) those frames pay a permanent ~-0.4/s tax on
    exactly the behaviour we're training. Conversely a body that is FAR from everything in the
    reference and yet reports contact in sim is a genuine fault (knee on the ground, torso into the
    table), whatever its name.

    The retargeting pipeline only bakes per-frame contact for the two hand ANCHORS
    (``object_ref_*`` / ``support_ref_*``), so the per-BODY mask is reconstructed here from the
    reference geometry, which is available for all bodies (``motion.body_pos_w`` is reindexed into
    the full robot body order). A body is exempt at a frame when, IN THE REFERENCE, it is:
      - within ``ground_margin`` of the clip ground (feet/ankles in stance), or
      - within ``ref_contact_margin`` of the box surface (box SDF, ``grasp_settle.box_half_extents``), or
      - within ``ref_contact_margin`` of the table surface (``motion.support_half_extents``).

    Everything is evaluated in the CLIP frame (reference bodies vs reference object/support poses),
    so no env-origin bookkeeping and the mask is well defined during the default-pose transitions
    prepended/appended to the clip (object far away there -> only the ground rule exempts).

    ``ref_contact_margin`` is deliberately generous: body positions are link ORIGINS, not collision
    surfaces, so the margin has to absorb the link radius. The asymmetry justifies erring wide -- a
    false exemption merely fails to penalise a contact, a false penalty actively fights the task.

    The regex stays as an unconditional floor (bodies never penalised whatever the reference does);
    the reference mask exempts more, per frame. Falls back to the pure-regex behaviour when the env
    has no motion command (non-WBT use).

    Opt-in via ``use_reference_mask`` (default False) so the object-free WBT preset keeps its exact
    historical behaviour; only ``g1_29dof_wbt_reward_w_object`` turns it on.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)
        self.ref_contact_margin = cfg.params.get("ref_contact_margin", 0.15)
        self.ground_margin = cfg.params.get("ground_margin", 0.10)
        self.use_reference_mask = cfg.params.get("use_reference_mask", False)
        # Indexes of the penalised bodies in the MOTION's own body axis, composed lazily on first
        # use (the command manager may not be set up yet at reward-term construction). Composing
        # them avoids `motion.body_pos_w`, a property that gathers the WHOLE clip (T, 32, 3) before
        # the frames are picked -- once per step, per term.
        self._ref_body_indexes_in_motion: torch.Tensor | None = None

    def _reference_contact_mask(self, env: WholeBodyTrackingManager) -> torch.Tensor | None:
        """(num_envs, n_penalised_bodies) bool: True where the REFERENCE expects contact."""
        from holosoma.utils.box_geometry import box_nearest_and_signed_distance
        from holosoma.utils.rotations import quat_rotate_inverse

        motion_command = env.command_manager.get_state("motion_command")
        if not isinstance(motion_command, MotionCommand):
            return None

        mc = motion_command
        ts = mc.time_steps
        if self._ref_body_indexes_in_motion is None:
            # motion._body_indexes maps robot body order -> motion body order; our indexes are in
            # robot body order (simulator.body_names == simulator._body_list, the same list the
            # loader was built against), so composing the two lands directly in the motion axis.
            self._ref_body_indexes_in_motion = mc.motion._body_indexes[self.undesired_contacts_body_indexes]
        # (N, B, 3) clip frame, only the bodies this term penalises.
        ref_pos = mc.motion._body_pos_w[ts][:, self._ref_body_indexes_in_motion]
        n, b = ref_pos.shape[0], ref_pos.shape[1]

        # Ground: the clip's floor is z=0 (reference frame), so stance feet/ankles sit just above it.
        allowed = ref_pos[..., 2] < self.ground_margin

        if mc.motion.has_object:
            obj_pos = mc.motion.object_pos_w[ts].unsqueeze(1)  # (N,1,3) clip frame
            obj_quat = mc.motion.object_quat_w[ts]  # (N,4) xyzw
            obj_quat_b = obj_quat.unsqueeze(1).expand(n, b, 4).reshape(n * b, 4)
            p_local = quat_rotate_inverse(obj_quat_b, (ref_pos - obj_pos).reshape(n * b, 3), w_last=True)
            half = torch.tensor(
                mc.grasp_settle_cfg.box_half_extents, device=ref_pos.device, dtype=ref_pos.dtype
            )
            d_obj, _ = box_nearest_and_signed_distance(p_local.reshape(n, b, 3), half)  # (N,B)
            allowed = allowed | (d_obj < self.ref_contact_margin)

        # getattr: MultiMotionLoader (multi-clip training) doesn't define the support fields at all.
        if getattr(mc.motion, "has_support", False):
            # Static table: its reference pose is constant over the clip, broadcast over frames.
            sup_pos = mc.motion._support_pos_w.view(1, 1, 3)
            sup_quat = mc.motion._support_quat_w.view(1, 4).expand(n * b, 4)
            p_local = quat_rotate_inverse(sup_quat, (ref_pos - sup_pos).reshape(n * b, 3), w_last=True)
            half = mc.motion.support_half_extents.to(ref_pos.dtype)
            d_sup, _ = box_nearest_and_signed_distance(p_local.reshape(n, b, 3), half)  # (N,B)
            allowed = allowed | (d_sup < self.ref_contact_margin)

        return allowed

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        if self.use_reference_mask:
            allowed = self._reference_contact_mask(env)
            if allowed is not None:
                is_contact = is_contact & ~allowed
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
