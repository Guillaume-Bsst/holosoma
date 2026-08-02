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


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

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


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def object_grasp_relative_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    """Dense grasp signal: track the hand<->object RELATIVE position during contact frames.

    The global object rewards above only constrain the object's world pose; while carrying, what
    actually matters is that the object stays where the grasp expects it, expressed in the hand
    frame. On frames where the REFERENCE is in contact -- ground-truth from the retargeting
    pipeline's own point-cloud interaction fields when the motion carries them (see
    MotionCommand._lookup_ref_contact), else the runtime nearest-anchor distance threshold -- this
    returns ``exp(-||rel_sim - rel_ref||^2 / sigma^2)``; on free frames it returns 1 (neutral) so
    the term never pushes the policy to break contact.
    """
    from holosoma.utils.grasp_settle import gather_anchor, grasp_relative_transform

    mc = _get_motion_command_and_assert_type(env)
    if mc._anchor_body_indexes is None:
        return torch.ones(env.num_envs, device=env.device)

    anchor_idx, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)

    a_pos_ref, a_quat_ref = gather_anchor(mc.anchor_pos_w, mc.anchor_quat_w, anchor_idx)
    rel_ref, _ = grasp_relative_transform(a_pos_ref, a_quat_ref, mc.object_pos_w, mc.object_quat_w)

    a_pos_sim, a_quat_sim = gather_anchor(mc.robot_anchor_pos_w, mc.robot_anchor_quat_w, anchor_idx)
    rel_sim, _ = grasp_relative_transform(
        a_pos_sim, a_quat_sim, mc.simulator_object_pos_w, mc.simulator_object_quat_w
    )

    error = torch.sum(torch.square(rel_sim - rel_ref), dim=-1)
    reward = torch.exp(-error / sigma**2)
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


def object_flat_contact_quality_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
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

    anchor_idx, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    a_pos, a_quat = gather_anchor(mc.robot_anchor_pos_w, mc.robot_anchor_quat_w, anchor_idx)  # (N,3),(N,4)

    offsets = torch.tensor(mc.grasp_settle_cfg.flat_contact_offsets, device=env.device, dtype=a_pos.dtype)  # (K,3)
    n, k = env.num_envs, offsets.shape[0]
    # chiral hands (rubber): the right palm is the y-mirror of the left -> per-anchor offsets.
    # None (half-sphere) = same offsets for both anchors, unchanged behaviour.
    offsets_r_cfg = mc.grasp_settle_cfg.flat_contact_offsets_right
    if offsets_r_cfg is not None:
        offsets_r = torch.tensor(offsets_r_cfg, device=env.device, dtype=a_pos.dtype)  # (K,3)
        both = torch.stack([offsets, offsets_r], dim=0)  # (2,K,3)
        off_env = both[anchor_idx.clamp(0, 1)]  # (N,K,3) per-env selection by contact anchor
    else:
        off_env = offsets.unsqueeze(0).expand(n, k, 3)
    # world keypoints: a_pos + R(a_quat) @ offset, per keypoint
    a_quat_k = a_quat.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
    off_k = off_env.reshape(n * k, 3)
    pts_w = a_pos.unsqueeze(1) + quat_apply(a_quat_k, off_k, w_last=True).reshape(n, k, 3)  # (N,K,3)

    box_pos = mc.simulator_object_pos_w.unsqueeze(1)  # (N,1,3)
    box_quat_k = mc.simulator_object_quat_w.unsqueeze(1).expand(n, k, 4).reshape(n * k, 4)
    pts_local = quat_rotate_inverse(box_quat_k, (pts_w - box_pos).reshape(n * k, 3), w_last=True).reshape(n, k, 3)

    half = torch.tensor(mc.grasp_settle_cfg.box_half_extents, device=env.device, dtype=pts_local.dtype)
    signed_dist, _ = box_nearest_and_signed_distance(pts_local, half)  # (N,K)
    err = torch.mean(torch.square(signed_dist), dim=-1)  # (N,)
    reward = torch.exp(-err / sigma**2)
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
# Undesired Contacts Rewards
# ================================================================================================


class ObjectGripForcePotential(RewardTermBase):
    """"You can always press harder" -- as POTENTIAL-BASED shaping, so it can never become the goal.

    Ng, Harada & Russell (1999): adding ``F(s,s') = gamma * Phi(s') - Phi(s)`` to the reward leaves
    the optimal policy EXACTLY unchanged for any ``Phi``. It only redistributes credit, so it
    expresses a hint ("more grip force is progress") without ever asserting how much grip is right.
    That property is why this is a separate mechanism and not another exponential: every other
    contact term here is added raw to the return and therefore DOES move the optimum, which is what
    made the "die before contact" attractor possible in the first place.

    Why contact FORCE and not penetration depth: with rigid contacts, pressing harder does not push
    the hand into the box -- PhysX penetration stays sub-millimetre whatever the policy does. The
    signed distances used by ``object_flat_contact_quality_exp`` saturate the instant the palm
    touches, so they are blind to grip effort. Net contact force on the hand is the only quantity
    that actually varies with how hard the robot squeezes.

    Phi = min(sum of contact-force magnitudes on the grasp anchors / ``force_ref``, 1), and 0 outside
    reference-contact frames (the hint only means anything while the clip says we are carrying).
    Saturation matters: past ``force_ref`` extra force stops counting as progress, so this cannot
    reward crushing. ``force_ref`` default 20 N is ~2x what the carry actually needs -- the box is
    0.811 kg (7.95 N) and the URDF gives mu = 0.9, so holding it between two palms needs a summed
    normal force of about mg/mu = 8.8 N.

    Two implementation points that the invariance theorem depends on:

      - ``Phi(terminal) = 0``. Over an episode the shaping telescopes to ``-Phi(s_0) + gamma^T
        Phi(s_T)``, so leaving ``Phi(s_T)`` free would pay a bonus for DYING in a high-force state.
        ``_check_termination()`` runs before ``_compute_reward()`` (base_task.py), so the true
        terminal flag is readable here. Timeouts are excluded: they are bootstrapped, not terminal.
      - ``gamma`` must match the algorithm's discount (0.99 for this WBT config, experiment.py) --
        a mismatched gamma silently breaks the invariance guarantee and turns this into an ordinary,
        optimum-moving reward.

    Unlike every other term in this file, the WEIGHT of this one cannot change what the policy
    converges to -- only how fast it gets there. Tuning it up is safe.
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        p = cfg.params
        self.gamma = float(p.get("gamma", 0.99))
        self.force_ref = float(p.get("force_ref", 20.0))
        self._body_names = p.get("body_names", None)
        self._body_indexes: torch.Tensor | None = None
        n, dev = env.num_envs, env.device
        self._phi_prev = torch.zeros(n, device=dev)
        # First step after a reset emits 0 and just seeds Phi_prev: Phi(s_0) is not observable here
        # (reset() runs before the state write). Starting the telescope one step late is still a
        # valid potential, and avoids the spurious spike that seeding with a stale Phi would give.
        self._needs_prime = torch.ones(n, dtype=torch.bool, device=dev)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._needs_prime[:] = True
            self._phi_prev[:] = 0.0
        elif env_ids.numel() > 0:
            self._needs_prime[env_ids] = True
            self._phi_prev[env_ids] = 0.0

    def _potential(self, env: WholeBodyTrackingManager) -> torch.Tensor:
        mc = _get_motion_command_and_assert_type(env)
        if self._body_indexes is None:
            names = self._body_names or mc.grasp_settle_cfg.anchor_body_names
            body_list = env.simulator.body_names  # type: ignore[attr-defined]
            self._body_indexes = torch.tensor(
                [body_list.index(bn) for bn in names], dtype=torch.long, device=env.device
            )

        # Prefer the sensor FILTERED on the box: it reports the robot<->box force alone, whereas
        # net_forces_w sums ground + table + box + self-collision into one vector. Phi built on the
        # unfiltered signal would rise when a foot hits the ground -- rewarding "progress" that has
        # nothing to do with gripping. Falls back to the unfiltered anchors when no object sensor
        # exists (non-object scenes), where the distinction is moot.
        obj_names = getattr(env.simulator, "_object_contact_body_names", [])  # type: ignore[attr-defined]
        if obj_names:
            mag = torch.norm(env.simulator.contact_forces_object, dim=-1).sum(dim=-1)  # type: ignore[attr-defined]
        else:
            # (num_envs, history, num_bodies, 3): max over the contact history smooths the
            # single-step solver dropouts, matching what UndesiredContacts does with this buffer.
            forces = env.simulator.contact_forces_history[:, :, self._body_indexes]  # type: ignore[attr-defined]
            mag = torch.norm(forces, dim=-1).max(dim=1).values.sum(dim=-1)  # (num_envs,)
        phi = (mag / self.force_ref).clamp(0.0, 1.0)

        if mc._anchor_body_indexes is None:
            return torch.zeros_like(phi)
        _, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
        return torch.where(ref_contact, phi, torch.zeros_like(phi))

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        phi = self._potential(env)

        # Phi(s') := 0 on a real termination so the telescoped total cannot reward dying well.
        # Timeouts stay bootstrapped, hence the ~time_out_buf.
        terminal = (env.reset_buf > 0) & ~env.time_out_buf
        phi_eff = torch.where(terminal, torch.zeros_like(phi), phi)

        shaping = self.gamma * phi_eff - self._phi_prev
        shaping = torch.where(self._needs_prime, torch.zeros_like(shaping), shaping)

        self._phi_prev = phi
        self._needs_prime = torch.zeros_like(self._needs_prime)
        return shaping


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
