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


def object_global_ref_lin_vel_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    """Track the object's LINEAR velocity against the reference clip.

    The global pose terms above constrain where the object is, not how it is moving to get there.
    Two trajectories that hit the same waypoints can differ by a jerked carry versus a smooth one,
    and only the second survives contact: velocity error is what the box's momentum does to the
    grasp. This is the object-side counterpart of ``motion_global_body_lin_vel``.

    Not part of HDMI, whose reward table carries only object pose and contact.
    """
    from holosoma.utils.object_interaction import velocity_tracking_reward

    mc = _get_motion_command_and_assert_type(env)
    return velocity_tracking_reward(mc.object_lin_vel_w, mc.simulator_object_lin_vel_w, sigma)


def object_global_ref_ang_vel_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    """Track the object's ANGULAR velocity against the reference clip.

    The rotational half of ``object_global_ref_lin_vel_error_exp``, and the one that bites: a box put
    into a spin escapes a fingerless palm, and the orientation-error term only sees it once the box
    has already turned.

    Returns 0 when the loaded motion carries no reference angular velocity (clips baked before the
    converter wrote ``object_ang_vel_w``) -- otherwise the term would train the policy to hold the
    box still against a zeros reference. 0 and not 1: an additive term paying a constant every step
    is a survival bonus, since in a discounted return its contribution grows with episode length.
    """
    from holosoma.utils.object_interaction import velocity_tracking_reward

    mc = _get_motion_command_and_assert_type(env)
    if not mc.motion.has_object_ang_vel:
        return torch.zeros(env.num_envs, device=env.device)
    return velocity_tracking_reward(mc.object_ang_vel_w, mc.simulator_object_ang_vel_w, sigma)


def object_contact_force_match_exp(
    env: WholeBodyTrackingManager,
    sigma_pos: float,
    sigma_force: float,
    force_threshold: float,
    max_force_bonus: float = 2.0,
) -> torch.Tensor:
    """HDMI's interaction reward: is the hand REALLY pressing on the box where the reference says?

    Port of the contact reward of HDMI (arXiv:2509.16757, Table I weight 5.0), gated by the binary
    reference contact indicator ``c_t``. See ``utils.object_interaction.hdmi_contact_reward`` for the
    formula and for the two deviations (capped force factor, 0 rather than 1 off-gate).

    What it adds over a kinematic contact term: geometry alone grades where the hand sits relative
    to the box surface, and a hand can satisfy that while resting a millimetre off the box, carrying
    nothing. This term reads the measured contact force, so it can tell a grip that bears load from
    a pose that merely looks like one.

    Returns 0 without resolved anchors (no object in the scene), the neutral value of an additive
    reward -- see ``utils.object_interaction.hdmi_contact_reward``.
    """
    from holosoma.utils.box_geometry import box_nearest_and_signed_distance
    from holosoma.utils.grasp_settle import gather_anchor
    from holosoma.utils.object_interaction import hdmi_contact_reward
    from holosoma.utils.rotations import quat_apply, quat_rotate_inverse

    mc = _get_motion_command_and_assert_type(env)
    if mc._anchor_body_indexes is None:
        return torch.zeros(env.num_envs, device=env.device)

    anchor_idx, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    a_pos, _ = gather_anchor(mc.robot_anchor_pos_w, mc.robot_anchor_quat_w, anchor_idx)

    # p_target, box-local, carried on the CURRENT sim box pose. Tracking the box's own pose is the
    # job of object_global_ref_*; this term grades the hand against the box that is actually there.
    box_pos, box_quat = mc.simulator_object_pos_w, mc.simulator_object_quat_w
    if mc.motion.has_gt_witness:
        witness_local = mc.motion.object_ref_witness_local[mc.time_steps]  # (N, 3)
    else:
        # No reference witness on this clip: fall back to the nearest point of the box surface, so
        # the distance becomes the hand-to-surface gap. Less faithful to HDMI, whose p_target is the
        # demo's own contact point, but the FORCE factor -- what this term adds over the purely
        # kinematic contact terms -- needs no witness at all, and requiring one would make the whole
        # term inert on every clip that does not carry one.
        p_local = quat_rotate_inverse(box_quat, a_pos - box_pos, w_last=True)
        half_extents = torch.tensor(mc.grasp_settle_cfg.box_half_extents, device=env.device, dtype=p_local.dtype)
        _, witness_local = box_nearest_and_signed_distance(p_local, half_extents)
    p_target = box_pos + quat_apply(box_quat, witness_local, w_last=True)
    distance = torch.norm(a_pos - p_target, dim=-1)

    # ||F|| on the contact hand, maxed over the sensor history like UndesiredContacts does: contact
    # forces are intermittent across substeps, so an instantaneous read makes the bonus flicker on a
    # grip that is in fact steady.
    forces = env.simulator.contact_forces_history[:, :, mc._anchor_body_indexes]  # (N, H, A, 3)
    force_per_anchor = torch.max(torch.norm(forces, dim=-1), dim=1)[0]  # (N, A)
    env_ids = torch.arange(env.num_envs, device=env.device)
    force = force_per_anchor[env_ids, anchor_idx.clamp(0, force_per_anchor.shape[1] - 1)]  # (N,)

    # A supplied contact schedule is the gate; with ramp frames it fades in, which a bare boolean
    # cannot express. Without one, HDMI's plain binary c_t.
    if mc.motion.has_contact_schedule:
        gate = mc.schedule_contact_weight(mc.time_steps)[env_ids, anchor_idx.clamp(0, 1)]
    else:
        gate = ref_contact

    return hdmi_contact_reward(
        distance,
        force,
        gate,
        sigma_pos=sigma_pos,
        sigma_force=sigma_force,
        force_threshold=force_threshold,
        max_force_bonus=max_force_bonus,
    )


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_names = undesired_contacts_body_names
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        # Per-body breakdown. The term only ever reported a COUNT, so a penalty of 5.4 per episode
        # said nothing about which bodies earn it -- and the answer decides whether the regex is
        # penalising a body that legitimately has to touch the box while carrying it (a forearm
        # against a 32 cm cube) rather than a genuine collision.
        log = getattr(self.env, "log_dict", None)
        if log is not None:
            frac = is_contact.float().mean(dim=0).detach().cpu()
            for name, value in zip(self.undesired_contacts_body_names, frac):
                log[f"undesired_contacts/{name}"] = value
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
