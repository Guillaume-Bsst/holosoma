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
    return torch.where(ref_contact, reward, torch.ones_like(reward))


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
    return torch.where(ref_contact, reward, torch.ones_like(reward))


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
