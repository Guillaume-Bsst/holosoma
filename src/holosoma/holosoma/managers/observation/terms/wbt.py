"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def support_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Table (static object) position in the torso frame -> the robot KNOWS where it is."""
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.support_pos_w,
        motion_command.support_quat_w,
    )
    return pos.view(env.num_envs, -1)


def support_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Table orientation in the torso frame (first 2 columns of the matrix)."""
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.support_pos_w,
        motion_command.support_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)


def obj_lin_vel_b_rotated(env: WholeBodyTrackingManager) -> torch.Tensor:
    """(num_envs, 3): box linear velocity in the robot reference frame, rotation ONLY.

    Corrected counterpart of ``obj_lin_vel_b`` above, which passes a velocity to
    ``subtract_frame_transforms``. That helper builds ``R^T (t02 - t01)`` -- correct for a POSE, but
    for a velocity it subtracts the reference body's world position from a free vector, returning
    ``R^T (v_obj - p_torso)`` instead of ``R^T v_obj``. The parasitic ``-R^T p_torso`` is ~1.8 m/s
    against a carry velocity of ~0.3 m/s and drifts with the robot's position in the scene; a box at
    a dead stop reads as moving. Upstream states the rule explicitly in
    ``managers/observation/terms/objects.py`` (``relative_to_root=True`` for positions, ``False`` for
    both velocities).

    Kept as a separate term rather than fixed in place: ``obj_lin_vel_b`` is in the critic group of
    every existing ``w_object`` preset, so changing it would silently redefine the input of runs
    already trained against it. Opt in via the object-velocity presets.
    """
    mc = _get_motion_command_and_assert_type(env)
    return quat_rotate_inverse(mc.robot_ref_quat_w, mc.simulator_object_lin_vel_w, w_last=True).view(
        env.num_envs, -1
    )


def obj_ang_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """(num_envs, 3): box angular velocity in the robot reference frame.

    The critic already gets ``obj_lin_vel_b``; omitting the angular half was an asymmetry, and box
    tumbling is exactly what ``object_flat_contact_quality_exp`` is meant to prevent.

    (Ported from 1045923 on feat/dynamics-aware-training.)
    """
    mc = _get_motion_command_and_assert_type(env)
    return quat_rotate_inverse(mc.robot_ref_quat_w, mc.simulator_object_ang_vel_w, w_last=True).view(
        env.num_envs, -1
    )


def obj_contact_flag(env: WholeBodyTrackingManager, force_threshold: float = 1.0) -> torch.Tensor:
    """(num_envs, 3): ``[sim contact on anchor 0, sim contact on anchor 1, reference contact]``.

    The binary state variable behind ``object_contact_force_match_exp``: what the hands are ACTUALLY
    bearing (measured contact force over the threshold, per candidate anchor) next to what the
    reference prescribes. Giving the critic both closes the loop -- it can see the realised contact
    instead of inferring it from poses, and it can see the mismatch that the reward is paying on.

    Critic-only: the G1 has no force sensing at the wrists, so none of this exists on hardware.

    All zeros when the motion carries no object / no resolved anchors, so the term keeps a fixed
    width whatever the clip.
    """
    mc = _get_motion_command_and_assert_type(env)
    out = torch.zeros(env.num_envs, 3, device=env.device)
    if mc._anchor_body_indexes is None:
        return out

    forces = env.simulator.contact_forces_history[:, :, mc._anchor_body_indexes]  # (N, H, A, 3)
    per_anchor = torch.max(torch.norm(forces, dim=-1), dim=1)[0]  # (N, A)
    n_anchor = min(per_anchor.shape[1], 2)
    out[:, :n_anchor] = (per_anchor[:, :n_anchor] > force_threshold).float()

    _, ref_contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    out[:, 2] = ref_contact.float()
    return out
