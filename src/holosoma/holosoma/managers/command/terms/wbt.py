from __future__ import annotations

import math
import re
from typing import Any, List

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            self.fps = data["fps"]

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            # The first 7 joints_pos are [xyz, wxyz] of the pelvis, omit them from the joint_pos
            # The first 6 joints_vel are [vel_xyz, vel_wxyz] of the pelvis, omit them from the joint_vel
            # We use joint_pos[:, :3] (free joint xyz) for root position at reset: body_pos_w stores the
            # pelvis COM position (xpos), which is offset from the free joint origin by body_ipos (-7.6cm z).
            # Using the COM as the free joint position would place the robot 7.6cm too low at reset.
            self._root_free_joint_pos = torch.tensor(data["joint_pos"][:, :3], dtype=torch.float32, device=device)
            self._joint_pos = torch.tensor(data["joint_pos"][:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(data["joint_vel"][:, 6:], dtype=torch.float32, device=device)
            assert len(joint_names) == self._joint_pos.shape[1], "Joint names in motion data does not match"

            self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            assert len(body_names) == self._body_pos_w.shape[1], "Body names in motion data does not match"

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

            # add object pos and quat
            self.has_object = "object_pos_w" in data
            if self.has_object:
                # NOTE: wxyz after loading from npz
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("root_pos", "_root_free_joint_pos"),
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        bin_size_s: float = 1.0,
        kernel_size: int = 3,
        decay_lambda: float = 0.001,
        kernel_lambda: float = 0.8,
    ):
        # TODO: think better about the decay_lambda, will 0.001 be too small?
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        # size of the bin in seconds
        self.bin_size_s = bin_size_s
        # size of the kernel for smoothing the sampling probabilities
        self.kernel_size = kernel_size
        self.kernel_lambda = kernel_lambda
        # exponential decay when updating the failure counts over training steps.

        self.decay_lambda = decay_lambda

        # number of bins in the motion
        self.num_bins = math.ceil((self.motion_time_step_total / self.env_fps) / self.bin_size_s)

        # initialize exponential 1d decay kernel, used for smoothing the failure counts over time.
        assert self.kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel = torch.tensor(
            [self.kernel_lambda ** abs(i) for i in range((-self.kernel_size + 1) // 2, (self.kernel_size + 1) // 2)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor):
        """Update the current bin failed count with terminated time steps."""
        failed_bin = torch.floor(failed_at_time_step / self.motion_time_step_total * self.num_bins).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        self.current_bin_failed_count[:] = torch.bincount(failed_bin, minlength=self.num_bins)

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.decay_lambda * self.current_bin_failed_count) + (
            1 - self.decay_lambda
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + 1e-6
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities += 0.01
        return sampling_probabilities / sampling_probabilities.sum()

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / np.log(self.num_bins)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


def _point_to_segment_distance(
    points: torch.Tensor,  # (..., 3)
    seg_start: torch.Tensor,  # (..., 3)
    seg_end: torch.Tensor,  # (..., 3)
) -> torch.Tensor:
    """Vectorized minimum distance from points to line segments. Returns (...,)."""
    seg_vec = seg_end - seg_start  # (..., 3)
    pt_vec = points - seg_start  # (..., 3)
    seg_len_sq = (seg_vec * seg_vec).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    t = ((pt_vec * seg_vec).sum(dim=-1, keepdim=True) / seg_len_sq).clamp(0.0, 1.0)
    closest = seg_start + t * seg_vec
    return torch.norm(points - closest, dim=-1)


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        self.motion: MotionLoader = MotionLoader(
            self.motion_cfg.motion_file,
            robot_body_names_alias,
            robot_joint_names,
            device=self.device,
        )

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )

        # Simulator body IDs for _rigid_body_pos/_rot/_vel/_ang_vel access.
        # These differ from tracked_body_indexes in MuJoCo where _rigid_body_pos is
        # MuJoCo-indexed (worldbody at 0), whereas tracked_body_indexes is 0-based
        # from body_names (worldbody excluded). find_rigid_body_indice() returns the
        # correct index for each simulator (MuJoCo body ID or Isaac local index).
        self._robot_ref_body_id: int = self._env.simulator.find_rigid_body_indice(  # type: ignore[attr-defined]
            self.motion_cfg.body_name_ref[0]
        )
        self._robot_tracked_body_ids: torch.Tensor = torch.tensor(
            [self._env.simulator.find_rigid_body_indice(name) for name in self.motion_cfg.body_names_to_track],  # type: ignore[attr-defined]
            dtype=torch.long,
            device=self.device,
        )

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            # cache the object_index_in_simulator
            # Use the name from scene config if available, fall back to "object" for legacy Isaac path
            sim_rigid_objects = self._env.simulator.simulator_config.scene.rigid_objects
            self.object_name = sim_rigid_objects[0].name if sim_rigid_objects else "object"
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() in (
                SimulatorType.ISAACSIM,
                SimulatorType.MUJOCO,
            ), "Object interaction is only supported in IsaacSim and MuJoCo"

            # Auto-detect object radius for collision checking
            self.object_collision_radius = self._autodetect_object_radius()

        # 4. get the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total, self.device, int(1 / (self._env.dt))
            )

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        # 6. Noise precomputation: indices for arm DOFs, hand bodies, and torso capsules.
        # Arm DOF indices — joints whose noise is suppressed when the robot grasps the object.
        if self.init_pose_cfg.arm_joint_pattern:
            arm_idx = [
                i for i, name in enumerate(robot_joint_names)
                if re.match(self.init_pose_cfg.arm_joint_pattern, name)
            ]
            self._arm_dof_indices: torch.Tensor | None = (
                torch.tensor(arm_idx, dtype=torch.long, device=self.device) if arm_idx else None
            )
        else:
            self._arm_dof_indices = None

        # Hand body indices — into the motion.body_pos_w second dimension (robot-body order).
        if self.init_pose_cfg.hand_body_names and self.motion.has_object:
            hand_idx = [
                i for i, name in enumerate(robot_body_names_alias)
                if name in self.init_pose_cfg.hand_body_names
            ]
            self._hand_body_robot_indices: torch.Tensor | None = (
                torch.tensor(hand_idx, dtype=torch.long, device=self.device) if hand_idx else None
            )
        else:
            self._hand_body_robot_indices = None

        # Torso capsule body-pair indices for geometric collision rejection.
        self._torso_capsule_pairs: list[tuple[int, int]] = []
        self._torso_capsule_radii_list: list[float] = []
        for cap_i, pair in enumerate(self.init_pose_cfg.torso_capsule_body_pairs):
            if len(pair) != 2:
                continue
            name_a, name_b = pair[0], pair[1]
            if name_a in robot_body_names_alias and name_b in robot_body_names_alias:
                self._torso_capsule_pairs.append(
                    (robot_body_names_alias.index(name_a), robot_body_names_alias.index(name_b))
                )
                cap_r = (
                    self.init_pose_cfg.torso_capsule_radii[cap_i]
                    if cap_i < len(self.init_pose_cfg.torso_capsule_radii)
                    else 0.15
                )
                self._torso_capsule_radii_list.append(cap_r)

        self.init_buffers()

        # 7. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        # 0. Sample the time steps
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            phase = self.adaptive_timesteps_sampler.sample(env_ids.numel())
        else:
            phase = torch.rand(env_ids.numel(), device=self.device)

        if self._env.is_evaluating:
            phase = torch.zeros_like(phase)

        self.time_steps[env_ids] = (phase * (self.motion.time_step_total - 1)).long()

        # Handle start_at_timestep_zero_prob
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = 0
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, torch.zeros_like(subset), subset)
            self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        already_last_timestep_mask = self.time_steps[env_ids] == self.motion.time_step_total - 1
        self.time_steps[env_ids] = torch.where(
            already_last_timestep_mask, self.motion.time_step_total - 2, self.time_steps[env_ids]
        )

        # 1. Get the root/body poses from the motion data
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 Prepare noise scales
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(self.init_pose_cfg.root_pos, device=self.device)
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(self.init_pose_cfg.root_rot, device=self.device)
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(self.init_pose_cfg.root_lin_vel, device=self.device)
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(self.init_pose_cfg.root_ang_vel, device=self.device)
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Smooth proximity masks (both in [0, 1]; 0 = suppress noise, 1 = full noise).
        # grasp_mask    — hand-to-object distance: suppresses arm DOF noise when grasping.
        # body_prox_mask — min distance from ANY robot body to object: suppresses ALL DOF noise
        #                  when any body part (e.g. shins/knees) is near the object, preventing
        #                  leg joints from swinging into the box at reset (e.g. floor-pickup pose).
        grasp_mask = self._compute_grasp_mask(env_ids)  # (num_envs,) in [0, 1]
        body_prox_mask = self._compute_body_proximity_mask(env_ids)  # (num_envs,) in [0, 1]

        # 2.3 Root state noise (always applied — root state does not affect grasp geometry directly)
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 2.4 DOF noise with two-level masking:
        #   • body_prox_mask applied to ALL DOFs — prevents any joint from moving body parts
        #     into the object when the robot is in close proximity (e.g. floor-pickup pose).
        #   • grasp_mask applied additionally to arm DOFs — finer suppression specifically
        #     for arm/wrist joints when the hands are grasping the object.
        dof_noise = (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        dof_noise = dof_noise * body_prox_mask.unsqueeze(1)
        if self._arm_dof_indices is not None and self._arm_dof_indices.numel() > 0:
            dof_noise[:, self._arm_dof_indices] = (
                dof_noise[:, self._arm_dof_indices] * grasp_mask.unsqueeze(1)
            )
        target_dof_pos = dof_pos + dof_noise  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])
        target_dof_vel = dof_vel  # no noise on velocities

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]

            # 4.1 Global anchoring: rotate the reference relative position by the root rotation
            # delta, then shift by the noisy root translation.  This guarantees that the grasp
            # geometry is exactly preserved after root-level noise, regardless of magnitude.
            rel_pos = obj_pos - root_pos  # (num_envs, 3)
            rotated_rel_pos = quat_apply(orientations_delta, rel_pos, w_last=True)  # (num_envs, 3)
            anchored_obj_pos = target_root_pos + rotated_rel_pos  # (num_envs, 3)

            # 4.2 Local object noise with batched rejection sampling.
            # Scaled by both grasp_mask (hand proximity) and body_prox_mask (any-body proximity).
            # body_prox_mask suppresses object noise when any robot body is near the box, so that
            # noise cannot push the box into the robot's geometry from a valid reference position.
            obj_pos_noise = (
                torch.tensor(self.init_pose_cfg.object_pos, device=self.device)
                * self.init_pose_cfg.overall_noise_scale
                * grasp_mask.unsqueeze(1)      # zero when hands are grasping
                * body_prox_mask.unsqueeze(1)  # zero when any body is near the object
            )
            K = self.init_pose_cfg.object_noise_num_proposals
            num_reset = len(env_ids)
            # Generate K-1 random proposals in the robot local frame.
            # Proposal 0 is always zero noise (= anchored_obj_pos) so the no-noise position is
            # explicitly tested by the collision filter rather than used blindly as fallback.
            local_proposals = (
                (torch.rand((num_reset, K, 3), device=self.device) - 0.5) * 2
                * obj_pos_noise.unsqueeze(1)  # broadcast: (num_envs, K, 3)
            )
            local_proposals[:, 0, :] = 0.0  # slot 0 = zero noise = anchored_obj_pos
            # Rotate proposals into world frame using the noisy root orientation.
            target_root_rot_exp = target_root_rot.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4)
            world_proposals = quat_apply(
                target_root_rot_exp, local_proposals.reshape(-1, 3), w_last=True
            ).reshape(num_reset, K, 3)
            candidate_positions = anchored_obj_pos.unsqueeze(1) + world_proposals  # (num_envs, K, 3)

            # 4.3 Geometric collision filter: reject proposals that penetrate torso capsules.
            capsules = self._get_torso_capsules_world(env_ids, target_root_pos, root_rot, target_root_rot)
            margins = self._compute_proposals_margin(candidate_positions, capsules)  # (N, K)

            # 4.4 Select proposal:
            # Priority 1: First valid proposal (margin > 0)
            # Priority 2: Least violating proposal (max margin) to minimize penetration
            valid_mask = margins > 0.0
            has_valid = valid_mask.any(dim=1)  # (num_envs,)

            first_valid_idx = torch.argmax(valid_mask.float(), dim=1)  # (num_envs,)
            best_fallback_idx = torch.argmax(margins, dim=1)  # (num_envs,)

            selected_idx = torch.where(has_valid, first_valid_idx, best_fallback_idx)
            selected_world_noise = world_proposals[
                torch.arange(num_reset, device=self.device), selected_idx
            ]  # (num_envs, 3)
            target_obj_pos = anchored_obj_pos + selected_world_noise

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 13)
            # 4.5 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at timestep 0, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == 0
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        self.time_steps += advance_mask.long()

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.update_bin_failed_count()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps][:, self.tracked_body_indexes]
            + self._env.simulator.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.ref_body_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.ref_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion._root_free_joint_pos[self.time_steps] + self._env.simulator.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, 0]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self._robot_tracked_body_ids, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self._robot_tracked_body_ids, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self._robot_tracked_body_ids, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self._robot_tracked_body_ids, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self._robot_ref_body_id, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self._robot_ref_body_id, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self._robot_ref_body_id, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self._robot_ref_body_id, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Applies env origins, but ideally we should rely on the simulator
        return self.motion.object_pos_w[self.time_steps] + self._env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.object_lin_vel_w[self.time_steps]

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.init_buffers()

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]

    #########################################################################################
    ## Internal helpers
    #########################################################################################

    # ------------------------------------------------------------------ #
    # Robust noise helpers
    # ------------------------------------------------------------------ #

    def _compute_grasp_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Smooth proximity mask in [0, 1].

        Returns 0 when the closest hand is within ``grasp_mask_min_dist`` of the
        object (suppress arm/object noise), and 1 when it is beyond
        ``grasp_mask_max_dist`` (full noise).  Linearly interpolated between.

        If hand bodies or object are not configured, returns all-ones (no masking).
        """
        if self._hand_body_robot_indices is None or not self.motion.has_object:
            return torch.ones(len(env_ids), device=self.device)

        ts = self.time_steps[env_ids]
        # motion.body_pos_w (MotionLoader property) shape: (T, num_robot_bodies, 3), no env_origins.
        hand_pos = self.motion.body_pos_w[ts][:, self._hand_body_robot_indices]  # (N, H, 3)
        obj_pos = self.motion.object_pos_w[ts]  # (N, 3)

        min_dist = torch.norm(hand_pos - obj_pos.unsqueeze(1), dim=-1).min(dim=1).values  # (N,)
        d_min = self.init_pose_cfg.grasp_mask_min_dist
        d_max = self.init_pose_cfg.grasp_mask_max_dist
        return torch.clamp((min_dist - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)

    def _compute_body_proximity_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Smooth proximity mask based on the closest robot body to the object.

        Returns 0 when ANY robot body is within ``grasp_mask_min_dist`` of the object
        (suppress all DOF noise), and 1 when all bodies are beyond ``grasp_mask_max_dist``
        (full noise).  Linearly interpolated between.

        This is broader than ``_compute_grasp_mask``: it catches cases where non-hand body
        parts (e.g. shins/knees in a floor-pickup pose) are near the object, preventing leg
        joints from swinging into the box at reset.

        Returns all-ones when no object is present.
        """
        if not self.motion.has_object:
            return torch.ones(len(env_ids), device=self.device)

        ts = self.time_steps[env_ids]
        # shape: (N, num_robot_bodies, 3) — robot bodies only, no env_origins, no largebox.
        all_body_pos = self.motion.body_pos_w[ts]
        obj_pos = self.motion.object_pos_w[ts]  # (N, 3)

        min_dist = torch.norm(all_body_pos - obj_pos.unsqueeze(1), dim=-1).min(dim=1).values  # (N,)
        d_min = self.init_pose_cfg.grasp_mask_min_dist
        d_max = self.init_pose_cfg.grasp_mask_max_dist
        return torch.clamp((min_dist - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)

    def _get_torso_capsules_world(
        self,
        env_ids: torch.Tensor,
        target_root_pos: torch.Tensor,  # (N, 3)
        root_rot_ref: torch.Tensor,  # (N, 4) xyzw — reference root orientation from motion data
        target_root_rot: torch.Tensor,  # (N, 4) xyzw — noisy root orientation
    ) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Build torso capsule endpoints in world frame adapted to the noisy root pose.

        Each capsule is specified as a pair of body names in root-local frame.  We:
          1. Read the reference body positions from motion data (no env_origins, same frame as root).
          2. Express them relative to the reference root position and orientation.
          3. Re-apply the *noisy* root pose so the capsules match the placed robot.

        Returns a list of (start_world, end_world, radius) tuples — empty when no capsules are
        configured.
        """
        if not self._torso_capsule_pairs:
            return []

        ts = self.time_steps[env_ids]
        # All body positions in the motion frame (no env_origins) — shape (N, num_robot_bodies, 3).
        body_pos_motion = self.motion.body_pos_w[ts]
        # Reference root position in motion frame (no env_origins) — shape (N, 3).
        root_pos_motion = self.motion._root_free_joint_pos[ts]
        root_rot_ref_inv = quat_inverse(root_rot_ref, w_last=True)  # (N, 4)

        capsules: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        for (idx_a, idx_b), radius in zip(self._torso_capsule_pairs, self._torso_capsule_radii_list):
            # Body positions relative to the reference root (motion frame).
            rel_a = body_pos_motion[:, idx_a] - root_pos_motion  # (N, 3)
            rel_b = body_pos_motion[:, idx_b] - root_pos_motion  # (N, 3)
            # Rotate into root-local frame using the reference root orientation.
            local_a = quat_apply(root_rot_ref_inv, rel_a, w_last=True)  # (N, 3)
            local_b = quat_apply(root_rot_ref_inv, rel_b, w_last=True)  # (N, 3)
            # Transform to the noisy world frame.
            start_w = target_root_pos + quat_apply(target_root_rot, local_a, w_last=True)  # (N, 3)
            end_w = target_root_pos + quat_apply(target_root_rot, local_b, w_last=True)  # (N, 3)
            capsules.append((start_w, end_w, radius))

        return capsules

    def _compute_proposals_margin(
        self,
        candidate_positions: torch.Tensor,  # (N, K, 3)
        capsules: list[tuple[torch.Tensor, torch.Tensor, float]],
    ) -> torch.Tensor:
        """Return (N, K) margins. Positive = safe, Negative = penetration.

        If no capsules are defined, returns infinity (safe).
        """
        N, K, _ = candidate_positions.shape
        # Initialize with infinity (safe)
        min_margins = torch.full((N, K), float("inf"), device=self.device)

        if not capsules:
            return min_margins

        obj_r = getattr(self, "object_collision_radius", self.init_pose_cfg.object_collision_radius)
        for start_w, end_w, cap_r in capsules:
            seg_start = start_w.unsqueeze(1).expand(-1, K, -1)  # (N, K, 3)
            seg_end = end_w.unsqueeze(1).expand(-1, K, -1)  # (N, K, 3)
            dist = _point_to_segment_distance(candidate_positions, seg_start, seg_end)  # (N, K)
            # Margin for this capsule: distance - min_safe_dist
            # If dist < min_safe_dist, margin is negative (penetration)
            margin = dist - (cap_r + obj_r)
            # We care about the *worst* margin across all capsules for a given proposal
            min_margins = torch.minimum(min_margins, margin)

        return min_margins

    def _autodetect_object_radius(self) -> float:
        """Auto-detect object radius from simulator geometry."""
        # Start with config value as fallback
        radius = self.init_pose_cfg.object_collision_radius

        try:
            simulator = self._env.simulator
            sim_type = simulator.get_simulator_type()

            if sim_type == SimulatorType.ISAACSIM:
                from pxr import Usd, UsdGeom
                import omni.usd
                import numpy as np

                stage = omni.usd.get_context().get_stage()
                # Find the object prim path for env 0.
                # The object wrapper in IsaacLab is at simulator.scene.rigid_objects[self.object_name]
                obj_wrapper = simulator.scene.rigid_objects[self.object_name]
                # The cfg has the prim_path regex, e.g. "/World/envs/env_.*/Object"
                prim_path = obj_wrapper.cfg.prim_path.replace(".*", "0")

                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    # Compute bounding box using BBoxCache to include children and transforms
                    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                    bound = bbox_cache.ComputeWorldBound(prim)
                    range_ = bound.ComputeAlignedBox()
                    size = np.array(range_.GetSize())
                    # Use the bounding sphere radius (half diagonal)
                    detected_radius = float(np.linalg.norm(size) / 2.0)
                    logger.info(f"Auto-detected object radius (Isaac): {detected_radius:.4f} m")
                    return detected_radius

            elif sim_type == SimulatorType.MUJOCO:
                import mujoco
                model = simulator.backend.model

                # Find body ID
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.object_name)
                if body_id == -1:
                    # Fallback to generic name if specific name not found
                    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")

                if body_id != -1:
                    max_rbound = 0.0
                    for i in range(model.ngeom):
                        if model.geom_bodyid[i] == body_id:
                            max_rbound = max(max_rbound, model.geom_rbound[i])
                    if max_rbound > 0:
                        logger.info(f"Auto-detected object radius (MuJoCo): {max_rbound:.4f} m")
                        return float(max_rbound)

        except Exception as e:
            logger.warning(f"Could not auto-detect object radius: {e}")

        return radius

    # ------------------------------------------------------------------ #

    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = -1 if use_motion_end else 0

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        sim_type = simulator.get_simulator_type()

        # ---------------------------------------------------------------------
        # ISAAC SIM Fork
        # ---------------------------------------------------------------------
        if sim_type == SimulatorType.ISAACSIM:
            env_id = 0
            env_origin = simulator.scene.env_origins[env_id].to(self.device)

            root_backup = simulator.robot_root_states[env_id].clone()
            dof_pos_backup = simulator.dof_pos[env_id].clone()
            dof_vel_backup = simulator.dof_vel[env_id].clone()

            try:
                simulator.robot_root_states[env_id, :3] = root_pos + env_origin
                simulator.robot_root_states[env_id, 3:7] = root_quat
                simulator.robot_root_states[env_id, 7:10] = root_lin_vel
                simulator.robot_root_states[env_id, 10:13] = root_ang_vel
                simulator.dof_pos[env_id] = joint_pos
                simulator.dof_vel[env_id] = joint_vel

                simulator.set_actor_root_state_tensor_robots()
                simulator.set_dof_state_tensor_robots()
                simulator.write_state_updates()
                simulator.refresh_sim_tensors()

                body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
                body_quat = simulator._rigid_body_rot[env_id].clone()
                body_lin_vel = simulator._rigid_body_vel[env_id].clone()
                body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
            finally:
                simulator.robot_root_states[env_id] = root_backup
                simulator.dof_pos[env_id] = dof_pos_backup
                simulator.dof_vel[env_id] = dof_vel_backup
                simulator.set_actor_root_state_tensor_robots()
                simulator.set_dof_state_tensor_robots()
                simulator.write_state_updates()
                simulator.refresh_sim_tensors()

            return {
                "pos": body_pos,
                "quat": body_quat,
                "lin_vel": body_lin_vel,
                "ang_vel": body_ang_vel,
            }

        # ---------------------------------------------------------------------
        # MJWARP Fork
        # ---------------------------------------------------------------------
        elif sim_type == SimulatorType.MUJOCO:
            import mujoco
            import numpy as np

            mj_model = simulator.backend.model
            mj_data = mujoco.MjData(mj_model)

            if not hasattr(self, '_mj_body_ids'):
                body_names = simulator._body_list
                body_ids = []
                for name in body_names:
                    b_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{name}")
                    if b_id == -1:
                        b_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                    
                    if b_id == -1:
                        logger.warning(f"[MuJoCo WBT] Body '{name}' not found, falling back to World (ID 0).")
                        body_ids.append(0)
                    else:
                        body_ids.append(b_id)
                        
                self._mj_body_ids = np.array(body_ids, dtype=np.int32)
                logger.info(f"[MuJoCo WBT] Fixed body mapping initialized for {len(self._mj_body_ids)} bodies.")

            qpos_addr = getattr(simulator, 'robot_qpos_addr', 0)
            qvel_addr = getattr(simulator, 'robot_qvel_addr', 0)
            
            mj_data.qpos[qpos_addr : qpos_addr + 3] = root_pos.detach().cpu().numpy()
            mj_data.qpos[qpos_addr + 3 : qpos_addr + 7] = root_quat[[3, 0, 1, 2]].detach().cpu().numpy()
            
            joint_start = qpos_addr + 7
            mj_data.qpos[joint_start : joint_start + joint_pos.shape[0]] = joint_pos.detach().cpu().numpy()

            mj_data.qvel[qvel_addr : qvel_addr + 3] = root_lin_vel.detach().cpu().numpy()
            mj_data.qvel[qvel_addr + 3 : qvel_addr + 6] = root_ang_vel.detach().cpu().numpy()
            
            joint_vel_start = qvel_addr + 6
            mj_data.qvel[joint_vel_start : joint_vel_start + joint_vel.shape[0]] = joint_vel.detach().cpu().numpy()

            mujoco.mj_kinematics(mj_model, mj_data)
            mujoco.mj_comPos(mj_model, mj_data)
            mujoco.mj_comVel(mj_model, mj_data)

            xpos_all = mj_data.xpos[self._mj_body_ids]
            xquat_all = mj_data.xquat[self._mj_body_ids]

            body_pos_out = torch.from_numpy(xpos_all).to(self.device).float()
            
            body_quat_out = torch.from_numpy(xquat_all[:, [1, 2, 3, 0]]).to(self.device).float()

            num_bodies = len(self._mj_body_ids)
            obj_vels = np.zeros((num_bodies, 6), dtype=np.float64)
            
            for idx, b_id in enumerate(self._mj_body_ids):
                mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, b_id, obj_vels[idx], 0)
            
            body_ang_vel_out = torch.from_numpy(obj_vels[:, 0:3]).to(self.device).float()
            body_lin_vel_out = torch.from_numpy(obj_vels[:, 3:6]).to(self.device).float()

            return {
                "pos": body_pos_out,
                "quat": body_quat_out,
                "lin_vel": body_lin_vel_out,
                "ang_vel": body_ang_vel_out,
            }

        else:
            raise NotImplementedError(f"Default-pose interpolation is not supported for simulator type: {sim_type}")

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "root_pos": self.motion._root_free_joint_pos[idx].to(device=device, dtype=dtype),
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "root_pos": default_state["root_pos"].to(device=device, dtype=dtype).squeeze(0),
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "root_pos": _lerp(start["root_pos"], target["root_pos"], alphas_joint),
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
