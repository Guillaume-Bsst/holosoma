"""Whole body tracking observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from holosoma.config_types.observation import ObsTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.utils import resolve_callable
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
    """Position de la table (objet statique) dans le repère torse -> le robot SAIT où elle est."""
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.support_pos_w,
        motion_command.support_quat_w,
    )
    return pos.view(env.num_envs, -1)


def support_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Orientation de la table dans le repère torse (2 premières colonnes de la matrice)."""
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


class PerceptionNoisyPosition(ObservationTermBase):
    """Wrap a position observation with the CORRELATED errors of a real pose estimator.

    The per-step uniform noise the observation manager already applies (``ObsTermCfg.noise``) models
    white measurement jitter, and only that. Two reasons it isn't enough for the box pose:

      - It is exactly what a policy can average away -- and ``actor_obs`` now stacks 3 frames, which
        makes averaging easier still. The errors that actually survive averaging are the correlated
        ones, and none of them are modelled.
      - Its magnitude is already far past the measured jitter. The FoundationPose finite-difference
        velocity jitter measured on these clips is ~0.13 m/s p95, i.e. 0.13 * 0.02 s = ~2.6 mm of
        per-frame position jitter at the 50 Hz control rate (200 Hz sim / decimation 4) -- against
        the +-2 cm of uniform noise on ``obj_pos_b``. Widening the white noise further would only
        add signal the policy learns to ignore.

    So this term adds what a camera-based pipeline really does wrong, in the ACTOR group only (the
    critic keeps the clean privileged pose -- observation-term instances are per group):

      - ``bias_range``: a per-EPISODE, per-env, per-axis constant offset (m), resampled on reset.
        Camera extrinsics calibration and the mesh-origin convention of the pose estimator. This is
        the dominant real error and the one white noise cannot emulate: it does not average out over
        frames, so the policy has to stay robust to a box that is systematically a couple of cm off
        for the whole clip rather than trusting the mean.
      - ``latency_step_range``: a per-EPISODE integer delay (control steps) applied via a ring
        buffer. FoundationPose runs well below the 50 Hz control rate, so the pose the policy acts
        on is stale and zero-order held; 0-3 steps is 0-60 ms.
      - ``dropout_prob`` / ``dropout_max_steps``: per-step probability of FREEZING the reported
        position for a few steps (the last good value is held). Models tracking loss under
        occlusion -- which is precisely what happens when the hands close on the box. Off by
        default: it changes the task, not just the sensor, so it deserves its own A/B.

    Positions only. An additive bias on the 6D rotation representation used by ``obj_ori_b`` would
    not be a rotation and would break the orthonormality the policy reads, so orientation keeps the
    plain white noise.

    ``source`` is the import path of the underlying position term to wrap, so the same class serves
    ``obj_pos_b`` and ``support_pos_b`` (the table is perceived by the same pipeline, and its error
    is a pure calibration bias since it never moves).

    The whole term is a pass-through when the owning group has ``enable_noise=False``, so eval/play
    runs get the clean pose from the same switch that already silences the white noise. ``enabled``
    forces it either way.
    """

    def __init__(self, cfg: ObsTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        p = cfg.params
        self._source = resolve_callable(p.get("source", ""), context="observation term")
        self.bias_range = float(p.get("bias_range", 0.03))
        lat = p.get("latency_step_range", (0, 3))
        self.latency_min, self.latency_max = int(lat[0]), int(lat[1])
        self.dropout_prob = float(p.get("dropout_prob", 0.0))
        self.dropout_max_steps = int(p.get("dropout_max_steps", 5))
        # None = follow the owning group's `enable_noise`, so the one conventional switch turns off
        # ALL observation corruption at eval/play time (the manager's own `_apply_noise` only ever
        # gated the white noise, and a term cannot see its group). Resolved lazily on first call --
        # the manager isn't attached to the env yet at construction. True/False forces it.
        self._enabled_override = p.get("enabled", None)
        self._enabled: bool | None = None if self._enabled_override is None else bool(self._enabled_override)

        n, dev = env.num_envs, env.device
        self._bias = torch.zeros(n, 3, device=dev)
        self._latency = torch.zeros(n, dtype=torch.long, device=dev)
        # Ring buffer of past (biased) readings, depth latency_max + 1 so index 0 == current step.
        self._ring = torch.zeros(self.latency_max + 1, n, 3, device=dev)
        self._ring_head = 0
        self._freeze_left = torch.zeros(n, dtype=torch.long, device=dev)
        self._held = torch.zeros(n, 3, device=dev)
        # Envs whose buffers must be re-primed from the CURRENT reading on the next call. Priming
        # lazily (rather than inside reset()) keeps this correct whichever order the task calls
        # observation_manager.reset() and the state write that follows a termination.
        self._needs_prime = torch.ones(n, dtype=torch.bool, device=dev)
        self._env_arange = torch.arange(n, device=dev)
        self.reset(None)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        n, dev = self.env.num_envs, self.env.device
        idx = torch.arange(n, device=dev) if env_ids is None else env_ids.to(dev)
        if idx.numel() == 0:
            return
        self._bias[idx] = (torch.rand(idx.numel(), 3, device=dev) * 2.0 - 1.0) * self.bias_range
        if self.latency_max > self.latency_min:
            self._latency[idx] = torch.randint(
                self.latency_min, self.latency_max + 1, (idx.numel(),), device=dev
            )
        else:
            self._latency[idx] = self.latency_min
        self._freeze_left[idx] = 0
        self._needs_prime[idx] = True

    def _resolve_enabled(self, env: WholeBodyTrackingManager) -> bool:
        """Follow the owning group's ``enable_noise``, found by identity on our own cfg object."""
        if self._enabled is not None:
            return self._enabled
        self._enabled = True
        manager = getattr(env, "observation_manager", None)
        if manager is not None:
            for group_cfg in manager.cfg.groups.values():
                if any(t is self.cfg for t in group_cfg.terms.values()):
                    self._enabled = bool(group_cfg.enable_noise)
                    break
        return self._enabled

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        if not self._resolve_enabled(env):
            return self._source(env)

        pos = self._source(env) + self._bias

        # Newly reset envs: fill the whole ring with the current reading so the delayed sample is a
        # real pose from this episode rather than a stale one from the previous clip.
        if bool(self._needs_prime.any()):
            prime = self._needs_prime.clone()  # clone: we write into _needs_prime while indexing by it
            self._ring[:, prime] = pos[prime].unsqueeze(0)
            self._held[prime] = pos[prime]
            self._needs_prime[prime] = False

        self._ring_head = (self._ring_head + 1) % self._ring.shape[0]
        self._ring[self._ring_head] = pos
        # Read `latency` steps back, wrapping.
        read_idx = (self._ring_head - self._latency) % self._ring.shape[0]
        out = self._ring[read_idx, self._env_arange]

        if self.dropout_prob > 0.0:
            # Start a new freeze only when not already frozen, so dropout_prob is the rate of
            # dropout EVENTS and the expected outage length stays dropout_max_steps/2.
            start = (torch.rand(env.num_envs, device=pos.device) < self.dropout_prob) & (self._freeze_left == 0)
            if bool(start.any()):
                self._freeze_left[start] = torch.randint(
                    1, self.dropout_max_steps + 1, (int(start.sum()),), device=pos.device
                )
                self._held[start] = out[start]
            frozen = self._freeze_left > 0
            out = torch.where(frozen.unsqueeze(-1), self._held, out)
            self._freeze_left = (self._freeze_left - 1).clamp(min=0)

        return out


# ================================================================================================
# Critic-only privileged observations
# ================================================================================================
#
# The critic is never deployed -- it exists to estimate the value during training -- so anything
# that reduces the variance of that estimate is free with respect to the real robot. This is the
# same asymmetry that already gives it base_lin_vel / obj_lin_vel_b / robot_body_pos_b.
#
# NONE of these may be added to the actor group: they are either unavailable on hardware (measured
# contact forces) or would let the policy key on clip position instead of state (phase).


def motion_phase(env: WholeBodyTrackingManager) -> torch.Tensor:
    """(num_envs, 2): normalised position in the clip, and normalised frames REMAINING.

    ``motion_command`` gives the reference joint pose at the current frame and nothing else, so
    neither network is told where in the clip it is. For the actor that is a deliberate choice (a
    phase-conditioned policy memorises a trajectory instead of tracking it, and recovers badly once
    it drifts). For the CRITIC it is a straight loss: episodes start at a uniformly random phase
    (RSI) and end at the clip's end, so the achievable return depends directly on how much clip is
    left -- two states with an identical robot configuration and different phase have genuinely
    different values, and the critic currently has to infer the phase by recognising the reference
    pose, which says nothing about the remaining horizon.

    Both components are normalised per motion, so multi-clip runs stay comparable.
    """
    mc = _get_motion_command_and_assert_type(env)
    start = mc.motion.motion_start_idx[mc.motion_ids]
    end = mc.motion.motion_end_idx[mc.motion_ids]
    length = (end - start).clamp_min(1).float()
    elapsed = (mc.time_steps - start).float()
    phase = elapsed / length
    remaining = 1.0 - phase
    return torch.stack([phase, remaining], dim=-1)


def ref_obj_contact_lr(env: WholeBodyTrackingManager) -> torch.Tensor:
    """(num_envs, 2) float: does the REFERENCE hold the box with [left, right] at this frame.

    Tells the critic whether the contact reward terms (3.0 of the 11.0 positive budget) are even
    reachable here -- the phase-dependent ceiling that reward/achievable measures. Without it the
    critic must predict a return whose maximum swings with the frame, from inputs that do not
    contain the swing.
    """
    mc = _get_motion_command_and_assert_type(env)
    if not mc.motion.has_object:
        return torch.zeros(env.num_envs, 2, device=env.device)
    return mc.ref_hand_contact_lr().float()


def ref_foot_contact_lr(env: WholeBodyTrackingManager) -> torch.Tensor:
    """(num_envs, 2) float: reference stance flags [left, right] (0 without the stage-05 sidecar)."""
    mc = _get_motion_command_and_assert_type(env)
    if not getattr(mc.motion, "has_dyn_contact", False):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return mc.dyn_foot_contact_lr.float()


def ref_grip_force_lr(env: WholeBodyTrackingManager) -> torch.Tensor:
    """(num_envs, 2): reference squeeze force per hand in N (0 without the sidecar).

    Scaled down at the config level -- these reach ~190 N while every other observation is O(1).
    """
    mc = _get_motion_command_and_assert_type(env)
    if not getattr(mc.motion, "has_dyn_grip", False):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return mc.dyn_grip_force_lr


def measured_contact_forces(
    env: WholeBodyTrackingManager,
    body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ),
) -> torch.Tensor:
    """(num_envs, len(body_names)): magnitude of the net contact force on each named body, in N.

    What the robot is ACTUALLY touching, as opposed to what the reference says it should be. This is
    the state variable the contact reward terms are computed from, so giving it to the critic closes
    the loop: it can see the realised contact rather than inferring it from poses. Not available on
    hardware (no force sensing on the G1 wrists), hence critic-only.
    """
    idx = torch.tensor(
        [env.simulator.body_names.index(n) for n in body_names], dtype=torch.long, device=env.device
    )
    return torch.norm(env.simulator.contact_forces[:, idx], dim=-1)


def obj_ang_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """(num_envs, 3): box angular velocity in the robot reference frame.

    The critic already gets ``obj_lin_vel_b``; omitting the angular half was an asymmetry, and box
    tumbling is exactly what object_flat_contact_quality_exp is meant to prevent.
    """
    mc = _get_motion_command_and_assert_type(env)
    return quat_rotate_inverse(mc.robot_ref_quat_w, mc.simulator_object_ang_vel_w, w_last=True).view(
        env.num_envs, -1
    )
