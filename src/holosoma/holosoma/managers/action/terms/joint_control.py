"""Action terms for joint-level control."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from holosoma.managers.action.base import ActionTermBase
from holosoma.utils.rotations import quat_apply

if TYPE_CHECKING:
    from holosoma.config_types.action import ActionTermCfg, GripForceCfg


class JointPositionActionTerm(ActionTermBase):
    """Action term for joint position control with PD controller.

    This term processes raw actions as joint position targets and computes
    torques using a PD controller. Supports:
    - Action scaling
    - Action clipping
    - Action delay (if configured)
    - Torque randomization (if configured)
    - Torque clipping
    """

    def __init__(self, cfg: ActionTermCfg, env: Any):
        """Initialize joint position action term.

        Args:
            cfg: Configuration for this action term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        super().__init__(cfg, env)

        # Get action dimension from environment
        self._action_dim = env.num_dof

        # Initialize action buffers
        self._raw_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self._action_dim, device=env.device)
        self._actions_after_delay = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Initialize torque buffer
        self.torques = torch.zeros(env.num_envs, self._action_dim, device=env.device)

        # Sub-step torque history: [num_envs, decimation, num_dof], allocated in setup().
        self._substep_idx: int = 0

        # Cache previous DOF velocities for derivative control
        self._prev_dof_vel = torch.zeros(env.num_envs, env.num_dof, device=env.device)

        # Default actuator scaling (may be overridden by randomization terms)
        self._kp_scale = torch.ones(env.num_envs, self._action_dim, device=env.device)
        self._kd_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim_scale = torch.ones_like(self._kp_scale)
        self._rfi_lim: float = 0.0
        self._randomize_torque_rfi: bool = False

        # PD gains and action scales
        self.p_gains = torch.zeros(self._action_dim, dtype=torch.float, device=env.device)
        self.d_gains = torch.zeros_like(self.p_gains)
        self.i_gains = torch.zeros_like(self.p_gains)
        self.action_scales = torch.zeros_like(self.p_gains)

        self._configure_pd_gains(env)
        self._configure_action_scales(env)

        # Expose references on the environment for backward compatibility
        env.p_gains = self.p_gains
        env.d_gains = self.d_gains
        env.i_gains = self.i_gains
        env.action_scales = self.action_scales

        # Action delay queue will be initialized in setup() after randomization manager is ready
        self.action_queue: torch.Tensor | None = None

        # Grip-force control (optional, see GripForceCfg): closed-loop wrist torque bias that
        # presses each hand into the carried object at a target force, replacing the old
        # curriculum/assist mechanisms.
        self._grip_cfg: GripForceCfg | None = self.cfg.params.get("grip_force")
        self._grip_enabled = bool(self._grip_cfg is not None and self._grip_cfg.enable)
        if self._grip_enabled:
            self._configure_grip_force(env)

    def _configure_grip_force(self, env: Any) -> None:
        cfg = self._grip_cfg
        assert cfg is not None
        self._left_wrist_dof_idx = torch.tensor(
            [env.dof_names.index(n) for n in cfg.left_wrist_joint_names], dtype=torch.long, device=env.device
        )
        self._right_wrist_dof_idx = torch.tensor(
            [env.dof_names.index(n) for n in cfg.right_wrist_joint_names], dtype=torch.long, device=env.device
        )
        self._left_chain_body_idx = [env.body_names.index(n) for n in cfg.left_chain_body_names]
        self._right_chain_body_idx = [env.body_names.index(n) for n in cfg.right_chain_body_names]
        self._left_wrist_yaw_body_idx = self._left_chain_body_idx[-1]
        self._right_wrist_yaw_body_idx = self._right_chain_body_idx[-1]

        self._grip_hand_offset_local = torch.tensor(cfg.hand_offset_local, dtype=torch.float32, device=env.device)
        self._grip_axis_roll = torch.tensor([1.0, 0.0, 0.0], device=env.device)
        self._grip_axis_pitch = torch.tensor([0.0, 1.0, 0.0], device=env.device)
        self._grip_axis_yaw = torch.tensor([0.0, 0.0, 1.0], device=env.device)

        self._grip_command_term = None  # resolved in setup(), once all managers exist

    def setup(self) -> None:
        """Setup action term after all managers are initialized.

        Initialize action delay queue if control delay randomization is enabled.
        This must be called after the randomization manager is set up.
        """
        super().setup()

        # Initialize action delay queue if randomization is enabled
        if getattr(self.env, "_randomize_ctrl_delay", False):
            max_delay = self.env._ctrl_delay_step_range[1]
            self.action_queue = torch.zeros(self.env.num_envs, max_delay + 1, self._action_dim, device=self.env.device)

        # Allocate sub-step torque history buffer
        decimation = self.env.simulator.simulator_config.sim.control_decimation
        self.torques_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)
        self.dof_pos_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)
        self.dof_vel_substep = torch.zeros(self.env.num_envs, decimation, self._action_dim, device=self.env.device)

        # IsaacGym creates randomization buffers before the action manager exists.
        # Once we reach setup(), try attaching any pre-created actuator scales.
        self._attach_actuator_randomizer_scales()

        enabled, rfi_lim = self.env._pending_torque_rfi
        self.configure_torque_rfi(enabled=enabled, rfi_lim=rfi_lim)
        self.env._pending_torque_rfi = (False, 0.0)

        if self._grip_enabled:
            assert self._grip_cfg is not None
            self._grip_command_term = self.env.command_manager.get_state(self._grip_cfg.command_term_name)

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._action_dim

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process raw actions: clip and apply delay if configured.

        Args:
            actions: Raw action tensor [num_envs, action_dim]
        """
        self._substep_idx = 0
        # Store raw actions
        assert self._raw_actions is not None
        self._raw_actions[:] = actions

        self.env.log_dict["action_mean_abs"] = actions.abs().mean()
        self.env.log_dict["action_std"] = actions.std()

        # Clip actions
        if self.env.robot_config.control.clip_actions:
            clip_limit = self.env.robot_config.control.action_clip_value
            assert self._processed_actions is not None
            self._processed_actions[:] = torch.clip(actions, -clip_limit, clip_limit)
            # Log clipping fraction
            self.env.log_dict["action_clip_frac"] = (
                self._processed_actions.abs() == clip_limit
            ).sum() / self._processed_actions.numel()
        else:
            assert self._processed_actions is not None
            self._processed_actions[:] = actions
            self.env.log_dict["action_clip_frac"] = torch.tensor(0.0)

        # Apply action delay if configured
        if getattr(self.env, "_randomize_ctrl_delay", False):
            self._apply_action_delay()
        else:
            assert self._processed_actions is not None
            self._actions_after_delay[:] = self._processed_actions

    def _apply_action_delay(self) -> None:
        """Apply action delay based on domain randomization settings."""
        assert self.action_queue is not None, "action_queue must be initialized in setup()"
        assert self._processed_actions is not None

        # Update action queue
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = self._processed_actions.clone()

        # Apply uniform delay
        self._actions_after_delay[:] = self.action_queue[
            torch.arange(self.env.num_envs), self.env.action_delay_idx
        ].clone()

    def apply_actions(self) -> None:
        """Apply processed actions by computing and applying torques."""
        # Compute torques using PD controller
        self.torques[:] = self._compute_torques(self._actions_after_delay)
        # Record sub-step torques/dof_pos/dof_vel
        self.torques_substep[:, self._substep_idx] = self.torques
        self.dof_pos_substep[:, self._substep_idx] = self.env.simulator.dof_pos
        self.dof_vel_substep[:, self._substep_idx] = self.env.simulator.dof_vel
        self._substep_idx += 1
        # Apply torques to simulator
        self.env.simulator.apply_torques_at_dof(self.torques)
        # Cache velocities for next derivative computation
        self._prev_dof_vel.copy_(self.env.simulator.dof_vel)

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute torques from actions using PD controller.

        Args:
            actions: Action tensor [num_envs, action_dim]

        Returns:
            Torque tensor [num_envs, action_dim]
        """
        # Scale actions
        actions_scaled = actions * self.action_scales

        # Compute torques based on control type
        control_type = self.env.robot_config.control.control_type

        if control_type == "P":
            # Position control
            torques = (
                self._kp_scale * self.p_gains * (actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos)
                - self._kd_scale * self.d_gains * self.env.simulator.dof_vel
            )
        elif control_type == "V":
            # Velocity control
            torques = (
                self._kp_scale * self.p_gains * (actions_scaled - self.env.simulator.dof_vel)
                - self._kd_scale * self.d_gains * (self.env.simulator.dof_vel - self._prev_dof_vel) / self.env.sim_dt
            )
        elif control_type == "T":
            # Torque control
            torques = actions_scaled
        else:
            raise ValueError(f"Unknown controller type: {control_type}")

        # Apply torque randomization if configured
        if self._randomize_torque_rfi:
            torques = (
                torques
                + (torch.rand_like(torques) * 2.0 - 1.0) * self._rfi_lim * self._rfi_lim_scale * self.env.torque_limits
            )

        # Grip-force bias: added to the policy's own wrist torques (not a replacement) so the
        # clip below still bounds the TOTAL commanded torque against the real actuator limits.
        if self._grip_enabled:
            torques = torques + self._compute_grip_force_bias()

        # Clip torques if configured
        if self.env.robot_config.control.clip_torques:
            torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)

        return torques

    def _compute_grip_force_bias(self) -> torch.Tensor:
        """Open-loop wrist torque bias pressing each hand into the carried object.

        No force sensor on the real wrist, only current-based torque sensing -- and on a
        torque-CONTROLLED joint, measured torque just reflects what was commanded, so there is no
        new information to close a loop on. Instead this commands target_force_n directly whenever
        the command term's contact flag is active: squeeze_dir = unit(live box center -
        wrist_yaw_link pos), force_vec = target_force_n * squeeze_dir, mapped to the 3 wrist DOF via
        the analytic revolute-joint Jacobian transpose (tau = J^T @ F). Once the wrist is actually
        blocked against the box (quasi-static), the delivered force converges to target_force_n by
        Newton's third law regardless of the box's mass/friction -- see GripForceCfg docstring.
        """
        cfg = self._grip_cfg
        assert cfg is not None and self._grip_command_term is not None
        sim = self.env.simulator
        grip_active: torch.Tensor = self._grip_command_term.grip_active
        box_pos: torch.Tensor = self._grip_command_term.simulator_object_pos_w
        target_force_n = min(cfg.target_force_n, cfg.force_command_max_n)

        bias = torch.zeros_like(self.torques)
        sides = (
            ("left", self._left_wrist_dof_idx, self._left_chain_body_idx, self._left_wrist_yaw_body_idx),
            ("right", self._right_wrist_dof_idx, self._right_chain_body_idx, self._right_wrist_yaw_body_idx),
        )
        grip_active_f = grip_active.float()
        n_active = grip_active_f.sum().clamp_min(1.0)
        self.env.log_dict["grip/contact_active_frac"] = grip_active_f.mean()
        for side, dof_idx, chain_body_idx, wrist_yaw_idx in sides:
            jacobian = self._wrist_jacobian(chain_body_idx)

            wrist_pos = sim._rigid_body_pos[:, wrist_yaw_idx]
            squeeze_dir = box_pos - wrist_pos
            squeeze_dir = squeeze_dir / squeeze_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)

            command_force = torch.where(
                grip_active, torch.full_like(grip_active_f, target_force_n), torch.zeros_like(grip_active_f)
            )

            force_vec = squeeze_dir * command_force.unsqueeze(-1)  # (E, 3)
            # tau[e, dof] = sum_axis jacobian[e, axis, dof] * force_vec[e, axis]  (J^T @ F)
            tau = torch.einsum("eac,ea->ec", jacobian, force_vec)
            bias[:, dof_idx] += tau

            # Averaged over envs currently gripping only -- diluting by the (usually majority) of
            # envs not in contact would wash out the signal on wandb.
            self.env.log_dict[f"grip/command_force_{side}"] = (command_force * grip_active_f).sum() / n_active

        return bias

    def _wrist_jacobian(self, chain_body_idx: list[int]) -> torch.Tensor:
        """Analytic linear-velocity Jacobian (E, 3, 3) of the hand contact point w.r.t. the 3 wrist
        DOF (roll, pitch, yaw, in that column order).

        chain_body_idx = [elbow_link, wrist_roll_link, wrist_pitch_link, wrist_yaw_link] indices
        into env.body_names / env.simulator._rigid_body_pos. Each revolute joint's pivot point is
        its CHILD link's current world position (translation is invariant to the joint's own
        rotation), and its world axis is the PARENT link's current world orientation applied to the
        URDF-local axis (joint origins here have zero rotation, so parent orientation == joint-frame
        orientation) -- see check_wrist_jacobian.py for a finite-difference validation of this
        formula against a synthetic copy of this exact kinematic chain.
        """
        sim = self.env.simulator
        elbow_i, roll_i, pitch_i, yaw_i = chain_body_idx
        p, q = sim._rigid_body_pos, sim._rigid_body_rot
        p_elbow, q_elbow = p[:, elbow_i], q[:, elbow_i]
        p_roll, q_roll = p[:, roll_i], q[:, roll_i]
        p_pitch, q_pitch = p[:, pitch_i], q[:, pitch_i]
        p_yaw, q_yaw = p[:, yaw_i], q[:, yaw_i]

        n = p_elbow.shape[0]
        hand_offset = self._grip_hand_offset_local.expand(n, 3)
        p_hand = p_yaw + quat_apply(q_yaw, hand_offset, w_last=True)

        a_roll = quat_apply(q_elbow, self._grip_axis_roll.expand(n, 3), w_last=True)
        a_pitch = quat_apply(q_roll, self._grip_axis_pitch.expand(n, 3), w_last=True)
        a_yaw = quat_apply(q_pitch, self._grip_axis_yaw.expand(n, 3), w_last=True)

        jacobian = torch.zeros(n, 3, 3, device=self.env.device)
        jacobian[:, :, 0] = torch.cross(a_roll, p_hand - p_roll, dim=-1)
        jacobian[:, :, 1] = torch.cross(a_pitch, p_hand - p_pitch, dim=-1)
        jacobian[:, :, 2] = torch.cross(a_yaw, p_hand - p_yaw, dim=-1)
        return jacobian

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action term state.

        Args:
            env_ids: Environment IDs to reset. If None, reset all.
        """
        super().reset(env_ids)

        # Reset action delay queue if applicable
        if self.env._randomize_ctrl_delay and self.action_queue is not None:
            if env_ids is None:
                self.action_queue.zero_()
            else:
                self.action_queue[env_ids] = 0.0

        # Reset torques
        if env_ids is None:
            self.torques.zero_()
        else:
            self.torques[env_ids] = 0.0

        # Reset cached velocities
        if env_ids is None:
            self._prev_dof_vel.zero_()
        else:
            self._prev_dof_vel[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Hooks for randomization manager

    def attach_actuator_scales(
        self, kp_scale: torch.Tensor, kd_scale: torch.Tensor, rfi_lim_scale: torch.Tensor
    ) -> None:
        """Attach shared actuator scaling tensors provided by the randomization manager."""
        self._kp_scale = kp_scale
        self._kd_scale = kd_scale
        self._rfi_lim_scale = rfi_lim_scale

    def update_pd_scales(self, env_ids: torch.Tensor, kp_values: torch.Tensor, kd_values: torch.Tensor) -> None:
        """Fallback PD-scale update when no shared buffers are registered."""
        self._kp_scale[env_ids] = kp_values
        self._kd_scale[env_ids] = kd_values

    def update_rfi_scales(self, env_ids: torch.Tensor, rfi_values: torch.Tensor) -> None:
        """Fallback RFI-scale update when no shared buffers are registered."""
        self._rfi_lim_scale[env_ids] = rfi_values

    def configure_torque_rfi(self, *, enabled: bool, rfi_lim: float | None = None) -> None:
        """Configure residual force injection behaviour."""
        self._randomize_torque_rfi = enabled
        if rfi_lim is not None:
            self._rfi_lim = float(rfi_lim)

    def get_pd_scale_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return references to the PD gain scale buffers."""
        return self._kp_scale, self._kd_scale

    def get_rfi_scale_tensor(self) -> torch.Tensor:
        """Return reference to the RFI limit scale buffer."""
        return self._rfi_lim_scale

    def get_prev_dof_vel(self) -> torch.Tensor:
        """Return cached previous DOF velocities."""
        return self._prev_dof_vel

    # ------------------------------------------------------------------
    # Internal helpers

    def _attach_actuator_randomizer_scales(self) -> None:
        """Attach shared actuator randomizer buffers if they exist."""
        rand_manager = getattr(self.env, "randomization_manager", None)
        if rand_manager is None:
            return

        get_state = getattr(rand_manager, "get_state", None)
        if not callable(get_state):
            return

        state = get_state("actuator_randomizer_state")
        if state is None:
            return

        self.attach_actuator_scales(state.kp_scale_tensor, state.kd_scale_tensor, state.rfi_lim_scale_tensor)

    def _configure_pd_gains(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        stiffness_cfg = control_cfg.stiffness
        damping_cfg = control_cfg.damping
        integral_cfg = getattr(control_cfg, "integral", {})

        for i, name in enumerate(env.dof_names):
            if name not in env.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")

            matched = False
            for dof_name, stiffness in stiffness_cfg.items():
                if dof_name in name:
                    self.p_gains[i] = stiffness
                    self.d_gains[i] = damping_cfg[dof_name]
                    self.i_gains[i] = integral_cfg.get(dof_name, 0.0)
                    matched = True
            if not matched:
                self.p_gains[i] = 0.0
                self.d_gains[i] = 0.0
                self.i_gains[i] = 0.0
                if control_cfg.control_type in ["P", "V"]:
                    raise ValueError(
                        f"PD gains for joint '{name}' were not defined. Please specify them in the YAML configuration."
                    )

    def _configure_action_scales(self, env: Any) -> None:
        control_cfg = env.robot_config.control
        if control_cfg.action_scales_by_effort_limit_over_p_gain:
            dof_effort_limit_list = env.robot_config.dof_effort_limit_list
            for i, effort in enumerate(dof_effort_limit_list):
                stiffness = self.p_gains[i]
                if stiffness == 0.0:
                    self.action_scales[i] = 0.0
                else:
                    self.action_scales[i] = control_cfg.action_scale * effort / stiffness
        else:
            self.action_scales[:] = control_cfg.action_scale
