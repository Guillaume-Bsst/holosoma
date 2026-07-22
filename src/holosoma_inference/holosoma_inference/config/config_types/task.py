"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from typing import Literal

from pydantic.dataclasses import dataclass

InputSource = Literal["keyboard", "interface", "joystick", "ros2"]

DEFAULT_VELOCITY_INPUT: InputSource = "keyboard"
DEFAULT_STATE_INPUT: InputSource = "keyboard"


@dataclass(frozen=True)
class DebugConfig:
    """Debug overrides for quick testing."""

    force_upright_imu: bool = False
    """Override projected_gravity with [0, 0, -1] (perfectly upright)."""

    force_zero_angular_velocity: bool = False
    """Override base_ang_vel with [0, 0, 0]."""

    force_zero_action: bool = False
    """Zero out the scaled policy action (robot holds default pose)."""


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    action_scales_by_effort_limit_over_p_gain: bool = False
    """Use per-joint scaling: ``policy_action_scale * effort_limit / p_gain``."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "auto"
    """Network interface name. Use ``"auto"`` to auto-detect, or specify explicitly (e.g. ``"eth0"``)."""

    velocity_input: InputSource = DEFAULT_VELOCITY_INPUT
    """Source for velocity commands."""

    state_input: InputSource = DEFAULT_STATE_INPUT
    """Source for non-velocity inputs (start/stop, walk/stand, tuning)."""

    use_keyboard: bool = False
    """Shortcut: set both velocity_input and state_input to "keyboard".

    Cannot be combined with explicit input settings.
    """

    use_joystick: bool = False
    """Shortcut: set both velocity_input and state_input to "joystick".

    Cannot be combined with explicit input settings.
    """

    switch_blend_steps: int = 50
    """Dual-mode only: number of control steps over which the commanded joint target is
    blended from the previous policy's last command to the incoming policy's target when
    switching (X/x). Removes the one-step target jump at the switch (most visible on the
    low-stiffness arms). 0 disables the blend (instant switch)."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    ros_cmd_vel_topic: str = "cmd_vel"
    """ROS2 topic name for velocity commands (used when velocity_input is "ros2")."""

    ros_state_input_topic: str = "holosoma/state_input"
    """ROS2 topic name for discrete commands (used when state_input is "ros2")."""

    ros_vel_timeout: float = 1.0
    """Seconds without a velocity message before zeroing commands. Set to 0 to disable."""

    auto_walk_on_vel_cmd: bool = False
    """Automatically enter walking mode when a non-zero velocity command is received."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    print_observations: bool = False
    """Print observation vectors for debugging."""

    motion_start_timestep: int = 0
    """Starting timestep for motion clip playback."""

    motion_end_timestep: int | None = None
    """Ending timestep for motion clip playback. If None, plays until the end."""

    object_motion_file: str | None = None
    """Path to the training motion NPZ (with object_pos_w/object_quat_w) for object-carry policies.
    Required by g1-29dof-wbt-w-object: the actor obs includes the box pose relative to the reference
    root (obj_pos_b + obj_ori_b, 9 dims). Since the box tracks the reference during contact
    (kinematic), that relative transform is derived directly from this clip, indexed by the motion
    timestep -- no external box-pose channel needed for sim-to-sim. For real deployment this is where
    a live mocap/RGB-D box pose would be substituted."""

    motion_prepend_timesteps: int = 0
    """Number of default-pose prepend frames the trained motion has ahead of the clip (matches the
    training MotionConfig default_pose_prepend). The object-obs lookup pads the clip's object/root
    trajectory by this many frames (holding frame 0) so the timestep indexing aligns with the ONNX."""

    zero_object_obs: bool = False
    """Debug: feed zeros for obj_pos_b/obj_ori_b instead of the clip lookup. Isolation test -- if the
    robot stands with this on but falls with it off, the object obs (frame/convention) is the culprit;
    if it falls either way, the object obs is not the cause (look at gains/URDF/general sim2sim gap)."""

    debug: DebugConfig = DebugConfig()
    """Debug overrides for quick testing."""

    def __post_init__(self):
        """Resolve use_keyboard/use_joystick shortcuts into velocity_input/state_input."""
        if self.use_keyboard and self.use_joystick:
            raise ValueError(
                "Cannot combine --task.use-keyboard with --task.use-joystick. "
                "Use one shortcut or set --task.velocity-input and --task.state-input individually."
            )

        shortcut: InputSource | None = None
        flag_name: str | None = None
        if self.use_joystick:
            shortcut = "interface"
            flag_name = "joystick"
        elif self.use_keyboard:
            shortcut = "keyboard"
            flag_name = "keyboard"

        if shortcut is not None:
            has_custom_input = self.velocity_input != DEFAULT_VELOCITY_INPUT or self.state_input != DEFAULT_STATE_INPUT
            if has_custom_input:
                raise ValueError(
                    f"Cannot combine --task.use-{flag_name} with --task.velocity-input or "
                    "--task.state-input. Use either the shortcut flag or the individual "
                    "input settings, not both."
                )
            object.__setattr__(self, "velocity_input", shortcut)
            object.__setattr__(self, "state_input", shortcut)
