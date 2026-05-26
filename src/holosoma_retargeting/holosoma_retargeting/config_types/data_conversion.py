"""Configuration types for data conversion."""

from __future__ import annotations

from dataclasses import dataclass, field

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig

_G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

_G1_27DOF_JOINT_NAMES = [
    n for n in _G1_29DOF_JOINT_NAMES if n not in ("waist_roll_joint", "waist_pitch_joint")
]

_ROBOT_JOINT_NAMES_DEFAULT: dict[str, dict[int, list[str]]] = {
    "g1": {
        29: _G1_29DOF_JOINT_NAMES,
        27: _G1_27DOF_JOINT_NAMES,
    },
}


@dataclass(frozen=True)
class DataConversionConfig:
    """Configuration for data conversion.

    This follows the pattern from holosoma's config_types.
    Uses a flat structure with all conversion parameters.
    """

    input_file: str
    """Path to input motion file."""

    robot: str = "g1"
    """Robot model to use. Use str to allow dynamic robot types."""

    data_format: str = "smplh"
    """Motion data format. Use str to allow dynamic data formats."""

    object_name: str | None = None
    """Override object name (default depends on robot and data type)."""

    input_fps: int = 30
    """FPS of the input motion."""

    output_fps: int = 50
    """FPS of the output motion."""

    line_range: tuple[int, int] | None = None
    """Line range (start, end) for loading data (both inclusive)."""

    has_dynamic_object: bool = False
    """Whether the motion has a dynamic object."""

    output_name: str | None = None
    """Name of the output motion npz file."""

    once: bool = False
    """Run the motion once and exit."""

    use_omniretarget_data: bool = False
    """Use OmniRetarget data format."""

    # --- Nested configs ---
    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    """Robot configuration (nested - can override robot_urdf_file, robot_dof, etc.
    via --robot-config.robot-urdf-file)."""

    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplh", robot_type="g1")
    )
    """Motion data configuration (nested - can override data_format, robot_type, etc.
    via --motion-data-config.data-format)."""

    # --- Joint names, follow the pattern in holosoma_retargeting/config_types/robot.py ---
    joint_names: list[str] | None = None
    """Joint names to use."""

    def _joint_names(self) -> list[str]:
        """Get joint names - use override if provided, else use default based on robot and DOF."""
        if self.joint_names is not None:
            return self.joint_names
        if self.robot not in _ROBOT_JOINT_NAMES_DEFAULT:
            raise ValueError(f"No joint names found for robot: {self.robot}")
        dof_configs = _ROBOT_JOINT_NAMES_DEFAULT[self.robot]
        robot_dof = self.robot_config.ROBOT_DOF
        if robot_dof not in dof_configs:
            available = ", ".join(str(d) for d in sorted(dof_configs.keys()))
            raise ValueError(
                f"No joint names found for robot '{self.robot}' with {robot_dof} DOF. "
                f"Available DOF configs: {available}"
            )
        return dof_configs[robot_dof]

    JOINT_NAMES = property(
        _joint_names,
        doc="Get joint names - use override if provided, else use default.",
    )
