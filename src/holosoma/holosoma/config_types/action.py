"""Configuration types for action manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ActionTermCfg:
    """Configuration for a single action-processing term."""

    func: str
    """Import path to the action term class (e.g. ``holosoma.managers.action.terms:JointPositionActionTerm``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments to initialize the action term."""

    scale: float | tuple[float, ...] = 1.0
    """Scaling factor(s) applied to the raw action values before processing."""

    clip: tuple[float, float] | None = None
    """Optional min/max clamp applied to the raw action values."""


@dataclass(frozen=True)
class ActionManagerCfg:
    """Configuration for the action manager."""

    terms: dict[str, ActionTermCfg] = field(default_factory=dict)
    """Mapping of action term name to configuration."""


@dataclass(frozen=True)
class GripForceCfg:
    """Open-loop wrist force control: press each hand into the carried object at ``target_force_n``
    whenever the command term reports contact.

    Replaces the old box-physicality curriculum: instead of faking the hold with a kinematic
    override or an external wrench on the OBJECT, this adds a torque bias to the ROBOT's wrist
    joints so the hold is produced by a real, physical contact force -- the same mechanism that
    must hold the box at deployment. Read by ``JointPositionActionTerm`` via
    ``self.cfg.params["grip_force"]``.

    Deliberately open-loop, not closed-loop on a force sensor: the real G1 wrist has no F/T sensor,
    only current-based torque sensing -- and for a torque-CONTROLLED joint, measured torque just
    reflects the commanded torque (no new information to close a loop on; that's what torque control
    means). The physical quantity that actually matters is the joint's motion: once the wrist is
    genuinely blocked against the box (quasi-static, q_dot ~ 0), ALL of the commanded bias torque
    goes into the contact reaction by Newton's third law / static equilibrium, so the delivered
    force converges to the commanded target on its own -- independent of the box's mass or friction,
    which only affect whether that force is SUFFICIENT to hold the box, not whether it is delivered.

    Per hand: squeeze_dir = unit(box_center - wrist_yaw_link_pos) (live sim poses); command_force =
    target_force_n whenever the command term's contact flag is True, else 0; turned into a 3D force
    along squeeze_dir and mapped to the 3 wrist DOF (roll/pitch/yaw) via the analytic revolute-joint
    Jacobian transpose (tau = J^T @ F) built from the live elbow_link/wrist_roll_link/
    wrist_pitch_link/wrist_yaw_link poses already tracked by the simulator.
    """

    enable: bool = False
    """Master switch. Requires an object-carry motion (has_object) and the wrist body names below
    to exist in ``env.simulator.body_names`` / ``env.dof_names``."""

    command_term_name: str = "motion_command"
    """Name under which the WBT command term is registered (``env.command_manager.get_state(...)``);
    read each substep for the contact gate (``grip_active``) and the live box position
    (``simulator_object_pos_w``)."""

    left_wrist_joint_names: tuple[str, str, str] = (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    )
    right_wrist_joint_names: tuple[str, str, str] = (
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    """DOF names of the 3-joint wrist chain (roll, pitch, yaw order matters -- matches the analytic
    Jacobian columns)."""

    left_chain_body_names: tuple[str, str, str, str] = (
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    )
    right_chain_body_names: tuple[str, str, str, str] = (
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    )
    """Rigid bodies of the chain, in order (parent-of-roll, then the 3 child links the roll/pitch/yaw
    joints each pivot at) -- see JointPositionActionTerm._wrist_jacobian for how these are used."""

    hand_offset_local: tuple[float, float, float] = (0.0415, 0.003, 0.0)
    """Hand contact point, as an offset (m) from wrist_yaw_link in its own local frame -- the
    left/right_hand_palm_joint fixed-joint origin in main_mesh_collision_rubberhand.urdf (identical
    for both hands; the URDF does not mirror this offset, only the arm's own orientation differs)."""

    target_force_n: float = 30.0
    """Target normal (squeeze) force per hand, in Newtons -- commanded directly (open-loop) whenever
    the contact gate is active."""

    force_command_max_n: float = 60.0
    """Safety clamp on target_force_n (in case of a config typo); has no effect while
    target_force_n stays below it."""
