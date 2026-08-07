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

    use_reference_profile: bool = False
    """Command the MEASURED per-hand squeeze force from the stage-05 dynamics sidecar
    (``motion.dyn_grip_force_lr``, N, per frame) instead of the constant ``target_force_n``.

    ``target_force_n`` is a single hand-tuned number applied to both hands for the whole carry; the
    real clip needs a lot of force at pick-up and much less while the box rides against the
    forearms, and it does not need the same on both hands. The profile replays what the physics
    solve actually measured, so the squeeze follows the task instead of being a constant tax.
    Still clamped by ``force_command_max_n``, and still gated by the same proximity ramp. Falls back
    to ``target_force_n`` automatically when the loaded clip has no sidecar (``has_dyn_grip``).

    Approximation: the sidecar stores the MAGNITUDE of the total hand<->box contact wrench, normal
    and tangential together, while the bias commands it purely along the squeeze direction. It
    therefore over-commands slightly whenever a large part of the real reaction was friction holding
    the box up rather than a squeeze into it. That errs toward gripping too hard, which
    ``force_command_max_n`` bounds and which is the safe direction for a task that fails by dropping
    the box."""


@dataclass(frozen=True)
class TorqueFeedforwardCfg:
    """Replay the stage-05 reference joint torque as a feed-forward term in the PD control law.

    The physics stage (SPIDER + MuJoCo) already solved what torque this motion demands at every
    frame, joint by joint, contacts included. Without this the policy has to rediscover all of it --
    starting with gravity compensation, which is most of the torque and is pure prior knowledge, not
    something worth spending samples on. With it the PD loop only has to produce the CORRECTION::

        tau = kp*(q_des - q) - kd*qd + grip_bias + scale * tau_ref(reference frame)

    then the usual clip against the real actuator limits, so the total commanded torque stays
    physically admissible whatever the feed-forward asks for.

    Why this is legitimate even though the solve used SPIDER's stock kp=500 actuators: at those
    gains the solve tracks the kinematic reference to ~0.02 rad, so the replayed trajectory IS
    essentially the reference and the recovered torque is a property of the motion (inertia,
    Coriolis, gravity, contact wrench), not of the controller that produced it. What it is NOT is a
    per-joint prediction of what a real-gain controller will command at this instant -- the policy's
    state will differ from the reference, and the feed-forward is blind to that. Hence ``scale``
    below 1 and the one-sided ``torque_envelope_penalty`` reward rather than torque tracking.

    Deployable: the Unitree low-level motor command carries a feed-forward torque field alongside
    q/dq/kp/kd, so the same phase-indexed profile can be replayed on hardware. It must be plumbed
    through ``holosoma_inference`` before a policy trained with this is deployed -- a policy trained
    with feed-forward and run without it will be commanding the wrong torque.
    """

    enable: bool = False
    """Master switch. Requires the loaded clip to carry ``dyn_tau`` (``motion.has_dyn_tau``);
    a no-op with a warning otherwise."""

    command_term_name: str = "motion_command"
    """Name of the WBT command term to read ``dyn_tau`` (already sampled at each env's own
    reference frame) from."""

    scale: float = 0.5
    """Fraction of the reference torque fed forward. Below 1 on purpose: the reference torque is
    exact for the reference STATE, and the policy is never exactly there, so a full-authority
    feed-forward injects the full modelling error too. Half the torque still removes most of the
    gravity-compensation burden while leaving the PD loop the authority to disagree."""

    joint_names: tuple[str, ...] | None = None
    """Restrict the feed-forward to these DOF (substring match against ``env.dof_names``, same
    convention as the PD stiffness config). ``None`` = all DOF."""

    exclude_joint_names: tuple[str, ...] = ()
    """DOF to EXCLUDE from the feed-forward (substring match), applied after ``joint_names``.

    This is the field that matters in practice, because ``dyn_tau`` is a CLIPPED signal: the physics
    solve ran with a torque cap at the URDF effort limits, so on any joint that hit its cap the
    stored value is not "what the motion demands" but "the most that joint can give". Feeding that
    forward commands a permanent rail-pinned bias the policy then has to fight.

    Measured on femto14_box36 (halfsphere_torquecap): 9.6% of all (frame, joint) samples sit at
    their cap, but it is very unevenly spread -- legs and waist 0.9-7%, arms 1.5-16%, and the wrist
    pitch/yaw 41-50%, pinned at their 5 N.m limit for roughly half the clip. That wrist figure is a
    kp=500 artifact rather than a property of the motion: a 1 kg box on a ~0.05 m lever needs well
    under 1 N.m, and what actually saturates the joint is the stiff actuator resolving the box
    contact. Excluding the wrists is therefore the default, not a tuning knob."""


@dataclass(frozen=True)
class TorqueReferenceNoiseCfg:
    """Per-step torque noise scaled by the torque the MOTION demands, not by the actuator limit.

    The existing RFI (``actuator_randomizer_state``, currently ``enable_rfi_lim: false``) perturbs
    the commanded torque by ``U(-1,1) * rfi_lim * torque_limits`` -- a fixed fraction of each
    joint's LIMIT, identical at every frame. That is the wrong yardstick for a tracking task: a
    joint idling at 2 N.m and the same joint carrying 80 N.m get the same absolute perturbation, so
    the noise is either negligible under load or dominant when free.

    ``dyn_tau`` gives the torque the motion actually demands per joint per frame, so the
    perturbation can follow the load::

        noise = U(-1,1) * (ref_scale * |tau_ref| + floor_scale * tau_limit)

    The ``floor_scale`` term is not a rounding detail, it is what keeps the scheme from degenerating.
    A purely proportional noise vanishes wherever ``tau_ref`` approaches zero -- a swinging leg,
    most obviously -- which removes exploration exactly where the policy has the most freedom to
    choose a different solution. The floor keeps a small limit-relative perturbation everywhere.

    Returns a no-op (with a warning) on clips without ``dyn_tau``, like the feed-forward.
    """

    enable: bool = False
    """Master switch. Requires the loaded clip to carry ``dyn_tau`` (``motion.has_dyn_tau``)."""

    command_term_name: str = "motion_command"
    """Name of the WBT command term to read ``dyn_tau`` from (already sampled at each env's own
    reference frame)."""

    ref_scale: float = 0.15
    """Noise amplitude as a fraction of ``|tau_ref|``. 0.15 = +/-15% of the demanded torque."""

    floor_scale: float = 0.01
    """Noise floor as a fraction of the joint's torque limit, applied everywhere including where
    ``tau_ref`` is ~0. Deliberately small: this is the exploration floor, not the main signal."""

    exclude_joint_names: tuple[str, ...] = ("wrist_pitch", "wrist_yaw")
    """DOF to exclude (substring match). Defaults to the wrists for the same reason the
    feed-forward excludes them, plus one specific to noise: those joints sit pinned at their 5 N.m
    cap for ~half the clip, and the final ``clip_torques`` then removes any positive excursion.
    Noise on an already-saturated joint is therefore not noise at all -- it is a one-sided
    downward bias, which is worse than no randomisation."""
