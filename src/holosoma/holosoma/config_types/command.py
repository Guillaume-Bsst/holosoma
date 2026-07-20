"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class GraspSettleConfig:
    """Grasp-consistent object initialisation + settling window for object-interaction clips.

    When a reset lands mid-manipulation (robot holding the object), RSI + independent per-actor
    noise breaks the hand<->object contact, so the object is ejected (penetration) or dropped and
    the episode dies on the object-tracking termination. This makes those contact resets stable:
      - place the object exactly at its reference pose (drop the independent object noise),
      - optionally scale down the robot init-pose noise on contact resets,
      - hold the clip frozen for ``settle_steps`` while the physics contact equilibrates,
      - suppress tracking termination during that settle window (grace period),
      - optionally weld the object to the nearest hand during the window, then release.

    All gated behind ``enable`` and only active when the motion actually has an object, so the
    default (disabled) reproduces the previous behaviour exactly.
    """

    enable: bool = False
    """Master switch. When False, reset/step/termination behave exactly as before."""

    contact_distance_threshold: float = 0.35
    """Nearest hand<->object distance (m) below which a reset frame counts as 'in contact'.
    Above it (object resting on the ground, hands away) the reset is left untouched."""

    box_half_extents: tuple[float, float, float] = (0.16, 0.16, 0.16)
    """Object half-extents (m), box-local axes -- must match the grasped object's URDF/mesh
    (box32.obj: 0.32m cube). Used by the GPU box SDF/geodesic (utils/box_geometry.py) for the
    surface-contact reward (object_surface_contact_error_exp): the live nearest-surface-point and
    signed distance of the current sim contact are computed against THIS box, then compared to the
    retargeting-pipeline's reference witness/distance (see gvhmr-fp-pipeline/contact_from_retarget.py)."""

    settle_steps: int = 12
    """Number of policy steps to freeze the clip + grace termination after a contact reset."""

    settle_robot_noise_scale: float = 0.0
    """Multiplier applied to the robot init-pose noise (dof/root) on contact resets only.
    0.0 = spawn exactly at the reference contact pose (most stable). 1.0 = keep full noise."""

    freeze_clip_during_settle: bool = True
    """Hold the motion counter fixed during the settle window (don't advance the clip)."""

    disable_termination_during_settle: bool = True
    """Suppress tracking-based termination (BadTracking) during the settle window."""

    weld_object_during_settle: bool = False
    """Kinematically weld the object to the nearest hand each policy step during the window,
    then release. Robustness upgrade for clips where the object still pops after settling.
    Applied per policy step (not per physics substep)."""

    # --- full-contact weld curriculum ("training wheels") -------------------------------------
    # Independently of the settle window, an episode can be "assisted": whenever the REFERENCE is
    # in contact (nearest reference hand within contact_distance_threshold of the reference object),
    # the object is kinematically carried at the reference grasp transform applied to the SIM hand.
    # The assist probability is drawn per episode at reset and annealed over training so early
    # learning sees a stable carry (no cold-start "box always falls -> no signal") while the final
    # policy holds the object fully physically.
    weld_contact_prob_start: float = 0.0
    """Episode-assist probability at step 0 of training. 0.0 disables the curriculum entirely."""

    weld_contact_prob_end: float = 0.0
    """Episode-assist probability after weld_anneal_steps env steps (keep 0.0 so the final policy
    is never assisted)."""

    weld_anneal_steps: int = 400_000
    """Env steps over which the assist probability anneals linearly from start to end.
    (~55% of a 30k-iteration PPO run at 24 steps/iteration.)"""

    # --- physicality curriculum: gradually make the box physical as the policy succeeds ---------
    physicality_curriculum: bool = False
    """Adaptive curriculum on top of kinematic_object_during_contact. The kinematic override is
    blended by ``alpha`` in [0,1]: box_state = alpha*reference + (1-alpha)*physical, per contact step.
    alpha=1 = fully kinematic (box forced to reference, current behaviour); alpha=0 = fully physical
    (box free, the robot must physically hold it); intermediate = partial assist (the box slips/droops
    between corrections, so the policy must grip to keep it near the reference). alpha starts at 1 and
    DECREASES by physicality_alpha_step whenever the success-rate EMA exceeds
    physicality_success_threshold (with a cooldown for the policy to re-adapt), down to
    physicality_alpha_min. Monotonic (alpha never rises); if success drops the curriculum just waits,
    so it plateaus at the most-physical box the policy can hold. Requires
    kinematic_object_during_contact=True. Teaches gripping while staying convergent."""

    physicality_success_threshold: float = 0.9
    """Success-rate EMA above which alpha is decreased (box made more physical)."""

    physicality_alpha_step: float = 0.1
    """Decrement applied to alpha (1->0) at each curriculum advance."""

    physicality_alpha_min: float = 0.0
    """Floor for alpha. 0.0 lets the curriculum reach a fully physical box."""

    physicality_cooldown_steps: int = 2000
    """Policy steps to wait after each alpha decrease before checking the threshold again (lets the
    policy re-adapt to the new physicality before advancing further)."""

    physicality_ema_beta: float = 0.02
    """EMA smoothing for the per-step success signal (higher = more reactive, noisier)."""

    # --- kinematic object during contact (the reliable "make it work" grasp) ------------------
    kinematic_object_during_contact: bool = False
    """When True, the object is KINEMATIC on every reference-contact frame: its pose+velocity are set
    to the REFERENCE trajectory (clip) each policy step, always on (no anneal, no probability). This is
    the standard manipulation-from-mocap treatment: a small-contact hand model cannot robustly hold a
    box by friction under imperfect tracking (empirically: box drifts ~25cm mid-carry and the episode
    dies on bad_object_pos), so instead the grasp is *assumed* (the real robot's hand grips for real at
    deployment) and the policy learns the BODY motion + hand PLACEMENT (via object_grasp_relative_error).
    Unlike the (buggy) assist weld this welds to the SMOOTH REFERENCE with the reference velocity (not to
    the lagging sim hand with zero velocity -> no tumble), so the box never drifts and never kills the
    episode. Supersedes weld_contact_prob_* when enabled."""

    anchor_body_names: list[str] = field(
        default_factory=lambda: ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    )
    """Candidate hand/anchor bodies; the nearest to the object at the reset frame is chosen."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files.
    When non-empty, takes precedence over motion_file."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    freeze_at_timestep_end_prob: float = 0.0
    """Probability of freezing the motion counter at the LAST frame (motion_end_idx - 1, the appended
    default pose) instead of advancing into the end-of-clip reset. Mirror of freeze_at_timestep_zero_prob
    for the end of the clip, so the robot practices holding the final (stiff) pose. Without it the end
    hold is never trained — the env resets the instant the counter reaches motion_end_idx — so holding
    the final pose is out-of-distribution at inference. Sampled independently each policy step; expected
    hold is roughly 1 / (1 - p) steps. Most useful together with enable_default_pose_append."""

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)

    # object-interaction: grasp-consistent init + settling window (no-op unless enabled + has_object)
    grasp_settle: GraspSettleConfig = field(default_factory=GraspSettleConfig)
