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
    DECREASES on a constant-difficulty schedule whenever the success-rate EMA exceeds
    physicality_success_threshold (with a cooldown for the policy to re-adapt), down to
    physicality_alpha_min. The step is NOT uniform in alpha: the residual box drift the policy must
    absorb scales like beta = (1-alpha)/alpha, which explodes as alpha->0, so a uniform alpha step is a
    tiny difficulty jump near 1 and an infinite one near 0. Instead we keep the difficulty RATIO
    constant: each advance multiplies beta by physicality_alpha_ratio (alpha_next = 1/(1+ratio*beta)),
    which is geometric in beta and auto-shrinks the alpha step near 0. Monotonic (alpha never rises); if
    success drops the curriculum just waits, so it plateaus at the most-physical box the policy can
    hold. Requires kinematic_object_during_contact=True. Teaches gripping while staying convergent."""

    physicality_success_threshold: float = 0.9
    """Success-rate EMA above which alpha is decreased (box made more physical)."""

    physicality_alpha_ratio: float = 1.2
    """Constant factor by which the difficulty beta=(1-alpha)/alpha grows at each advance
    (alpha_next = 1/(1 + ratio*beta)). >1 makes the box more physical each step; larger = coarser,
    faster, riskier. Anchored on the old uniform schedule, whose clean middle steps sat near a beta
    ratio of ~1.5-1.7; 1.2 is deliberately gentler for the hard approach to the floor."""

    physicality_alpha_start: float = 0.9
    """Alpha reached on the FIRST advance out of the fully-kinematic warmup (alpha=1). Needed because
    beta=(1-alpha)/alpha is 0 at alpha=1, so the geometric-in-beta update cannot leave 1 on its own."""

    physicality_alpha_min: float = 0.05
    """Floor for alpha (curriculum barrier). Kept > 0 on purpose: the geometric schedule never reaches
    0 anyway, and alpha<=1e-4 is treated as a fully-free box (no override), which the fingerless hand
    cannot hold -> episodes die on bad_object_pos. 0.05 leaves a light kinematic leash (~95% physical
    box) while staying well under the object termination threshold. This kinematic floor is a probe /
    scaffold, NOT a sim2real endpoint: a faithful deployment needs a physical grasp constraint."""

    physicality_cooldown_steps: int = 2000
    """Policy steps to wait after each alpha decrease before checking the threshold again (lets the
    policy re-adapt to the new physicality before advancing further)."""

    physicality_ema_beta: float = 0.02
    """EMA smoothing for the per-step success signal (higher = more reactive, noisier)."""

    # --- force-mode assist: bounded PD wrench instead of the state blend -----------------------
    physicality_force_mode: bool = False
    """Replace the state-blend assist with a BOUNDED PD WRENCH toward the reference on contact
    frames. The blend is an infinite-gain controller: it rewrites the object STATE regardless of
    the force that would require, so its residual difficulty scales like beta=(1-alpha)/alpha and
    the final step to alpha=0 is an infinite difficulty jump — the curriculum structurally cannot
    finish (observed: stuck at the alpha floor, policy leans on the crutch, sim2sim drops the box
    at deposit). A capped wrench cannot rescue arbitrary drift: difficulty is ~linear in the cap
    and the cap->0 limit is CONTINUOUS (a human helper progressively letting go — same nature as
    the hand contact forces that must replace it). alpha keeps its curriculum role: 1 = kinematic
    warmup (state override, unchanged); <1 = PD wrench with caps alpha*force_assist_fmax /
    alpha*force_assist_tmax; 0 = fully free box. Ladder in force mode is multiplicative
    (physicality_force_alpha_decay, snap to exactly 0 below physicality_force_alpha_snap) and
    ignores physicality_alpha_min. Success is measured on OBJECT TRACKING, not survival
    (physicality_success_obj_err) — required because object terminations are gated off at low
    alpha (object_term_min_alpha), so survival saturates. Requires physicality_curriculum and
    kinematic_object_during_contact."""

    force_assist_gravity_comp: bool = True
    """Feed the object's WEIGHT forward instead of making the tracking PD fight it.

    Without this the assist is ``clamp(PD, alpha*fmax)`` and what actually governs the carry is
    the authority left ABOVE gravity, ``alpha*fmax - m*g`` — which hits zero at
    ``alpha = m*g/fmax`` and goes NEGATIVE below it (the assist can no longer even levitate the
    box). So the ladder had a hard cliff well before alpha=0 and the curriculum could never reach
    the fully-physical box: the same structural failure as the state blend, merely relocated from
    alpha->0 to alpha->m*g/fmax. A pure PD also carries a permanent sag of ``m*g/kp`` against that
    constant load.

    With gravity compensation the wrench is ``alpha * (m*g_up + clamp(PD, fmax))``: the weight the
    POLICY must carry is ``(1-alpha)*m*g`` — linear in alpha, zero at alpha=1, the full box at
    alpha=0 — and ``fmax`` becomes a pure tracking-authority budget rather than a levitation
    budget. The mass is read PER ENV from the simulator, so the object mass domain randomisation
    (``randomize_object_rigid_body_mass_startup``, +U(1,4) kg here — the trained box is 1.8-4.8 kg,
    not the 0.811 kg of the URDF) is respected exactly."""

    force_assist_track_accel: float = 8.0
    """Tracking authority in m/s^2: the PD cap becomes ``m_env * this`` (per-env mass), instead of
    the absolute ``force_assist_fmax``. Set to 0 to fall back to the absolute cap.

    A FIXED cap in newtons is not mass-invariant, and the object mass is randomised over a 2.7x
    range (1.8-4.8 kg): the tracking term has to supply ``m*a`` to follow the reference, so a 12 N
    cap leaves 6.0 m/s2 of authority on the lightest box but only 2.2 m/s2 on the heaviest. The
    heavy tail then saturates during the lift and drifts past the 10 cm success radius NO MATTER
    the alpha — which pins ``obj_track_success`` at ~0.89 and starves the curriculum of a signal
    it could otherwise earn. Scaling the cap with the mass gives every env the same tracking
    authority, so the metric measures the POLICY rather than the mass it happened to draw."""

    force_assist_fmax: float = 12.0
    """Absolute TRACKING force cap (N) at alpha=1, applied to the PD term only. Used only when
    ``force_assist_track_accel`` is 0 (otherwise the mass-proportional cap wins).

    With ``force_assist_gravity_comp`` this is a positioning budget, not a lifting one: the weight
    is fed forward separately, so this no longer has to exceed m*g to keep the box up."""

    force_assist_tmax: float = 1.5
    """Torque cap (N.m) at alpha=1 for the orientation PD."""

    force_assist_kp: float = 200.0
    """Assist stiffness (N/m). With gravity compensation the PD only sees the tracking residual
    (no constant m*g load to fight), so the steady-state sag it used to carry is gone."""

    force_assist_kd: float = 20.0
    """Assist damping (N/(m/s)) toward the reference velocity."""

    force_assist_kp_rot: float = 8.0
    """Orientation stiffness (N.m/rad)."""

    force_assist_kd_rot: float = 0.5
    """Angular damping (N.m/(rad/s)); damps to zero angular velocity (carry reference is quasi-static)."""

    physicality_force_alpha_decay: float = 0.75
    """Force-mode ladder: alpha (hence both caps) multiplies by this on each advance."""

    physicality_force_alpha_snap: float = 0.05
    """Below this alpha the force-mode ladder snaps to exactly 0 (fully free box) — the residual
    cap (<0.6 N) is noise-level, holding a rung there teaches nothing."""

    physicality_success_obj_err: float = 0.10
    """Force mode: curriculum success = fraction of ref-contact envs whose object position error
    is under this (m), EMA'd, combined (min) with the survival rate so the warmup still requires
    surviving episodes."""

    object_term_min_alpha: float = 0.2
    """Below this assist alpha, bad_object_pos/ori terminations are DISABLED: a drop stops paying
    object rewards for the rest of the clip instead of killing the episode. Makes drops learnable
    (recovery gradient exists) and keeps the curriculum signal meaningful near alpha=0. Force mode
    only; 0.0 restores always-on object terminations."""

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

    flat_contact_offsets: list[list[float]] = field(
        default_factory=lambda: [
            [0.089, -0.009, 0.002],
            [0.054, -0.009, 0.002],
            [0.124, -0.009, 0.002],
            [0.089, -0.009, 0.037],
            [0.089, -0.009, -0.033],
        ]
    )
    """Contact-patch keypoints on the hand's flat face, as offsets (m) in the anchor (wrist_yaw_link)
    frame. Used by the contact-quality reward object_flat_contact_quality_exp: rewarding ALL of them to
    be flush against the box (signed distance ~0) drives a PATCH contact -- >=3 non-collinear points
    touching == a flat face against the box face, which resists the rotational escape a single contact
    point cannot (the 155deg box tumble). Independent of the reference witness; teaches HOW to grip.
    Default = the LEFT rubber-hand PALM (flat -y side of left_rubber_hand.STL, plane y~-0.009 in the
    wrist frame, centre + 4 spread over the palm; measured from the mesh). The old half-sphere disk was
    [[0.029,-0.003,0],[0.029,0.032,0],[0.029,-0.038,0],[0.029,-0.003,0.035],[0.029,-0.003,-0.035]].
    The rubber hands are y-mirrors, so the right hand needs flat_contact_offsets_right."""

    flat_contact_offsets_right: list[list[float]] | None = field(
        default_factory=lambda: [
            [0.089, 0.015, 0.002],
            [0.054, 0.015, 0.002],
            [0.124, 0.015, 0.002],
            [0.089, 0.015, 0.037],
            [0.089, 0.015, -0.033],
        ]
    )
    """Contact-patch keypoints for the RIGHT anchor (anchor_body_names[1]) when the hand geometry is
    chiral (rubber hand: palm is -y on the left mesh, +y on the right, both palm planes offset by the
    +0.003 joint origin -> y~+0.015). None = use flat_contact_offsets for both anchors (correct for
    the symmetric half-sphere hand)."""

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

    eval_start_at_zero: bool = False
    """En eval (is_evaluating) : False (défaut) = départ à une frame ALÉATOIRE (RSI, comme
    en training) ; True = forcer la frame 0 (eval déterministe/reproductible). N'affecte QUE
    l'eval — en training la phase est déjà aléatoire indépendamment de ce flag."""

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

    # C-D lite : proximité relative mains↔objet (terme reward gaté sur le variant _actor)
    beta_scale: float = 0.1
    """Échelle (m) de la décroissance du poids β = exp(-d_demo/beta_scale) ; ~1 au contact."""

    hand_body_names: list[str] = field(default_factory=lambda: ["left_wrist_yaw_link", "right_wrist_yaw_link"])
    """Liens des mains suivis par le terme C-D lite (repère objet)."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)

    # object-interaction: grasp-consistent init + settling window (no-op unless enabled + has_object)
    grasp_settle: GraspSettleConfig = field(default_factory=GraspSettleConfig)
