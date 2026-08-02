"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import (
    CommandManagerCfg,
    CommandTermCfg,
    GraspSettleConfig,
    MotionConfig,
    NoiseToInitialPoseConfig,
)

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    # Re-enable the default-pose transitions and the initial-pose hold practice.
    # Upstream commit 470fd78 (PR #95) flipped these dataclass defaults off; we restore them
    # explicitly here so timestep 0 is a stable standing pose the policy learns to hold,
    # which keeps the inference START hold ("]") parkable regardless of the clip's first frame.
    enable_default_pose_prepend=True,
    enable_default_pose_append=True,
    start_at_timestep_zero_prob=0.2,
    freeze_at_timestep_zero_prob=0.95,
    freeze_at_timestep_end_prob=0.95,  # symmetric end-of-clip hold: practice holding the final default pose
    noise_to_initial_pose=init_pose_config,
)

# Object clip: enable grasp-consistent init + settling so mid-clip contact resets don't eject/drop
# the box. A+B here (contact-consistent placement + freeze + termination grace); weld is left off and
# can be toggled for the hardest clips (or exercised via probe_grasp_settle.py).
grasp_settle_config = GraspSettleConfig(
    enable=True,
    contact_distance_threshold=0.35,  # wrist<->box ~0.21-0.28m when carried, ~0.5-0.8m when free
    settle_steps=12,  # ~0.24s at 50Hz
    settle_robot_noise_scale=0.0,  # contact resets spawn exactly at the reference pose
    freeze_clip_during_settle=True,
    disable_termination_during_settle=True,
    weld_object_during_settle=False,
    # Assist-weld curriculum ("training wheels"): DISABLED BY DEFAULT — it is a known-harmful
    # mechanism, kept only so the probe harness can still exercise it explicitly.
    # It welds the box to the SIM hand (lagging, jittery under an imperfect policy) with the
    # velocity forced to ZERO, and it runs AFTER — so it OVERWRITES — the kinematic /
    # force-mode assist. Measured cost when left at 1.0 (run mub6qh0i vs 2hizgun6, same clip,
    # same hand, weld the only functional difference): the box sits ~8 cm off the reference at
    # alpha=1 where the kinematic override should put it at ~0, the object tracking reward is
    # cut to a third, and the success rate is essentially just the fraction of NON-welded
    # episodes (succ ~= 1 - weld_assist_prob, r^2 = 0.94 over the anneal) — i.e. the "learning
    # curve" was the anneal schedule, costing ~13k iterations.
    # Superseded by kinematic_object_during_contact (welds to the SMOOTH REFERENCE instead).
    weld_contact_prob_start=0.0,
    weld_contact_prob_end=0.0,
    weld_anneal_steps=400_000,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
    grasp_settle=grasp_settle_config,
)

g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

g1_29dof_wbt_command_w_object = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

# Clip femto14 box36 + TABLE. Le seul (avec sa variante gaitfix2) a porter les champs de contact
# table complets -- support_ref_contact / support_ref_witness_local / support_half_extents -- sans
# lesquels support_surface_contact_error_exp renvoie 0 sur tout le run.
# Mesures sur ce clip : 327 frames a 50 Hz (6.5 s), caisse de (-0.550, 0.222, 0.180) au sol jusqu'a
# (-1.592, 0.422, 0.928) sur la table (plateau z=0.750, +0.18 de demi-caisse = 0.930). Jambes dans
# les bornes et sans discontinuite, mais rasantes : appui 95 % du temps et glissement median
# 0.020 m/s en appui, contre 88 % / 0.006 m/s pour la variante gaitfix2 -- le robot traine les pieds
# plus qu'il ne marche. A garder en tete si la locomotion apprise glisse en sim2sim.
motion_config_w_object_femto14_box36 = replace(
    motion_config_w_object,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/femto14_box36_w_obj_gtcontact_nobj.npz",
)

g1_29dof_wbt_command_w_object_femto14_box36 = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_femto14_box36,
            },
        )
    },
)

# G1 27-DOF: waist_roll/pitch joints are fixed in the URDF; their bodies (waist_roll_link,
# torso_link) are preserved as rigid bodies in the sim (collapse_fixed_joints=False).
# body_names_to_track and body_name_ref are identical to 29-DOF.
motion_config_27dof = motion_config

motion_config_27dof_w_object = replace(
    motion_config_27dof,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
    grasp_settle=grasp_settle_config,
)

g1_27dof_wbt_command = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_27dof,
            },
        ),
    },
)

g1_27dof_wbt_command_w_object = replace(
    g1_27dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_27dof_w_object,
            },
        ),
    },
)

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_command_w_object",
    "g1_27dof_wbt_command",
    "g1_27dof_wbt_command_w_object",
]
