"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.1, 0.1, 0.05],
    root_ang_vel=[0.1, 0.1, 0.1],
    object_pos=[0.05, 0.05, 0.05],
    # -- Robust noise fields --
    # Wrist links are the closest controlled points to the grasped object.
    hand_body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"],
    # Matches all arm DOFs (shoulders, elbows, wrists) on both sides.
    arm_joint_pattern=r"^(left|right)_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow|wrist_roll|wrist_pitch|wrist_yaw)_joint$",
    grasp_mask_min_dist=0.10,
    grasp_mask_max_dist=0.40,
    # Capsule geometry for object-placement rejection sampling.
    # Upper-leg capsules start from hip_pitch_link (laterally offset from pelvis center) so
    # the inter-leg space is NOT falsely covered.  Using pelvis as origin would place the
    # capsule axis through the center and reject valid between-legs placements.
    torso_capsule_body_pairs=[
        ["pelvis", "torso_link"],                              # spine / waist volume
        ["left_hip_pitch_link", "left_knee_link"],            # left thigh (laterally offset)
        ["right_hip_pitch_link", "right_knee_link"],          # right thigh (laterally offset)
        ["left_knee_link", "left_ankle_roll_link"],           # left shin
        ["right_knee_link", "right_ankle_roll_link"],         # right shin
    ],
    torso_capsule_radii=[0.20, 0.09, 0.09, 0.07, 0.07],
    object_noise_num_proposals=5,
    object_collision_radius=0.05,
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
    use_adaptive_timesteps_sampler=False,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
    # Disable prepend/append transitions when an object is present.
    # The transition interpolates the robot from its default upright pose to the motion-start pose
    # while the object stays fixed at its motion-start position (on the floor, near the robot).
    # Intermediate frames have the upright robot's legs passing through the stationary box,
    # creating physically invalid states that cause MuJoCo contact explosions when sampled.
    enable_default_pose_prepend=False,
    enable_default_pose_append=False,
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

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_command_w_object",
]
