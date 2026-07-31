"""Whole Body Tracking reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        # Fine companion to the term above (coarse+fine). The wide sigma=0.3 term gives far-field
        # guidance during the reach/lift, but is nearly flat near the target (~0.89 at 10 cm) so the
        # policy stagnates a few cm below the carry height. This narrow sigma=0.12 term restores a
        # strong gradient over the last few cm without starving the far field (the wide term stays).
        "object_global_ref_position_error_fine_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.12},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Dense grasp signal: hand<->object relative position on contact frames (neutral on free
        # frames). Complements the global object terms, which say nothing about WHERE in the hand
        # frame the object should be while carried.
        "object_grasp_relative_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_grasp_relative_error_exp",
            params={"sigma": 0.1},
            weight=1.0,
        ),
        # WHERE on the box surface + how deep the current contact is, vs the retargeting reference
        # (HoloV2's own witness/distance point-cloud contact fields, baked per-frame -- see
        # gvhmr-fp-pipeline/contact_from_retarget.py). Neutral automatically if the loaded motion
        # doesn't carry a reference witness (older/synthetic clips).
        "object_surface_contact_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_surface_contact_error_exp",
            params={"sigma_geodesic": 0.1, "sigma_dist": 0.05},
            weight=1.0,
        ),
        # Contact QUALITY (option 2): reward the hand's flat-face keypoints to be flush against the
        # box (patch contact that resists rotation). Independent of the reference witness -- teaches
        # HOW to grip, pairs with the physicality curriculum. sigma=0.03 m (flush within a few cm).
        "object_flat_contact_quality_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_flat_contact_quality_exp",
            params={"sigma": 0.03},
            weight=1.0,
        ),
        # robot<->TABLE : même reward de contact de surface (witness + SDF) que pour la box, mais
        # sur l'objet statique table. Apprend au robot à placer sa main au bon endroit près de la
        # table (approche/dépôt) et à ne pas foncer dedans. Neutre si le clip ne porte pas de table.
        # Poids plus faible que la box (signal secondaire de placement, pas la tâche principale).
        "support_surface_contact_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:support_surface_contact_error_exp",
            params={"sigma_geodesic": 0.1, "sigma_dist": 0.05},
            weight=0.5,
        ),
    }
)

__all__ = ["g1_29dof_wbt_fast_sac_reward", "g1_29dof_wbt_reward", "g1_29dof_wbt_reward_w_object"]
