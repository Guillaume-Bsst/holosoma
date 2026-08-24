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
        # robot<->TABLE: same surface-contact reward (witness + SDF) as for the box, but on the
        # static table object. Teaches the robot to put its hand in the right spot near the table
        # (approach/drop-off) and not to barge into it. Neutral if the clip carries no table.
        # Lower weight than the box (secondary placement signal, not the main task).
        "support_surface_contact_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:support_surface_contact_error_exp",
            params={"sigma_geodesic": 0.1, "sigma_dist": 0.05},
            weight=0.5,
        ),
    }
)

g1_29dof_wbt_reward_w_object_actor = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
        # C-D lite: relative hand<->object proximity, beta-weighted (refiner, small weight).
        "motion_relative_hand_object_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_hand_object_position_error_exp",
            params={"sigma": 0.3},
            weight=0.3,
        ),
    }
)

# ================================================================================================
# Optional object-training feature blocks
# ================================================================================================
# Two independent blocks, meant to be run alone or together so their effect can be told apart.
# Both are opt-in: the base w_object / w_object_actor presets above stay untouched.
_REW = "holosoma.managers.reward.terms.wbt:"

# Block A -- object VELOCITY tracking. The global pose terms constrain where the box is, not how it
# moves to get there; two trajectories through the same waypoints differ by a jerked carry versus a
# smooth one, and only the second survives contact. Sigmas mirror the robot-side body velocity terms
# (motion_global_body_lin_vel / _ang_vel) so the scales stay comparable. Weight 0.5: a shaping term
# next to the pose terms at 1.0, not a competitor.
# Not from HDMI, whose reward table carries only object pose and contact.
object_velocity_reward_terms = {
    "object_global_ref_lin_vel_error_exp": RewardTermCfg(
        func=f"{_REW}object_global_ref_lin_vel_error_exp",
        params={"sigma": 1.0},
        weight=0.5,
    ),
    "object_global_ref_ang_vel_error_exp": RewardTermCfg(
        func=f"{_REW}object_global_ref_ang_vel_error_exp",
        params={"sigma": 3.14},
        weight=0.5,
    ),
}

# Block B -- HDMI CONTACT reward (arXiv:2509.16757). The contact terms already in the base preset are
# purely kinematic: a hand can satisfy them resting a millimetre off the box, carrying nothing. This
# one reads the measured contact force, so it separates a grip that bears load from a pose that only
# looks like one.
#   sigma_pos 0.1 m      matches object_surface_contact_error_exp's geodesic sigma.
#   force_threshold 10 N "actually bearing something" rather than brushing past.
#   sigma_force 20 N     with max_force_bonus 2.0 the bonus saturates near 24 N, i.e. a firm grip
#                        gets the full factor and a crush gets nothing more.
# HDMI weights contact at 2.5x its object pose term; that would be 2.5 here, which is aggressive
# against the four object terms already present. Start at 1.0 and tune up.
object_contact_reward_terms = {
    "object_contact_force_match_exp": RewardTermCfg(
        func=f"{_REW}object_contact_force_match_exp",
        params={
            "sigma_pos": 0.1,
            "sigma_force": 20.0,
            "force_threshold": 10.0,
            "max_force_bonus": 2.0,
        },
        weight=1.0,
    ),
}


def _with_blocks(base: RewardManagerCfg, *blocks: dict) -> RewardManagerCfg:
    """base preset + the opted-in feature blocks, base terms untouched."""
    terms = dict(base.terms)
    for block in blocks:
        terms.update(block)
    return RewardManagerCfg(terms=terms)


# On the w_object base -- what the 27dof object experiments actually run.
g1_29dof_wbt_reward_w_object_objvel = _with_blocks(g1_29dof_wbt_reward_w_object, object_velocity_reward_terms)
g1_29dof_wbt_reward_w_object_objcontact = _with_blocks(g1_29dof_wbt_reward_w_object, object_contact_reward_terms)
g1_29dof_wbt_reward_w_object_objvel_objcontact = _with_blocks(
    g1_29dof_wbt_reward_w_object, object_velocity_reward_terms, object_contact_reward_terms
)

# On the w_object_actor base (keeps the C-D lite term) -- what the 29dof actor experiments run.
g1_29dof_wbt_reward_w_object_actor_objvel = _with_blocks(
    g1_29dof_wbt_reward_w_object_actor, object_velocity_reward_terms
)
g1_29dof_wbt_reward_w_object_actor_objcontact = _with_blocks(
    g1_29dof_wbt_reward_w_object_actor, object_contact_reward_terms
)
g1_29dof_wbt_reward_w_object_actor_objvel_objcontact = _with_blocks(
    g1_29dof_wbt_reward_w_object_actor, object_velocity_reward_terms, object_contact_reward_terms
)

__all__ = [
    "g1_29dof_wbt_reward_w_object_objvel",
    "g1_29dof_wbt_reward_w_object_objcontact",
    "g1_29dof_wbt_reward_w_object_objvel_objcontact",
    "g1_29dof_wbt_reward_w_object_actor_objvel",
    "g1_29dof_wbt_reward_w_object_actor_objcontact",
    "g1_29dof_wbt_reward_w_object_actor_objvel_objcontact",
    "object_contact_reward_terms",
    "object_velocity_reward_terms",
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_actor",
]
