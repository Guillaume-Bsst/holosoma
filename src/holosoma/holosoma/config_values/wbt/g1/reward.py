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
# smooth one, and only the second survives contact.
# Not from HDMI, whose reward table carries only object pose and contact.
#
# Sigmas are set from the CARRIED BOX's own velocity distribution over the 14 object clips, not
# copied from the robot-side body velocity terms (motion_global_body_lin_vel sigma 1.0,
# _ang_vel 3.14). A swinging limb reaches several rad/s; a carried box has |w| p50 0.164 and p95
# 0.546 rad/s, so the body sigmas leave the term flat. Measured on the frames where the box actually
# moves, against a do-nothing policy (box held still, error = |reference|):
#
#     sigma_ang  3.14 -> 0.993 reward for doing nothing, i.e. 0.007 of usable range (dead)
#                1.00 -> 0.939
#                0.30 -> 0.650, 0.35 of range          <- chosen
#     sigma_lin  1.00 -> 0.834, 0.166 of range
#                0.50 -> 0.610, 0.39 of range          <- chosen
#
# A term whose value barely moves is a constant, and a constant paid every step is a survival bonus
# rather than a signal -- the same failure the neutral guards were fixed for.
#
# Weight 0.5 each: a shaping term next to the pose terms at 1.0, not a competitor. Note the object
# side of the base preset now weighs 2.3 (pose 1.0 + 1.0, C-D lite 0.3), so this block is a ~43%
# increase in object-side mass -- it was ~17% when the removed contact rewards were still there.
object_velocity_reward_terms = {
    "object_global_ref_lin_vel_error_exp": RewardTermCfg(
        func=f"{_REW}object_global_ref_lin_vel_error_exp",
        params={"sigma": 0.5},
        weight=0.5,
    ),
    "object_global_ref_ang_vel_error_exp": RewardTermCfg(
        func=f"{_REW}object_global_ref_ang_vel_error_exp",
        params={"sigma": 0.3},
        weight=0.5,
    ),
}

# Block B -- HDMI CONTACT reward (arXiv:2509.16757). The contact terms already in the base preset are
# purely kinematic: a hand can satisfy them resting a millimetre off the box, carrying nothing. This
# one reads the measured contact force, so it separates a grip that bears load from a pose that only
# looks like one.
#   sigma_pos 0.1 m      Measured right: the reference hand-to-surface distance on contact frames
#                        is 12 mm median / 28 mm p90 over the seven clips that carry it, so a
#                        perfect tracker scores 0.87, a hand 5 cm off 0.61, one 10 cm off 0.37.
#   force_threshold 2 N  The box weighs 0.811 kg = 7.96 N, so a bimanual carry puts ~4 N on each
#                        hand. The threshold has to sit BELOW that or the bonus never fires: at the
#                        10 N this started with, a CORRECT carry scored 1.00 and the term collapsed
#                        to its proximity factor, losing the one thing it was added for -- telling a
#                        hand that bears load from a hand that hovers.
#   sigma_force 4 N      A correct bimanual carry (4 N) then scores 1.65 and 8 N saturates the cap.
# Weight 2.5, set from HDMI's own table rather than picked. HDMI weights Contact 5.0 against Object
# Pose 2.0, a ratio of 2.5. Our term ranges over [0, 2] where theirs is [0, 1] at nominal force and
# unbounded above (the capped force bonus multiplies a [0, 1] proximity), so matching at the CEILING
# -- the conservative reading, since the cap is what protects us -- gives 2*w = 5, w = 2.5. Object
# pose here is 1.0 + 1.0 = 2.0, the same as theirs, so the ratio lands on 2.5 exactly.
#
# The risk, stated: HDMI spends that 5.0 inside a reward that also regulates FOOT contact (Feet Air
# Time 5.0, Feet Impact 1.0, Feet Slip 0.5), none of which exists here. At 2.5 this becomes the
# heaviest term in the reward -- ceiling 5.0, against 5.0 for the whole body-tracking block. What
# holds it in proportion is the gate: 0 off contact, which is ~55% of frames.
object_contact_reward_terms = {
    "object_contact_force_match_exp": RewardTermCfg(
        func=f"{_REW}object_contact_force_match_exp",
        params={
            "sigma_pos": 0.1,
            "sigma_force": 4.0,
            "force_threshold": 2.0,
            "max_force_bonus": 2.0,
        },
        weight=2.5,
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
