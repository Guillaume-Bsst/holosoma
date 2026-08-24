"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

actor_obs_w_object = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        **actor_obs_shared.terms,
        # Object pose is available at deployment via mocap/RGB-D perception; add measurement-level
        # noise so the policy is robust to that pipeline (~2 cm position, ~0.05 orientation).
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.02,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        # Table (static support): the robot SEES it, so it can place itself correctly (drop-off)
        # and not barge into it. Pose relative to the torso, with the same perception noise as the
        # box.
        "support_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_pos_b",
            scale=1.0,
            noise=0.02,
        ),
        "support_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_ori_b",
            scale=1.0,
            noise=0.05,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
        "support_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "support_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:support_ori_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_w_object,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

# ================================================================================================
# Optional object-training feature blocks (critic-only)
# ================================================================================================
# The critic is never deployed -- it exists to estimate the value during training -- so anything
# that reduces the variance of that estimate is free with respect to the real robot. That is the
# same asymmetry that already gives it base_lin_vel / obj_lin_vel_b / robot_body_pos_b.
#
# NONE of these may be added to the actor group: measured contact forces do not exist on hardware
# (no force sensing at the G1 wrists), and object velocity is not something the perception pipeline
# delivers cleanly.
_OBS = "holosoma.managers.observation.terms.wbt:"

# Block A -- object VELOCITY. Pairs with object_velocity_reward_terms: the critic sees the quantity
# those rewards are computed from.
critic_obs_object_velocity_terms = {
    # Same term NAME as the inherited one, so the critic vector keeps its width and its alphabetical
    # slot, but pointed at the rotation-only implementation. The inherited obj_lin_vel_b passes a
    # velocity to subtract_frame_transforms, which also subtracts the reference body's world
    # position: it returns R^T (v_obj - p_torso), a ~1.8 m/s offset on a ~0.3 m/s carry that drifts
    # with the robot's position -- a motionless box reads as moving. Scoped to these presets rather
    # than fixed in place so existing w_object runs keep the input they were trained on.
    "obj_lin_vel_b": ObsTermCfg(
        func=f"{_OBS}obj_lin_vel_b_rotated",
        scale=1.0,
        noise=0.0,
    ),
    "obj_ang_vel_b": ObsTermCfg(
        func=f"{_OBS}obj_ang_vel_b",
        scale=1.0,
        noise=0.0,
    ),
}

# Block B -- CONTACT. Pairs with object_contact_reward_terms: what the hands actually bear, next to
# what the reference prescribes. 1 N threshold = touching at all (the reward uses a much higher
# threshold, 10 N, to mean bearing load).
critic_obs_object_contact_terms = {
    "obj_contact_flag": ObsTermCfg(
        func=f"{_OBS}obj_contact_flag",
        params={"force_threshold": 1.0},
        scale=1.0,
        noise=0.0,
    ),
}


def _w_object_actor_observation(*critic_blocks: dict) -> ObservationManagerCfg:
    """Actor sees the noisy object pose; critic gets the privileged set plus the opted-in blocks."""
    critic_terms = critic_obs_w_object_terms.copy()
    for block in critic_blocks:
        critic_terms.update(block)
    return ObservationManagerCfg(
        groups={
            "actor_obs": actor_obs_w_object,
            "critic_obs": ObsGroupCfg(
                concatenate=True,
                enable_noise=False,
                history_length=1,
                terms=critic_terms,
            ),
        },
    )


g1_29dof_wbt_observation_w_object_actor_objvel = _w_object_actor_observation(critic_obs_object_velocity_terms)
g1_29dof_wbt_observation_w_object_actor_objcontact = _w_object_actor_observation(critic_obs_object_contact_terms)
g1_29dof_wbt_observation_w_object_actor_objvel_objcontact = _w_object_actor_observation(
    critic_obs_object_velocity_terms, critic_obs_object_contact_terms
)

__all__ = [
    "critic_obs_object_contact_terms",
    "critic_obs_object_velocity_terms",
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_actor",
    "g1_29dof_wbt_observation_w_object_actor_objvel",
    "g1_29dof_wbt_observation_w_object_actor_objcontact",
    "g1_29dof_wbt_observation_w_object_actor_objvel_objcontact",
]
