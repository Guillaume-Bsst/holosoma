"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

# NOTE: history_length=3 sur l'ACTEUR (le critic reste a 1). L'obs acteur est empilee sur 3 frames
# -> dim x3 (172 -> 516 pour le groupe w_object). Donne au policy une notion de vitesse/derive de la
# box et du corps que l'obs instantanee ne porte pas (obj_pos_b seul est ambigu entre "posee" et "en
# train de glisser"). Cout: retrain FROM SCRATCH (les checkpoints h1 ne se chargent plus) + cote
# inference il faut le preset h3 (holosoma_inference .. config_values/observation.py:
# wbt_w_object_support_h3).
actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=3,
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
    history_length=3,  # idem actor_obs_shared -- voir la note la-haut
    terms={
        **actor_obs_shared.terms,
        # Object pose is available at deployment via mocap/RGB-D perception; add measurement-level
        # noise so the policy is robust to that pipeline (~2 cm position, ~0.05 orientation).
        #
        # The position goes through PerceptionNoisyPosition, which adds the CORRELATED errors a real
        # estimator makes on top of that white noise: a per-episode constant bias (calibration /
        # mesh origin) and a per-episode latency (FoundationPose runs below the 50 Hz control rate).
        # White noise alone is exactly what the policy averages away -- and actor_obs now stacks 3
        # frames, which makes averaging easier -- while a constant bias never averages out.
        # NOTE: the term keeps the name "obj_pos_b" on purpose. Groups are concatenated in
        # ALPHABETICAL term order (ObservationManager.compute_group), so renaming it would reshuffle
        # the observation vector and desync every inference preset. Only `func` changes.
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:PerceptionNoisyPosition",
            params={
                "source": "holosoma.managers.observation.terms.wbt:obj_pos_b",
                "bias_range": 0.03,
                "latency_step_range": (0, 3),
                # Occlusion dropout is OFF: freezing the box pose while the hands close on it
                # changes the task, not just the sensor. Worth its own A/B, not this run.
                "dropout_prob": 0.0,
            },
            scale=1.0,
            noise=0.02,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        # Table (support statique) : le robot la VOIT pour se placer correctement (dépôt) et ne pas
        # foncer dedans. Pose relative au torse, avec le même bruit de perception que la box.
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

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_w_object_actor",
]
