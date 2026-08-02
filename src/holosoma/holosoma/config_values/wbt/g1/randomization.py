"""Whole Body Tracking randomization presets for the G1 robot."""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

robot_state_dr_at_setup = {
    "randomize_robot_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_robot_rigid_body_material_startup",
        params={
            "static_friction_range": [0.3, 1.6],
            "dynamic_friction_range": [0.3, 1.2],
            "restitution_range": [0.0, 0.5],
        },
    ),
    "randomize_base_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
        params={
            "base_com_range": {"x": [-0.1, 0.1], "y": [-0.1, 0.1], "z": [-0.1, 0.1]},
            "enabled": True,
        },
    ),
    "setup_dof_pos_bias": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
        params={
            "dof_pos_bias_range": [-0.025, 0.025],
            "enabled": True,
        },
    ),
}

object_state_dr_at_setup = {
    "randomize_object_rigid_body_material_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_material_startup",
        params={
            "static_friction_range": [0.1, 0.6],
            "dynamic_friction_range": [0.1, 0.6],
            "restitution_range": [0.0, 1.0],
        },
    ),
    "randomize_object_rigid_body_mass_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_mass_startup",
        params={
            # ADD to the 0.811 kg URDF base -> 0.81-1.21 kg (mean ~1.0), la vraie box36 pese ~1 kg.
            # Ancien [1.0, 4.0] => 1.8-4.8 kg : une queue lourde IRREELLE dont le poids residuel
            # (1-alpha)*m*g depassait la capacite de prise -> les boites lourdes derivaient dans la
            # bande morte [0.10, 0.25] m (survie OK, tracking KO), clouant obj_track_success a ~0.83
            # sous le gate 0.85 -> curriculum d'alpha gele. Cf. run vcbkx2mm.
            "mass_distribution_params": [0.0, 0.4],
        },
    ),
    "randomize_object_com_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_com_startup",
        params={
            # Une vraie caisse a un contenu : son COM n'est pas au centre geometrique. Le decalage
            # cree un couple permanent autour de la prise que la policy doit compenser -- effet du
            # premier ordre pour du portage. +-4 cm sur un cube de 0.36 m (demi-cote 0.18) = +-22%
            # du demi-cote : sensible mais pas absurde pour une caisse mal remplie. Pendant objet du
            # randomize_base_com_startup du robot.
            "com_range": {"x": [-0.04, 0.04], "y": [-0.04, 0.04], "z": [-0.04, 0.04]},
            "enabled": True,
        },
    ),
    "randomize_object_rigid_body_inertia_startup": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_object_rigid_body_inertia_startup",
        params={
            "inertia_distribution_params_dict": {
                # In beyondmimic, only Ixx is randomized, which is probably a bug instead of a feature.
                # Here, we want to reproduce their work. User should feel free to randomize all terms.
                "Ixx": [0.5, 1.5],
                "Iyy": [1.0, 1.0],
                "Izz": [1.0, 1.0],
                "Ixy": [1.0, 1.0],
                "Iyz": [1.0, 1.0],
                "Ixz": [1.0, 1.0],
            }
        },
    ),
}

base_setup_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
        params={
            "push_interval_s": [1.0, 3.0],
            "max_push_vel": [0.75, 0.75, 0.3, 0.52, 0.52, 0.78],
            "enabled": True,
        },
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
        params={
            "kp_range": [0.9, 1.1],
            "kd_range": [0.9, 1.1],
            "rfi_lim_range": [1.0, 1.0],
            "enable_pd_gain": True,
            "enable_rfi_lim": False,
        },
    ),
    "setup_action_delay_buffers": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
        params={
            "ctrl_delay_step_range": [0, 1],
            "enabled": True,
        },
    ),
    **robot_state_dr_at_setup,
}

base_reset_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "randomize_push_schedule": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
    ),
    "randomize_action_delay": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
    ),
    "actuator_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
    ),
    "randomize_dof_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
        params={
            "joint_pos_scale_range": [1.0, 1.0],
            "joint_vel_range": [0.0, 0.0],
            "joint_pos_bias_range": [-0.025, 0.025],
            "randomize_dof_pos_bias": True,
        },
    ),
}

base_step_terms = {
    "push_randomizer_state": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
    ),
    "apply_pushes": RandomizationTermCfg(
        func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
    ),
}

g1_29dof_wbt_randomization = RandomizationManagerCfg(
    setup_terms={**base_setup_terms},
    reset_terms={**base_reset_terms},
    step_terms={**base_step_terms},
)

g1_29dof_wbt_randomization_w_object = RandomizationManagerCfg(
    setup_terms={
        **base_setup_terms,
        **object_state_dr_at_setup,
    },
    reset_terms={
        **base_reset_terms,
    },
    step_terms={
        **base_step_terms,
    },
)

__all__ = ["g1_29dof_wbt_randomization", "g1_29dof_wbt_randomization_w_object"]
