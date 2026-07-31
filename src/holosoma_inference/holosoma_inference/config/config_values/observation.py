"""Default observation configurations for holosoma_inference.

This module provides pre-configured observation spaces for different
robot types and tasks, converted from the original YAML configurations.
"""

from __future__ import annotations

from dataclasses import replace

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_types.observation import ObservationConfig

# =============================================================================
# Locomotion Observation Configurations
# =============================================================================

loco_g1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

loco_t1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 1.0,  # T1 uses 1.0 (vs G1's 2.0)
        "base_ang_vel": 1.0,  # T1 uses 1.0 (vs G1's 0.25)
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.1,  # T1 uses 0.1 (vs G1's 0.05)
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)


# =============================================================================
# WBT (Whole Body Tracking) Observation Configurations
# =============================================================================

wbt = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

wbt_g1_27dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
        ]
    },
    obs_dims={
        "motion_command": 54,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 27,
        "dof_vel": 27,
        "actions": 27,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# Object-carry variant: same actor_obs as `wbt` plus the box pose relative to the reference root
# (obj_pos_b 3 + obj_ori_b 6 = 9). Terms are assembled in ALPHABETICAL order (see base.py
# obs_terms_sorted), matching the training export, so obj_ori_b/obj_pos_b land at the end after the
# motion_* terms -> total 154 + 9 = 163. Used by the g1-29dof-wbt-w-object inference config.
wbt_w_object = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "obj_pos_b",
            "obj_ori_b",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "obj_pos_b": 3,
        "obj_ori_b": 6,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
        "obj_pos_b": 1.0,
        "obj_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# Object-carry + static support table variant: same as `wbt_w_object` (163) plus the support
# table's pose relative to the reference root (support_pos_b 3 + support_ori_b 6 = 9) -> 172.
# Alphabetical assembly (see base.py obs_terms_sorted) places support_* after obj_*. Used by
# g1-29dof-wbt-w-object-support; requires a clip NPZ with support_pos_w/support_quat_w (see
# WholeBodyTrackingPolicy._load_object_motion).
wbt_w_object_support = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "motion_command",
            "motion_ref_ori_b",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "obj_pos_b",
            "obj_ori_b",
            "support_pos_b",
            "support_ori_b",
        ]
    },
    obs_dims={
        "motion_command": 58,
        "motion_ref_pos_b": 3,
        "motion_ref_ori_b": 6,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "obj_pos_b": 3,
        "obj_ori_b": 6,
        "support_pos_b": 3,
        "support_ori_b": 6,
    },
    obs_scales={
        "actions": 1.0,
        "motion_command": 1.0,
        "motion_ref_pos_b": 1.0,
        "motion_ref_ori_b": 1.0,
        "base_lin_vel": 1.0,
        "base_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 1.0,
        "robot_body_pos_b": 1.0,
        "robot_body_ori_b": 1.0,
        "obj_pos_b": 1.0,
        "obj_ori_b": 1.0,
        "support_pos_b": 1.0,
        "support_ori_b": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)

# Same terms as wbt_w_object_support, but actor_obs stacks 3 frames of history instead of 1 --
# matches checkpoints trained with observation.groups.actor_obs.history_length=3 (e.g. run
# bzwhv8kk). Total actor_obs dim = 3 * 172 = 516.
wbt_w_object_support_h3 = replace(
    wbt_w_object_support,
    history_length_dict={
        "actor_obs": 3,
    },
)

# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-29dof": loco_t1_29dof,
    "wbt": wbt,
    "wbt-g1-27dof": wbt_g1_27dof,
    "wbt-w-object": wbt_w_object,
}
"""Dictionary of all available observation configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""

# Auto-discover observation configs from installed extensions
for ep in entry_points(group="holosoma.config.observation"):
    DEFAULTS[ep.name] = ep.load()
