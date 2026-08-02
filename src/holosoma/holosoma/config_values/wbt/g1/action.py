"""Whole Body Tracking action presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.action import GripForceCfg
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos

# Object-carry training with real physical grip force instead of the box-physicality curriculum
# (see config_values/wbt/g1/command.py): the box is fully physical from step 0, and each hand
# presses it at grip_force.target_force_n whenever the command term's GT contact flag is active.
#
# 60 N (not 30): with object mass randomized abs [0.8, 1.2] kg and independently-randomized
# box/hand friction (box static/dynamic in [0.1, 0.6], hand in [0.3, 1.6]/[0.3, 1.2]), the worst-case
# combine="average" effective mu is ~0.20 -- 2 * 0.20 * 30N = 12.0N vs a max box weight of
# 1.2*9.81 = 11.77N is only a ~2% margin. 60N/hand gives 2 * 0.20 * 60N = 24N, a ~2x margin even in
# that worst-case draw (still within the wrist torque limits: ~3 N.m at 60N given the ~0.05m lever
# arm, under the 5 N.m pitch/yaw and 25 N.m roll effort limits).
grip_force_cfg = GripForceCfg(
    enable=True,
    target_force_n=60.0,
    force_command_max_n=90.0,
    # Recale sur la main HALF-SPHERE : origine de son hand_palm_joint dans le repere poignet.
    # Le defaut (0.0415, 0.003, 0) est celui de la main RUBBER -- c'est la valeur qu'utilisait la
    # run de reference c4k7xrin, qui tournait pourtant en half-sphere : elle appliquait donc sa
    # force 1.25 cm devant la paume reelle, soit ~25% d'erreur sur le bras de levier du couple
    # poignet. Corrige ici.
    hand_offset_local=(0.029, -0.003, 0.0),
)

g1_29dof_joint_pos_grip_force = replace(
    g1_29dof_joint_pos,
    terms={
        **g1_29dof_joint_pos.terms,
        "joint_control": replace(
            g1_29dof_joint_pos.terms["joint_control"],
            params={"grip_force": grip_force_cfg},
        ),
    },
)

__all__ = ["g1_29dof_joint_pos_grip_force"]
