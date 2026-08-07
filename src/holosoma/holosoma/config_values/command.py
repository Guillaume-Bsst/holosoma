"""Default command manager configurations."""

from holosoma.config_values.loco.g1.command import g1_29dof_command
from holosoma.config_values.loco.t1.command import t1_29dof_command
from holosoma.config_values.wbt.g1.command import (
    g1_27dof_wbt_command,
    g1_27dof_wbt_command_w_object,
    g1_29dof_wbt_command,
    g1_29dof_wbt_command_w_object,
    g1_29dof_wbt_command_w_object_femto14_box36,
    g1_29dof_wbt_command_w_object_femto14_box36_dyn,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_command,
    "g1_29dof": g1_29dof_command,
    "g1_27dof": g1_29dof_command,
    "g1_29dof_wbt": g1_29dof_wbt_command,
    "g1_27dof_wbt": g1_27dof_wbt_command,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_command_w_object,
    "g1_27dof_wbt_w_object": g1_27dof_wbt_command_w_object,
}
