"""Default observation manager configurations."""

from holosoma.config_values.loco.g1.observation import g1_29dof_loco_single_wolinvel
from holosoma.config_values.loco.t1.observation import t1_29dof_loco_single_wolinvel
from holosoma.config_values.wbt.g1.observation import (
    g1_29dof_wbt_observation,
    g1_29dof_wbt_observation_w_object,
    g1_29dof_wbt_observation_w_object_actor,
    g1_29dof_wbt_observation_w_object_actor_objcontact,
    g1_29dof_wbt_observation_w_object_actor_objvel,
    g1_29dof_wbt_observation_w_object_actor_objvel_objcontact,
    g1_29dof_wbt_observation_w_object_objvel_objcontact,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof_loco_single_wolinvel": t1_29dof_loco_single_wolinvel,
    "g1_29dof_loco_single_wolinvel": g1_29dof_loco_single_wolinvel,
    "g1_27dof_loco_single_wolinvel": g1_29dof_loco_single_wolinvel,
    "g1_29dof_wbt": g1_29dof_wbt_observation,
    "g1_27dof_wbt": g1_29dof_wbt_observation,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_observation_w_object,
    "g1_27dof_wbt_w_object": g1_29dof_wbt_observation_w_object,
    "g1_29dof_wbt_w_object_actor": g1_29dof_wbt_observation_w_object_actor,
    "g1_27dof_wbt_w_object_actor": g1_29dof_wbt_observation_w_object_actor,
    "g1_29dof_wbt_w_object_actor_objvel": g1_29dof_wbt_observation_w_object_actor_objvel,
    "g1_27dof_wbt_w_object_actor_objvel": g1_29dof_wbt_observation_w_object_actor_objvel,
    "g1_29dof_wbt_w_object_actor_objcontact": g1_29dof_wbt_observation_w_object_actor_objcontact,
    "g1_27dof_wbt_w_object_actor_objcontact": g1_29dof_wbt_observation_w_object_actor_objcontact,
    "g1_29dof_wbt_w_object_actor_objvel_objcontact": g1_29dof_wbt_observation_w_object_actor_objvel_objcontact,
    "g1_27dof_wbt_w_object_actor_objvel_objcontact": g1_29dof_wbt_observation_w_object_actor_objvel_objcontact,
    "g1_29dof_wbt_w_object_objvel_objcontact": g1_29dof_wbt_observation_w_object_objvel_objcontact,
}
