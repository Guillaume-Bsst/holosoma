from holosoma.config_values.wbt.g1 import reward
from holosoma.config_values.wbt.g1 import experiment

TERM = "motion_relative_hand_object_position_error_exp"


def test_base_w_object_preset_unchanged():
    # the base training must NOT carry the C-D term
    assert TERM not in reward.g1_29dof_wbt_reward_w_object.terms


def test_actor_preset_has_cd_term_small_weight():
    actor = reward.g1_29dof_wbt_reward_w_object_actor
    assert TERM in actor.terms
    t = actor.terms[TERM]
    assert t.weight == 0.3
    assert t.params["sigma"] == 0.3
    assert t.func == "holosoma.managers.reward.terms.wbt:motion_relative_hand_object_position_error_exp"
    # base left intact inside the actor preset
    assert "motion_relative_body_position_error_exp" in actor.terms


def test_actor_experiment_uses_actor_reward():
    assert experiment.g1_29dof_wbt_w_object_actor.reward is reward.g1_29dof_wbt_reward_w_object_actor
    # base experiment unchanged
    assert experiment.g1_29dof_wbt_w_object.reward is reward.g1_29dof_wbt_reward_w_object
