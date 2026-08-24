"""Composition of the optional object-training presets: objvel, objcontact, and both.

The two blocks must be independent (either one alone, or both) and must leave every pre-existing
preset bit-identical, so a run carrying one feature can be compared against the base.
"""

import pytest
from holosoma.config_values import experiment as experiment_registry
from holosoma.config_values import observation as observation_registry
from holosoma.config_values import reward as reward_registry
from holosoma.config_values.wbt.g1 import experiment, observation, reward

VEL_TERMS = ["object_global_ref_lin_vel_error_exp", "object_global_ref_ang_vel_error_exp"]
CONTACT_TERMS = ["object_contact_force_match_exp"]
SUFFIXES = ["objvel", "objcontact", "objvel_objcontact"]

BASE_REWARDS = [reward.g1_29dof_wbt_reward_w_object, reward.g1_29dof_wbt_reward_w_object_actor]


def _critic(cfg):
    return cfg.observation.groups["critic_obs"].terms


#########################################################################################
## the base presets stay untouched
#########################################################################################
@pytest.mark.parametrize("base", BASE_REWARDS)
def test_base_reward_presets_carry_no_new_term(base):
    for term in VEL_TERMS + CONTACT_TERMS:
        assert term not in base.terms


def test_base_observation_presets_carry_no_new_term():
    for cfg in [observation.g1_29dof_wbt_observation_w_object, observation.g1_29dof_wbt_observation_w_object_actor]:
        critic = cfg.groups["critic_obs"].terms
        assert "obj_ang_vel_b" not in critic
        assert "obj_contact_flag" not in critic
        # and the inherited (uncorrected) linear velocity is left exactly as it was
        assert critic["obj_lin_vel_b"].func.endswith(":obj_lin_vel_b")


def test_base_experiments_are_unchanged_objects():
    assert experiment.g1_29dof_wbt_w_object_actor.reward is reward.g1_29dof_wbt_reward_w_object_actor
    assert experiment.g1_27dof_wbt_w_object_actor.reward is reward.g1_29dof_wbt_reward_w_object
    for cfg in [experiment.g1_29dof_wbt_w_object_actor, experiment.g1_27dof_wbt_w_object_actor]:
        assert cfg.observation is observation.g1_29dof_wbt_observation_w_object_actor


#########################################################################################
## each block adds exactly its own terms
#########################################################################################
@pytest.mark.parametrize("base", BASE_REWARDS)
def test_reward_blocks_are_additive_and_independent(base):
    name = "g1_29dof_wbt_reward_w_object_actor" if base is BASE_REWARDS[1] else "g1_29dof_wbt_reward_w_object"
    vel = getattr(reward, name + "_objvel").terms
    contact = getattr(reward, name + "_objcontact").terms
    both = getattr(reward, name + "_objvel_objcontact").terms

    assert set(vel) - set(base.terms) == set(VEL_TERMS)
    assert set(contact) - set(base.terms) == set(CONTACT_TERMS)
    assert set(both) - set(base.terms) == set(VEL_TERMS + CONTACT_TERMS)
    # blocks do not leak into each other
    assert not set(CONTACT_TERMS) & set(vel)
    assert not set(VEL_TERMS) & set(contact)
    # base terms are carried through untouched
    for term, cfg in base.terms.items():
        assert both[term] is cfg


def test_velocity_reward_weights_and_sigmas():
    terms = reward.g1_29dof_wbt_reward_w_object_actor_objvel.terms
    lin, ang = terms[VEL_TERMS[0]], terms[VEL_TERMS[1]]
    assert (lin.weight, lin.params["sigma"]) == (0.5, 1.0)
    assert (ang.weight, ang.params["sigma"]) == (0.5, 3.14)
    assert lin.func.endswith(":object_global_ref_lin_vel_error_exp")
    assert ang.func.endswith(":object_global_ref_ang_vel_error_exp")


def test_contact_reward_weight_and_params():
    t = reward.g1_29dof_wbt_reward_w_object_actor_objcontact.terms[CONTACT_TERMS[0]]
    assert t.weight == 1.0
    assert t.params == {"sigma_pos": 0.1, "sigma_force": 20.0, "force_threshold": 10.0, "max_force_bonus": 2.0}
    # the reward threshold means "bearing load", well above the obs "touching at all" threshold
    assert t.params["force_threshold"] > observation.critic_obs_object_contact_terms[
        "obj_contact_flag"
    ].params["force_threshold"]


#########################################################################################
## observations: critic only, and the linear-velocity fix is scoped to the velocity block
#########################################################################################
@pytest.mark.parametrize("suffix", SUFFIXES)
def test_new_observations_never_reach_the_actor(suffix):
    cfg = getattr(observation, "g1_29dof_wbt_observation_w_object_actor_" + suffix)
    actor = cfg.groups["actor_obs"]
    assert actor is observation.actor_obs_w_object
    assert "obj_ang_vel_b" not in actor.terms
    assert "obj_contact_flag" not in actor.terms
    assert "obj_lin_vel_b" not in actor.terms


def test_velocity_block_swaps_in_the_rotation_only_linear_velocity():
    critic = observation.g1_29dof_wbt_observation_w_object_actor_objvel.groups["critic_obs"].terms
    # same term NAME (so the critic vector keeps its width and alphabetical slot)...
    assert "obj_lin_vel_b" in critic
    # ...but the corrected implementation
    assert critic["obj_lin_vel_b"].func.endswith(":obj_lin_vel_b_rotated")
    assert critic["obj_ang_vel_b"].func.endswith(":obj_ang_vel_b")


def test_contact_block_leaves_the_linear_velocity_alone():
    # the fix is scoped to the velocity block: a contact-only run keeps the inherited input
    critic = observation.g1_29dof_wbt_observation_w_object_actor_objcontact.groups["critic_obs"].terms
    assert critic["obj_lin_vel_b"].func.endswith(":obj_lin_vel_b")
    assert "obj_ang_vel_b" not in critic
    assert "obj_contact_flag" in critic


def test_observation_blocks_are_noise_free():
    # critic group has enable_noise=False anyway; keep the terms declaring 0 so it stays true if
    # the group is ever reused with noise on
    for block in [observation.critic_obs_object_velocity_terms, observation.critic_obs_object_contact_terms]:
        for term in block.values():
            assert term.noise == 0.0


#########################################################################################
## experiments: all six registered, each dof keeping its own reward base
#########################################################################################
@pytest.mark.parametrize("suffix", SUFFIXES)
def test_experiments_registered_for_both_robots(suffix):
    for dof in ["29", "27"]:
        key = f"g1_{dof}dof_wbt_w_object_actor_{suffix}"
        assert key in experiment_registry.DEFAULTS
        assert key in observation_registry.DEFAULTS
        assert key in reward_registry.DEFAULTS
        assert experiment_registry.DEFAULTS[key] is getattr(experiment, key)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_each_robot_keeps_its_own_reward_base(suffix):
    # 29dof carries the actor (C-D lite) reward, 27dof the plain w_object one -- as the base
    # experiments already do; the feature blocks must not silently change that
    e29 = getattr(experiment, "g1_29dof_wbt_w_object_actor_" + suffix)
    e27 = getattr(experiment, "g1_27dof_wbt_w_object_actor_" + suffix)
    assert e29.reward is getattr(reward, "g1_29dof_wbt_reward_w_object_actor_" + suffix)
    assert e27.reward is getattr(reward, "g1_29dof_wbt_reward_w_object_" + suffix)
    assert "motion_relative_hand_object_position_error_exp" in e29.reward.terms
    assert "motion_relative_hand_object_position_error_exp" not in e27.reward.terms


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_experiments_share_the_matching_observation(suffix):
    obs = getattr(observation, "g1_29dof_wbt_observation_w_object_actor_" + suffix)
    for dof in ["29", "27"]:
        assert getattr(experiment, f"g1_{dof}dof_wbt_w_object_actor_{suffix}").observation is obs


def test_both_block_experiment_is_the_union():
    both = _critic(experiment.g1_29dof_wbt_w_object_actor_objvel_objcontact)
    vel = _critic(experiment.g1_29dof_wbt_w_object_actor_objvel)
    contact = _critic(experiment.g1_29dof_wbt_w_object_actor_objcontact)
    assert set(both) == set(vel) | set(contact)
