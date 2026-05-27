"""Tests for WBT observation presets."""

from holosoma.config_values.observation import DEFAULTS
from holosoma.config_values.wbt.g1.observation import (
    actor_obs_shared,
    actor_obs_w_object,
    g1_29dof_wbt_observation_w_object_actor,
)


def test_actor_obs_w_object_contains_shared_terms():
    """actor_obs_w_object must include every term from actor_obs_shared."""
    for key in actor_obs_shared.terms:
        assert key in actor_obs_w_object.terms, (
            f"Expected term '{key}' from actor_obs_shared in actor_obs_w_object"
        )


def test_actor_obs_w_object_has_object_terms():
    """actor_obs_w_object must add obj_pos_b and obj_ori_b."""
    assert "obj_pos_b" in actor_obs_w_object.terms
    assert "obj_ori_b" in actor_obs_w_object.terms


def test_actor_obs_w_object_term_funcs():
    """obj_pos_b / obj_ori_b must point to the correct term functions."""
    assert actor_obs_w_object.terms["obj_pos_b"].func == (
        "holosoma.managers.observation.terms.wbt:obj_pos_b"
    )
    assert actor_obs_w_object.terms["obj_ori_b"].func == (
        "holosoma.managers.observation.terms.wbt:obj_ori_b"
    )


def test_actor_obs_w_object_noise_disabled_for_object_terms():
    """Object terms use noise=0.0 (matched to critic convention)."""
    assert actor_obs_w_object.terms["obj_pos_b"].noise == 0.0
    assert actor_obs_w_object.terms["obj_ori_b"].noise == 0.0


def test_preset_has_actor_and_critic_groups():
    """g1_29dof_wbt_observation_w_object_actor exposes actor_obs and critic_obs."""
    assert "actor_obs" in g1_29dof_wbt_observation_w_object_actor.groups
    assert "critic_obs" in g1_29dof_wbt_observation_w_object_actor.groups


def test_preset_actor_group_is_actor_obs_w_object():
    """Preset actor group is actor_obs_w_object (same object, not a copy)."""
    assert (
        g1_29dof_wbt_observation_w_object_actor.groups["actor_obs"]
        is actor_obs_w_object
    )


def test_defaults_registry_contains_new_presets():
    """New actor-with-object presets are accessible by name from DEFAULTS."""
    assert "g1_29dof_wbt_w_object_actor" in DEFAULTS
    assert "g1_27dof_wbt_w_object_actor" in DEFAULTS


def test_defaults_registry_new_presets_resolve_to_correct_cfg():
    """Both G1 DOF variants resolve to the same preset object."""
    assert DEFAULTS["g1_29dof_wbt_w_object_actor"] is g1_29dof_wbt_observation_w_object_actor
    assert DEFAULTS["g1_27dof_wbt_w_object_actor"] is g1_29dof_wbt_observation_w_object_actor
