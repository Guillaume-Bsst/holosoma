"""The object half of the bad-tracking termination can be switched off from the CLI.

A dropped or mistracked box then costs reward instead of killing the episode, which is what you
want when the question is whether a reward signal is learnable rather than whether the policy can
already hold the box.
"""

from types import SimpleNamespace

import torch
from holosoma.config_types.termination import TerminationTermCfg
from holosoma.config_values import experiment as experiment_registry
from holosoma.config_values.wbt.g1 import termination
from holosoma.managers.termination.terms.wbt import BadTracking

PARAM = "enable_object_termination"
BODIES = ["pelvis", "torso_link"]
N = 2


def _term(enabled):
    cfg = TerminationTermCfg(
        func="holosoma.managers.termination.terms.wbt:BadTracking",
        params={
            "bad_ref_pos_threshold": 100.0,          # robot checks slack: only the object matters here
            "bad_ref_ori_threshold": 100.0,
            "bad_motion_body_pos_threshold": 100.0,
            "body_names_to_track": BODIES,
            "bad_motion_body_pos_body_names": ["pelvis"],
            "bad_object_pos_threshold": 0.15,
            "bad_object_ori_threshold": 0.8,
            PARAM: enabled,
        },
    )
    env = SimpleNamespace(device="cpu", num_envs=N, up_axis_idx=2)
    return BadTracking(cfg, env)


def _motion_command(object_drift):
    """A box drifted `object_drift` metres from its reference; the robot tracks perfectly."""
    zeros, quat = torch.zeros(N, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1)
    drift = torch.zeros(N, 3)
    drift[:, 0] = object_drift
    return SimpleNamespace(
        motion_cfg=SimpleNamespace(body_names_to_track=BODIES),
        motion=SimpleNamespace(has_object=True),
        ref_pos_w=zeros, robot_ref_pos_w=zeros,
        ref_quat_w=quat, robot_ref_quat_w=quat,
        body_pos_relative_w=torch.zeros(N, len(BODIES), 3),
        robot_body_pos_w=torch.zeros(N, len(BODIES), 3),
        object_pos_w=zeros, simulator_object_pos_w=drift,
        object_quat_w=quat, simulator_object_quat_w=quat,
    )


def test_object_drift_kills_the_episode_when_enabled():
    term = _term(True)
    term.env.command_manager = SimpleNamespace(get_state=lambda _: _motion_command(0.5))
    assert term(term.env).all()


def test_object_drift_is_survivable_when_disabled():
    term = _term(False)
    term.env.command_manager = SimpleNamespace(get_state=lambda _: _motion_command(0.5))
    assert not term(term.env).any()


def test_disabling_does_not_touch_the_robot_checks():
    term = _term(False)
    mc = _motion_command(0.0)
    mc.robot_ref_pos_w = torch.full((N, 3), 999.0)   # robot wildly off its reference
    term.bad_ref_pos_threshold = 0.5
    term.env.command_manager = SimpleNamespace(get_state=lambda _: mc)
    assert term(term.env).all(), "the switch must only gate the OBJECT half"


def test_within_threshold_never_terminates():
    for enabled in (True, False):
        term = _term(enabled)
        term.env.command_manager = SimpleNamespace(get_state=lambda _: _motion_command(0.05))
        assert not term(term.env).any()


#########################################################################################
## config surface
#########################################################################################
def test_preset_declares_the_parameter_so_the_cli_exposes_it():
    # declared in the preset, not merely read with a default in the term: tyro builds the flag from
    # the default config, so a key absent there cannot be set on the command line
    params = termination.g1_29dof_wbt_termination.terms["bad_tracking"].params
    assert params[PARAM] is True


def test_every_wbt_experiment_carries_it():
    wbt = {k: v for k, v in experiment_registry.DEFAULTS.items() if v is not None and "wbt" in k}
    assert wbt
    for key, exp in wbt.items():
        assert PARAM in exp.termination.terms["bad_tracking"].params, key
