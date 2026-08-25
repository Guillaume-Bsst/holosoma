"""UndesiredContacts reports which bodies earn the penalty, not just how many.

The term returned a count, so a penalty of ~5.4 per episode said nothing about its source. The
distinction matters: a forearm pressed against a 32 cm box while carrying it is not the same event
as a knee hitting the floor, and only one of them should be discouraged.
"""

import re
from types import SimpleNamespace

import torch
from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.reward.terms.wbt import UndesiredContacts

BODIES = ["pelvis", "left_elbow_link", "left_wrist_yaw_link", "left_knee_link"]
PATTERN = "^(?!left_wrist_yaw_link$).+$"  # everything but the wrist is undesired
N, H = 3, 2


def _term(pattern=PATTERN, threshold=1.0):
    env = SimpleNamespace(
        device="cpu",
        num_envs=N,
        simulator=SimpleNamespace(body_names=BODIES, contact_forces_history=torch.zeros(N, H, len(BODIES), 3)),
        log_dict={},
    )
    cfg = RewardTermCfg(
        func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
        params={"threshold": threshold, "undesired_contacts_body_names": pattern},
    )
    return UndesiredContacts(cfg, env), env


def _push(env, body, envs, force):
    i = BODIES.index(body)
    for e in envs:
        env.simulator.contact_forces_history[e, 0, i, 0] = force


def test_the_excluded_body_is_never_counted_nor_logged():
    term, env = _term()
    _push(env, "left_wrist_yaw_link", range(N), 500.0)
    assert term(env).sum() == 0
    assert "undesired_contacts/left_wrist_yaw_link" not in env.log_dict


def test_each_body_reports_its_own_contact_fraction():
    term, env = _term()
    _push(env, "left_elbow_link", [0, 1], 50.0)   # 2 envs sur 3
    _push(env, "left_knee_link", [2], 50.0)       # 1 env sur 3
    term(env)
    assert env.log_dict["undesired_contacts/left_elbow_link"] == torch.tensor(2 / 3)
    assert env.log_dict["undesired_contacts/left_knee_link"] == torch.tensor(1 / 3)
    assert env.log_dict["undesired_contacts/pelvis"] == torch.tensor(0.0)


def test_the_count_still_matches_the_breakdown():
    # the returned penalty must stay the sum of the per-body flags, not drift from what is logged
    term, env = _term()
    _push(env, "left_elbow_link", [0], 50.0)
    _push(env, "left_knee_link", [0], 50.0)
    counts = term(env)
    assert counts.tolist() == [2, 0, 0]
    logged = sum(env.log_dict[f"undesired_contacts/{b}"] for b in BODIES if b != "left_wrist_yaw_link")
    assert float(logged) * N == counts.sum()


def test_below_threshold_is_not_a_contact():
    term, env = _term(threshold=10.0)
    _push(env, "left_elbow_link", range(N), 5.0)
    assert term(env).sum() == 0
    assert env.log_dict["undesired_contacts/left_elbow_link"] == torch.tensor(0.0)


def test_it_survives_an_env_without_a_log_dict():
    # the breakdown is a diagnostic; it must never be the reason a term raises
    term, env = _term()
    del env.log_dict
    _push(env, "left_elbow_link", [0], 50.0)
    assert term(env).tolist() == [1, 0, 0]


def test_the_pattern_selects_the_same_bodies_the_term_uses():
    term, _ = _term()
    assert term.undesired_contacts_body_names == [b for b in BODIES if re.match(PATTERN, b)]
