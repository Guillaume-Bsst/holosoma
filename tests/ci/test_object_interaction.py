"""Unit tests for the pure math behind the optional object-interaction reward terms."""

import math

import torch

from holosoma.utils.object_interaction import hdmi_contact_reward, velocity_tracking_reward

CONTACT_KW = dict(sigma_pos=0.1, sigma_force=20.0, force_threshold=10.0, max_force_bonus=2.0)


def _contact(distance, force, gate, **overrides):
    kw = {**CONTACT_KW, **overrides}
    return hdmi_contact_reward(
        torch.tensor(distance), torch.tensor(force), torch.tensor(gate), **kw
    )


#########################################################################################
## velocity_tracking_reward
#########################################################################################
def test_velocity_reward_is_one_at_zero_error():
    v = torch.tensor([[0.3, -0.1, 0.05]])
    assert torch.allclose(velocity_tracking_reward(v, v.clone(), sigma=1.0), torch.ones(1), atol=1e-6)


def test_velocity_reward_decreases_with_error():
    v_ref = torch.zeros(1, 3)
    small = velocity_tracking_reward(torch.full((1, 3), 0.05), v_ref, sigma=1.0)
    large = velocity_tracking_reward(torch.full((1, 3), 0.5), v_ref, sigma=1.0)
    assert small > large
    assert 0.0 < large < 1.0


def test_velocity_reward_matches_closed_form():
    # sigma == ||dv|| -> exp(-1)
    v_ref = torch.zeros(1, 3)
    v_sim = torch.tensor([[0.6, 0.8, 0.0]])  # norm 1.0
    got = velocity_tracking_reward(v_ref, v_sim, sigma=1.0)
    assert torch.allclose(got, torch.tensor([math.exp(-1.0)]), atol=1e-6)


def test_velocity_reward_is_batched_over_envs():
    v_ref = torch.zeros(4, 3)
    v_sim = torch.zeros(4, 3)
    v_sim[2] = 1.0
    r = velocity_tracking_reward(v_ref, v_sim, sigma=1.0)
    assert r.shape == (4,)
    assert torch.allclose(r[[0, 1, 3]], torch.ones(3), atol=1e-6)
    assert r[2] < 1.0


#########################################################################################
## hdmi_contact_reward
#########################################################################################
def test_contact_reward_is_zero_off_gate():
    # zero, NOT one: an additive term must not pay a constant on non-contact frames
    r = _contact([0.0], [500.0], [False])
    assert torch.allclose(r, torch.zeros(1), atol=1e-6)


def test_contact_force_factor_is_flat_below_threshold():
    # HDMI uses max(exp(.), 1): below F_thres the factor saturates at 1, it does NOT penalise
    no_force = _contact([0.0], [0.0], [True])
    half = _contact([0.0], [5.0], [True])
    at_threshold = _contact([0.0], [10.0], [True])
    assert torch.allclose(no_force, at_threshold, atol=1e-6)
    assert torch.allclose(half, at_threshold, atol=1e-6)
    assert torch.allclose(at_threshold, torch.ones(1), atol=1e-6)


def test_contact_force_factor_grows_above_threshold():
    at_threshold = _contact([0.0], [10.0], [True])
    above = _contact([0.0], [25.0], [True])
    assert above > at_threshold


def test_contact_force_factor_is_capped():
    # the paper's factor diverges with ||F||; the cap keeps a collision spike from swamping the run
    spike = _contact([0.0], [2400.0], [True])
    assert torch.allclose(spike, torch.tensor([CONTACT_KW["max_force_bonus"]]), atol=1e-6)


def test_contact_reward_decreases_with_distance():
    near = _contact([0.0], [10.0], [True])
    far = _contact([0.3], [10.0], [True])
    assert near > far
    assert 0.0 < far < 1.0


def test_contact_reward_matches_closed_form():
    # distance == sigma_pos, force == threshold -> exp(-1) * 1
    r = _contact([0.1], [10.0], [True])
    assert torch.allclose(r, torch.tensor([math.exp(-1.0)]), atol=1e-6)


def test_contact_reward_gate_is_per_env():
    r = _contact([0.0, 0.0], [10.0, 10.0], [True, False])
    assert torch.allclose(r, torch.tensor([1.0, 0.0]), atol=1e-6)
