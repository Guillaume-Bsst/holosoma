import torch

from holosoma.utils.contact_targets import (
    beta_from_distance,
    beta_weighted_position_reward,
    relative_position_in_object_frame,
)


def test_relative_position_object_at_origin_identity():
    # objet à l'origine, quat identité (xyzw) -> rel == point
    point = torch.tensor([[1.0, 2.0, 3.0]])
    obj_pos = torch.zeros(1, 3)
    obj_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # xyzw identité
    rel = relative_position_in_object_frame(point, obj_pos, obj_quat)
    assert torch.allclose(rel, point, atol=1e-6)


def test_relative_position_yaw_90():
    # objet à l'origine, +90° autour de z. Point monde (1,0,0) -> objet-local (0,-1,0).
    point = torch.tensor([[1.0, 0.0, 0.0]])
    obj_pos = torch.zeros(1, 3)
    s = 0.5**0.5
    obj_quat = torch.tensor([[0.0, 0.0, s, s]])  # xyzw, +90° autour de z
    rel = relative_position_in_object_frame(point, obj_pos, obj_quat)
    assert torch.allclose(rel, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5)


def test_relative_position_translation_only():
    point = torch.tensor([[2.0, 0.0, 0.0]])
    obj_pos = torch.tensor([[1.0, 0.0, 0.0]])
    obj_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    rel = relative_position_in_object_frame(point, obj_pos, obj_quat)
    assert torch.allclose(rel, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_beta_one_at_contact_and_decays():
    d = torch.tensor([0.0, 0.1, 1.0])
    beta = beta_from_distance(d, beta_scale=0.1)
    assert torch.isclose(beta[0], torch.tensor(1.0))
    assert beta[1] < beta[0] and beta[2] < beta[1]
    assert torch.isclose(beta[1], torch.exp(torch.tensor(-1.0)), atol=1e-6)


def test_beta_clamps_negative_distance():
    beta = beta_from_distance(torch.tensor([-0.5]), beta_scale=0.1)
    assert torch.isclose(beta[0], torch.tensor(1.0))  # clamp(<0)->0 -> beta=1


def test_reward_one_at_zero_error():
    rel = torch.zeros(2, 2, 3)
    beta = torch.ones(2, 2)
    r = beta_weighted_position_reward(rel, rel.clone(), beta, sigma=0.3)
    assert torch.allclose(r, torch.ones(2), atol=1e-6)


def test_reward_decreases_with_error():
    rel_ref = torch.zeros(1, 2, 3)
    beta = torch.ones(1, 2)
    small = beta_weighted_position_reward(torch.full((1, 2, 3), 0.05), rel_ref, beta, sigma=0.3)
    big = beta_weighted_position_reward(torch.full((1, 2, 3), 0.5), rel_ref, beta, sigma=0.3)
    assert big < small < 1.0


def test_reward_ignores_hand_with_zero_beta():
    # main 0 fautive mais β=0 ; main 1 parfaite -> reward ~ 1
    rel_cur = torch.zeros(1, 2, 3)
    rel_cur[0, 0] = torch.tensor([1.0, 0.0, 0.0])
    rel_ref = torch.zeros(1, 2, 3)
    beta = torch.tensor([[0.0, 1.0]])
    r = beta_weighted_position_reward(rel_cur, rel_ref, beta, sigma=0.3)
    assert torch.allclose(r, torch.ones(1), atol=1e-6)


def test_reward_neutral_when_all_beta_zero():
    # espace libre : toutes les mains loin -> β=0 -> reward neutre (=1), sans NaN
    rel_cur = torch.full((1, 2, 3), 5.0)
    rel_ref = torch.zeros(1, 2, 3)
    beta = torch.zeros(1, 2)
    r = beta_weighted_position_reward(rel_cur, rel_ref, beta, sigma=0.3)
    assert torch.isfinite(r).all()
    assert torch.allclose(r, torch.ones(1), atol=1e-6)


def test_relative_position_rotation_and_translation():
    # objet en (1,0,0), tourné +90° autour de z ; point monde (1,1,0) -> objet-local (1,0,0)
    point = torch.tensor([[1.0, 1.0, 0.0]])
    obj_pos = torch.tensor([[1.0, 0.0, 0.0]])
    s = 0.5**0.5
    obj_quat = torch.tensor([[0.0, 0.0, s, s]])  # xyzw, +90° autour de z
    rel = relative_position_in_object_frame(point, obj_pos, obj_quat)
    assert torch.allclose(rel, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5)
