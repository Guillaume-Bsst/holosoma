"""Unit tests for the grasp-settle helpers (holosoma.utils.grasp_settle).

Pure torch, no IsaacSim — run with `pytest src/holosoma/holosoma/utils/tests/test_grasp_settle.py`.
"""

from __future__ import annotations

import torch

from holosoma.utils.grasp_settle import (
    anneal_prob,
    apply_grasp_transform,
    gather_anchor,
    grasp_relative_transform,
    select_grasp_anchor,
)
from holosoma.utils.rotations import quat_from_euler_xyz, quat_mul


def _rand_quat(n: int) -> torch.Tensor:
    rpy = (torch.rand(n, 3) - 0.5) * 6.0
    return quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])  # xyzw


def test_round_trip_identity():
    """Feeding the same anchor pose back must recover the object pose exactly."""
    torch.manual_seed(0)
    n = 64
    anchor_pos = torch.randn(n, 3)
    anchor_quat = _rand_quat(n)
    obj_pos = torch.randn(n, 3)
    obj_quat = _rand_quat(n)

    rel_pos, rel_quat = grasp_relative_transform(anchor_pos, anchor_quat, obj_pos, obj_quat)
    rec_pos, rec_quat = apply_grasp_transform(anchor_pos, anchor_quat, rel_pos, rel_quat)

    assert torch.allclose(rec_pos, obj_pos, atol=1e-5)
    # quaternion equality up to sign
    dot = (rec_quat * obj_quat).sum(dim=-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)


def test_weld_preserves_relative_transform():
    """Moving the anchor and reapplying the stored transform keeps the hand->object relation fixed."""
    torch.manual_seed(1)
    n = 32
    anchor_pos = torch.randn(n, 3)
    anchor_quat = _rand_quat(n)
    obj_pos = torch.randn(n, 3)
    obj_quat = _rand_quat(n)

    rel_pos, rel_quat = grasp_relative_transform(anchor_pos, anchor_quat, obj_pos, obj_quat)

    # perturb the anchor (translation + rotation), then weld the object onto it
    new_anchor_pos = anchor_pos + torch.randn(n, 3) * 0.1
    delta = _rand_quat(n)
    new_anchor_quat = quat_mul(delta, anchor_quat, w_last=True)
    welded_pos, welded_quat = apply_grasp_transform(new_anchor_pos, new_anchor_quat, rel_pos, rel_quat)

    # the relative transform recomputed from the welded pose must equal the original
    rel_pos2, rel_quat2 = grasp_relative_transform(new_anchor_pos, new_anchor_quat, welded_pos, welded_quat)
    assert torch.allclose(rel_pos2, rel_pos, atol=1e-5)
    dot = (rel_quat2 * rel_quat).sum(dim=-1).abs()
    assert torch.allclose(dot, torch.ones_like(dot), atol=1e-5)


def test_select_grasp_anchor_picks_nearest():
    anchor_pos = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # object closer to anchor 0
            [[2.0, 0.0, 0.0], [0.1, 0.0, 0.0]],  # object closer to anchor 1
        ]
    )
    object_pos = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    idx, dist = select_grasp_anchor(anchor_pos, object_pos)
    assert idx.tolist() == [0, 1]
    assert torch.allclose(dist, torch.tensor([0.2, 0.1]), atol=1e-6)


def test_anneal_prob():
    # ramp 1.0 -> 0.0 over 100 steps
    assert anneal_prob(0, 1.0, 0.0, 100) == 1.0
    assert abs(anneal_prob(50, 1.0, 0.0, 100) - 0.5) < 1e-9
    assert anneal_prob(100, 1.0, 0.0, 100) == 0.0
    assert anneal_prob(10_000, 1.0, 0.0, 100) == 0.0  # clamped after ramp
    assert anneal_prob(-5, 1.0, 0.0, 100) == 1.0  # clamped before ramp
    assert anneal_prob(42, 1.0, 0.0, 0) == 0.0  # no ramp -> end immediately
    assert anneal_prob(42, 0.0, 0.0, 100) == 0.0  # disabled curriculum stays 0


def test_gather_anchor():
    anchor_pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])
    anchor_quat = torch.tensor([[[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]])
    idx = torch.tensor([1])
    p, q = gather_anchor(anchor_pos, anchor_quat, idx)
    assert torch.allclose(p, torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.allclose(q, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
