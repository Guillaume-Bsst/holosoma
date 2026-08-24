"""The reference object angular velocity is optional in the motion NPZ.

The retargeting converter (convert_data_format_mj.py) writes ``object_ang_vel_w``, but clips baked
before it did not. The loader must accept both, and flag which case it is so the angular tracking
reward can stay neutral rather than train against zeros.
"""

import numpy as np
import pytest

from holosoma.managers.command.terms.wbt import MotionLoader

BODY_NAMES = ["pelvis", "torso_link"]
JOINT_NAMES = ["joint_a", "joint_b"]
T = 5


def _write_npz(path, *, with_object, with_ang_vel):
    data = {
        "fps": np.array(50),
        # holosoma format: joint_pos carries 7 root DOFs up front, joint_vel 6
        "joint_pos": np.zeros((T, len(JOINT_NAMES) + 7), dtype=np.float32),
        "joint_vel": np.zeros((T, len(JOINT_NAMES) + 6), dtype=np.float32),
        "body_pos_w": np.zeros((T, 2, 3), dtype=np.float32),
        # wxyz on disk, identity
        "body_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 2, 1)),
        "body_lin_vel_w": np.zeros((T, 2, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((T, 2, 3), dtype=np.float32),
        "body_names": np.array(BODY_NAMES),
        "joint_names": np.array(JOINT_NAMES),
    }
    if with_object:
        data["object_pos_w"] = np.zeros((T, 3), dtype=np.float32)
        data["object_quat_w"] = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))
        data["object_lin_vel_w"] = np.zeros((T, 3), dtype=np.float32)
        if with_ang_vel:
            data["object_ang_vel_w"] = np.arange(T * 3, dtype=np.float32).reshape(T, 3)
    np.savez(path, **data)
    return str(path)


def _load(tmp_path, name, **kw):
    f = _write_npz(tmp_path / name, **kw)
    return MotionLoader(f, BODY_NAMES, JOINT_NAMES, device="cpu")


def test_object_ang_vel_is_loaded_when_present(tmp_path):
    m = _load(tmp_path, "with.npz", with_object=True, with_ang_vel=True)
    assert m.has_object_ang_vel is True
    assert m.object_ang_vel_w.shape == (T, 3)
    # values round-trip, not silently zeroed
    assert pytest.approx(m.object_ang_vel_w[1].tolist()) == [3.0, 4.0, 5.0]


def test_object_ang_vel_falls_back_to_zeros_when_absent(tmp_path):
    m = _load(tmp_path, "without.npz", with_object=True, with_ang_vel=False)
    assert m.has_object_ang_vel is False
    # zeros of the RIGHT shape: every concat / interpolation path stays unconditional
    assert m.object_ang_vel_w.shape == (T, 3)
    assert m.object_ang_vel_w.abs().sum() == 0.0


def test_object_ang_vel_empty_without_object(tmp_path):
    m = _load(tmp_path, "noobj.npz", with_object=False, with_ang_vel=False)
    assert m.has_object is False
    assert m.has_object_ang_vel is False
    assert m.object_ang_vel_w.shape == (0, 3)


def test_object_ang_vel_matches_lin_vel_shape(tmp_path):
    # the fallback must track the linear tensor, which the transition padding concatenates alongside
    for name, ang in [("a.npz", True), ("b.npz", False)]:
        m = _load(tmp_path, name, with_object=True, with_ang_vel=ang)
        assert m.object_ang_vel_w.shape == m.object_lin_vel_w.shape
