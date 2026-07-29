import numpy as np
from scipy.spatial.transform import Rotation


def _run(n, overrides, tmp_path, name):
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(n, overrides)
    out = str(tmp_path / f"{name}.npz")
    r.retarget_motion(dest_res_path=out, **mk)
    return np.load(out)["qpos"], mk


def test_flag_off_bit_identical(tmp_path):
    qpos, _ = _run(15, {}, tmp_path, "off")
    base = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"]
    assert np.array_equal(qpos, base), np.abs(qpos - base).max()


def test_strong_anchor_recovers_fixed_object(tmp_path):
    qpos, mk = _run(15, {"object_variable": True, "w_object_tracking": 1e6}, tmp_path, "strong")
    ref = mk["object_poses_augmented"][:15]        # mujoco order [pos(3), quat wxyz(4)]
    dpos = np.abs(qpos[:, -7:-4] - ref[:, :3]).max()
    # 2.5 mm threshold (not 1 mm): the largebox_003 reference grazes the ground/the knee
    # to within less than penetration_tolerance (1e-3) -- masks_ok now lets object<->ground
    # (and object<->body) into the QP when object_variable is active, so those HARD
    # constraints bound the anchor (checked: phi object-ground/object-knee pins exactly at
    # -penetration_tolerance, and the residual is invariant to w_object_tracking 1e6->1e9
    # -- this is not an under-weighted anchor, it is a collision stop).
    assert dpos < 2.5e-3, dpos
    for t in range(15):                            # quat angle < 0.2 deg
        qa = qpos[t, -4:] / np.linalg.norm(qpos[t, -4:])
        qb = ref[t, 3:7] / np.linalg.norm(ref[t, 3:7])
        ang = 2 * np.degrees(np.arccos(np.clip(abs(qa @ qb), -1, 1)))
        assert ang < 0.2, (t, ang)


def test_free_object_actually_moves(tmp_path):
    qpos, mk = _run(15, {"object_variable": True, "w_object_tracking": 0.0}, tmp_path, "free")
    ref = mk["object_poses_augmented"][:15]
    dpos = np.abs(qpos[:, -7:-4] - ref[:, :3]).max()
    assert dpos > 1e-4, "the object did not move -- the variable is not wired up"
