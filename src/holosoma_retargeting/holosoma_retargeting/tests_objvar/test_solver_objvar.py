import numpy as np


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


def test_anchor_quat_hemisphere_insensitive(tmp_path):
    # Over the first 15 frames of largebox_003 (the default fixture of the other tests),
    # the object barely rotates (angular speed ~0.04 deg/frame): a sign flip there pushes
    # the buggy anchor toward a purely RADIAL target in quat space (target = -2*current),
    # which the post-step renormalization absorbs without any rotational effect at all
    # (checked: dot(ref,cur) stays at exactly -1.0000 iteration after iteration, cost
    # unchanged -> convergence stop as early as the 2nd iteration). The bug only visibly
    # flips the object where a REAL frame-to-frame rotation already exists (there the
    # non-radial component of the buggy residual becomes significant): in largebox_003
    # that starts around frame 25 (up to ~2 deg/frame). So we flip from frame 25 onward
    # over a 50-frame window -- without the fix (checked by hand) the object makes a
    # transient flip of nearly 160 deg before re-locking (same physical rotation, opposite
    # hemisphere); with the fix the peak angle stays under 0.1 deg.
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(50, {"object_variable": True, "w_object_tracking": 1e6})
    ref = mk["object_poses_augmented"]
    flipped = ref.copy()
    flipped[25:, 3:7] *= -1.0            # same rotation, antipodal representation
    mk["object_poses_augmented"] = flipped
    out = str(tmp_path / "hemi.npz")
    r.retarget_motion(dest_res_path=out, **mk)
    qpos = np.load(out)["qpos"]
    for t in range(50):                  # the object follows the reference ROTATION
        qa = qpos[t, -4:] / np.linalg.norm(qpos[t, -4:])
        qb = ref[t, 3:7] / np.linalg.norm(ref[t, 3:7])
        ang = 2 * np.degrees(np.arccos(np.clip(abs(qa @ qb), -1, 1)))
        assert ang < 0.5, (t, ang)
