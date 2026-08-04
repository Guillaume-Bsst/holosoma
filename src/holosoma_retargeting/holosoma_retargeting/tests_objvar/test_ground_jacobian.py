"""Jacobian of the GROUND vertices in object frame, validated by finite differences.

The ground is FIXED in the world: in object frame its coordinates move when the object
pose moves, so its J_V rows cannot stay zero while object_variable is active (unlike the
object rows, which are constant in the object's own frame).

Same template as test_relative_jacobian.py: same fixture, centred eps=1e-6, atol=5e-5.
The ground truth is simpler here -- R_o^T (g - t_o) is computed directly, without FK.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _obj_frame_pos_of_world_point(q, g_world):
    """Ground truth: position of a fixed WORLD point, expressed in the object frame."""
    quat = q[-4:]
    R_o = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return R_o.T @ (g_world - q[-7:-4])


def test_ground_jacobian_finite_differences():
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(2, {"object_variable": True})
    q = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"][5].copy()
    # non-trivial object pose (rotation + translation) to excite every term
    q[-7:-4] += np.array([0.11, -0.07, 0.05])
    q[-4:] = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_quat()[[3, 0, 1, 2]]

    import mujoco
    r.robot_data.qpos[:] = q
    mujoco.mj_forward(r.robot_model, r.robot_data)

    # an arbitrary ground point, off-axis so every rotation term is excited
    g_world = np.array([0.8, -1.3, 0.0])

    J = r._calc_ground_vertex_jacobian(g_world)          # (3, n_opt)
    assert J.shape == (3, r.n_opt), J.shape

    eps = 1e-6
    for col, qi in enumerate(r.q_opt_indices):
        qp, qm = q.copy(), q.copy()
        qp[qi] += eps
        qm[qi] -= eps
        for qq in (qp, qm):               # renormalize the perturbed quats (base + object)
            qq[3:7] /= np.linalg.norm(qq[3:7])
            qq[-4:] /= np.linalg.norm(qq[-4:])
        fd = (_obj_frame_pos_of_world_point(qp, g_world)
              - _obj_frame_pos_of_world_point(qm, g_world)) / (2 * eps)
        np.testing.assert_allclose(J[:, col], fd, atol=5e-5,
                                   err_msg=f"column {col} (qpos {qi})")


def test_ground_jacobian_is_insensitive_to_robot_joints():
    """The ground does not move with the robot: the ROBOT dof columns must be zero.

    This is what tells the ground case apart from the manipulator case, and it is also the
    guard against a partial sign error: a Jacobian that mixed the two blocks up would have
    non-zero robot columns.
    """
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(2, {"object_variable": True})
    q = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"][5].copy()
    import mujoco
    r.robot_data.qpos[:] = q
    mujoco.mj_forward(r.robot_model, r.robot_data)

    J = r._calc_ground_vertex_jacobian(np.array([0.8, -1.3, 0.0]))
    n_obj_cols = 7                                        # object pose = 7 qpos coords
    np.testing.assert_allclose(J[:, :-n_obj_cols], 0.0, atol=1e-12)
    assert np.abs(J[:, -n_obj_cols:]).max() > 1e-6, "the object columns must NOT be zero"
