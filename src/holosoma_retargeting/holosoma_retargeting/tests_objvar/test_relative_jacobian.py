import numpy as np
from scipy.spatial.transform import Rotation


def _obj_frame_pos(r, q, link_name):
    """Position monde->repere objet du lien, par FK mujoco (verite terrain du test)."""
    import mujoco
    r.robot_data.qpos[:] = q
    mujoco.mj_forward(r.robot_model, r.robot_data)
    bid = mujoco.mj_name2id(r.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
    quat = q[-4:]
    R_o = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return R_o.T @ (r.robot_data.xpos[bid] - q[-7:-4])


def test_object_relative_jacobian_finite_differences():
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, mk = build_largebox_inputs(2, {"object_variable": True})
    q = np.load("tests_objvar/data/baseline_largebox_15f.npz")["qpos"][5].copy()
    # pose objet non triviale (rotation + translation) pour exciter tous les termes
    q[-7:-4] += np.array([0.11, -0.07, 0.05])
    q[-4:] = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_quat()[[3, 0, 1, 2]]

    name, link = next(iter(r.laplacian_match_links.items()))
    J_dict, p_dict, _ = r._calc_manipulator_jacobians(q, {name: link}, obj_frame=True)
    J = J_dict[name]                      # (3, n_opt)
    assert J.shape == (3, r.n_opt)

    eps = 1e-6
    for col, qi in enumerate(r.q_opt_indices):
        qp, qm = q.copy(), q.copy()
        qp[qi] += eps
        qm[qi] -= eps
        for qq in (qp, qm):               # renorm des quats perturbes (base + objet)
            qq[3:7] /= np.linalg.norm(qq[3:7])
            qq[-4:] /= np.linalg.norm(qq[-4:])
        fd = (_obj_frame_pos(r, qp, link) - _obj_frame_pos(r, qm, link)) / (2 * eps)
        np.testing.assert_allclose(J[:, col], fd, atol=5e-5,
                                   err_msg=f"colonne {col} (qpos {qi})")
