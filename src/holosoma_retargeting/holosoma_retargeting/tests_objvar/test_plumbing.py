import numpy as np


def test_objvar_indices_on():
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, _ = build_largebox_inputs(2, {"object_variable": True})
    assert r.object_variable
    assert r.n_opt == r.nq_a + 7
    assert np.array_equal(r.q_opt_indices[-7:], np.arange(r.nq - 7, r.nq))
    # the object free joint is the model's 2nd FREE joint -> consistent body id
    import mujoco
    free = [j for j in range(r.robot_model.njnt)
            if r.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert r.object_body_id == int(r.robot_model.jnt_bodyid[free[1]])


def test_objvar_indices_off():
    from tests_objvar.largebox_fixture import build_largebox_inputs
    r, _ = build_largebox_inputs(2, {})
    assert not r.object_variable
    assert r.q_opt_indices is r.q_a_indices and r.n_opt == r.nq_a
