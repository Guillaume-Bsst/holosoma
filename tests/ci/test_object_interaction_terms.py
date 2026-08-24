"""Tensor plumbing of the object-interaction reward and observation terms.

Exercises the real term functions against a stubbed simulator: shapes, per-env anchor gathering,
world-frame mapping of the reference witness, and the guards that keep the terms neutral on clips
that do not carry the fields they need. The scalar math itself is covered in
``test_object_interaction.py``.
"""

import torch
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import obj_ang_vel_b, obj_contact_flag, obj_lin_vel_b_rotated
from holosoma.managers.reward.terms.wbt import (
    object_contact_force_match_exp,
    object_global_ref_ang_vel_error_exp,
    object_global_ref_lin_vel_error_exp,
)

N = 2  # envs
A = 2  # candidate anchors (left/right hand)
H = 3  # contact sensor history length
NB = 4  # robot bodies
ANCHORS = [2, 3]  # anchor body indexes within the robot body list


class _Motion:
    has_object = True
    has_gt_contact = True
    has_gt_witness = True
    has_object_ang_vel = True
    has_contact_schedule = False

    def __init__(self, **over):
        self.object_ref_anchor_idx = torch.tensor([0, 1])
        self.object_ref_contact = torch.tensor([True, True])
        self.object_ref_witness_local = torch.zeros(N, 3)
        self.object_lin_vel_w = torch.zeros(N, 3)
        self.object_ang_vel_w = torch.zeros(N, 3)
        self.object_pos_w = torch.zeros(N, 3)
        self.body_pos_w = torch.zeros(N, NB, 3)
        for k, v in over.items():
            setattr(self, k, v)


class _Sim:
    def __init__(self, contact_forces, obj_state, anchor_pos):
        self.contact_forces_history = contact_forces  # (N, H, NB, 3)
        self.all_root_states = obj_state  # (N, 13), indexed by object_indices_in_simulator
        self._rigid_body_pos = torch.zeros(N, NB, 3)
        self._rigid_body_pos[:, ANCHORS, :] = anchor_pos
        self._rigid_body_rot = torch.zeros(N, NB, 4)
        self._rigid_body_rot[..., 3] = 1.0  # identity xyzw
        self.scene = type("S", (), {"env_origins": torch.zeros(N, 3)})()


class _Env:
    device = "cpu"
    num_envs = N

    def __init__(self, mc, sim):
        self.simulator = sim
        self.command_manager = type("CM", (), {"get_state": staticmethod(lambda _: mc)})()


def _build(*, forces=None, anchor_pos=None, ref_anchor_pos=None, obj_lin=None, obj_ang=None,
           witness=None, motion_over=None):
    """A MotionCommand wired to stub tensors, plus the env that carries it."""
    mc = object.__new__(MotionCommand)
    mc.motion = _Motion(**(motion_over or {}))
    if ref_anchor_pos is not None:
        mc.motion.body_pos_w[:, ANCHORS, :] = torch.as_tensor(ref_anchor_pos, dtype=torch.float32)
    if witness is not None:
        mc.motion.object_ref_witness_local = witness
    mc.time_steps = torch.arange(N)
    mc.device = "cpu"
    mc._anchor_body_indexes = torch.tensor(ANCHORS)
    mc.grasp_settle_cfg = type("G", (), {"box_half_extents": (0.1, 0.1, 0.1)})()
    mc.object_indices_in_simulator = torch.arange(N)

    root = torch.zeros(N, 13)
    root[:, 6] = 1.0  # identity quat, xyzw
    if obj_lin is not None:
        root[:, 7:10] = obj_lin
    if obj_ang is not None:
        root[:, 10:13] = obj_ang

    sim = _Sim(
        torch.zeros(N, H, NB, 3) if forces is None else forces,
        root,
        torch.zeros(N, A, 3) if anchor_pos is None else anchor_pos,
    )
    env = _Env(mc, sim)
    mc._env = env
    # robot_ref_* is the reference BODY (torso) pose; identity here so body frame == world frame
    mc.ref_body_index = 0
    return env, mc


def _forces(per_env_per_anchor):
    """(N, H, NB, 3) with the given |F| on each anchor body, spread over the history."""
    f = torch.zeros(N, H, NB, 3)
    for env_i, mags in enumerate(per_env_per_anchor):
        for a_i, mag in enumerate(mags):
            f[env_i, 0, ANCHORS[a_i], 0] = mag
    return f


#########################################################################################
## velocity rewards
#########################################################################################
def test_lin_vel_reward_is_one_when_object_matches_reference():
    env, _ = _build(obj_lin=torch.zeros(N, 3))
    assert torch.allclose(object_global_ref_lin_vel_error_exp(env, sigma=1.0), torch.ones(N), atol=1e-6)


def test_lin_vel_reward_drops_when_object_drifts():
    env, _ = _build(obj_lin=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    r = object_global_ref_lin_vel_error_exp(env, sigma=1.0)
    assert r.shape == (N,)
    assert torch.isclose(r[0], torch.tensor(1.0), atol=1e-6)
    assert r[1] < r[0]


def test_ang_vel_reward_tracks_the_angular_reference():
    env, _ = _build(obj_ang=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]))
    r = object_global_ref_ang_vel_error_exp(env, sigma=3.14)
    assert torch.isclose(r[0], torch.tensor(1.0), atol=1e-6)
    assert r[1] < r[0]


def test_ang_vel_reward_is_zero_without_a_reference():
    # Clip baked before the converter wrote object_ang_vel_w. ZERO, not one: a constant paid every
    # step is a survival bonus in a discounted return, not a neutral value.
    env, _ = _build(obj_ang=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
                    motion_over={"has_object_ang_vel": False})
    assert torch.allclose(object_global_ref_ang_vel_error_exp(env, sigma=3.14), torch.zeros(N), atol=1e-6)


#########################################################################################
## HDMI contact reward
#########################################################################################
CONTACT_KW = dict(sigma_pos=0.1, sigma_force=20.0, force_threshold=10.0, max_force_bonus=2.0)


def test_contact_reward_gathers_the_force_of_the_CONTACT_hand():
    # env 0 uses anchor 0, env 1 uses anchor 1 (motion.object_ref_anchor_idx = [0, 1]).
    # Only the non-contact hand is loaded -> the reward must NOT see that force.
    env, _ = _build(forces=_forces([[0.0, 500.0], [500.0, 0.0]]))
    r = object_contact_force_match_exp(env, **CONTACT_KW)
    assert torch.allclose(r, torch.ones(N), atol=1e-6)  # distance 0, force below threshold -> 1.0

    # now load the contact hand instead: both envs get the capped bonus
    env, _ = _build(forces=_forces([[500.0, 0.0], [0.0, 500.0]]))
    r = object_contact_force_match_exp(env, **CONTACT_KW)
    assert torch.allclose(r, torch.full((N,), CONTACT_KW["max_force_bonus"]), atol=1e-6)


def test_contact_reward_uses_the_max_over_the_sensor_history():
    f = torch.zeros(N, H, NB, 3)
    f[:, 2, ANCHORS[0], 0] = 500.0  # spike two steps ago only
    f[1, 2, ANCHORS[0], 0] = 0.0
    f[1, 2, ANCHORS[1], 0] = 500.0
    env, _ = _build(forces=f)
    assert torch.allclose(
        object_contact_force_match_exp(env, **CONTACT_KW),
        torch.full((N,), CONTACT_KW["max_force_bonus"]),
        atol=1e-6,
    )


def test_contact_reward_falls_with_distance_to_the_reference_witness():
    # witness is box-local; the box sits at the origin with identity rotation, so world == local
    near, far = 0.0, 0.5
    env, _ = _build(anchor_pos=torch.tensor([[[near, 0, 0], [0, 0, 0]], [[far, 0, 0], [0, 0, 0]]]),
                    motion_over={"object_ref_anchor_idx": torch.tensor([0, 0])})
    r = object_contact_force_match_exp(env, **CONTACT_KW)
    assert r[0] > r[1]
    assert torch.isclose(r[0], torch.tensor(1.0), atol=1e-6)


def test_contact_reward_is_zero_where_the_reference_says_no_contact():
    env, _ = _build(motion_over={"object_ref_contact": torch.tensor([True, False])})
    r = object_contact_force_match_exp(env, **CONTACT_KW)
    assert r[1] == 0.0
    assert r[0] > 0.0


def test_contact_reward_falls_back_to_the_box_surface_without_a_witness():
    # box at the origin, half extents 0.1 -> the +x face sits at x = 0.1
    on_surface, far = 0.1, 0.6
    env, _ = _build(
        anchor_pos=torch.tensor([[[on_surface, 0, 0], [0, 0, 0]], [[far, 0, 0], [0, 0, 0]]]),
        motion_over={"has_gt_witness": False, "object_ref_anchor_idx": torch.tensor([0, 0])},
    )
    r = object_contact_force_match_exp(env, **CONTACT_KW)
    # NOT neutral: the term works without a witness, which is what makes it usable on clips that
    # carry no reference contact at all
    assert torch.isclose(r[0], torch.tensor(1.0), atol=1e-6)   # hand on the surface -> distance 0
    assert r[1] < r[0]                                          # 0.5 m off the surface


def test_contact_reward_is_zero_without_resolved_anchors():
    # no object in the scene -> the term contributes nothing, which is 0 and not 1
    env, mc = _build()
    mc._anchor_body_indexes = None
    assert torch.allclose(object_contact_force_match_exp(env, **CONTACT_KW), torch.zeros(N), atol=1e-6)


#########################################################################################
## the supplied contact schedule drives the gate
#########################################################################################
def _with_schedule(weights, **kw):
    """weights: (N, 2) hand activation in [0, 1] for the current frames."""
    env, mc = _build(**kw)
    mc.motion.has_contact_schedule = True
    mc.motion._schedule_hand_contact = torch.tensor(weights, dtype=torch.float32)
    return env, mc


def test_schedule_gate_replaces_the_binary_reference():
    # reference says contact on both envs; the schedule says only the first
    env, _ = _with_schedule([[1.0, 1.0], [0.0, 0.0]])
    r = object_contact_force_match_exp(env, **CONTACT_KW)
    assert r[0] > 0.0
    assert r[1] == 0.0


def test_schedule_ramp_scales_the_reward_continuously():
    full, half = _with_schedule([[1.0, 1.0], [1.0, 1.0]])[0], _with_schedule([[0.5, 0.5], [0.5, 0.5]])[0]
    r_full = object_contact_force_match_exp(full, **CONTACT_KW)
    r_half = object_contact_force_match_exp(half, **CONTACT_KW)
    assert torch.allclose(r_half, r_full * 0.5, atol=1e-6)


def test_schedule_picks_the_nearer_of_the_hands_it_closes():
    # both hands closed by the schedule; the anchor must be the one nearest the box (env 1: right)
    env, mc = _with_schedule(
        [[1.0, 1.0], [1.0, 1.0]],
        ref_anchor_pos=torch.tensor([[[0.1, 0, 0], [0.9, 0, 0]], [[0.9, 0, 0], [0.1, 0, 0]]]),
    )
    idx, contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    assert contact.all()
    assert idx.tolist() == [0, 1]


def test_schedule_ignores_a_hand_it_leaves_open_even_if_it_is_nearer():
    env, mc = _with_schedule(
        [[0.0, 1.0], [0.0, 1.0]],
        ref_anchor_pos=torch.tensor([[[0.1, 0, 0], [0.9, 0, 0]], [[0.1, 0, 0], [0.9, 0, 0]]]),
    )
    idx, contact = mc._lookup_ref_contact(mc.time_steps, mc.anchor_pos_w, mc.object_pos_w)
    assert contact.all()
    assert idx.tolist() == [1, 1]  # the far hand, because it is the one the schedule closes


#########################################################################################
## observation terms
#########################################################################################
def test_rotated_lin_vel_is_zero_for_a_motionless_object():
    # the whole point of the corrected term: the inherited obj_lin_vel_b returns ~1.8 m/s here
    env, _ = _build(obj_lin=torch.zeros(N, 3))
    env.simulator._rigid_body_pos[:, 0, :] = torch.tensor([1.5, -0.4, 0.9])  # torso away from origin
    out = obj_lin_vel_b_rotated(env)
    assert out.shape == (N, 3)
    assert torch.allclose(out, torch.zeros(N, 3), atol=1e-6)


def test_rotated_lin_vel_passes_the_velocity_through_in_an_identity_frame():
    v = torch.tensor([[0.3, -0.1, 0.05], [0.0, 0.0, 0.0]])
    env, _ = _build(obj_lin=v)
    assert torch.allclose(obj_lin_vel_b_rotated(env), v, atol=1e-6)


def test_ang_vel_obs_shape_and_passthrough():
    w = torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]])
    env, _ = _build(obj_ang=w)
    out = obj_ang_vel_b(env)
    assert out.shape == (N, 3)
    assert torch.allclose(out, w, atol=1e-6)


def test_contact_flag_reports_both_hands_and_the_reference():
    env, _ = _build(forces=_forces([[50.0, 0.0], [0.0, 50.0]]),
                    motion_over={"object_ref_contact": torch.tensor([True, False])})
    out = obj_contact_flag(env, force_threshold=1.0)
    assert out.shape == (N, 3)
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.allclose(out[1], torch.tensor([0.0, 1.0, 0.0]))


def test_contact_flag_respects_its_threshold():
    env, _ = _build(forces=_forces([[0.5, 0.0], [5.0, 0.0]]))
    out = obj_contact_flag(env, force_threshold=1.0)
    assert out[0, 0] == 0.0  # 0.5 N is noise
    assert out[1, 0] == 1.0  # 5 N is a touch


def test_contact_flag_is_all_zeros_without_anchors():
    env, mc = _build()
    mc._anchor_body_indexes = None
    assert torch.allclose(obj_contact_flag(env), torch.zeros(N, 3))
