"""Reading an MPC contact schedule and putting it on the training clip's timeline."""

import numpy as np
import pytest
from holosoma.utils.contact_schedule import (
    inferred_fps,
    load_mpc_schedule,
    ramp_activation,
    resample_nearest,
)

PAIRS = ["left_foot|ground", "left_hand|box32", "right_foot|ground", "right_hand|box32",
         "obj0|ground", "obj0|support"]
T = 10


def _write(tmp_path, name, pairs, active, **extra):
    active = np.asarray(active, dtype=bool)
    np.savez(tmp_path / name, pair_names=np.array(pairs), active=active,
             pair_frames=np.array(["f"] * len(pairs)), pair_mu=np.full(len(pairs), 0.7),
             pair_types=np.array(["6D"] * len(pairs)), pair_objects=np.zeros(len(pairs), int),
             **extra)
    return str(tmp_path / (name if name.endswith(".npz") else name + ".npz"))


def _columns(**per_pair):
    """(T, P) with the named pairs True over the given frame slices."""
    a = np.zeros((T, len(PAIRS)), dtype=bool)
    for name, sl in per_pair.items():
        a[sl, PAIRS.index(name.replace("__", "|"))] = True
    return a


#########################################################################################
## load_mpc_schedule
#########################################################################################
def test_pairs_land_in_their_channels(tmp_path):
    a = _columns(left_hand__box32=slice(2, 6), right_hand__box32=slice(3, 6),
                 left_foot__ground=slice(0, 8), obj0__support=slice(6, 10))
    s = load_mpc_schedule(_write(tmp_path, "s.npz", PAIRS, a))

    assert s.num_frames == T
    assert list(np.flatnonzero(s.hand_object[:, 0])) == [2, 3, 4, 5]
    assert list(np.flatnonzero(s.hand_object[:, 1])) == [3, 4, 5]
    assert list(np.flatnonzero(s.foot_ground[:, 0])) == list(range(8))
    assert not s.foot_ground[:, 1].any()
    assert list(np.flatnonzero(s.object_support)) == [6, 7, 8, 9]
    assert not s.object_ground.any()


def test_hands_are_independent_not_collapsed_to_one_anchor(tmp_path):
    # the whole reason for reading the schedule: a bimanual carry is two contacts, not the nearest one
    a = _columns(left_hand__box32=slice(0, 5), right_hand__box32=slice(0, 5))
    s = load_mpc_schedule(_write(tmp_path, "s.npz", PAIRS, a))
    assert s.hand_object[:5].all()
    assert not s.hand_object[5:].any()


def test_object_token_name_is_not_hardcoded(tmp_path):
    # a schedule baked against box36 (or any other object name) must still resolve
    pairs = ["left_hand|box36", "right_hand|largebox"]
    a = np.zeros((T, 2), dtype=bool)
    a[1:4, 0] = True
    a[5:7, 1] = True
    s = load_mpc_schedule(_write(tmp_path, "s.npz", pairs, a))
    assert list(np.flatnonzero(s.hand_object[:, 0])) == [1, 2, 3]
    assert list(np.flatnonzero(s.hand_object[:, 1])) == [5, 6]


def test_unmapped_but_inactive_pair_is_tolerated(tmp_path):
    pairs = PAIRS + ["left_toe|ground"]
    a = np.zeros((T, len(pairs)), dtype=bool)
    s = load_mpc_schedule(_write(tmp_path, "s.npz", pairs, a))
    assert s.unmapped == ("left_toe|ground",)


def test_unmapped_ACTIVE_pair_is_refused(tmp_path):
    pairs = PAIRS + ["left_toe|ground"]
    a = np.zeros((T, len(pairs)), dtype=bool)
    a[3:5, -1] = True
    with pytest.raises(ValueError, match="maps to no channel"):
        load_mpc_schedule(_write(tmp_path, "s.npz", pairs, a))


def test_inconsistent_file_is_refused(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, pair_names=np.array(PAIRS), active=np.zeros((T, 3), dtype=bool))
    with pytest.raises(ValueError, match="inconsistent"):
        load_mpc_schedule(str(p))


def test_non_schedule_npz_is_refused(tmp_path):
    p = tmp_path / "other.npz"
    np.savez(p, joint_pos=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="not an MPC contact schedule"):
        load_mpc_schedule(str(p))


#########################################################################################
## resample_nearest / inferred_fps
#########################################################################################
def test_resample_keeps_the_phase_across_a_30_to_50_fps_pair():
    src = np.zeros((197, 1), dtype=bool)
    src[60:120] = True                                   # 2.000 s -> 4.000 s at 30 fps
    out = resample_nearest(src, 327)                     # onto a 327-frame, 50 fps clip
    assert out.shape == (327, 1)
    on = np.flatnonzero(out[:, 0])
    one_src_frame = 1 / 30.0
    assert abs(on[0] / 50.0 - 60 / 30.0) <= one_src_frame          # edges land within one source frame
    assert abs((on[-1] + 1) / 50.0 - 120 / 30.0) <= one_src_frame


def test_resample_matches_an_explicit_fps_mapping_on_phase_structured_contact():
    # The reason no cadence is asked for. A one-frame index shift only shows up at a phase EDGE, so
    # this must be measured on a phase-structured signal like a real schedule (contiguous contacts),
    # not on noise, where every frame is an edge.
    src = np.zeros((197, 1), dtype=bool)
    src[20:70] = True
    src[95:150] = True
    src[180:] = True
    proportional = resample_nearest(src, 327)
    i = np.arange(327)
    explicit = src[np.rint(i / 50.0 * 30.0).astype(int).clip(0, 196)]
    assert (proportional != explicit).sum() <= 6         # 3 phases -> at most a couple of edges each


def test_resample_is_nearest_never_blended():
    src = np.array([[True], [False], [True]], dtype=bool)
    out = resample_nearest(src, 6)
    assert out.dtype == bool
    assert set(np.unique(out)) <= {True, False}


def test_resample_identity_when_lengths_match():
    src = np.random.default_rng(0).random((20, 3)) > 0.5
    assert np.array_equal(resample_nearest(src, 20), src)


def test_resample_handles_downsampling_too():
    src = np.zeros((100, 1), dtype=bool)
    src[50:] = True
    out = resample_nearest(src, 10)
    assert out.shape == (10, 1)
    assert list(np.flatnonzero(out[:, 0])) == [5, 6, 7, 8, 9]


def test_resample_refuses_degenerate_lengths():
    with pytest.raises(ValueError, match="no frames"):
        resample_nearest(np.zeros((0, 2), dtype=bool), 10)
    with pytest.raises(ValueError, match="at least one frame"):
        resample_nearest(np.zeros((5, 1), dtype=bool), 0)


def test_inferred_fps_reads_as_a_familiar_rate_on_the_real_pair():
    # 197 schedule frames onto femto14_w_obj (327 frames at 50 fps)
    assert inferred_fps(197, 327, 50.0) == pytest.approx(30.12, abs=0.01)


def test_inferred_fps_exposes_a_borrowed_schedule():
    # the same schedule against femto14_box36 (555 frames at 50 fps, 11.1 s): no capture ran at 17.7
    assert inferred_fps(197, 555, 50.0) == pytest.approx(17.75, abs=0.01)


#########################################################################################
## ramp_activation
#########################################################################################
def test_ramp_zero_is_the_plain_booleans():
    a = _columns(left_hand__box32=slice(2, 6))
    assert np.array_equal(ramp_activation(a, 0), a.astype(float))


def test_ramp_rises_over_n_frames_from_the_phase_start():
    a = np.zeros((8, 1), dtype=bool)
    a[2:7] = True
    r = ramp_activation(a, 4)[:, 0]
    assert np.allclose(r[:2], 0.0)
    assert np.allclose(r[2:6], [0.25, 0.5, 0.75, 1.0])
    assert r[6] == 1.0


def test_ramp_release_is_immediate():
    a = np.zeros((6, 1), dtype=bool)
    a[1:4] = True
    r = ramp_activation(a, 3)[:, 0]
    assert r[3] == 1.0
    assert r[4] == 0.0        # no decay: a hand off the box is off the box


def test_ramp_leaves_a_contact_open_at_frame_zero_at_full_weight():
    a = np.ones((5, 1), dtype=bool)
    a[3:] = False
    assert np.allclose(ramp_activation(a, 4)[:, 0], [1.0, 1.0, 1.0, 0.0, 0.0])


def test_ramp_restarts_on_each_phase():
    a = np.zeros((10, 1), dtype=bool)
    a[1:4] = True
    a[6:9] = True
    r = ramp_activation(a, 2)[:, 0]
    assert np.allclose(r[1:4], [0.5, 1.0, 1.0])
    assert np.allclose(r[6:9], [0.5, 1.0, 1.0])   # second phase ramps again, not carried over


def test_ramp_is_per_channel():
    a = np.zeros((6, 2), dtype=bool)
    a[1:5, 0] = True
    a[3:5, 1] = True
    r = ramp_activation(a, 2)
    assert np.allclose(r[:, 0], [0, 0.5, 1, 1, 1, 0])
    assert np.allclose(r[:, 1], [0, 0, 0, 0.5, 1, 0])
