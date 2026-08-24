"""Read an MPC contact-schedule NPZ and put it on the training clip's timeline.

The MPC contact schedule is the interchange format the planner already uses for per-frame contact
booleans, and it is explicitly meant to be written from booleans obtained anywhere else. Reading it
here means the RL side consumes the same contact truth the planner does, instead of a second,
independently derived one.

On-disk format (P pairs, T frames)::

    pair_names   (P,)   str    "left_hand|box32"
    pair_frames  (P,)   str    pinocchio frame of the robot-side contact point
    pair_mu      (P,)   float  Coulomb coefficient
    pair_types   (P,)   str    "6D" (flat support) | "3D" (point)
    pair_objects (P,)   int    object index, -1 = static (ground)
    pair_normals (P,3)  float  default world normal, NaN row => +z
    active       (T,P)  bool   contact closed at frame t
    normals_t    (T,P,3) float optional, world normal at the start of the phase covering t

Everything here is plain numpy so it can be unit-tested without a simulator or a motion loader.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Robot-side token of a pair name -> the semantic channel it feeds. Matched on the token BEFORE the
# "|", so the object token stays free ("box32", "box36", ...): a schedule baked for a differently
# named box still resolves. The right-hand token only matters for the object pairs, where it says
# WHAT the box rests on.
_HAND_TOKENS = {"left_hand": 0, "right_hand": 1}
_FOOT_TOKENS = {"left_foot": 0, "right_foot": 1}
_OBJECT_TOKEN = "obj0"


@dataclass(frozen=True)
class ContactSchedule:
    """Per-frame contact booleans on the schedule's OWN timeline (not yet resampled)."""

    hand_object: np.ndarray  # (T, 2) bool, [left, right] <-> the carried object
    foot_ground: np.ndarray  # (T, 2) bool, [left, right] <-> ground
    object_ground: np.ndarray  # (T,) bool, object resting on the ground
    object_support: np.ndarray  # (T,) bool, object resting on the support (table)
    pair_names: tuple[str, ...]
    unmapped: tuple[str, ...]  # pairs carried by the file that no channel consumes

    @property
    def num_frames(self) -> int:
        return self.hand_object.shape[0]


def load_mpc_schedule(path: str) -> ContactSchedule:
    """Read the NPZ and fold its pairs into the semantic channels above.

    A pair the classifier does not recognise is tolerated only when it is never active: silently
    dropping a contact that IS closed somewhere would plan a fall, the same argument
    ``MPC2/mpc2/schedule.py:load_schedule`` makes when it refuses an unmapped active pair.
    """
    data = np.load(path, allow_pickle=False)
    for key in ("pair_names", "active"):
        if key not in data.files:
            raise ValueError(f"{path}: not an MPC contact schedule (missing {key!r})")

    names = [str(n) for n in data["pair_names"]]
    active = np.asarray(data["active"], dtype=bool)
    if active.ndim != 2 or active.shape[1] != len(names):
        raise ValueError(
            f"{path}: `active` is {active.shape} for {len(names)} pairs -- the file is inconsistent"
        )

    n_frames = active.shape[0]
    hand = np.zeros((n_frames, 2), dtype=bool)
    foot = np.zeros((n_frames, 2), dtype=bool)
    obj_ground = np.zeros(n_frames, dtype=bool)
    obj_support = np.zeros(n_frames, dtype=bool)
    unmapped: list[str] = []

    for col, name in enumerate(names):
        robot_token, _, other_token = name.partition("|")
        column = active[:, col]
        if robot_token in _HAND_TOKENS and other_token != "ground":
            hand[:, _HAND_TOKENS[robot_token]] |= column
        elif robot_token in _FOOT_TOKENS and other_token == "ground":
            foot[:, _FOOT_TOKENS[robot_token]] |= column
        elif robot_token == _OBJECT_TOKEN and other_token == "ground":
            obj_ground |= column
        elif robot_token == _OBJECT_TOKEN and other_token == "support":
            obj_support |= column
        elif column.any():
            raise ValueError(
                f"{path}: pair {name!r} is active on {int(column.sum())} frames but maps to no "
                f"channel. Losing a closed contact in silence is worse than refusing the file."
            )
        else:
            unmapped.append(name)

    return ContactSchedule(
        hand_object=hand,
        foot_ground=foot,
        object_ground=obj_ground,
        object_support=obj_support,
        pair_names=tuple(names),
        unmapped=tuple(unmapped),
    )


def resample_nearest(active: np.ndarray, dst_num_frames: int) -> np.ndarray:
    """Put ``active`` (T_src, ...) on a (dst_num_frames, ...) timeline: proportional, nearest frame.

    Nearest, never interpolated: these are booleans, and a blended contact is not a contact. Same
    choice ``merge_training_npz.py`` makes for the bool/index fields.

    No cadence is needed. Both files describe the same take, so their durations are equal and the
    frame-rate-consistent mapping ``round(i * T_src / T_dst)`` falls out of that alone. Passing the
    schedule's fps explicitly was measured to move at most one source frame (1 to 5 contact frames
    out of 327 on the femto14 pair, all at phase edges) -- sub-frame noise, not worth a required
    parameter.

    Nothing here verifies that the two files ARE the same take, because nothing cheap can: a
    duration check is necessary but not sufficient, and a geometric check (are the hands near the
    box when the schedule says contact?) was measured to score a MISMATCHED clip higher than the
    correct one, since a similar motion stretched onto a similar motion still lines up. The caller
    passes both paths on one command line; ``inferred_fps`` gives them the diagnostic to eyeball.
    """
    src_num_frames = active.shape[0]
    if src_num_frames == 0:
        raise ValueError("contact schedule has no frames")
    if dst_num_frames <= 0:
        raise ValueError(f"target timeline must have at least one frame, got {dst_num_frames}")
    idx = np.rint(np.arange(dst_num_frames) * src_num_frames / dst_num_frames)
    return active[idx.astype(np.int64).clip(0, src_num_frames - 1)]


def inferred_fps(src_num_frames: int, dst_num_frames: int, dst_fps: float) -> float:
    """The cadence the schedule must have had for both files to cover the same duration.

    Diagnostic, not a gate. On a correctly paired schedule it reads as a familiar rate (30.12 for
    femto14); on a schedule borrowed from another take it reads as something no capture ever ran at
    (17.7 for the same schedule against an 11.1 s clip), which is visible in the run log.
    """
    if dst_num_frames <= 0:
        raise ValueError(f"target timeline must have at least one frame, got {dst_num_frames}")
    return src_num_frames * dst_fps / dst_num_frames


def ramp_activation(active: np.ndarray, num_ramp_frames: int) -> np.ndarray:
    """Continuous activation in [0, 1]: ramped onset, IMMEDIATE release. (T, C) bool -> (T, C) float.

    Ported from ``MPC2/mpc2/schedule.py:Schedule.rampe``, including its asymmetry, which is the
    whole point: a foot in the air IS in the air, so release cannot be ramped. Onset is ramped
    because a hard 0->1 step is a cliff in the cost/reward exactly where a smooth surface is needed.

    Counted from the start of the phase containing t, not over a sliding window: a sliding window
    wider than a short phase straddles two of them and starts the ramp partway up.

    A phase that is already open at frame 0 starts at 1: the contact is established, not beginning.

    ``num_ramp_frames <= 0`` returns the plain booleans as floats, bit-identical to ``active``.
    """
    out = np.asarray(active, dtype=float)
    if num_ramp_frames <= 0:
        return out
    for channel in range(out.shape[1]):
        column = np.asarray(active[:, channel], dtype=bool)
        starts = np.flatnonzero(column & ~np.concatenate(([False], column[:-1])))
        for start in starts:
            if start == 0:
                continue  # already-established contact: no ramp
            end = start + int(np.argmin(column[start:])) if not column[start:].all() else len(column)
            span = np.arange(start, end)
            out[span, channel] = np.minimum(1.0, (span - start + 1) / num_ramp_frames)
    return out
