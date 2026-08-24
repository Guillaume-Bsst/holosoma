from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.contact_targets import beta_from_distance, relative_position_in_object_frame
from holosoma.utils.file_cache import cached_open
from holosoma.utils.grasp_settle import (
    anneal_prob,
    apply_grasp_transform,
    gather_anchor,
    grasp_relative_transform,
    select_grasp_anchor,
)
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quat_to_angle_axis,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        contact_schedule_file: str = "",
        contact_schedule_ramp_frames: int = 0,
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(motion_file, device)
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, device)

        self._joint_indexes = joint_indexes
        self._body_indexes = body_indexes
        self.time_step_total = self._joint_pos.shape[0]
        self._load_contact_schedule(contact_schedule_file, contact_schedule_ramp_frames, device)

    def _load_contact_schedule(self, path: str, ramp_frames: int, device: str) -> None:
        """Fold an externally supplied MPC contact schedule onto this clip's timeline.

        Takes precedence over whatever contact fields the motion NPZ carries: passing a schedule is
        how you say which contact truth to train against.
        """
        n_frames = self.time_step_total
        self.has_contact_schedule = bool(path)
        if not self.has_contact_schedule:
            self._schedule_hand_contact = torch.zeros(0, 2, device=device)
            self._schedule_foot_ground = torch.zeros(0, 2, device=device)
            self._schedule_object_ground = torch.zeros(0, device=device)
            self._schedule_object_support = torch.zeros(0, device=device)
            return

        from holosoma.utils.contact_schedule import (
            inferred_fps,
            load_mpc_schedule,
            ramp_activation,
            resample_nearest,
        )

        schedule = load_mpc_schedule(resolve_data_file_path(path))
        clip_fps = float(self.fps)

        def to_clip(active: np.ndarray) -> torch.Tensor:
            flat = active.reshape(active.shape[0], -1)
            weights = ramp_activation(resample_nearest(flat, n_frames), ramp_frames)
            return torch.tensor(weights, dtype=torch.float32, device=device).view(n_frames, -1)

        self._schedule_hand_contact = to_clip(schedule.hand_object)  # (T, 2)
        self._schedule_foot_ground = to_clip(schedule.foot_ground)  # (T, 2)
        self._schedule_object_ground = to_clip(schedule.object_ground[:, None])[:, 0]  # (T,)
        self._schedule_object_support = to_clip(schedule.object_support[:, None])[:, 0]  # (T,)
        # The inferred cadence is the one diagnostic worth printing: a correctly paired schedule
        # reads as a familiar rate, a borrowed one as a rate no capture ever ran at.
        logger.info(
            f"[contact_schedule] {path}: {schedule.num_frames} frames -> {n_frames} at "
            f"{clip_fps:g} fps (inferred schedule cadence "
            f"{inferred_fps(schedule.num_frames, n_frames, clip_fps):.2f} fps), hands in contact on "
            f"{float((self._schedule_hand_contact.amax(dim=-1) > 0).float().mean()):.0%} of frames"
            + (f", ramp {ramp_frames} frames" if ramp_frames > 0 else "")
            + (f", pairs ignored (never active): {list(schedule.unmapped)}" if schedule.unmapped else "")
        )

    def _pad_contact_schedule(self, n_seg: int, prepend: bool) -> None:
        """Pad the schedule channels over the default-pose transition with "no contact".

        Same convention as the GT-contact padding above: the interpolated transition between the
        default pose and the clip is not part of the captured motion, so nothing is in contact
        there. Without this the schedule and the clip drift apart by n_seg frames and every
        time_steps lookup lands on the wrong phase.
        """
        if not self.has_contact_schedule:
            return
        for attr in (
            "_schedule_hand_contact",
            "_schedule_foot_ground",
            "_schedule_object_ground",
            "_schedule_object_support",
        ):
            existing = getattr(self, attr)
            shape = (n_seg,) + tuple(existing.shape[1:])
            pad = torch.zeros(shape, dtype=existing.dtype, device=existing.device)
            setattr(self, attr, torch.cat((pad, existing) if prepend else (existing, pad), dim=0))

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    # Expected holosoma NPZ keys
    _REQUIRED_KEYS = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
    }

    def _load_data_from_motion_npz(self, motion_file: str, device: str) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            # Sanity check: warn if not in expected holosoma format
            keys = set(data.files)
            missing = self._REQUIRED_KEYS - keys
            if missing:
                logger.warning(
                    f"Motion NPZ '{motion_file}' is missing expected holosoma keys: {missing}. "
                    f"All motion data should be in holosoma format (with body_names, joint_names, "
                    f"and root DOFs in joint_pos). Convert from TML/BeyondMimic first."
                )
                raise ValueError(
                    f"Unsupported motion format in '{motion_file}': missing keys {missing}. "
                    f"Please convert to holosoma format."
                )

            self.fps = data["fps"]

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            joint_pos_raw = data["joint_pos"]
            joint_vel_raw = data["joint_vel"]
            body_pos_w_raw = data["body_pos_w"]
            body_quat_w_raw = data["body_quat_w"]
            body_lin_vel_w_raw = data["body_lin_vel_w"]
            body_ang_vel_w_raw = data["body_ang_vel_w"]

            # Holosoma format: joint_pos includes root DOFs [xyz, wxyz] as first 7 values
            # joint_vel includes root velocity [vel_xyz, vel_wxyz] as first 6 values
            num_joint_cols = joint_pos_raw.shape[1]
            num_vel_cols = joint_vel_raw.shape[1]
            num_bodies = body_pos_w_raw.shape[1]

            if num_joint_cols != len(joint_names) + 7:
                logger.warning(
                    f"Unexpected joint_pos columns: got {num_joint_cols}, expected {len(joint_names) + 7} "
                    f"(= {len(joint_names)} joints + 7 root DOFs). File: {motion_file}"
                )
            if num_vel_cols != len(joint_names) + 6:
                logger.warning(
                    f"Unexpected joint_vel columns: got {num_vel_cols}, expected {len(joint_names) + 6} "
                    f"(= {len(joint_names)} joints + 6 root DOFs). File: {motion_file}"
                )
            if num_bodies != len(body_names):
                logger.warning(
                    f"Body count mismatch: body_pos_w has {num_bodies} bodies but body_names has "
                    f"{len(body_names)}. File: {motion_file}"
                )

            # Strip root DOFs
            self._joint_pos = torch.tensor(joint_pos_raw[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel_raw[:, 6:], dtype=torch.float32, device=device)

            assert len(joint_names) == self._joint_pos.shape[1], (
                f"Joint names ({len(joint_names)}) != joint_pos columns ({self._joint_pos.shape[1]}) in {motion_file}"
            )
            assert len(body_names) == body_pos_w_raw.shape[1], (
                f"Body names ({len(body_names)}) != body_pos_w bodies ({body_pos_w_raw.shape[1]}) in {motion_file}"
            )

            self._body_pos_w = torch.tensor(body_pos_w_raw, dtype=torch.float32, device=device)

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(body_quat_w_raw, dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w_raw, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w_raw, dtype=torch.float32, device=device)

            # add object pos and quat
            self.has_object = "object_pos_w" in data
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                # NOTE: wxyz after loading from npz
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
                # Reference ANGULAR velocity. Written by the retargeting converter
                # (convert_data_format_mj.py) but absent from older clips, so it is optional on top
                # of has_object: fall back to zeros of the right shape, which keeps every concat /
                # interpolation path below unconditional. Consumers guard on has_object_ang_vel
                # rather than on the tensor being non-empty.
                self.has_object_ang_vel = "object_ang_vel_w" in data
                if self.has_object_ang_vel:
                    self._object_ang_vel_w = torch.tensor(
                        data["object_ang_vel_w"], dtype=torch.float32, device=device
                    )
                else:
                    self._object_ang_vel_w = torch.zeros_like(self._object_lin_vel_w)
            else:
                self.has_object_ang_vel = False
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
                self._object_ang_vel_w = torch.zeros(0, 3, device=device)

            # Ground-truth hand<->object contact from the retargeting pipeline's own point-cloud
            # interaction fields (witness/distance/normal + active-in-margin, real mesh-to-mesh SDF
            # against the captured demo -- see gvhmr-fp-pipeline/contact_from_retarget.py). Optional:
            # motions without it fall back to the runtime nearest-anchor distance threshold
            # (MotionCommand._lookup_ref_contact).
            self.has_gt_contact = self.has_object and "object_ref_contact" in data
            if self.has_gt_contact:
                self._object_ref_contact = torch.tensor(data["object_ref_contact"], dtype=torch.bool, device=device)
                self._object_ref_contact_dist = torch.tensor(
                    data["object_ref_contact_dist"], dtype=torch.float32, device=device
                )
                self._object_ref_anchor_idx = torch.tensor(
                    data["object_ref_anchor_idx"], dtype=torch.long, device=device
                )
            else:
                self._object_ref_contact = torch.zeros(0, dtype=torch.bool, device=device)
                self._object_ref_contact_dist = torch.zeros(0, device=device)
                self._object_ref_anchor_idx = torch.zeros(0, dtype=torch.long, device=device)

            # Reference witness point (nearest box-surface point to the contact hand, BOX-LOCAL frame
            # -- see contact_from_retarget.py) for the surface-geodesic reward. Optional on top of
            # has_gt_contact: older GT-contact NPZs without it just don't get the geodesic term.
            self.has_gt_witness = self.has_gt_contact and "object_ref_witness_local" in data
            if self.has_gt_witness:
                self._object_ref_witness_local = torch.tensor(
                    data["object_ref_witness_local"], dtype=torch.float32, device=device
                )
            else:
                self._object_ref_witness_local = torch.zeros(0, 3, device=device)

            # Static support (table) carried by the clip: a fixed-pose scene object (clip-world
            # frame, same as body_pos_w / object_pos_w) with its own centered mesh, spawned once
            # per env. Replaces the terrain-baked table + the run_sim add_support AABB hack: the
            # box is DEPOSITED on this surface, so it needs real collision + SDF, not floor.
            self.has_support = "support_pos_w" in data
            if self.has_support:
                self._support_pos_w = torch.tensor(data["support_pos_w"], dtype=torch.float32, device=device)
                sq = torch.tensor(data["support_quat_w"], dtype=torch.float32, device=device)  # wxyz
                self._support_quat_w = sq[[1, 2, 3, 0]]  # -> xyzw (runtime convention)
                self.support_mesh = str(data["support_mesh"]) if "support_mesh" in data else ""
                self._support_half_extents = torch.tensor(
                    data["support_half_extents"] if "support_half_extents" in data else [0.0, 0.0, 0.0],
                    dtype=torch.float32, device=device,
                )
            else:
                self._support_pos_w = torch.zeros(3, device=device)
                self._support_quat_w = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
                self.support_mesh = ""
                self._support_half_extents = torch.zeros(3, device=device)

            # Reference robot<->support contact (SYMMETRIC to object_ref_*: same shape, computed
            # by add_support_contact.py = wrist -> table surface, witness + distance). Feeds the
            # support_surface_contact_error_exp reward (the robot learns to place its hand near the
            # table, not to barge into it). Frames T (T,) / witness (T,3) table-LOCAL.
            self.has_support_contact = self.has_support and "support_ref_contact" in data
            if self.has_support_contact:
                self._support_ref_contact = torch.tensor(data["support_ref_contact"], dtype=torch.bool, device=device)
                self._support_ref_contact_dist = torch.tensor(
                    data["support_ref_contact_dist"], dtype=torch.float32, device=device
                )
                self._support_ref_anchor_idx = torch.tensor(
                    data["support_ref_anchor_idx"], dtype=torch.long, device=device
                )
                self._support_ref_witness_local = torch.tensor(
                    data["support_ref_witness_local"], dtype=torch.float32, device=device
                )
            else:
                self._support_ref_contact = torch.zeros(0, dtype=torch.bool, device=device)
                self._support_ref_contact_dist = torch.zeros(0, device=device)
                self._support_ref_anchor_idx = torch.zeros(0, dtype=torch.long, device=device)
                self._support_ref_witness_local = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self._object_ang_vel_w[:]

    @property
    def object_ref_contact(self) -> torch.Tensor:
        return self._object_ref_contact[:]

    @property
    def object_ref_contact_dist(self) -> torch.Tensor:
        return self._object_ref_contact_dist[:]

    @property
    def object_ref_anchor_idx(self) -> torch.Tensor:
        return self._object_ref_anchor_idx[:]

    @property
    def object_ref_witness_local(self) -> torch.Tensor:
        return self._object_ref_witness_local[:]

    @property
    def support_ref_contact(self) -> torch.Tensor:
        return self._support_ref_contact[:]

    @property
    def support_ref_contact_dist(self) -> torch.Tensor:
        return self._support_ref_contact_dist[:]

    @property
    def support_ref_anchor_idx(self) -> torch.Tensor:
        return self._support_ref_anchor_idx[:]

    @property
    def support_ref_witness_local(self) -> torch.Tensor:
        return self._support_ref_witness_local[:]

    @property
    def support_half_extents(self) -> torch.Tensor:
        return self._support_half_extents[:]

    @property
    def num_motions(self) -> int:
        return 1

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self._joint_pos.device)

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return torch.tensor([self.time_step_total], dtype=torch.long, device=self._joint_pos.device)

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                    ("object_ang_vel", "_object_ang_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self._pad_contact_schedule(segments["joint_pos"].shape[0], prepend)

        if self.has_gt_contact:
            # The prepended/appended transition (default pose <-> clip) has no real demo contact --
            # pad with "no contact" (False / large distance / anchor 0), same convention `active`
            # uses outside its margin (see targets/interaction/fields.py in HoloV2).
            n_seg = segments["joint_pos"].shape[0]
            pad_contact = torch.zeros(n_seg, dtype=torch.bool, device=self._object_ref_contact.device)
            pad_dist = torch.full((n_seg,), 999.0, device=self._object_ref_contact_dist.device)
            pad_anchor = torch.zeros(n_seg, dtype=torch.long, device=self._object_ref_anchor_idx.device)
            self._object_ref_contact = torch.cat(
                (pad_contact, self._object_ref_contact) if prepend else (self._object_ref_contact, pad_contact),
                dim=0,
            )
            self._object_ref_contact_dist = torch.cat(
                (pad_dist, self._object_ref_contact_dist)
                if prepend
                else (self._object_ref_contact_dist, pad_dist),
                dim=0,
            )
            self._object_ref_anchor_idx = torch.cat(
                (pad_anchor, self._object_ref_anchor_idx)
                if prepend
                else (self._object_ref_anchor_idx, pad_anchor),
                dim=0,
            )
            if self.has_gt_witness:
                pad_witness = torch.zeros(n_seg, 3, device=self._object_ref_witness_local.device)
                self._object_ref_witness_local = torch.cat(
                    (pad_witness, self._object_ref_witness_local)
                    if prepend
                    else (self._object_ref_witness_local, pad_witness),
                    dim=0,
                )

        # support (table) contact reference: same "no contact" padding on the transition.
        if getattr(self, "has_support_contact", False):
            n_seg = segments["joint_pos"].shape[0]
            s_contact = torch.zeros(n_seg, dtype=torch.bool, device=self._support_ref_contact.device)
            s_dist = torch.full((n_seg,), 999.0, device=self._support_ref_contact_dist.device)
            s_anchor = torch.zeros(n_seg, dtype=torch.long, device=self._support_ref_anchor_idx.device)
            s_witness = torch.zeros(n_seg, 3, device=self._support_ref_witness_local.device)
            self._support_ref_contact = torch.cat(
                (s_contact, self._support_ref_contact) if prepend else (self._support_ref_contact, s_contact), dim=0
            )
            self._support_ref_contact_dist = torch.cat(
                (s_dist, self._support_ref_contact_dist) if prepend else (self._support_ref_contact_dist, s_dist), dim=0
            )
            self._support_ref_anchor_idx = torch.cat(
                (s_anchor, self._support_ref_anchor_idx) if prepend else (self._support_ref_anchor_idx, s_anchor), dim=0
            )
            self._support_ref_witness_local = torch.cat(
                (s_witness, self._support_ref_witness_local)
                if prepend
                else (self._support_ref_witness_local, s_witness),
                dim=0,
            )

        self.time_step_total = self._joint_pos.shape[0]
        return self


class MultiMotionLoader:
    """Loads multiple NPZ motion files from a directory and concatenates them at runtime.

    Tracks per-motion boundaries so environments can sample within individual clips.
    Compatible with the same interface as MotionLoader.
    """

    def __init__(
        self,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        contact_schedule_file: str = "",
        contact_schedule_ramp_frames: int = 0,
    ):
        # A schedule is baked for ONE take. Applied to a concatenation of clips it would line up
        # with the first one and be pure noise on the rest -- refuse instead of silently mislabeling
        # every contact frame after the first clip.
        if contact_schedule_file:
            raise ValueError(
                "contact_schedule_file is not supported with motion_dir: a schedule is baked for a "
                "single take, so there is no correct way to apply one to a concatenation of clips. "
                "Use motion_file with the clip the schedule was baked for."
            )
        del contact_schedule_ramp_frames  # only meaningful with a file
        self.has_contact_schedule = False
        # Support comma-separated directories for combining multiple datasets
        dirs = [d.strip() for d in motion_dir.split(",")]
        motion_files = []
        for d in dirs:
            expanded = os.path.expanduser(d)
            files = sorted(str(p) for p in Path(expanded).glob("*.npz"))
            logger.info(f"MultiMotionLoader: found {len(files)} .npz files in {expanded}")
            motion_files.extend(files)
        assert len(motion_files) > 0, f"No .npz files found in {motion_dir}"
        logger.info(f"MultiMotionLoader: loading {len(motion_files)} total motion files")

        loaders = []
        skipped = 0
        for mf in motion_files:
            try:
                loader = MotionLoader(mf, robot_body_names, robot_joint_names, device=device)
                loaders.append(loader)
            except (KeyError, AssertionError, ValueError) as e:  # noqa: PERF203
                # Skip files with incompatible format (e.g., missing body_names, wrong body count)
                skipped += 1
                if skipped <= 3:
                    logger.warning(f"MultiMotionLoader: skipping {mf}: {e}")
        if skipped > 3:
            logger.warning(f"MultiMotionLoader: skipped {skipped} files total due to format issues")
        assert len(loaders) > 0, f"No compatible motion files found (skipped {skipped})"

        # Track per-motion boundaries
        lengths = [loader.time_step_total for loader in loaders]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(loaders)

        # Concatenate all motion data
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld._body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld._body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld._body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld._body_ang_vel_w for ld in loaders], dim=0)

        # Use indexes from first loader (all loaders share the same robot)
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = loaders[0]._body_indexes
        self.fps = loaders[0].fps
        self.time_step_total = self._joint_pos.shape[0]

        # Object support: only if ALL motions have objects
        self.has_object = all(ld.has_object for ld in loaders)
        if self.has_object:
            self._object_pos_w = torch.cat([ld._object_pos_w for ld in loaders], dim=0)
            self._object_quat_w = torch.cat([ld._object_quat_w for ld in loaders], dim=0)
            self._object_lin_vel_w = torch.cat([ld._object_lin_vel_w for ld in loaders], dim=0)
            self._object_ang_vel_w = torch.cat([ld._object_ang_vel_w for ld in loaders], dim=0)
            # Usable only if EVERY clip carries it (same rule as has_gt_contact): a mix would train
            # the angular term against zeros on the clips that lack it.
            self.has_object_ang_vel = all(ld.has_object_ang_vel for ld in loaders)
        else:
            self.has_object_ang_vel = False
            self._object_pos_w = torch.zeros(0, 3, device=device)
            self._object_quat_w = torch.zeros(0, 4, device=device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=device)
            self._object_ang_vel_w = torch.zeros(0, 3, device=device)

        # Ground-truth contact (see MotionLoader): only usable if ALL clips carry it.
        self.has_gt_contact = self.has_object and all(ld.has_gt_contact for ld in loaders)
        if self.has_gt_contact:
            self._object_ref_contact = torch.cat([ld._object_ref_contact for ld in loaders], dim=0)
            self._object_ref_contact_dist = torch.cat([ld._object_ref_contact_dist for ld in loaders], dim=0)
            self._object_ref_anchor_idx = torch.cat([ld._object_ref_anchor_idx for ld in loaders], dim=0)
        else:
            self._object_ref_contact = torch.zeros(0, dtype=torch.bool, device=device)
            self._object_ref_contact_dist = torch.zeros(0, device=device)
            self._object_ref_anchor_idx = torch.zeros(0, dtype=torch.long, device=device)

        self.has_gt_witness = self.has_gt_contact and all(ld.has_gt_witness for ld in loaders)
        if self.has_gt_witness:
            self._object_ref_witness_local = torch.cat([ld._object_ref_witness_local for ld in loaders], dim=0)
        else:
            self._object_ref_witness_local = torch.zeros(0, 3, device=device)

        logger.info(f"MultiMotionLoader: {self._num_motions} motions, {self.time_step_total} total frames")

    @property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return self._motion_start_idx

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return self._motion_end_idx

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos[:, self._joint_indexes]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel[:, self._joint_indexes]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w[:]

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w[:]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w[:]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self._object_ang_vel_w[:]

    @property
    def object_ref_contact(self) -> torch.Tensor:
        return self._object_ref_contact[:]

    @property
    def object_ref_contact_dist(self) -> torch.Tensor:
        return self._object_ref_contact_dist[:]

    @property
    def object_ref_anchor_idx(self) -> torch.Tensor:
        return self._object_ref_anchor_idx[:]

    @property
    def object_ref_witness_local(self) -> torch.Tensor:
        return self._object_ref_witness_local[:]

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MultiMotionLoader:
        """Merge interpolated segments with motion data, mutating this MultiMotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                    ("object_ang_vel", "_object_ang_vel_w"),
                ]
            )

        added_frames = 0
        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))
            if added_frames == 0:
                added_frames = segments[seg_key].shape[0]

        if self.has_gt_contact:
            # See MotionLoader.extend_with_segments: the transition has no real demo contact.
            pad_contact = torch.zeros(added_frames, dtype=torch.bool, device=self._object_ref_contact.device)
            pad_dist = torch.full((added_frames,), 999.0, device=self._object_ref_contact_dist.device)
            pad_anchor = torch.zeros(added_frames, dtype=torch.long, device=self._object_ref_anchor_idx.device)
            self._object_ref_contact = torch.cat(
                (pad_contact, self._object_ref_contact) if prepend else (self._object_ref_contact, pad_contact),
                dim=0,
            )
            self._object_ref_contact_dist = torch.cat(
                (pad_dist, self._object_ref_contact_dist)
                if prepend
                else (self._object_ref_contact_dist, pad_dist),
                dim=0,
            )
            self._object_ref_anchor_idx = torch.cat(
                (pad_anchor, self._object_ref_anchor_idx)
                if prepend
                else (self._object_ref_anchor_idx, pad_anchor),
                dim=0,
            )
            if self.has_gt_witness:
                pad_witness = torch.zeros(added_frames, 3, device=self._object_ref_witness_local.device)
                self._object_ref_witness_local = torch.cat(
                    (pad_witness, self._object_ref_witness_local)
                    if prepend
                    else (self._object_ref_witness_local, pad_witness),
                    dim=0,
                )

        # Update boundaries — shift all motion boundaries if prepending
        if prepend:
            self._motion_start_idx = self._motion_start_idx + added_frames
            self._motion_end_idx = self._motion_end_idx + added_frames
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [torch.tensor([0], dtype=torch.long, device=dev), self._motion_start_idx]
            )
            self._motion_end_idx = torch.cat(
                [torch.tensor([added_frames], dtype=torch.long, device=dev), self._motion_end_idx]
            )
        else:
            old_total = self.time_step_total
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [self._motion_start_idx, torch.tensor([old_total], dtype=torch.long, device=dev)]
            )
            self._motion_end_idx = torch.cat(
                [self._motion_end_idx, torch.tensor([old_total + added_frames], dtype=torch.long, device=dev)]
            )

        self.time_step_total = self._joint_pos.shape[0]
        self._num_motions = len(self._motion_start_idx)
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often."""

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_uniform_ratio = adaptive_uniform_ratio
        self.adaptive_alpha = adaptive_alpha

        # Match BeyondMimic binning: ~1 second bins at env FPS, with +1 tail bin.
        self.num_bins = int(self.motion_time_step_total // max(self.env_fps, 1)) + 1

        # Match BeyondMimic non-causal kernel.
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}

    def init_buffers(self):
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor):
        """Update the current bin failed count with terminated time steps."""
        failed_bin = torch.clamp(
            (failed_at_time_step * self.num_bins) // max(self.motion_time_step_total, 1),
            0,
            self.num_bins - 1,
        ).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        self.current_bin_failed_count[:] = torch.bincount(failed_bin, minlength=self.num_bins)

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = self.bin_failed_count + self.adaptive_uniform_ratio / float(self.num_bins)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        return sampling_probabilities / sampling_probabilities.sum()

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def get_stats(self):
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        H_norm = H / np.log(self.num_bins)
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = imax.float() / self.num_bins


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # self.motion_cfg: MotionConfig = cfg.params["motion_config"]
        # TODO(jchen):temporary fix for motion_config being a dict after tyro.cli
        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        assert self.motion_cfg.motion_file or self.motion_cfg.motion_dir, (
            "Either motion_file or motion_dir must be set in MotionConfig"
        )
        self.motion: MotionLoader | MultiMotionLoader
        if self.motion_cfg.motion_dir:
            self.motion = MultiMotionLoader(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                contact_schedule_file=self.motion_cfg.contact_schedule_file,
                contact_schedule_ramp_frames=self.motion_cfg.contact_schedule_ramp_frames,
            )
        else:
            self.motion = MotionLoader(
                self.motion_cfg.motion_file,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                contact_schedule_file=self.motion_cfg.contact_schedule_file,
                contact_schedule_ramp_frames=self.motion_cfg.contact_schedule_ramp_frames,
            )

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )

        # 3. get the name of the object, or indices of the object
        if self.motion.has_object:
            # cache the object_index_in_simulator
            self.object_name = "object"  # hardcoded object name
            self.object_indices_in_simulator = self._env.simulator.get_actor_indices(self.object_name, env_ids=None)

            assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
                "Object is only supported in IsaacSim"
            )

        # static support (table): if the clip carries one AND the sim spawned it, cache its actor
        # index. Posed once here at setup (static -> no per-reset write needed). Per-env pos =
        # clip support_pos_w + env_origin (like the box's object_pos_w property).
        self.support_name = "support"
        self.has_support_actor = (
            self.motion.has_support
            and self._env.robot_config.object.support_urdf_path is not None
            and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM
            and "support" in self._env.simulator.scene.rigid_objects
        )
        if self.has_support_actor:
            self.support_indices_in_simulator = self._env.simulator.get_actor_indices("support", env_ids=None)
            self._plant_support(torch.arange(self._env.num_envs, device=self.device))
            logger.info(
                f"[support] static table spawned at clip pos {self.motion._support_pos_w.tolist()} "
                f"(mesh {self.motion.support_mesh})"
            )

        # 3b. grasp-settle: resolve the candidate hand/anchor bodies (robot body order).
        # Resolved whenever the motion has an object (independent of the `enable` flag) so the
        # probe harness can toggle settling on at runtime via _settle_enabled_override.
        self.grasp_settle_cfg = self.motion_cfg.grasp_settle
        self._anchor_body_indexes: torch.Tensor | None = None
        if self.motion.has_object:
            anchor_idxs = []
            for name in self.grasp_settle_cfg.anchor_body_names:
                resolved = FAKE_BODY_NAME_ALIASES.get(name, name)
                if resolved in robot_body_names:
                    anchor_idxs.append(robot_body_names.index(resolved))
                else:
                    logger.warning(f"[grasp_settle] anchor body '{name}' not in robot bodies; skipping")
            assert len(anchor_idxs) > 0, (
                "grasp_settle is enabled but none of anchor_body_names were found in the robot bodies: "
                f"{self.grasp_settle_cfg.anchor_body_names}"
            )
            self._anchor_body_indexes = torch.tensor(anchor_idxs, dtype=torch.long, device=self.device)
            if self.grasp_settle_cfg.enable:
                logger.info(
                    f"[grasp_settle] enabled: anchors={self.grasp_settle_cfg.anchor_body_names}, "
                    f"settle_steps={self.grasp_settle_cfg.settle_steps}, "
                    f"contact_thr={self.grasp_settle_cfg.contact_distance_threshold}m, "
                    f"weld={self.grasp_settle_cfg.weld_object_during_settle}"
                )

        # 3c. C-D lite: derive the relative hand<->object reference and the beta weight once.
        # Done AFTER the default-pose transitions (motion already extended) -> indexed by time_steps.
        if getattr(self.motion, "has_object", False):
            self.hand_body_indexes = self._get_index_of_a_in_b(
                self.motion_cfg.hand_body_names, robot_body_names, self.device
            )  # (H,) sim body order, aligned with _rigid_body_pos AND self.motion.body_pos_w
            n_hand = self.hand_body_indexes.numel()
            wrist_pos_ref = self.motion.body_pos_w[:, self.hand_body_indexes]           # (T, H, 3)
            obj_pos_ref = self.motion.object_pos_w[:, None, :].repeat(1, n_hand, 1)     # (T, H, 3)
            obj_quat_ref = self.motion.object_quat_w[:, None, :].repeat(1, n_hand, 1)   # (T, H, 4)
            self._hand_obj_rel_pos_ref = relative_position_in_object_frame(
                wrist_pos_ref, obj_pos_ref, obj_quat_ref
            )  # (T, H, 3)
            d_demo = torch.norm(wrist_pos_ref - obj_pos_ref, dim=-1)                    # (T, H)
            self._hand_obj_beta = beta_from_distance(d_demo, self.motion_cfg.beta_scale)  # (T, H)

        # 4. get the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total, self.device, int(1 / (self._env.dt))
            )

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()

        # 6. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        # 0. Sample the time steps
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            # Match BeyondMimic behavior: update failed bins from environments
            # that terminated before this reset, then sample new phases.
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                failed_at_time_step = self.time_steps[env_ids][episode_failed]
                self.adaptive_timesteps_sampler.update_current_bin_failed_count(failed_at_time_step)
            phase = self.adaptive_timesteps_sampler.sample(env_ids.numel())
        else:
            phase = torch.rand(env_ids.numel(), device=self.device)

        if self._env.is_evaluating and self.motion_cfg.eval_start_at_zero:
            phase = torch.zeros_like(phase)

        # For multi-motion: randomly assign each env to a motion, sample within that motion's range
        n = env_ids.numel()
        num_motions = self.motion.num_motions
        self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)
        start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
        motion_len = end_idx - start_idx

        self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()

        # Handle start_at_timestep_zero_prob (reset to start of assigned motion)
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = start_idx
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, start_idx, subset)
            self.time_steps[env_ids] = subset

        # Debug hook (probe harness): force a specific absolute start frame per env instead of
        # sampling a phase, so a mid-clip contact frame can be reproduced deterministically.
        if self._force_start_timesteps is not None:
            forced = self._force_start_timesteps.to(device=self.device, dtype=torch.long)
            end_clamp = self.motion.motion_end_idx[self.motion_ids[env_ids]] - 2
            self.time_steps[env_ids] = torch.minimum(forced[env_ids].clamp(min=0), end_clamp)

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
        self.time_steps[env_ids] = torch.where(already_last_timestep_mask, end_idx - 2, self.time_steps[env_ids])

        # --- grasp-settle: detect contact resets & prepare per-env init scaling -------------------
        # A reset frame is "in contact" if the nearest tracked hand is within contact_distance_threshold
        # of the object. For those envs we (a) spawn the object at its reference pose with no independent
        # noise, (b) optionally scale down the robot init noise, and (c) arm a settle window (see step()).
        settle_on = self._settle_enabled()
        contact_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        robot_noise_scale: torch.Tensor | float = 1.0
        if settle_on:
            anchor_pos = self.anchor_pos_w[env_ids]  # (n, A, 3) reference hand positions
            anchor_quat = self.anchor_quat_w[env_ids]  # (n, A, 4)
            obj_pos_ref = self.object_pos_w[env_ids]  # (n, 3)
            obj_quat_ref = self.object_quat_w[env_ids]  # (n, 4)

            anchor_idx, contact_mask = self._lookup_ref_contact(self.time_steps[env_ids], anchor_pos, obj_pos_ref)

            # grasp transform from the chosen (reference, un-noised) hand -> stored for the weld in step()
            a_pos, a_quat = gather_anchor(anchor_pos, anchor_quat, anchor_idx)
            rel_pos, rel_quat = grasp_relative_transform(a_pos, a_quat, obj_pos_ref, obj_quat_ref)
            self.settle_anchor_idx[env_ids] = anchor_idx
            self.settle_grasp_rel_pos[env_ids] = rel_pos
            self.settle_grasp_rel_quat[env_ids] = rel_quat
            self.settle_counter[env_ids] = torch.where(
                contact_mask,
                torch.full_like(anchor_idx, self.grasp_settle_cfg.settle_steps),
                torch.zeros_like(anchor_idx),
            )
            # reduce robot init noise on contact resets (default 0.0 -> spawn exactly at the reference)
            robot_noise_scale = torch.where(
                contact_mask.unsqueeze(-1),
                torch.full((n, 1), float(self.grasp_settle_cfg.settle_robot_noise_scale), device=self.device),
                torch.ones(n, 1, device=self.device),
            )  # (n, 1), broadcasts over dof/root noise

            # full-contact weld curriculum: draw the per-episode assist flag with the annealed prob.
            # Assisted episodes carry the object kinematically at the reference grasp during ALL
            # reference-contact frames (see step()), so early training never sees an unholdable box.
            self.weld_assist_prob = anneal_prob(
                self._env_step_counter,
                self.grasp_settle_cfg.weld_contact_prob_start,
                self.grasp_settle_cfg.weld_contact_prob_end,
                self.grasp_settle_cfg.weld_anneal_steps,
            )
            self.weld_assist[env_ids] = torch.rand(n, device=self.device) < self.weld_assist_prob
        else:
            # keep stale settle state from lingering on these envs when settling is off
            self.settle_counter[env_ids] = 0
            self.weld_assist[env_ids] = False

        # 1. Get the root/body poses from the motion data
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # robot_noise_scale is (n, 1) and 1.0 for normal resets; on grasp-settle contact resets it is
        # settle_robot_noise_scale (default 0.0) so the robot spawns exactly at the reference contact pose.
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise * robot_noise_scale
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0) * robot_noise_scale  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (
            (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy * robot_noise_scale
        )
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0) * robot_noise_scale  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0) * robot_noise_scale  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        self._env.simulator.robot_root_states[env_ids, 7:10] = target_root_lin_vel
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]
            # Angular velocity used to be hard-zeroed here while the linear half took the reference,
            # because object_ang_vel_w was never loaded. On clips that do not carry the field the
            # accessor returns zeros, so those degrade to exactly the previous behaviour.
            obj_ang_vel = self.object_ang_vel_w[env_ids]

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            obj_noise = (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise
            if settle_on:
                # Contact resets: no independent object noise and start at rest, so the object stays
                # exactly in the reference grasp (contact-consistent placement). Free-frame resets
                # (object on the ground, hands away) keep the original noisy behaviour.
                keep = (~contact_mask).float().unsqueeze(-1)  # 1.0 free frame, 0.0 contact frame
                obj_noise = obj_noise * keep
                obj_lin_vel = obj_lin_vel * keep
                obj_ang_vel = obj_ang_vel * keep
            target_obj_pos = obj_pos + obj_noise

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, obj_ang_vel], dim=-1
            )  # (num_envs, 13)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

        # 5. re-plant the static support (the box may have nudged it during the episode)
        if getattr(self, "has_support_actor", False):
            self._plant_support(env_ids)

    def _plant_support(self, env_ids: torch.Tensor) -> None:
        """Write the static support (table) root pose = clip support pose + env origin, zero vel."""
        state = torch.zeros(env_ids.numel(), 13, device=self.device)
        state[:, :3] = self.motion._support_pos_w.unsqueeze(0) + self._env.simulator.scene.env_origins[env_ids]
        state[:, 3:7] = self.motion._support_quat_w.unsqueeze(0)
        self._env.simulator.set_actor_states(["support"], env_ids, state)

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # grasp-settle: hold the clip frozen on envs whose settle window is still open, so contact
        # equilibrates before the tracked motion resumes.
        settle_on = self._settle_enabled()
        if settle_on and self.grasp_settle_cfg.freeze_clip_during_settle:
            advance_mask = advance_mask & (self.settle_counter == 0)

        # Handle freeze_at_timestep_zero_prob: for envs at their motion's start, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == self.motion.motion_start_idx[self.motion_ids]
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        # Handle freeze_at_timestep_end_prob: mirror of the start freeze for the LAST frame
        # (motion_end_idx - 1, the appended default pose). Without this the end-of-clip hold is
        # never trained — the block below resets the instant the counter reaches motion_end_idx —
        # so holding the final pose is out-of-distribution at inference. Randomly hold at the last
        # frame instead of advancing into the reset, so the policy practices keeping the final pose.
        end_freeze_prob = self.motion_cfg.freeze_at_timestep_end_prob
        if end_freeze_prob > 0.0:
            last_mask = self.time_steps == (self.motion.motion_end_idx[self.motion_ids] - 1)
            if last_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                end_freeze_mask = (rand_vals < end_freeze_prob) & last_mask
                advance_mask = advance_mask & ~end_freeze_mask

        self.time_steps += advance_mask.long()

        # BeyondMimic-style behavior: when the clip ends, resample motion and
        # reset robot/object state without terminating the whole episode.
        per_motion_end = self.motion.motion_end_idx[self.motion_ids]
        ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() > 0:
            self.reset(ended_env_ids)
            # Flush the mutated root/dof state into the simulator so that
            # rigid-body positions are up-to-date for downstream consumers
            # (termination checks, observations, rewards).
            sim = self._env.simulator
            sim.set_actor_root_state_tensor_robots(ended_env_ids, sim.robot_root_states)
            sim.set_dof_state_tensor_robots(ended_env_ids, sim.dof_state)  # type: ignore[attr-defined]
            sim.refresh_sim_tensors()

        # grasp-settle welds, one unified path (per policy step):
        #  - settle-window weld (weld_object_during_settle): hold the object through the reset
        #    transient. The clip is frozen during the window, so the current-reference grasp equals
        #    the reset-time grasp.
        #  - full-contact assist weld (curriculum): on assisted episodes, kinematically carry the
        #    object at the CURRENT reference grasp during all reference-contact frames, so early
        #    training never sees an unholdable box. The assist probability anneals to 0 -> the
        #    final policy holds the object fully physically.
        # Both compute: object_sim = T(hand_sim) o T(hand_ref)^-1 o T(object_ref) at the current frame.
        # NOTE: must run AFTER the end-of-clip reset above — time_steps can transiently equal
        # motion_end (out of bounds for the motion arrays) until that reset resamples them.
        if settle_on:
            self._env_step_counter += 1

            anchor_pos_ref = self.anchor_pos_w  # (N, A, 3) reference hand poses at current frame
            anchor_quat_ref = self.anchor_quat_w
            obj_pos_ref = self.object_pos_w
            obj_quat_ref = self.object_quat_w
            anchor_idx_now, ref_contact = self._lookup_ref_contact(self.time_steps, anchor_pos_ref, obj_pos_ref)

            # Kinematic object during contact: box follows the SMOOTH REFERENCE trajectory (pos+quat+
            # vel from the clip) on every ref-contact frame, always on. The grasp is assumed (real hand
            # grips at deployment); the policy learns body motion + hand placement. Bulletproof: no
            # drift, no tumble, box never triggers bad_object_pos. Supersedes the assist weld.
            if self.grasp_settle_cfg.kinematic_object_during_contact:
                kin_ids = torch.where(ref_contact)[0]
                cfg_gs = self.grasp_settle_cfg
                if kin_ids.numel() > 0:
                    alpha = self._physicality_alpha
                    ref_pos = obj_pos_ref[kin_ids]
                    ref_quat = obj_quat_ref[kin_ids]
                    ref_lin_vel = self.object_lin_vel_w[kin_ids]
                    ref_ang_vel = self.object_ang_vel_w[kin_ids]
                    if alpha >= 0.999:
                        # fully kinematic (fast path): box forced onto the reference
                        kin_state = torch.cat([ref_pos, ref_quat, ref_lin_vel, ref_ang_vel], dim=-1)
                        self._env.simulator.set_actor_states([self.object_name], kin_ids, kin_state)
                    elif cfg_gs.physicality_force_mode:
                        # FORCE-MODE assist: gravity feedforward + bounded tracking PD, the whole
                        # wrench scaled by alpha:
                        #     F = alpha * ( m*g_up + clamp(PD, fmax) )
                        # so the weight the POLICY must carry is (1-alpha)*m*g — linear in alpha,
                        # zero at alpha=1, the whole box at alpha=0.
                        #
                        # Feeding the weight forward is what makes the ladder finishable. With the
                        # cap alone (clamp(PD, alpha*fmax)) the quantity that governs the carry is
                        # the authority left ABOVE gravity, alpha*fmax - m*g: it vanishes at
                        # alpha = m*g/fmax and is negative below, so the assist stops being able to
                        # even levitate the box long before alpha=0 — the state blend's divergence,
                        # merely relocated. A pure PD also sags a permanent m*g/kp against that
                        # constant load. Both disappear once the weight is fed forward exactly.
                        cur = self._env.simulator.all_root_states[self.object_indices_in_simulator][kin_ids]
                        force = cfg_gs.force_assist_kp * (ref_pos - cur[:, :3]) + cfg_gs.force_assist_kd * (
                            ref_lin_vel - cur[:, 7:10]
                        )
                        rel = quat_mul(ref_quat, quat_inverse(cur[:, 3:7], w_last=True), w_last=True)
                        # NB: quat_to_angle_axis's 2nd return is already the rotation VECTOR (axis*angle)
                        rotvec = quat_to_angle_axis(rel)[1]
                        torque = cfg_gs.force_assist_kp_rot * rotvec - cfg_gs.force_assist_kd_rot * (
                            cur[:, 10:13] - ref_ang_vel
                        )
                        # Cache the per-env masses / weights once: both come from the STARTUP mass
                        # randomisation, so they never change afterwards.
                        if self._object_mass is None:
                            sim = self._env.simulator
                            self._object_mass = sim.get_object_masses(self.object_name)
                            self._object_gravity_w = sim.get_object_gravity_force(self.object_name)
                        # Cap the TRACKING term only; the alpha scaling is applied to the whole
                        # wrench below, so the ladder stays linear in alpha. The cap is per-env and
                        # proportional to the mass (equal m/s^2 of authority whatever mass an env
                        # drew), falling back to the absolute cap when track_accel is 0.
                        if cfg_gs.force_assist_track_accel > 0.0:
                            fmax = (self._object_mass[kin_ids] * cfg_gs.force_assist_track_accel).unsqueeze(-1)
                        else:
                            fmax = torch.full(
                                (kin_ids.numel(), 1), cfg_gs.force_assist_fmax, device=self.device
                            )
                        tmax = cfg_gs.force_assist_tmax
                        f_norm = torch.norm(force, dim=-1, keepdim=True)
                        force = force * (fmax / torch.maximum(f_norm, fmax).clamp_min(1e-6))
                        torque = torque * (tmax / torch.norm(torque, dim=-1, keepdim=True).clamp_min(tmax)).clamp(max=1.0)
                        if cfg_gs.force_assist_gravity_comp:
                            # gravity points DOWN -> subtract it to support the weight
                            force = force - self._object_gravity_w[kin_ids]
                        force = alpha * force
                        torque = alpha * torque
                        # zeros for non-contact envs: the wrench PERSISTS in IsaacLab until
                        # overwritten, so the full buffer is rewritten every step.
                        forces_all = torch.zeros(self.num_envs, 3, device=self.device)
                        torques_all = torch.zeros(self.num_envs, 3, device=self.device)
                        if alpha > 1e-4:
                            forces_all[kin_ids] = force
                            torques_all[kin_ids] = torque
                        self._env.simulator.set_object_external_wrench(self.object_name, forces_all, torques_all)
                        # curriculum signal: fraction of contact envs tracking the object
                        err = torch.norm(ref_pos - cur[:, :3], dim=-1)
                        self._obj_track_success = float((err < cfg_gs.physicality_success_obj_err).float().mean())
                    elif alpha > 1e-4:
                        # PHYSICALITY CURRICULUM partial assist: blend the reference with the box's
                        # current PHYSICAL state. box = alpha*ref + (1-alpha)*physical. The box slips/
                        # droops between corrections (physics acts in the substeps), so the policy must
                        # grip to keep it near the reference. As alpha->0 the box becomes fully physical.
                        cur = self._env.simulator.all_root_states[self.object_indices_in_simulator][kin_ids]
                        t = torch.tensor(alpha, device=self.device)
                        blend = torch.cat(
                            [
                                alpha * ref_pos + (1.0 - alpha) * cur[:, :3],
                                slerp(cur[:, 3:7], ref_quat, t),
                                alpha * ref_lin_vel + (1.0 - alpha) * cur[:, 7:10],
                                alpha * ref_ang_vel + (1.0 - alpha) * cur[:, 10:13],
                            ],
                            dim=-1,
                        )
                        self._env.simulator.set_actor_states([self.object_name], kin_ids, blend)
                    # alpha <= 1e-4: fully physical -> no override, the box is free
                elif cfg_gs.physicality_force_mode and self._physicality_alpha < 0.999:
                    # no env in ref-contact this step: clear any persistent wrench
                    zeros = torch.zeros(self.num_envs, 3, device=self.device)
                    self._env.simulator.set_object_external_wrench(self.object_name, zeros, zeros)

            weld_mask = self.weld_assist & ref_contact
            if self.grasp_settle_cfg.weld_object_during_settle:
                weld_mask = weld_mask | (self.settle_counter > 0)
            weld_ids = torch.where(weld_mask)[0]
            if weld_ids.numel() > 0:
                a_pos_ref, a_quat_ref = gather_anchor(
                    anchor_pos_ref[weld_ids], anchor_quat_ref[weld_ids], anchor_idx_now[weld_ids]
                )
                rel_pos, rel_quat = grasp_relative_transform(
                    a_pos_ref, a_quat_ref, obj_pos_ref[weld_ids], obj_quat_ref[weld_ids]
                )
                a_pos_sim, a_quat_sim = gather_anchor(
                    self.robot_anchor_pos_w[weld_ids],
                    self.robot_anchor_quat_w[weld_ids],
                    anchor_idx_now[weld_ids],
                )
                op, oq = apply_grasp_transform(a_pos_sim, a_quat_sim, rel_pos, rel_quat)
                zeros6 = torch.zeros(weld_ids.numel(), 6, device=self.device)
                self._env.simulator.set_actor_states(
                    [self.object_name], weld_ids, torch.cat([op, oq, zeros6], dim=-1)
                )
            # advance the settle window: once the counter hits 0 the clip resumes and the weld releases
            self.settle_counter = torch.clamp(self.settle_counter - 1, min=0)

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.update_bin_failed_count()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self.motion.body_pos_w[self.time_steps][:, self.tracked_body_indexes]
            + self._env.simulator.scene.env_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps][:, self.tracked_body_indexes]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.ref_body_index] + self._env.simulator.scene.env_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.ref_body_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.ref_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, 0] + self._env.simulator.scene.env_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, 0]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Applies env origins, but ideally we should rely on the simulator
        return self.motion.object_pos_w[self.time_steps] + self._env.simulator.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.object_lin_vel_w[self.time_steps]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self.motion.object_ang_vel_w[self.time_steps]

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    @property
    def simulator_object_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 10:13]

    #########################################################################################
    ## Support (table): STATIC object. World pose = clip pose + env origin (the actor is pinned
    ## at that pose on reset, so there is no need to read its root state -> less plumbing).
    ## Consumed by the obs (the robot sees the table) and by the robot<->table reward.
    #########################################################################################
    @property
    def support_pos_w(self) -> torch.Tensor:
        """World position of the static support per env: clip pose + env origin. (num_envs, 3)."""
        return self.motion._support_pos_w.unsqueeze(0) + self._env.simulator.scene.env_origins

    @property
    def support_quat_w(self) -> torch.Tensor:
        """World orientation (xyzw) of the static support, broadcast per env. (num_envs, 4)."""
        return self.motion._support_quat_w.unsqueeze(0).expand(self._env.num_envs, 4)

    #########################################################################################
    ## Grasp-settle: anchor (hand) poses from motion data & from simulator
    #########################################################################################
    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Reference (motion) world positions of the candidate anchor bodies. (num_envs, A, 3)."""
        return (
            self.motion.body_pos_w[self.time_steps][:, self._anchor_body_indexes]
            + self._env.simulator.scene.env_origins[:, None, :]
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Reference (motion) world orientations of the candidate anchor bodies (xyzw). (num_envs, A, 4)."""
        return self.motion.body_quat_w[self.time_steps][:, self._anchor_body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Live simulator world positions of the candidate anchor bodies. (num_envs, A, 3)."""
        return self._env.simulator._rigid_body_pos[:, self._anchor_body_indexes, :]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Live simulator world orientations of the candidate anchor bodies (xyzw). (num_envs, A, 4)."""
        return self._env.simulator._rigid_body_rot[:, self._anchor_body_indexes, :]

    def _settle_enabled(self) -> bool:
        """Whether grasp-settle is active this run (config, overridable at runtime by the probe)."""
        if self._settle_enabled_override is not None:
            base = self._settle_enabled_override
        else:
            base = self.grasp_settle_cfg.enable
        return bool(base) and self.motion.has_object and self._anchor_body_indexes is not None

    def _lookup_ref_contact(
        self, time_idx: torch.Tensor, anchor_pos: torch.Tensor, obj_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(anchor_idx, ref_contact) at absolute motion frame(s) ``time_idx``.

        Ground-truth-first: uses the retargeting pipeline's own point-cloud interaction fields
        (motion.object_ref_contact / object_ref_anchor_idx -- real mesh-to-mesh SDF against the
        captured demo, see gvhmr-fp-pipeline/contact_from_retarget.py) when the loaded motion
        carries them. Falls back to the runtime nearest-anchor distance threshold
        (select_grasp_anchor) for motions without it -- keeps older/synthetic clips working.
        """
        if self.motion.has_contact_schedule:
            closed = self.schedule_contact_weight(time_idx) > 0.0  # (N, A)
            dist = torch.norm(anchor_pos - obj_pos.unsqueeze(1), dim=-1)  # (N, A)
            # Among the hands the schedule closes, take the NEAREST. A schedule is per-hand and the
            # carries here are bimanual on every contact frame, so a fixed side would half the time
            # grade the wrist that is further from the box.
            masked = torch.where(closed, dist, torch.full_like(dist, float("inf")))
            any_closed = closed.any(dim=-1)
            anchor_idx = torch.where(any_closed, masked.argmin(dim=1), dist.argmin(dim=1))
            return anchor_idx, any_closed
        if self.motion.has_gt_contact:
            return self.motion.object_ref_anchor_idx[time_idx], self.motion.object_ref_contact[time_idx]
        anchor_idx, anchor_dist = select_grasp_anchor(anchor_pos, obj_pos)
        return anchor_idx, anchor_dist < self.grasp_settle_cfg.contact_distance_threshold

    def schedule_contact_weight(self, time_idx: torch.Tensor) -> torch.Tensor:
        """(N, A) hand<->object activation in [0, 1] from the supplied schedule, per candidate anchor.

        All zeros without a schedule, so callers can multiply unconditionally. With
        contact_schedule_ramp_frames == 0 the values are exactly 0.0 / 1.0 and multiplying by them
        is the same as gating on the boolean.
        """
        n_anchor = 2 if self._anchor_body_indexes is None else int(self._anchor_body_indexes.numel())
        if not self.motion.has_contact_schedule:
            return torch.zeros(time_idx.shape[0], n_anchor, device=self.device)
        weights = self.motion._schedule_hand_contact[time_idx]  # (N, 2)
        out = torch.zeros(time_idx.shape[0], n_anchor, device=self.device)
        k = min(n_anchor, weights.shape[1])
        out[:, :k] = weights[:, :k]
        return out

    @property
    def object_termination_enabled(self) -> bool:
        """False when the force-mode assist is below object_term_min_alpha: dropping the box no
        longer kills the episode (the policy absorbs the loss of object reward and can learn to
        recover) -- otherwise the success EMA collapses on the low-alpha transition and the
        curriculum re-stalls exactly as it did with the blend. Read by the bad_tracking
        termination."""
        cfg = self.grasp_settle_cfg
        return not (
            cfg.physicality_force_mode
            and cfg.object_term_min_alpha > 0.0
            and self._physicality_alpha < cfg.object_term_min_alpha
        )

    def update_physicality_curriculum(self, success_rate: float) -> None:
        """Advance the box-physicality curriculum from the env's aggregate success rate.

        EMA-smooths the per-step success signal; when the EMA clears the threshold AND the cooldown
        has elapsed, lower the box blend factor ``_physicality_alpha`` by one step (box gets more
        physical). The step keeps the DIFFICULTY ratio constant rather than the alpha step: the
        residual drift the policy must absorb scales like ``beta = (1-alpha)/alpha``, so each advance
        multiplies beta by ``physicality_alpha_ratio`` (``alpha_next = 1/(1+ratio*beta)``). That is
        geometric in beta, so the alpha step auto-shrinks near the floor where the task is hardest.
        The first advance out of the kinematic warmup (alpha=1, beta=0) jumps to
        ``physicality_alpha_start`` since the geometric update cannot leave beta=0 on its own.
        Monotonic and cooldown-gated so the policy re-adapts between advances; after each advance the
        EMA is reset so the policy must re-earn the threshold at the new physicality. No-op unless both
        physicality_curriculum and kinematic_object_during_contact are enabled.
        """
        cfg = self.grasp_settle_cfg
        if not (cfg.physicality_curriculum and cfg.kinematic_object_during_contact):
            return
        beta = cfg.physicality_ema_beta
        # force mode: success = survival AND object tracking. Each signal is smoothed SEPARATELY,
        # then we take the min of the TWO EMAs -- never the instantaneous min before smoothing.
        #
        # Why: `success_rate` is computed over only the envs that reset on THAT step, i.e.
        # ~4096/914 ~= 4.5 envs -- a Bernoulli mean over ~5 samples, hence extremely noisy (whereas
        # obj_track covers ~2000 envs in contact, nearly smooth). And the min is concave:
        # E[min(X,y)] << min(E[X],y) as soon as X is noisy. Measured on run qsh0cwdo: both
        # components held ~0.89 while the min-before-smoothing capped out at 0.830 (binomial model
        # prediction: 0.836) -- 6 points of free penalty, which made a 0.90 threshold unreachable
        # (it would have taken ~0.96 on each component).
        # Smoothing being linear, each signal's EMA converges to its TRUE mean; the min of two
        # converged quantities expresses the intent ("both must be good") without importing the
        # noise. NB: at alpha=1 there is no min, hence no bias -- that is why the warmup cleared its
        # gate normally and the stall only showed up afterwards.
        self._success_ema = (1.0 - beta) * self._success_ema + beta * float(success_rate)
        self._obj_track_ema = (1.0 - beta) * self._obj_track_ema + beta * float(self._obj_track_success)
        gate_signal = self._success_ema
        if cfg.physicality_force_mode and self._physicality_alpha < 1.0 - 1e-6:
            gate_signal = min(gate_signal, self._obj_track_ema)
        self._steps_since_alpha_change += 1
        alpha_floor = 0.0 if cfg.physicality_force_mode else cfg.physicality_alpha_min
        if (
            self._physicality_alpha > alpha_floor
            and gate_signal >= cfg.physicality_success_threshold
            and self._steps_since_alpha_change >= cfg.physicality_cooldown_steps
        ):
            alpha = self._physicality_alpha
            if alpha >= 1.0 - 1e-6:
                # seed the ladder: leave the kinematic warmup (both modes)
                new_alpha = cfg.physicality_alpha_start
            elif cfg.physicality_force_mode:
                # force mode: difficulty ~ linear in the cap -> multiplicative ladder on alpha,
                # snap to exactly 0 once the residual cap is noise-level. Finite increments all
                # the way down — the fully-physical box is REACHABLE.
                new_alpha = alpha * cfg.physicality_force_alpha_decay
                if new_alpha < cfg.physicality_force_alpha_snap:
                    new_alpha = 0.0
            else:
                # constant difficulty ratio: grow beta=(1-alpha)/alpha by physicality_alpha_ratio
                new_beta = (1.0 - alpha) / alpha * cfg.physicality_alpha_ratio
                new_alpha = 1.0 / (1.0 + new_beta)
            self._physicality_alpha = max(alpha_floor, new_alpha)
            self._steps_since_alpha_change = 0
            # re-earn the threshold at the new physicality before advancing (both EMAs)
            self._success_ema = 0.0
            self._obj_track_ema = 0.0
            mode = "force cap" if cfg.physicality_force_mode else "blend"
            logger.info(
                f"[physicality curriculum] success EMA cleared {cfg.physicality_success_threshold:.2f} "
                f"-> box alpha ({mode}) -> {self._physicality_alpha:.3f} (1=kinematic, 0=fully physical)"
            )

    #########################################################################################
    ## C-D lite: relative hand<->object proximity
    #########################################################################################
    @property
    def hand_obj_rel_pos_ref(self) -> torch.Tensor:
        """(E, H, 3) baked reference: hand position in the object frame, indexed by time_steps."""
        return self._hand_obj_rel_pos_ref[self.time_steps]

    @property
    def hand_obj_beta(self) -> torch.Tensor:
        """(E, H) baked proximity weight, indexed by time_steps."""
        return self._hand_obj_beta[self.time_steps]

    @property
    def hand_obj_rel_pos_cur(self) -> torch.Tensor:
        """(E, H, 3) live: sim hand position in the sim object frame (one rigid transform per hand)."""
        wrist_pos = self._env.simulator._rigid_body_pos[:, self.hand_body_indexes, :]   # (E, H, 3) world
        n_hand = self.hand_body_indexes.numel()
        obj_pos = self.simulator_object_pos_w[:, None, :].repeat(1, n_hand, 1)           # (E, H, 3)
        obj_quat = self.simulator_object_quat_w[:, None, :].repeat(1, n_hand, 1)         # (E, H, 4) xyzw
        return relative_position_in_object_frame(wrist_pos, obj_pos, obj_quat)

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self):
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # grasp-settle state (per env). Populated in reset() for object clips.
        self.settle_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.settle_anchor_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.settle_grasp_rel_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.settle_grasp_rel_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.settle_grasp_rel_quat[:, 3] = 1.0  # identity (xyzw)

        # full-contact weld curriculum ("training wheels"): per-episode assist flag drawn at reset
        # with an annealed probability, and a global env-step counter driving the anneal.
        self.weld_assist = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.weld_assist_prob = 0.0
        self._env_step_counter = 0

        # physicality curriculum: box blend factor (1=kinematic, 0=physical), success EMA, cooldown.
        self._physicality_alpha = 1.0
        self._success_ema = 0.0
        self._steps_since_alpha_change = 0
        # force-mode assist: per-step object-tracking success (fraction of ref-contact envs with
        # object pos error < physicality_success_obj_err), fed to the curriculum instead of pure
        # survival (which saturates once object terminations are gated off at low alpha).
        self._obj_track_success = 1.0
        # EMA of _obj_track_success, smoothed SEPARATELY from _success_ema so the curriculum gate can
        # take min(EMA, EMA) instead of EMA(min(...)) — see update_physicality_curriculum.
        self._obj_track_ema = 0.0
        # Per-env object mass (num_envs,) and weight m*g in world frame (num_envs, 3) — filled
        # lazily on the first force-mode step so the startup mass randomisation is already applied,
        # then cached (the masses are fixed for the whole run).
        self._object_mass: torch.Tensor | None = None
        self._object_gravity_w: torch.Tensor | None = None

        # debug hooks used by the offline probe harness (probe_grasp_settle.py):
        #   _force_start_timesteps: if set (long tensor, per env, absolute frame index), reset()
        #     starts every env at that frame instead of sampling a phase.
        #   _settle_enabled_override: if set (bool), overrides grasp_settle.enable at runtime so the
        #     harness can A/B baseline vs settle in a single process.
        self._force_start_timesteps: torch.Tensor | None = None
        self._settle_enabled_override: bool | None = None
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.init_buffers()

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        # MPJPE: Mean Per Joint Position Error (radians), averaged over joints
        self.metrics["motion/mpjpe"] = (self.joint_pos - self.robot_joint_pos).abs().mean(dim=-1)

        # MPKPE: Mean Per Keypoint Position Error (metres), averaged over tracked bodies
        self.metrics["motion/mpkpe"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        # Object/grasp health metrics (object clips only). Read together:
        #   object_held / object_ref_contact = fraction of in-contact frames where the sim box is
        #   actually near a sim hand. weld_assist_prob tracks the training-wheels anneal.
        if self.motion.has_object and self._anchor_body_indexes is not None:
            sim_obj = self.simulator_object_pos_w
            hand_dist = torch.norm(self.robot_anchor_pos_w - sim_obj.unsqueeze(1), dim=-1).min(dim=1).values
            _, ref_contact = self._lookup_ref_contact(self.time_steps, self.anchor_pos_w, self.object_pos_w)
            thr = self.grasp_settle_cfg.contact_distance_threshold
            self.metrics["motion/object_hand_dist"] = hand_dist
            self.metrics["motion/object_ref_contact"] = ref_contact.float()
            self.metrics["motion/object_held"] = ((hand_dist < thr) & ref_contact).float()
            # box tracking vs the reference clip: these are the EXACT quantities the bad_object_pos
            # (>0.25 m) and bad_object_ori (>0.8 rad) terminations threshold, so you can watch the
            # margin to death directly. object_height catches a drop (box slipped out of the hands).
            self.metrics["motion/object_pos_error"] = torch.norm(self.object_pos_w - sim_obj, dim=-1)
            self.metrics["motion/object_ori_error"] = quat_error_magnitude(
                self.object_quat_w, self.simulator_object_quat_w
            )
            self.metrics["motion/object_lin_vel_error"] = torch.norm(
                self.object_lin_vel_w - self.simulator_object_lin_vel_w, dim=-1
            )
            self.metrics["motion/object_ang_vel_error"] = torch.norm(
                self.object_ang_vel_w - self.simulator_object_ang_vel_w, dim=-1
            )
            self.metrics["motion/object_height"] = sim_obj[:, 2]
            self.metrics["motion/weld_assist_prob"] = torch.full_like(hand_dist, float(self.weld_assist_prob))
            # box physicality curriculum: alpha (1=kinematic, 0=physical) + the success EMA driving it
            self.metrics["motion/physicality_alpha"] = torch.full_like(hand_dist, float(self._physicality_alpha))
            self.metrics["motion/physicality_success_ema"] = torch.full_like(hand_dist, float(self._success_ema))
            if self.grasp_settle_cfg.physicality_force_mode:
                self.metrics["motion/obj_track_success"] = torch.full_like(hand_dist, float(self._obj_track_success))
                # smoothed counterpart; the gate compares min(this, physicality_success_ema)
                self.metrics["motion/obj_track_ema"] = torch.full_like(hand_dist, float(self._obj_track_ema))
                # effective tracking cap actually applied (mass-proportional when enabled, so this
                # is the MEAN over envs; gravity support is fed forward on top and not counted here)
                cfg_gs = self.grasp_settle_cfg
                if cfg_gs.force_assist_track_accel > 0.0 and self._object_mass is not None:
                    cap = float(self._object_mass.mean()) * cfg_gs.force_assist_track_accel
                else:
                    cap = cfg_gs.force_assist_fmax
                self.metrics["motion/force_assist_fmax_eff"] = torch.full_like(
                    hand_dist, float(self._physicality_alpha * cap)
                )

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)

        action = "prepend" if prepend else "append"
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the state dict representing the robot's default standing pose.

        By default, anchor root pos/yaw to the motion start; when use_motion_end is True, anchor to motion end.
        """
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = -1 if use_motion_end else 0

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[motion_idx].to(self.device)
            object_ang_vel = self.motion._object_ang_vel_w[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_ang_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
            "object_ang_vel": object_ang_vel,
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
            "Default-pose interpolation only supports IsaacSim; IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        env_origin = simulator.scene.env_origins[env_id].to(self.device)

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
            state["object_ang_vel"] = self.motion._object_ang_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
            state["object_ang_vel"] = default_state["object_ang_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Linearly/spherically interpolate between start and target states."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        segments = {
            "joint_pos": _lerp(start["joint_pos"], target["joint_pos"], alphas_joint),
            "joint_vel": _lerp(start["joint_vel"], target["joint_vel"], alphas_joint),
            "body_pos": _lerp(start["body_pos"], target["body_pos"], alphas_body),
            "body_lin_vel": _lerp(start["body_lin_vel"], target["body_lin_vel"], alphas_body),
            "body_ang_vel": _lerp(start["body_ang_vel"], target["body_ang_vel"], alphas_body),
            "body_quat": self._slerp_quat_sequence(start["body_quat"], target["body_quat"], alphas),
        }

        if self.motion.has_object:
            segments["object_pos"] = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_lin_vel"] = _lerp(start["object_lin_vel"], target["object_lin_vel"], alphas_joint)
            segments["object_ang_vel"] = _lerp(start["object_ang_vel"], target["object_ang_vel"], alphas_joint)
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
