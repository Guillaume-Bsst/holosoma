"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_tracking = bad_ref_pos | bad_ref_ori | bad_motion_body_pos

        causes = {
            "ref_pos": bad_ref_pos,
            "ref_ori": bad_ref_ori,
            "motion_body_pos": bad_motion_body_pos,
        }

        # object terminations gated off at low force-mode assist (object_term_min_alpha): a drop
        # must be a reward loss the policy can learn from, not an episode kill that collapses the
        # curriculum's success signal right when the box goes fully physical.
        if motion_command.motion.has_object and getattr(motion_command, "object_termination_enabled", True):
            bad_object_pos = self.bad_object_pos(motion_command)
            bad_object_ori = self.bad_object_ori(motion_command)
            bad_tracking |= bad_object_pos | bad_object_ori
            causes["object_pos"] = bad_object_pos
            causes["object_ori"] = bad_object_ori

        # grasp-settle grace period: while a contact reset is still settling, suppress tracking
        # termination so the hand<->object contact can equilibrate without killing the episode.
        settle_cfg = getattr(motion_command, "grasp_settle_cfg", None)
        settle_counter = getattr(motion_command, "settle_counter", None)
        if (
            settle_cfg is not None
            and settle_cfg.disable_termination_during_settle
            and settle_counter is not None
            and motion_command._settle_enabled()
        ):
            bad_tracking = bad_tracking & (settle_counter == 0)

        self._log_termination_causes(bad_tracking, causes)
        return bad_tracking

    def _log_termination_causes(self, bad_tracking: torch.Tensor, causes: dict[str, torch.Tensor]) -> None:
        """Break the aggregate termination rate down by which condition fired.

        ``bad_tracking`` is an OR over three (or five, with an object) independent conditions, so a
        run whose ``Env/termination/bad_tracking`` sits at 1.0 says nothing about WHY episodes die:
        losing the box and falling over are the same number. That distinction decides what to fix --
        an object-driven failure is a carrying/grasp-diversity problem, a ref_pos/ref_ori one is a
        locomotion problem, and they call for opposite changes.

        Logged as the fraction OF TERMINATING ENVS tripping each condition, which is the directly
        readable quantity. The values can sum above 1: a failing episode usually trips several at
        once (a fall breaks ref_pos and ref_ori together, and drags the box down with it). Read them
        as "present at the moment of death", not as an exclusive attribution -- the useful signal is
        which one is ~1.0 and which one is ~0.
        """
        log_dict = getattr(self.env, "log_dict", None)
        if log_dict is None:
            return
        # clamp_min(1) rather than a Python `if n_term > 0`: reading a GPU tensor in a host-side
        # branch forces a device sync on EVERY control step, which measured ~12x slower end to end.
        # When nothing terminated the numerator is 0 as well, so 0/1 gives the same 0.
        n_term = bad_tracking.sum().clamp_min(1)
        for name, mask in causes.items():
            log_dict[f"termination_cause/{name}"] = (mask & bad_tracking).float().sum() / n_term

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        return (
            torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        return (
            quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments."""

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks for parity with BM Wo-State-Estimation."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference z position is too far from the robot's z position."""
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if tracked bodies have too much z-axis position error."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)
