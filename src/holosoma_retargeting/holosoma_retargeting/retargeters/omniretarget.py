"""OmniRetarget: wraps InteractionMeshRetargeter (SQP + interaction mesh).

This is the original holosoma_retargeting algorithm.
Supports all task types: robot_only, object_interaction, climbing.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from holosoma_retargeting.config_types.retargeters.omniretarget import OmniRetargeterConfig
from holosoma_retargeting.retargeters.base import BaseRetargeter


class OmniRetargeter(BaseRetargeter):
    """Wraps InteractionMeshRetargeter with the BaseRetargeter interface."""

    def __init__(self, inner) -> None:
        """Args:
            inner: Initialised InteractionMeshRetargeter instance.
        """
        self._inner = inner

    @property
    def supported_task_types(self) -> frozenset[str]:
        return frozenset({"robot_only", "object_interaction", "climbing"})

    @classmethod
    def from_config(cls, cfg: OmniRetargeterConfig, constants: SimpleNamespace) -> "OmniRetargeter":
        from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

        object_urdf_path = getattr(constants, "OBJECT_URDF_FILE", None)
        task_type = getattr(constants, "_task_type", "robot_only")

        kwargs = {
            "task_constants": constants,
            "object_urdf_path": object_urdf_path,
            "q_a_init_idx": cfg.q_a_init_idx,
            "activate_joint_limits": cfg.activate_joint_limits,
            "activate_obj_non_penetration": cfg.activate_obj_non_penetration,
            "activate_foot_sticking": cfg.activate_foot_sticking,
            "penetration_tolerance": cfg.penetration_tolerance,
            "foot_sticking_tolerance": cfg.foot_sticking_tolerance,
            "step_size": cfg.step_size,
            "visualize": cfg.visualize,
            "debug": cfg.debug,
            "w_nominal_tracking_init": cfg.w_nominal_tracking_init,
            "foot_lock": cfg.foot_lock,
            "self_collision": cfg.self_collision,
        }
        if task_type == "climbing":
            kwargs["nominal_tracking_tau"] = cfg.nominal_tracking_tau

        return cls(InteractionMeshRetargeter(**kwargs))

    def retarget_motion(
        self,
        human_joint_motions: np.ndarray,
        object_poses: np.ndarray,
        object_poses_augmented: np.ndarray,
        object_points_local_demo: np.ndarray | None,
        object_points_local: np.ndarray | None,
        foot_sticking_sequences: list[dict[str, bool]],
        q_a_init: np.ndarray | None,
        q_nominal_list: np.ndarray | None,
        original: bool,
        dest_res_path: str,
    ) -> None:
        self._inner.retarget_motion(
            human_joint_motions=human_joint_motions,
            object_poses=object_poses,
            object_poses_augmented=object_poses_augmented,
            object_points_local_demo=object_points_local_demo,
            object_points_local=object_points_local,
            foot_sticking_sequences=foot_sticking_sequences,
            q_a_init=q_a_init,
            q_nominal_list=q_nominal_list,
            original=original,
            dest_res_path=dest_res_path,
        )
