"""Fixture: real largebox pipeline inputs, truncated, for objvar tests.

Mirrors examples/robot_retarget.py:main() (task sub3_largebox_003, scaled world,
no augmentation) up to the retarget_motion call -- the PROD path, not a replica.

The default RetargetingConfig/TaskConfig/RetargeterConfig fields used here
(with_ground=False, native_scene=False, augmentation=False, robot="g1" with
matching robot_config/motion_data_config) make several branches in main() dead
code for this fixture (the "ensure configs match" reassignment, the
native_scene object_poses_native handling, and the with_ground ground-mesh
injection all no-op under these defaults), so they are omitted here rather
than reproduced -- everything that DOES execute mirrors main() call-for-call.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1]


def build_largebox_inputs(n_frames: int = 15, retargeter_overrides: dict | None = None):
    os.chdir(PKG_DIR)  # demo_data/... paths are package-relative
    np.random.seed(0)

    from holosoma_retargeting.config_types.retargeting import RetargetingConfig
    from holosoma_retargeting.examples.robot_retarget import (
        _AUGMENTATION_TRANSLATION,
        _OBJECT_SCALE_AUGMENTED,
        InteractionMeshRetargeter,
        build_retargeter_kwargs_from_config,
        create_task_constants,
        extract_foot_sticking_sequence_velocity,
        initialize_robot_pose,
        load_motion_data,
        preprocess_motion_data,
        setup_object_data,
    )

    cfg = RetargetingConfig(task_type="object_interaction", task_name="sub3_largebox_003",
                            data_path=Path("demo_data/OMOMO_new"), data_format="smplh")
    constants = create_task_constants(robot_config=cfg.robot_config,
                                      motion_data_config=cfg.motion_data_config,
                                      task_config=cfg.task_config, task_type=cfg.task_type)
    human_joints, object_poses, smpl_scale = load_motion_data(
        cfg.task_type, cfg.data_format, cfg.data_path, cfg.task_name, constants, cfg.motion_data_config)
    toe_names = cfg.motion_data_config.toe_names
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        cfg.task_type, constants, cfg.task_config.object_dir, smpl_scale,
        cfg.task_config, cfg.augmentation, object_scale_augmented=_OBJECT_SCALE_AUGMENTED)

    kwargs = build_retargeter_kwargs_from_config(cfg.retargeter, constants, object_urdf_path, cfg.task_type)
    kwargs.update(retargeter_overrides or {})
    retargeter = InteractionMeshRetargeter(**kwargs)

    human_joints, object_poses, _ = preprocess_motion_data(
        human_joints, retargeter, toe_names, scale=smpl_scale, object_poses=object_poses)
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        cfg.task_type, cfg.data_format, human_joints, object_poses, constants, retargeter,
        cfg.task_config, cfg.augmentation, Path("/tmp"), cfg.task_name,
        augmentation_translation=_AUGMENTATION_TRANSLATION)
    foot_sticking = extract_foot_sticking_sequence_velocity(human_joints, retargeter.demo_joints, toe_names)
    foot_sticking[0][toe_names[0]] = False
    foot_sticking[0][toe_names[1]] = False

    n = n_frames
    motion_kwargs = dict(
        human_joint_motions=human_joints[:n],
        object_poses=object_poses[:n],
        object_poses_augmented=object_poses_augmented[:n],
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        ground_points_world=None,
        foot_sticking_sequences=foot_sticking[:n],
        q_a_init=q_init,
        q_nominal_list=(q_nominal[:n] if q_nominal is not None else None),
        original=True,
    )
    return retargeter, motion_kwargs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=15)
    a = ap.parse_args()
    retargeter, mk = build_largebox_inputs(a.frames)
    retargeter.retarget_motion(dest_res_path=a.out, **mk)
