"""Resolve where/what free box to spawn in run_sim for object-carry sim-to-sim.

Two independent things are derived here so the spawned box always matches whatever object-carry
checkpoint is being played back, instead of relying on hand-tuned box_half_extent/box_mass/box_pos
config values that drift out of sync with the actual training data:

1. Geometry + mass: read directly from the object URDF (robot.object.object_urdf_path) the checkpoint
   was trained with -- the same file passed to training's --robot.object.object-urdf-path.
2. Spawn pose: read the box pose relative to the robot root at a given clip frame (the same clip
   passed to holosoma_inference's --task.object-motion-file) and re-anchor it to this scene's actual
   robot init_state pose, so the box lands where the policy expects it relative to the robot
   regardless of where the robot happens to stand in this particular MuJoCo scene.

Falls back to the static sim.box_pos/box_half_extent/box_mass config when the object URDF / motion
file aren't configured (e.g. plain decorative box, no object-carry checkpoint).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from holosoma.utils.path import resolve_data_file_path

if TYPE_CHECKING:
    from holosoma.config_types.robot import RobotConfig
    from holosoma.config_types.simulator import SimEngineConfig


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mul_wxyz(_quat_mul_wxyz(q, qv), _quat_conj_wxyz(q))[1:]


def _parse_object_urdf_box(urdf_path: str) -> tuple[float, float]:
    """Derive (half_extent_m, mass_kg) from a single-link box object URDF + its mesh.

    Matches the objects_box32.urdf / objects_box36.urdf / objects_largebox.urdf shape used for
    object-carry training: one <link> with a box-shaped mesh centered at the origin and a
    <mass value=.../> tag. half_extent is measured directly off the mesh vertices rather than assumed,
    so it stays correct for any object of this shape without a config knob per box size.
    """
    resolved = resolve_data_file_path(urdf_path)
    root = ET.parse(resolved).getroot()

    mass_el = root.find(".//mass")
    if mass_el is None:
        raise ValueError(f"No <mass> tag found in object URDF: {resolved}")
    mass = float(mass_el.get("value"))

    mesh_el = root.find(".//collision/geometry/mesh")
    if mesh_el is None:
        mesh_el = root.find(".//visual/geometry/mesh")
    if mesh_el is None:
        raise ValueError(f"No mesh geometry found in object URDF: {resolved}")
    mesh_path = os.path.join(os.path.dirname(resolved), mesh_el.get("filename"))

    vertices = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in object mesh: {mesh_path}")
    verts = np.asarray(vertices, dtype=np.float64)
    half_extent = float((verts.max(axis=0) - verts.min(axis=0)).max() / 2.0)

    return half_extent, mass


def _compute_box_spawn_pose(
    motion_file: str,
    timestep: int,
    robot_init_pos: tuple[float, float, float],
    robot_init_rot_xyzw: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (world_pos[3], world_quat_wxyz[4]) for the box at `timestep` of `motion_file`.

    Reads the clip's object pose relative to the robot root (the same relative transform
    holosoma_inference feeds the policy as obj_pos_b/obj_ori_b) and re-expresses it using the robot's
    ACTUAL init_state pose in this scene, so the box appears where the policy expects it relative to
    the robot even though the robot's spawn pose here is unrelated to the clip's own world frame.
    """
    data = np.load(resolve_data_file_path(motion_file))
    joint_pos = data["joint_pos"]
    obj_pos_w = data["object_pos_w"]
    obj_quat_w = data["object_quat_w"]  # wxyz

    t = int(np.clip(timestep, 0, joint_pos.shape[0] - 1))
    return _reanchor_yawxy(obj_pos_w[t], obj_quat_w[t], joint_pos[t, :7], robot_init_pos, robot_init_rot_xyzw)


def _yaw_of_wxyz(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _reanchor_yawxy(
    clip_pos: np.ndarray,
    clip_quat_wxyz: np.ndarray,
    clip_root: np.ndarray,
    robot_init_pos,
    robot_init_rot_xyzw,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a clip-world pose into this scene, anchoring by yaw + XY only.

    Rotates/translates about the clip's root-vs-robot-init delta restricted to yaw and the ground
    plane: heights stay EXACTLY as in the clip (both floors are at z=0), so nothing sinks into or
    hovers over the ground when the robot init z differs slightly from the clip's root z.
    """
    root_yaw = _yaw_of_wxyz(clip_root[3:7])
    robot_yaw = _yaw_of_wxyz(
        np.array([robot_init_rot_xyzw[3], robot_init_rot_xyzw[0], robot_init_rot_xyzw[1], robot_init_rot_xyzw[2]])
    )
    dyaw = robot_yaw - root_yaw
    c, s = np.cos(dyaw), np.sin(dyaw)

    rel_xy = np.asarray(clip_pos[:2], dtype=np.float64) - np.asarray(clip_root[:2], dtype=np.float64)
    world_xy = np.asarray(robot_init_pos[:2], dtype=np.float64) + np.array(
        [c * rel_xy[0] - s * rel_xy[1], s * rel_xy[0] + c * rel_xy[1]]
    )
    world_pos = np.array([world_xy[0], world_xy[1], float(clip_pos[2])])

    dyaw_quat = np.array([np.cos(dyaw / 2.0), 0.0, 0.0, np.sin(dyaw / 2.0)])
    world_quat = _quat_mul_wxyz(dyaw_quat, np.asarray(clip_quat_wxyz, dtype=np.float64))

    return world_pos, world_quat


def resolve_support_spawn(
    sim_cfg: SimEngineConfig, robot_config: RobotConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Resolve (vertices[N,3], faces[M,3], pos[3], quat_wxyz[4]) for the support table, or None.

    Implements SimEngineConfig.add_support: loads the REAL support mesh (clip-world coordinates,
    e.g. femto14_support_world.obj -- rotated plateau + legs, triangulated) and computes the rigid
    transform that re-anchors the whole clip world relative to the robot's init pose (same yaw+XY
    mapping as the free box, heights untouched). The caller attaches the mesh under a body at that
    transform, so the table keeps its exact shape and orientation -- an AABB box approximation is
    wrong here because the table is rotated ~-105 deg in the clip world.
    """
    if not (sim_cfg.add_support and sim_cfg.support_obj_file and sim_cfg.object_motion_file):
        return None

    mesh_path = resolve_data_file_path(sim_cfg.support_obj_file)
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                # fan-triangulate just in case (trimesh exports are already triangles)
                faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    vertices = np.asarray(verts, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int32)

    # Rigid transform mapping clip-world coordinates into this scene (yaw+XY anchoring):
    # world_pt = t + Rz(dyaw) @ clip_pt, derived from mapping the clip root onto the robot init.
    data = np.load(resolve_data_file_path(sim_cfg.object_motion_file))
    t_idx = int(np.clip(sim_cfg.object_motion_start_timestep, 0, data["joint_pos"].shape[0] - 1))
    origin, quat = _reanchor_yawxy(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        data["joint_pos"][t_idx, :7],
        robot_config.init_state.pos,
        robot_config.init_state.rot,
    )
    logger.info(
        f"Support table mesh '{sim_cfg.support_obj_file}': {len(vertices)} verts, "
        f"{len(faces_arr)} tris, scene transform pos={origin.round(3).tolist()}"
    )
    return vertices, faces_arr, origin, quat


def robot_init_state_from_clip(sim_cfg: SimEngineConfig, robot_config: RobotConfig) -> RobotConfig | None:
    """Return a robot config whose init_state matches the clip's frame-0 root pose, or None.

    Implements SimEngineConfig.spawn_robot_at_clip_start: x, y and yaw come from
    object_motion_file's frame 0 (the same anchoring training's default-pose prepend uses);
    z, roll and pitch are kept from the configured init_state. Returns None when the flag or
    the motion file is absent.
    """
    if not (sim_cfg.spawn_robot_at_clip_start and sim_cfg.object_motion_file):
        return None

    import dataclasses  # noqa: PLC0415 -- only needed on this path

    data = np.load(resolve_data_file_path(sim_cfg.object_motion_file))
    root = np.asarray(data["joint_pos"], dtype=np.float64)[0, :7]  # [pos(3), quat wxyz(4)]
    w, x, y, z = root[3:7]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    init = robot_config.init_state
    new_pos = [float(root[0]), float(root[1]), init.pos[2]]
    # yaw-only rotation, xyzw (init roll/pitch are flat for humanoid spawn configs)
    new_rot = [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]
    logger.info(
        f"Robot init overridden from clip frame 0: pos={[round(v, 3) for v in new_pos]}, "
        f"yaw={np.degrees(yaw):.1f} deg (spawn_robot_at_clip_start)"
    )
    return dataclasses.replace(
        robot_config, init_state=dataclasses.replace(init, pos=new_pos, rot=new_rot)
    )


def resolve_box_spawn(
    sim_cfg: SimEngineConfig, robot_config: RobotConfig
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Resolve (pos[3], quat_wxyz[4], half_extent, mass) for the free box spawned by run_sim.

    See module docstring: prefers the object URDF / motion clip when configured, falls back to the
    static sim.box_* values otherwise.
    """
    urdf_path = robot_config.object.object_urdf_path
    if urdf_path:
        half_extent, mass = _parse_object_urdf_box(urdf_path)
        logger.info(
            f"Box geometry/mass derived from object URDF '{urdf_path}': "
            f"half_extent={half_extent:.3f} m, mass={mass:.3f} kg"
        )
    else:
        half_extent, mass = sim_cfg.box_half_extent, sim_cfg.box_mass

    if sim_cfg.object_motion_file:
        pos, quat = _compute_box_spawn_pose(
            sim_cfg.object_motion_file,
            sim_cfg.object_motion_start_timestep,
            robot_config.init_state.pos,
            robot_config.init_state.rot,
        )
        logger.info(
            f"Box spawn pose anchored from clip '{sim_cfg.object_motion_file}' "
            f"@t={sim_cfg.object_motion_start_timestep}: pos={pos.round(3).tolist()}"
        )
    else:
        pos = np.asarray(sim_cfg.box_pos, dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0])

    return pos, quat, half_extent, mass
