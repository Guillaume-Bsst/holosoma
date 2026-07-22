# Sim-to-Sim Whole Body Tracking Workflow

> **See also:** [Inference & Deployment Guide](../../README.md) for all deployment options

This guide provides a complete workflow for running whole body tracking (WBT) policies in MuJoCo simulation.

## Overview

The sim-to-sim workflow allows you to replay IsaacSim/IsaacGym-trained WBT checkpoints inside MuJoCo for evaluation and testing.

## Prerequisites

- MuJoCo environment set up (`scripts/source_mujoco_setup.sh`)
- Holosoma inference environment set up (`scripts/source_inference_setup.sh`)
- ONNX model checkpoint
- Keyboard for control

**Note:** Always use `--task.interface lo` (loopback) when inference and MuJoCo run on the same machine.

---

## Unitree G1 (29-DOF)

### 1. Start MuJoCo Environment

In one terminal, launch the MuJoCo environment:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

The robot will spawn in the simulator, hanging from a gantry.

### 2. Launch the Policy

In another terminal, run the policy inference:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50 \
    --task.interface lo
```

### 3. Initialize Stiff Control Mode

In policy terminal, press `Enter` when prompted. The robot enters stiff control mode and holds its initial pose.

### 4. Deploy the Robot

- In MuJoCo window, press `8` to lower the gantry until robot touches ground
- In MuJoCo window, press `9` to remove gantry
- Wait a few seconds for the stiff controller to stabilize the robot

### 5. Start the Policy

In policy terminal, press `]` to activate the policy.

### 6. Start Motion Clip

In policy terminal, press `m` to start the motion clip. The robot will begin tracking the whole body motion.

---

## Object-Carry Variant (WBT + box)

For checkpoints trained with `exp:g1-29dof-wbt-w-object-actor` (actor observes a carried box). The box's
physical presence in MuJoCo and the policy's box observation are two independent things:

- **MuJoCo side**: `run_sim.py` can spawn a free (physical, unheld-by-default) rigid box so you can see the
  robot actually interact with something. Its geometry/mass are read straight from the object URDF the
  checkpoint was trained with, and — if you also point it at the training clip — its spawn pose is
  auto-anchored to the robot's actual spawn position in this scene (see `simulator/mujoco/object_spawn.py`).
- **Policy side**: `WholeBodyTrackingPolicy` does NOT perceive the physical box live. `--task.object-motion-file`
  feeds it the box pose recorded in the training clip, indexed by motion timestep, exactly like training's
  kinematic/contact-assisted box. This is a sim-to-sim shortcut (no perception pipeline needed) — a real
  deployment would substitute a live mocap/RGB-D box pose here instead.

### 1. Start MuJoCo Environment (with box)

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py simulator:mujoco robot:g1-29dof \
    --robot.object.object-urdf-path holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf \
    --simulator.config.sim.add-box True \
    --simulator.config.sim.object-motion-file holosoma/data/motions/g1_29dof/whole_body_tracking/femto14_box36_w_obj_gtcontact_slow16.npz
```

Swap `objects_box36.urdf` / the clip for whatever object-carry checkpoint you're playing back — geometry and
mass are derived from the URDF automatically, no need to hand-tune a half-extent/mass per object.

### 2. Launch the Policy (object-obs variant)

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-w-object \
    --task.model-path <path-to-exported-object-checkpoint>.onnx \
    --task.object-motion-file src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/femto14_box36_w_obj_gtcontact_slow16.npz \
    --task.motion-prepend-timesteps <N> \
    --task.no-use-joystick --task.use-sim-time --task.rl-rate 50 --task.interface lo
```

- `--task.model-path`: export via `eval_agent.py`'s `--training.export-onnx` (writes to
  `<checkpoint_dir>/exported/model_XXXXX.onnx`) or `algo.export(...)`.
- `--task.motion-prepend-timesteps <N>`: must match the training clip's `default_pose_prepend` window in
  POLICY control steps (`default_pose_prepend_duration_s * rl_rate`, not seconds) — see
  `TaskConfig.motion_prepend_timesteps` docstring. Get this from the training run's
  `MotionConfig.default_pose_prepend_duration_s`.
- Steps 3-6 above (stiff mode, gantry, start policy, start clip) are unchanged.
- `--task.zero-object-obs True` is a debugging isolation switch: if the robot holds up fine with it but falls
  with real object obs, the bug is in the object-obs frame/convention rather than a general sim2sim gap.

---

## MuJoCo Controls Reference

**Enter these commands in the MuJoCo window** (not the policy terminal):

### Gantry Controls

- `7`: Lift the gantry
- `8`: Lower the gantry
- `9`: Disable/remove the gantry

### General Controls

- `Backspace`: Reset simulation

---

## Policy Controls Reference

**Enter these commands in the policy terminal** (where you ran `run_policy.py`):

### General Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start the policy | `]` | A button |
| Stop the policy | `o` | B button |
| Set robot to default pose | `i` | Y button |

### Whole Body Tracking Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start motion clip | `m` | Select+A |

**Default pose**: Standing with raised arms

---

## Tips and Troubleshooting

- **Reset anytime**: Press `Backspace` in the MuJoCo window to reset the simulation
- **Interface**: Always use `lo` (loopback) for sim-to-sim on the same machine
- **Stiff mode**: The `Enter` prompt initializes stiff control mode - this is required for WBT policies to maintain balance before the policy starts
- **Stabilization**: Wait a few seconds after removing the gantry (step 3) before starting the policy to let the stiff controller stabilize
- **RL rate**: Use `--task.rl-rate 50` for WBT policies (50 Hz control rate)
- **Sim time**: Use `--task.use-sim-time` to synchronize with MuJoCo's simulation time
