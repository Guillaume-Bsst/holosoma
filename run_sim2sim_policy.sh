#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

source scripts/source_inference_setup.sh

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-w-object-support-h3 \
    --task.model-path wandb://guibsst-inria/WholeBodyTracking/bzwhv8kk/model_29999.onnx \
    --task.object-motion-file src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/femto14_box36_gaitfix2_w_obj_gtcontact.npz \
    --task.motion-prepend-timesteps 100 \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.live-object-obs \
    --task.rl-rate 50 \
    --task.interface lo
