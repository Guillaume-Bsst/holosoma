#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

source scripts/source_mujoco_setup.sh

python src/holosoma/holosoma/run_sim.py simulator:mujoco robot:g1-29dof \
    --robot.object.object-urdf-path holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf \
    --robot.asset.xml-file g1/g1_29dof_halfspherehand.xml \
    --robot.asset.enable-self-collisions True \
    --simulator.config.sim.add-box True \
    --simulator.config.sim.add-support True \
    --simulator.config.sim.support-obj-file holosoma/data/motions/g1_29dof/whole_body_tracking/femto14_support_world.obj \
    --simulator.config.sim.object-motion-file holosoma/data/motions/g1_29dof/whole_body_tracking/femto14_box36_gaitfix2_w_obj_gtcontact.npz
