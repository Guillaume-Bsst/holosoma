#!/bin/bash
#SBATCH --job-name=g1_wbt_mjwarp_wobj
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -p willow
#SBATCH -A willow

BASE_DIR="/home/gbesset/WILLOW/holosoma"

cd ${BASE_DIR}

mkdir -p slurm_logs
mkdir -p ${BASE_DIR}/logs
mkdir -p ${BASE_DIR}/wandb

export APPTAINER_CACHEDIR="${BASE_DIR}/.apptainer_cache"
export APPTAINER_TMPDIR="${BASE_DIR}/.apptainer_tmp"
mkdir -p ${APPTAINER_CACHEDIR} ${APPTAINER_TMPDIR}

apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind ${BASE_DIR}/logs:/workspace/holosoma/logs \
    --bind ${BASE_DIR}/wandb:/workspace/holosoma/wandb \
    --bind ${BASE_DIR}/src:/workspace/holosoma/src \
    ${BASE_DIR}/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-w-object \
        simulator:mjwarp-w-object \
        logger:wandb \
        terrain:terrain-locomotion-plane \
        --logger.video.enabled=False \
        --simulator.config.debug-viz=False \
        --simulator.config.mujoco-warp.ccd-iterations=200 \
        --training.num-envs=8192 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz\" \
        --simulator.config.scene.rigid-objects.0.name=\"largebox\" \
        --simulator.config.scene.rigid-objects.0.urdf-path=\"src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf\" \
        --simulator.config.scene.rigid-objects.0.position=\"[1.0,0.0,1.0]\" \
        --terrain.terrain-term.mesh-type=\"PLANE\" \
        --terrain.terrain-term.obj-file-path=\"\"
"