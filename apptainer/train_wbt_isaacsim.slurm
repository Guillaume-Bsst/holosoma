#!/bin/bash
#SBATCH --job-name=g1_wbt_isaac
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

PROJECT_ROOT="/home/gbesset/WILLOW/holosoma"
SIF_FILE="$PROJECT_ROOT/apptainer/holosoma.sif"

mkdir -p slurm_logs
mkdir -p ${PROJECT_ROOT}/logs
mkdir -p ${PROJECT_ROOT}/wandb
mkdir -p $SCRATCH/.isaac_cache
mkdir -p $SCRATCH/isaac_tmp
mkdir -p $SCRATCH/isaac_pkg_cache
mkdir -p $SCRATCH/isaac_pkg_data
mkdir -p $SCRATCH/isaac_pkg_logs
mkdir -p $SCRATCH/holosoma_converted_robots

export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR

apptainer exec --nv \
    --bind /run \
    --bind /dev/shm \
    --bind ${PROJECT_ROOT}/src:/workspace/holosoma/src \
    --bind ${PROJECT_ROOT}/logs:/workspace/holosoma/logs \
    --bind ${PROJECT_ROOT}/wandb:/workspace/holosoma/wandb \
    --bind $SCRATCH/.isaac_cache:/root/.local/share/ov \
    --bind $SCRATCH/isaac_tmp:/tmp \
    --bind $SCRATCH/isaac_pkg_cache:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/cache \
    --bind $SCRATCH/isaac_pkg_data:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/data \
    --bind $SCRATCH/isaac_pkg_logs:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/logs \
    --bind $SCRATCH/holosoma_converted_robots:/workspace/holosoma/src/holosoma/holosoma/data/robots/converted_rank0 \
    $SIF_FILE bash -c "
    export HTTP_PROXY=\$http_proxy && \
    export HTTPS_PROXY=\$https_proxy && \
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    export TMPDIR=/tmp && \
    export TEMP=/tmp && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt \
        simulator:isaacsim \
        logger:wandb \
        --logger.video.enabled=False \
        --training.num-envs=8192 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz\"
"
