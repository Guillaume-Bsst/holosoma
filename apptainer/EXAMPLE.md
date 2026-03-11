
# Holosoma command example

## Retargeting

From `holosoma/` directory:

```bash

# Robot-only (OMOMO)
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/OMOMO_new \
        --task-type robot_only \
        --task-name sub3_largebox_003 \
        --data_format smplh
"

# Object interaction (OMOMO)
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/OMOMO_new \
        --task-type object_interaction \
        --task-name sub3_largebox_003 \
        --data_format smplh
"

# Climbing
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    export MUJOCO_GL=egl && \
    export PYOPENGL_PLATFORM=egl && \
    unset DISPLAY && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    python examples/robot_retarget.py \
        --data_path demo_data/climb \
        --task-type climbing \
        --task-name mocap_climb_seq_0 \
        --data_format mocap \
        --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf 
"

```

## Data conversion

From `holosoma/` directory:

```bash

# Robot-Only Setting
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    sed -i 's/viewer = mjv/#viewer = mjv/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.opt/#viewer.opt/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.cam/#viewer.cam/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.sync/#viewer.sync/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.close/#viewer.close/g' data_conversion/convert_data_format_mj.py && \
    python data_conversion/convert_data_format_mj.py \
        --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz \
        --output_fps 50 \
        --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz \
        --data_format smplh \
        --object_name 'ground' \
        --once
"

# Robot-Object Setting
apptainer exec --nv --writable-tmpfs --bind /run apptainer/holosoma.sif bash -c "
    mkdir -p /home/gbesset && \
    ln -sf /root/.holosoma_deps /home/gbesset/.holosoma_deps && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hsretargeting && \
    cd src/holosoma_retargeting/holosoma_retargeting && \
    sed -i 's/viewer = mjv/#viewer = mjv/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.opt/#viewer.opt/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.cam/#viewer.cam/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.sync/#viewer.sync/g' data_conversion/convert_data_format_mj.py && \
    sed -i 's/viewer.close/#viewer.close/g' data_conversion/convert_data_format_mj.py && \
    python data_conversion/convert_data_format_mj.py \
        --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz \
        --output_fps 50 \
        --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz \
        --data_format smplh \
        --object_name 'largebox' \
        --has_dynamic_object \
        --once
"

```

## Training

```bash

# Locomotion Mujoco FastSAC
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    export MUJOCO_GL=egl && \
    cd /workspace/holosoma && \
    python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-fast-sac \
        simulator:mjwarp \
        logger:wandb \
        --logger.video.enabled=False
"

# WBT Isaac PPO / FastSAC (g1-29dof-wbt-fast-sac)
apptainer exec --nv \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/.isaac_cache:/root/.local/share/ov \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $SCRATCH/isaac_tmp:/tmp \
    --bind $SCRATCH/isaac_pkg_cache:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/cache \
    --bind $SCRATCH/isaac_pkg_data:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/data \
    --bind $SCRATCH/isaac_pkg_logs:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/logs \
    --bind $SCRATCH/holosoma_converted_robots:/workspace/holosoma/src/holosoma/holosoma/data/robots/converted_rank0 \
    apptainer/holosoma.sif bash -c "
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
        logger:wandb \
        --logger.video.enabled=False \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz\"
"

# WBT Isaac PPO / FastSAC (g1-29dof-wbt-fast-sac) WITH OBJECT
apptainer exec --nv \
    --bind /run \
    --bind /dev/shm \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    --bind $SCRATCH/.isaac_cache:/root/.local/share/ov \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind $SCRATCH/isaac_tmp:/tmp \
    --bind $SCRATCH/isaac_pkg_cache:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/cache \
    --bind $SCRATCH/isaac_pkg_data:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/data \
    --bind $SCRATCH/isaac_pkg_logs:/root/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/isaacsim/kit/logs \
    --bind $SCRATCH/holosoma_converted_robots:/workspace/holosoma/src/holosoma/holosoma/data/robots/converted_rank0 \
    apptainer/holosoma.sif bash -c "
    export HTTP_PROXY=\$http_proxy && \
    export HTTPS_PROXY=\$https_proxy && \
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt && \
    export OMNI_KIT_ALLOW_ROOT=1 && \
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-w-object \
        simulator:isaacsim-w-object \
        logger:wandb \
        --logger.video.enabled=False \
        --training.num-envs=512 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz\" \
        --simulator.config.scene.rigid-objects.0.name=\"largebox\" \
        --simulator.config.scene.rigid-objects.0.urdf-path=\"src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf\" \
        --simulator.config.scene.rigid-objects.0.position=\"[1.0,0.0,1.0]\"
"
# The 0.5 small delta is just a security to not have an explosive computing at the simulation reset. The object position is adjusted randomly right after the reset. 

# WBT Mujoco PPO / FastSAC (g1-29dof-wbt-fast-sac)
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    /home/gbesset/WILLOW/holosoma/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt \
        simulator:mjwarp \
        logger:wandb \
        --logger.video.enabled=False \
        --reward.terms.undesired_contacts.weight=-0.1 \
        --training.num-envs=512 \
        --algo.config.actor_learning_rate=0.001 \
        --algo.config.critic_learning_rate=0.001 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz\"
"

#WBT Mujoco Multi GPU de l'extreme
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    /home/gbesset/WILLOW/holosoma/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a torchrun --nproc_per_node=2 src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt \
        simulator:mjwarp \
        logger:wandb \
        --logger.video.enabled=False \
        --training.multigpu=True \
        --training.num-envs=8192 \
        --reward.terms.undesired_contacts.weight=-0.1 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz\"
"

# WBT Mujoco PPO / FastSAC (g1-29dof-wbt-fast-sac) WITH OBJECT
apptainer exec --nv --writable-tmpfs \
    --bind /run \
    --bind /dev/shm \
    --bind $SCRATCH/holosoma_logs:/workspace/holosoma/logs \
    --bind $SCRATCH/wandb_cache:/workspace/holosoma/wandb \
    --bind /home/gbesset/WILLOW/holosoma/src:/workspace/holosoma/src \
    /home/gbesset/WILLOW/holosoma/apptainer/holosoma.sif bash -c "
    source /root/.holosoma_deps/miniconda3/etc/profile.d/conda.sh && \
    conda activate hssim && \
    export MUJOCO_GL=egl && \
    export WANDB_INSECURE_DISABLE_SSL=true && \
    cd /workspace/holosoma && \
    xvfb-run -a python src/holosoma/holosoma/train_agent.py \
        exp:g1-29dof-wbt-w-object \
        simulator:mjwarp-w-object \
        logger:wandb \
        terrain:terrain-locomotion-plane \
        --logger.video.enabled=False \
        --training.num-envs=8192 \
        --command.setup-terms.motion-command.params.motion-config.motion-file=\"holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz\" \
        --simulator.config.scene.rigid-objects.0.name=\"largebox\" \
        --simulator.config.scene.rigid-objects.0.urdf-path=\"src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf\" \
        --simulator.config.scene.rigid-objects.0.position=\"[1.0,0.0,1.0]\" \
        --terrain.terrain-term.mesh-type=\"PLANE\" \
        --terrain.terrain-term.obj-file-path=\"\" \
        --simulator.config.mujoco-warp.ccd-iterations=200
"

```