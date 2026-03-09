# Holosoma Apptainer Container Installation

Containerized environment for Holosoma full pipeline with IsaacSim & MuJoCo

## Requirements

* Linux OS (Ubuntu 22.04+)
* 128 GB RAM (for building)
* NVIDIA GPU with CUDA support
* 150 GB disk space

## Building

From `holosoma/` directory:

```bash
export APPTAINER_CACHEDIR=$SCRATCH/.apptainer_cache
mkdir -p $APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=$TMP_DIR/.apptainer_tmp
mkdir -p $APPTAINER_TMPDIR
apptainer build --fakeroot --force apptainer/holosoma.sif apptainer/setup_apptainer.def
```

This accepts the NVIDIA Isaac Sim EULA. Build time: ~30-60 minutes

## Running

```bash
# Interactive shell
apptainer exec --nv --bind /run apptainer/holosoma.sif bash

# Run Python script
apptainer exec --nv --bind /run apptainer/holosoma.sif python my_script.py

# With home directory access
apptainer exec --nv --bind /run --bind $HOME:/home/$USER apptainer/holosoma.sif bash
```

Other example usages in EXAMPLE.md

## Installed

* Isaac Sim 5.1.0 + Isaac Lab
* MuJoCo 3.5.0
* Holosoma + tools