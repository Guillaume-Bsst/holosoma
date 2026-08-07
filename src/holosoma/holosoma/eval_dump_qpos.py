"""Roll out a trained checkpoint in the real object-WBT env and dump viser-playable qpos excerpts.

Companion to probe_grasp_settle.py, but policy-driven: instead of pinning the robot kinematically,
this loads a training checkpoint (config embedded in the .pt), rebuilds the exact training env with
a handful of envs, forces a chosen RSI start frame per env, forces the assist-weld ON or OFF per
env, and records every env's [root pos, root quat wxyz, joints URDF order, obj pos, obj quat wxyz]
at each control step. Output plays in holosoma_retargeting's viser_player / viser_compare — this is
how we "see inside" the 4096-env training run without RTX rendering (broken on this driver).

Terminations behave exactly as in training (bad_tracking etc.); on reset the env restarts at the
same forced frame, so a rollout shows repeated attempts from the same RSI moment.

Example
-------
    OMNI_KIT_ACCEPT_EULA=YES python -u -m holosoma.eval_dump_qpos \
        --evald-checkpoint logs/WholeBodyTracking/<run>/model_04000.pt \
        --evald-frames 140,140,200,200 --evald-weld on,off,on,off \
        --evald-steps 300 --evald-out /path/prefix \
        --training.num-envs 4 --training.headless True

Flags (popped before tyro; the rest overrides the saved training config):
    --evald-checkpoint PATH   checkpoint .pt (its embedded experiment_config is the base config)
    --evald-frames A,B,...    absolute start frame per env, round-robin (default 200)
    --evald-weld m1,m2,...    per-env assist-weld forcing: on|off|auto, round-robin (default auto)
    --evald-steps N           control steps to record (default 300)
    --evald-out PREFIX        writes <prefix>_env<i>_<weld>.npz + <prefix>_meta.json
"""

from __future__ import annotations

import json
import os
import sys

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_sim_imports,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


class EvalDumpArgs:
    checkpoint: str = ""
    frames: list[int] = [200]
    weld: list[str] = ["auto"]  # on | off | auto (auto = whatever reset() draws)
    steps: int = 300
    out: str = "eval_dump"


def _pop_args(argv: list[str]) -> tuple[EvalDumpArgs, list[str]]:
    args = EvalDumpArgs()
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--evald-checkpoint":
            args.checkpoint = argv[i + 1]
            i += 2
        elif tok == "--evald-frames":
            args.frames = [int(x) for x in argv[i + 1].split(",") if x != ""]
            i += 2
        elif tok == "--evald-weld":
            args.weld = [x.strip().lower() for x in argv[i + 1].split(",") if x != ""]
            i += 2
        elif tok == "--evald-steps":
            args.steps = int(argv[i + 1])
            i += 2
        elif tok == "--evald-out":
            args.out = argv[i + 1]
            i += 2
        else:
            remaining.append(tok)
            i += 1
    if not args.checkpoint:
        raise SystemExit("--evald-checkpoint is required")
    return args, remaining


def main() -> None:
    evald, remaining = _pop_args(sys.argv[1:])

    saved_cfg, _wandb_path = load_saved_experiment_config(CheckpointConfig(checkpoint=evald.checkpoint))
    cfg = tyro.cli(
        ExperimentConfig,
        default=saved_cfg,
        args=remaining,
        description="Overrides on top of the checkpoint's saved config.",
        config=TYRO_CONIFG,
    )

    simulation_app = init_sim_imports(cfg)

    import numpy as np
    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)
    env.set_is_evaluating()

    algo = get_class(cfg.algo._target_)(
        device=device, env=env, config=cfg.algo.config, log_dir=os.path.dirname(evald.out) or ".", multi_gpu_cfg=None
    )
    algo.setup()
    local_checkpoint = load_checkpoint(evald.checkpoint, log_dir=os.path.dirname(evald.out) or ".")
    algo.load(str(local_checkpoint))
    policy = algo.get_inference_policy()

    num_envs = env.num_envs
    all_ids = torch.arange(num_envs, device=env.device)
    frames = [evald.frames[e % len(evald.frames)] for e in range(num_envs)]
    weld_modes = [evald.weld[e % len(evald.weld)] for e in range(num_envs)]
    force_on = torch.tensor([m == "on" for m in weld_modes], device=env.device)
    force_off = torch.tensor([m == "off" for m in weld_modes], device=env.device)

    # Boot (reset_all re-inits motion-command buffers, clearing the debug hooks) THEN force frames.
    env.reset_all()
    mc = env.command_manager.get_state("motion_command")
    assert mc.motion.has_object, "eval_dump_qpos expects an object-interaction experiment"
    mc._force_start_timesteps = torch.tensor(frames, dtype=torch.long, device=env.device)
    env.reset_envs_idx(all_ids)
    env._refresh_envs_after_reset(all_ids)
    env._compute_observations()
    env._post_compute_observations_callback()
    env._clip_observations()
    obs_dict = env.obs_buf_dict

    joint_to_urdf = mc.motion._joint_indexes
    origins = env.simulator.scene.env_origins

    qpos = [[] for _ in range(num_envs)]  # per env, per step: 3+4+29+3+4
    drift = [[] for _ in range(num_envs)]
    dones_trace = [[] for _ in range(num_envs)]
    weld_trace = [[] for _ in range(num_envs)]
    frame_trace = [[] for _ in range(num_envs)]
    restarts = [0] * num_envs

    with torch.no_grad():
        for _step in range(evald.steps):
            mc.weld_assist[force_on] = True
            mc.weld_assist[force_off] = False
            actor_obs = torch.cat([obs_dict[k] for k in algo.actor_obs_keys], dim=1)
            actions = policy({"actor_obs": actor_obs})
            obs_dict, _rew, dones, _extras = env.step({"actions": actions})

            rs = env.simulator.robot_root_states
            obj_p = mc.simulator_object_pos_w
            obj_q = mc.simulator_object_quat_w
            d = torch.norm(obj_p - mc.object_pos_w, dim=-1)
            joints = torch.zeros(num_envs, int(joint_to_urdf.max().item()) + 1, device=env.device)
            joints[:, joint_to_urdf] = env.simulator.dof_pos
            row = torch.cat(
                [
                    rs[:, :3] - origins,
                    rs[:, [6, 3, 4, 5]],  # xyzw -> wxyz
                    joints,
                    obj_p - origins,
                    obj_q[:, [3, 0, 1, 2]],  # xyzw -> wxyz
                ],
                dim=-1,
            ).detach().cpu().numpy()
            done_np = dones.detach().cpu().numpy()
            weld_np = mc.weld_assist.detach().cpu().numpy()
            frame_np = mc.time_steps.detach().cpu().numpy()
            drift_np = d.detach().cpu().numpy()
            for e in range(num_envs):
                qpos[e].append(row[e])
                drift[e].append(float(drift_np[e]))
                dones_trace[e].append(bool(done_np[e]))
                weld_trace[e].append(bool(weld_np[e]))
                frame_trace[e].append(int(frame_np[e]))
                if done_np[e]:
                    restarts[e] += 1

    fps_out = int(round(1.0 / env.dt))
    meta = {
        "checkpoint": evald.checkpoint,
        "steps": evald.steps,
        "fps": fps_out,
        "envs": [],
    }
    for e in range(num_envs):
        tag = weld_modes[e]
        pth = f"{evald.out}_env{e}_f{frames[e]}_weld-{tag}.npz"
        np.savez(pth, qpos=np.asarray(qpos[e], np.float64), fps=np.int64(fps_out))
        meta["envs"].append(
            {
                "env": e,
                "npz": pth,
                "start_frame": frames[e],
                "weld_mode": tag,
                "restarts": restarts[e],
                "final_drift": drift[e][-1],
                "peak_drift": max(drift[e]),
                "drift": drift[e],
                "dones": [i for i, v in enumerate(dones_trace[e]) if v],
                "weld_on_frac": sum(weld_trace[e]) / len(weld_trace[e]),
                "ref_frame_trace": frame_trace[e][:: max(1, evald.steps // 100)],
            }
        )
        print(
            f"[evald] env{e} f{frames[e]} weld={tag}: restarts={restarts[e]} "
            f"final_drift={drift[e][-1]:.3f} peak={max(drift[e]):.3f} -> {pth}",
            flush=True,
        )
    with open(f"{evald.out}_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"[evald] meta -> {evald.out}_meta.json", flush=True)
    sys.stdout.flush()

    close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
