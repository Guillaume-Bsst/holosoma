"""Roll a trained checkpoint under the REAL training reset distribution (natural RSI, no forced
frames) and record, at every episode death, WHY it died -- so we can separate:

  * RSI/init failure  -> deaths cluster at low episode-age (a few steps after reset), often with box
    drift spiking immediately: the reset lands in an unrecoverable state.
  * carry/holding failure -> deaths at moderate age, high box drift, during the clip's contact window:
    the box can't be held once free.
  * body-tracking failure -> deaths with high body mpkpe but LOW box drift: the robot loses the body
    pose regardless of the box.

No viser, no qpos dump -- pure numeric aggregation printed to a report file (IsaacSim eats stdout on
shutdown). Usage:

    OMNI_KIT_ACCEPT_EULA=YES python -u -m holosoma.diagnose_deaths \
        --diag-checkpoint logs/.../model_29999.pt --diag-steps 500 \
        --diag-out /path/report.json --training.num-envs 512 --training.headless True
"""

from __future__ import annotations

import json
import os
import sys

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.eval_utils import CheckpointConfig, init_sim_imports, load_saved_experiment_config
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


class DiagArgs:
    checkpoint: str = ""
    steps: int = 500
    out: str = "diagnose_deaths.json"


def _pop(argv):
    a = DiagArgs()
    rem = []
    i = 0
    while i < len(argv):
        if argv[i] == "--diag-checkpoint":
            a.checkpoint = argv[i + 1]; i += 2
        elif argv[i] == "--diag-steps":
            a.steps = int(argv[i + 1]); i += 2
        elif argv[i] == "--diag-out":
            a.out = argv[i + 1]; i += 2
        else:
            rem.append(argv[i]); i += 1
    if not a.checkpoint:
        raise SystemExit("--diag-checkpoint required")
    return a, rem


def main():
    diag, remaining = _pop(sys.argv[1:])
    saved_cfg, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=diag.checkpoint))
    cfg = tyro.cli(ExperimentConfig, default=saved_cfg, args=remaining,
                   description="Overrides on top of checkpoint config.", config=TYRO_CONIFG)

    simulation_app = init_sim_imports(cfg)
    import numpy as np
    import torch
    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)
    env.set_is_evaluating()
    algo = get_class(cfg.algo._target_)(device=device, env=env, config=cfg.algo.config,
                                        log_dir=os.path.dirname(diag.out) or ".", multi_gpu_cfg=None)
    algo.setup()
    algo.load(diag.checkpoint)
    policy = algo.get_inference_policy()

    num_envs = env.num_envs
    env.reset_all()
    mc = env.command_manager.get_state("motion_command")
    assert mc.motion.has_object

    settle_steps = int(mc.grasp_settle_cfg.settle_steps)
    T_clip = int(mc.motion.time_step_total)
    has_gt = bool(mc.motion.has_gt_contact)
    obs_dict = env.obs_buf_dict

    age = torch.zeros(num_envs, dtype=torch.long, device=device)  # steps alive since last reset

    # per-death records
    rec_age, rec_frame, rec_drift, rec_mpkpe, rec_refcontact, rec_handdist = [], [], [], [], [], []
    # post-reset early drift: box drift at age 1..settle+4, to see if RSI breaks contact immediately
    early_drift_by_age = {k: [] for k in range(1, settle_steps + 5)}

    def snapshot():
        """Diagnostic quantities at the CURRENT (pre-step) reference frame, per env."""
        frame = mc.time_steps.clone()
        drift = torch.norm(mc.simulator_object_pos_w - mc.object_pos_w, dim=-1)
        # nearest sim-hand to sim-object distance
        hand_dist = torch.norm(mc.robot_anchor_pos_w - mc.simulator_object_pos_w.unsqueeze(1), dim=-1).min(dim=1).values
        mpkpe = mc.metrics.get("motion/mpkpe")
        if mpkpe is None:
            mpkpe = torch.full((num_envs,), float("nan"), device=device)
        _, refc = mc._lookup_ref_contact(frame, mc.anchor_pos_w, mc.object_pos_w)
        return frame, drift, hand_dist, mpkpe.clone(), refc

    with torch.no_grad():
        for step in range(diag.steps):
            frame, drift, hand_dist, mpkpe, refc = snapshot()

            # record early-post-reset drift bucketed by current age (contact frames only -- that's
            # where RSI has to preserve the grasp)
            for k in early_drift_by_age:
                sel = (age == k) & refc
                if sel.any():
                    early_drift_by_age[k].extend(drift[sel].cpu().tolist())

            actor_obs = torch.cat([obs_dict[k] for k in algo.actor_obs_keys], dim=1)
            actions = policy({"actor_obs": actor_obs})
            obs_dict, _rew, dones, _extras = env.step({"actions": actions})

            dead = dones.bool()
            if dead.any():
                idx = torch.where(dead)[0]
                rec_age.extend(age[idx].cpu().tolist())
                rec_frame.extend(frame[idx].cpu().tolist())
                rec_drift.extend(drift[idx].cpu().tolist())
                rec_mpkpe.extend(mpkpe[idx].cpu().tolist())
                rec_refcontact.extend(refc[idx].cpu().tolist())
                rec_handdist.extend(hand_dist[idx].cpu().tolist())

            age = age + 1
            age[dead] = 0

    rec_age = np.array(rec_age); rec_frame = np.array(rec_frame)
    rec_drift = np.array(rec_drift); rec_mpkpe = np.array(rec_mpkpe)
    rec_refcontact = np.array(rec_refcontact, dtype=bool); rec_handdist = np.array(rec_handdist)
    n = len(rec_age)

    def pct(mask):
        return round(100.0 * mask.sum() / max(n, 1), 1)

    rsi_window = rec_age <= (settle_steps + 2)
    contact_death = rec_refcontact
    high_drift = rec_drift > 0.20  # box clearly left its reference pose
    low_drift = rec_drift <= 0.20

    report = {
        "checkpoint": diag.checkpoint,
        "steps": diag.steps,
        "num_envs": num_envs,
        "T_clip": T_clip,
        "settle_steps": settle_steps,
        "has_gt_contact": has_gt,
        "total_deaths": int(n),
        "deaths_in_rsi_window_pct": pct(rsi_window),
        "deaths_during_ref_contact_pct": pct(contact_death),
        "deaths_high_boxdrift_pct": pct(high_drift),
        "median_age_at_death": float(np.median(rec_age)) if n else None,
        "median_boxdrift_at_death": float(np.median(rec_drift)) if n else None,
        "median_mpkpe_at_death": float(np.nanmedian(rec_mpkpe)) if n else None,
        # split diagnostics
        "contact_deaths_median_drift": float(np.median(rec_drift[contact_death])) if contact_death.any() else None,
        "freeframe_deaths_median_drift": float(np.median(rec_drift[~contact_death])) if (~contact_death).any() else None,
        "high_drift_deaths_median_mpkpe": float(np.nanmedian(rec_mpkpe[high_drift])) if high_drift.any() else None,
        "low_drift_deaths_median_mpkpe": float(np.nanmedian(rec_mpkpe[low_drift])) if low_drift.any() else None,
        # early post-reset drift on contact frames (RSI health): median drift at each age
        "early_contact_drift_median_by_age": {
            k: (round(float(np.median(v)), 4) if v else None) for k, v in early_drift_by_age.items()
        },
        # death age histogram (coarse buckets)
        "age_hist": {
            "0-2": int((rec_age <= 2).sum()),
            "3-settle": int(((rec_age > 2) & (rec_age <= settle_steps)).sum()),
            "settle-50": int(((rec_age > settle_steps) & (rec_age <= 50)).sum()),
            "50-150": int(((rec_age > 50) & (rec_age <= 150)).sum()),
            "150+": int((rec_age > 150).sum()),
        },
        # death by clip phase (10 bins over the clip)
        "frame_hist": [int(((rec_frame >= b * T_clip // 10) & (rec_frame < (b + 1) * T_clip // 10)).sum())
                       for b in range(10)],
    }

    with open(diag.out, "w") as f:
        json.dump(report, f, indent=1)
    print("=" * 70, flush=True)
    print(json.dumps(report, indent=1), flush=True)
    print(f"[diag] wrote {diag.out}", flush=True)
    sys.stdout.flush()
    close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
