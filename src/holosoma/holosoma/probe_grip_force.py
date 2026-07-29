"""Offline probe: isolate whether the grip-force controller alone can hold the box, GIVEN perfect
whole-body reference tracking -- no training, no policy.

Motivation
----------
The grip-force-no-curriculum run is stuck (object_hand_dist plateaued ~50%, success_rate near zero):
the whole-body policy isn't reliably getting its hands to the box in the first place, so we can't
yet tell from that run whether the FORCE MAGNITUDE (target_force_n) is even sufficient once contact
is made. This probe removes the whole-body-tracking bootstrap problem entirely: every policy step,
it feeds the EXACT scripted action that makes the PD position target equal the reference joint
pose (actions = (ref_dof_pos - default_dof_pos) / action_scales), for ALL DOF including the wrists.
This is the best case a perfectly-trained policy could produce. The wrist DOF then evolve under
REAL torque control (PD tracking + the REAL grip-force bias from JointPositionActionTerm, unmodified
production code -- not a reimplementation), the box is a free rigid body under real physics, and the
motion clip plays forward normally (MotionCommand.step() runs for real, so grip_active turns on/off
exactly as in training).

If the box stays near the reference under this best-case scripted tracking, the grip-force mechanism
itself is sound and the bottleneck is purely the whole-body-tracking bootstrap (motivates the
alpha=1 kinematic curriculum + later force switch). If the box still drifts/falls even here, the
force magnitude (or squeeze geometry) itself needs fixing before spending another training run on it.

Example
-------
    OMNI_KIT_ACCEPT_EULA=YES python -m holosoma.probe_grip_force \
        exp:g1-29dof-wbt-w-object-grip-force \
        --probe-frames 140,155,170,200 --probe-steps 150 \
        --training.num-envs 8 --training.headless True
"""

from __future__ import annotations

import json
import sys

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


class ProbeArgs:
    frames: list[int] = [155]
    steps: int = 150
    out: str = "grip_force_probe.json"
    drift_tol: float = 0.15  # metres; "held" if final drift below this


def _pop_probe_args(argv: list[str]) -> tuple[ProbeArgs, list[str]]:
    args = ProbeArgs()
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--probe-frames":
            args.frames = [int(x) for x in argv[i + 1].split(",") if x != ""]
            i += 2
        elif tok == "--probe-steps":
            args.steps = int(argv[i + 1])
            i += 2
        elif tok == "--probe-out":
            args.out = argv[i + 1]
            i += 2
        elif tok == "--probe-drift-tol":
            args.drift_tol = float(argv[i + 1])
            i += 2
        else:
            remaining.append(tok)
            i += 1
    return args, remaining


def main() -> None:
    probe_args, remaining = _pop_probe_args(sys.argv[1:])

    cfg = tyro.cli(
        AnnotatedExperimentConfig,
        args=remaining,
        description="Grip-force isolation probe: perfect scripted tracking + real grip-force torque.",
        config=TYRO_CONIFG,
    )

    simulation_app = init_sim_imports(cfg)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(cfg.env_class)(get_tyro_env_config(cfg), device=device)

    num_envs = env.num_envs
    all_ids = torch.arange(num_envs, device=env.device)
    frames = [probe_args.frames[e % len(probe_args.frames)] for e in range(num_envs)]

    env.reset_all()
    mc = env.command_manager.get_state("motion_command")
    assert mc.motion.has_object, "probe_grip_force expects an object-interaction experiment"
    mc._force_start_timesteps = torch.tensor(frames, dtype=torch.long, device=env.device)
    env.reset_envs_idx(all_ids)
    env._refresh_envs_after_reset(all_ids)
    env._compute_observations()
    env._post_compute_observations_callback()
    env._clip_observations()

    drift_trace = [[] for _ in range(num_envs)]
    height_trace = [[] for _ in range(num_envs)]
    hand_dist_trace = [[] for _ in range(num_envs)]
    grip_active_trace = [[] for _ in range(num_envs)]
    command_force_trace = []

    action_scales = env.action_scales  # (num_dof,)
    safe_scales = torch.where(action_scales.abs() > 1e-8, action_scales, torch.ones_like(action_scales))

    with torch.no_grad():
        for _step in range(probe_args.steps):
            ref_dof = mc.joint_pos  # (num_envs, num_dof) reference pose at the CURRENT frame
            actions = (ref_dof - env.default_dof_pos) / safe_scales
            actions = torch.where(action_scales.abs() > 1e-8, actions, torch.zeros_like(actions))

            env.step({"actions": actions})

            sim_obj = mc.simulator_object_pos_w
            ref_obj = mc.object_pos_w
            drift = torch.norm(sim_obj - ref_obj, dim=-1)
            hand_dist = torch.norm(mc.robot_anchor_pos_w - sim_obj.unsqueeze(1), dim=-1).min(dim=1).values

            for e in range(num_envs):
                drift_trace[e].append(float(drift[e]))
                height_trace[e].append(float(sim_obj[e, 2]))
                hand_dist_trace[e].append(float(hand_dist[e]))
                grip_active_trace[e].append(bool(mc.grip_active[e]))

            cf_left = env.log_dict.get("grip/command_force_left")
            cf_right = env.log_dict.get("grip/command_force_right")
            command_force_trace.append(
                [float(cf_left) if cf_left is not None else None, float(cf_right) if cf_right is not None else None]
            )

    report = {"envs": []}
    for e in range(num_envs):
        final_drift = drift_trace[e][-1]
        max_drift = max(drift_trace[e])
        held = final_drift < probe_args.drift_tol
        report["envs"].append(
            {
                "env": e,
                "start_frame": frames[e],
                "final_drift_m": final_drift,
                "max_drift_m": max_drift,
                "held": held,
                "final_height_m": height_trace[e][-1],
                "min_height_m": min(height_trace[e]),
                "grip_active_frac": sum(grip_active_trace[e]) / len(grip_active_trace[e]),
                "drift_trace": drift_trace[e],
                "hand_dist_trace": hand_dist_trace[e],
                "height_trace": height_trace[e],
            }
        )
        print(
            f"[probe] env{e} f{frames[e]}: held={held} final_drift={final_drift:.3f}m "
            f"max_drift={max_drift:.3f}m grip_active_frac={report['envs'][-1]['grip_active_frac']:.2f}"
        )

    n_held = sum(1 for r in report["envs"] if r["held"])
    print(f"[probe] {n_held}/{num_envs} envs held the box (final drift < {probe_args.drift_tol}m)")

    with open(probe_args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] wrote {probe_args.out}")

    close_simulation_app(simulation_app)


if __name__ == "__main__":
    main()
