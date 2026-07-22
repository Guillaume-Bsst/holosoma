"""Offline probe: preview a mid-clip contact reset WITH vs WITHOUT grasp-settling — no training.

Motivation
----------
Object-interaction clips (robot carrying a box) break under the standard reset: RSI teleports the
robot and the object independently and adds independent noise, so a mid-clip reset lands with the
hand<->object contact broken -> the box is ejected (penetration) or dropped, and the episode dies on
the object-tracking termination. Verifying a fix normally means a 10-16h training run. This probe
lets you *see the reset moment itself* in seconds:

  1. build the object-interaction WBT env,
  2. force a reset at a chosen mid-clip contact frame,
  3. pin the robot at its post-reset pose (frozen frame) and let the box be a free rigid body,
  4. step physics and log how far the box drifts from its reference pose over time,

first with grasp-settling OFF (current behaviour) then ON (contact-consistent placement + settle
window, optionally + weld). A good fix keeps the box near the reference; the baseline lets it fly.

The robot is pinned kinematically at the frame it was reset to, which is the *best case* a trained
policy could achieve (hands exactly where the clip wants them). If the box still leaves the grasp
under that best case, the reset itself is broken — which is exactly what we are diagnosing.

Example
-------
    python -m holosoma.probe_grasp_settle exp:g1-29dof-wbt-w-object \
        --training.num-envs 5 \
        --probe-frames 64,129,155,194,259 --probe-steps 150 --probe-weld

    # add your usual viewer flag to watch it live; omit it for a headless numeric report.

Probe flags (parsed out before tyro sees the rest of the CLI):
    --probe-frames A,B,C   absolute frame indices, assigned round-robin across envs (default 155)
    --probe-steps N        physics steps to run per rollout (default 150)
    --probe-weld           enable the kinematic weld during the settle window (default off)
    --probe-drift-tol M    "box held" if final drift < M metres (default 0.15)
"""

from __future__ import annotations

import json
import sys

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import init_sim_imports
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


class ProbeArgs:
    frames: list[int] = [155]
    steps: int = 150
    weld: bool = False
    drift_tol: float = 0.15
    hold: str = "pin"  # "pin" = kinematic freeze at reference; "pd" = PD-control joints to reference (root pinned)
    squeeze: float = 0.0  # pd only: extra inward shoulder-roll target bias (rad) to press the box between hands
    follow: bool = False  # play the clip FORWARD: robot kinematically tracks the advancing reference through
    # the real MotionCommand.step() path (settle/assist-weld training code runs), box under physics.
    # Validates the full-contact weld curriculum end-to-end (baseline rollout = same replay, no assist).
    video: bool = False  # record the rollouts to IsaacSim MP4 (offscreen replicator; GUI viewer segfaults on this GPU)
    dump_qpos: str | None = None  # path PREFIX: dump env-0 rollouts as qpos NPZs (<prefix>_baseline.npz /
    # <prefix>_settle.npz, layout [root pos, root quat wxyz, joints(URDF order), obj pos, obj quat wxyz])
    # playable in holosoma_retargeting's viser_player.py — lets you scrub the simulated RSI rollout in a browser.
    out: str = "grasp_settle_probe.json"  # traces + report written here (IsaacSim eats stdout on shutdown)


def _pop_probe_args(argv: list[str]) -> tuple[ProbeArgs, list[str]]:
    """Extract --probe-* flags from argv so the remainder can go to tyro untouched."""
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
        elif tok == "--probe-drift-tol":
            args.drift_tol = float(argv[i + 1])
            i += 2
        elif tok == "--probe-out":
            args.out = argv[i + 1]
            i += 2
        elif tok == "--probe-hold":
            args.hold = argv[i + 1]
            i += 2
        elif tok == "--probe-squeeze":
            args.squeeze = float(argv[i + 1])
            i += 2
        elif tok == "--probe-video":
            args.video = True
            i += 1
        elif tok == "--probe-dump-qpos":
            args.dump_qpos = argv[i + 1]
            i += 2
        elif tok == "--probe-follow":
            args.follow = True
            i += 1
        elif tok == "--probe-weld":
            args.weld = True
            i += 1
        else:
            remaining.append(tok)
            i += 1
    return args, remaining


def _run_rollout(env, torch, mc, frames_tensor, *, settle: bool, weld: bool, steps: int, hold: str = "pin", squeeze: float = 0.0, record: bool = False, follow: bool = False):
    """Reset all envs to the forced frames with settling on/off, hold the robot, run physics.

    hold="pin": kinematically freeze the robot at its post-reset pose each substep (best case for a
        perfect tracker; isolates whether the box is held by geometry alone).
    hold="pd": pin only the root, and PD-control the joints toward the reference with the real gains
        (arms exert real torque); optional `squeeze` biases the shoulder-roll targets inward to press
        the box between the hands -> tests whether a learnable grip can keep the box up.

    Returns a dict of per-env, per-step traces (numpy-friendly python lists).
    """
    from holosoma.utils.grasp_settle import apply_grasp_transform, gather_anchor

    num_envs = env.num_envs
    all_ids = torch.arange(num_envs, device=env.device)

    # Configure the forced start frame + settle override, then reset through the real reset path so
    # the contact-consistent placement / settle arming in MotionCommand.reset() actually runs.
    mc._force_start_timesteps = frames_tensor
    mc._settle_enabled_override = bool(settle)
    env.reset_envs_idx(all_ids)
    # Push the freshly written reset state into the sim (normally done at the top of env.step()).
    env._refresh_envs_after_reset(all_ids)

    # Freeze reference for the probe: pin the robot at exactly the pose the reset produced (noised for
    # baseline, clean for settle), and remember the reference object pose at the frozen frame.
    hold_dof = env.simulator.dof_pos.clone()
    hold_root = env.simulator.robot_root_states[:, :7].clone()
    ref_obj_pos = mc.object_pos_w.clone()  # (num_envs, 3) reference object position at the frozen frame

    # Settle state that reset() populated (used for the weld and to size the settle window).
    settle_counter = mc.settle_counter.clone()
    rel_pos = mc.settle_grasp_rel_pos.clone()
    rel_quat = mc.settle_grasp_rel_quat.clone()
    anchor_idx = mc.settle_anchor_idx.clone()

    drift_trace: list[list[float]] = []
    z_trace: list[list[float]] = []
    handdist_trace: list[list[float]] = []
    body_trace: list = []  # per step: (num_envs, n_tracked_bodies, 3) tracked-body world positions
    objpos_trace: list = []  # per step: (num_envs, 3) object world position
    objquat_trace: list = []  # per step: (num_envs, 4) object world orientation (xyzw)
    qpos_trace: list = []  # per step (env 0): [root pos(3), root quat wxyz(4), joints URDF order, obj pos(3), obj quat wxyz(4)]

    # sim dof k -> its column in the motion NPZ joint order (= URDF order for our converted clips)
    joint_to_urdf = mc.motion._joint_indexes
    env0_origin = env.simulator.scene.env_origins[0]

    decimation = int(env.simulator.simulator_config.sim.control_decimation)

    # PD-hold target = the reference joints (post-reset dof), optionally with an inward squeeze bias.
    pd_target = hold_dof.clone()
    if hold == "pd" and squeeze != 0.0:
        dof_names = list(env.simulator.dof_names)
        for name, sign in (("left_shoulder_roll_joint", +1.0), ("right_shoulder_roll_joint", -1.0)):
            if name in dof_names:
                pd_target[:, dof_names.index(name)] += sign * squeeze

    def _pin_root():
        env.simulator.robot_root_states[:, :7] = hold_root
        env.simulator.robot_root_states[:, 7:13] = 0.0
        env.simulator.set_actor_root_state_tensor_robots(all_ids, env.simulator.robot_root_states)

    def _pin_robot():
        env.simulator.dof_pos[:] = hold_dof
        env.simulator.dof_vel[:] = 0.0
        _pin_root()
        env.simulator.set_dof_state_tensor(all_ids, env.simulator.dof_state)

    def _pd_hold():
        # real PD torque toward the reference (+squeeze), root kept fixed so the robot doesn't tip.
        _pin_root()
        tau = env.p_gains * (pd_target - env.simulator.dof_pos) - env.d_gains * env.simulator.dof_vel
        env.simulator.apply_torques_at_dof(tau)

    def _weld_object():
        weld_ids = torch.where(settle_counter > 0)[0]
        if weld_ids.numel() > 0:
            a_pos, a_quat = gather_anchor(
                mc.robot_anchor_pos_w[weld_ids], mc.robot_anchor_quat_w[weld_ids], anchor_idx[weld_ids]
            )
            op, oq = apply_grasp_transform(a_pos, a_quat, rel_pos[weld_ids], rel_quat[weld_ids])
            zeros6 = torch.zeros(weld_ids.numel(), 6, device=env.device)
            env.simulator.set_actor_states(["object"], weld_ids, torch.cat([op, oq, zeros6], dim=-1))

    rec = getattr(env.simulator, "video_recorder", None) if record else None
    if rec is not None:
        rec.start_recording(episode_id=0)

    for _ in range(steps):
        # Pin the robot at the reference EACH physics substep so contact is the only free dof, and
        # advance physics with the same call the env uses (simulate_at_each_physics_step, not
        # sim.step which does not integrate PhysX here). The object is a free rigid body -> it
        # falls / is ejected if the hand<->object contact is broken.
        # follow mode: the pin target is the CURRENT (advancing) reference frame — kinematic replay.
        if follow:
            hold_dof = mc.joint_pos.clone()
            hold_root = torch.cat([mc.root_pos_w, mc.root_quat_w], dim=-1)
        for _ in range(decimation):
            if hold == "pd":
                _pd_hold()
            else:
                _pin_robot()
            if settle and weld and not follow:
                _weld_object()
            env.simulator.simulate_at_each_physics_step()
        env.simulator.refresh_sim_tensors()

        # measure how far the box left its reference pose + nearest-hand distance.
        # follow mode: compare to the advancing reference (and let MotionCommand.step() run the real
        # training-path settle/assist-weld code + advance the clip).
        if follow:
            ref_obj_pos = mc.object_pos_w
        obj_pos = mc.simulator_object_pos_w
        drift = torch.norm(obj_pos - ref_obj_pos, dim=-1)
        hand_dist = torch.norm(mc.robot_anchor_pos_w - obj_pos.unsqueeze(1), dim=-1).min(dim=1).values
        drift_trace.append(drift.detach().cpu().tolist())
        z_trace.append(obj_pos[:, 2].detach().cpu().tolist())
        handdist_trace.append(hand_dist.detach().cpu().tolist())
        # skeleton + object pose for the offline animation renderer
        body_trace.append(mc.robot_body_pos_w.detach().cpu().tolist())
        objpos_trace.append(obj_pos.detach().cpu().tolist())
        objquat_trace.append(mc.simulator_object_quat_w.detach().cpu().tolist())
        # env-0 qpos snapshot for the viser dump (xyzw -> wxyz reorder for quats)
        rs0 = env.simulator.robot_root_states[0]
        joints_urdf = torch.zeros(int(joint_to_urdf.max().item()) + 1, device=env.device)
        joints_urdf[joint_to_urdf] = env.simulator.dof_pos[0]
        oq0 = mc.simulator_object_quat_w[0]
        qpos_trace.append(
            torch.cat(
                [
                    rs0[:3] - env0_origin,
                    rs0[[6, 3, 4, 5]],  # xyzw -> wxyz
                    joints_urdf,
                    obj_pos[0] - env0_origin,
                    oq0[[3, 0, 1, 2]],  # xyzw -> wxyz
                ]
            )
            .detach()
            .cpu()
            .tolist()
        )

        if follow:
            # real training code path: advances time_steps, applies settle/assist welds, decrements
            # the settle window (MotionCommand.step()).
            mc.step()
        else:
            settle_counter = torch.clamp(settle_counter - 1, min=0)

    if rec is not None:
        rec.stop_recording()

    return {
        "drift": drift_trace,
        "z": z_trace,
        "hand_dist": handdist_trace,
        "body": body_trace,
        "objpos": objpos_trace,
        "objquat": objquat_trace,
        "qpos": qpos_trace,
    }


def _report(frames: list[int], baseline: dict, settle: dict, drift_tol: float) -> str:
    def final(trace, e):
        return trace[-1][e]

    def peak(trace, e):
        return max(row[e] for row in trace)

    num_envs = len(baseline["drift"][0])
    lines = []
    lines.append("=" * 88)
    lines.append("GRASP-SETTLE PROBE — object drift from reference pose (metres). Lower = contact preserved.")
    lines.append("=" * 88)
    header = f"{'env':>3} {'frame':>6} | {'baseline final':>14} {'peak':>7} | {'settle final':>12} {'peak':>7} | verdict"
    lines.append(header)
    lines.append("-" * len(header))
    for e in range(num_envs):
        fr = frames[e % len(frames)]
        b_fin, b_pk = final(baseline["drift"], e), peak(baseline["drift"], e)
        s_fin, s_pk = final(settle["drift"], e), peak(settle["drift"], e)
        held_b = b_fin < drift_tol
        held_s = s_fin < drift_tol
        if held_s and not held_b:
            verdict = "FIXED  (box flew in baseline, held with settle)"
        elif held_s and held_b:
            verdict = "both held"
        elif not held_s:
            verdict = "STILL BROKEN (try --probe-weld / larger settle_steps)"
        else:
            verdict = "regressed?"
        lines.append(f"{e:>3} {fr:>6} | {b_fin:>14.3f} {b_pk:>7.3f} | {s_fin:>12.3f} {s_pk:>7.3f} | {verdict}")
    lines.append("-" * len(header))
    lines.append(f"drift tolerance for 'held' = {drift_tol} m. steps per rollout = {len(baseline['drift'])}.")
    lines.append("=" * 88)
    return "\n".join(lines)


def probe(tyro_config: ExperimentConfig, probe_args: ProbeArgs) -> None:
    if probe_args.video:
        # Enable IsaacSim's offscreen replicator recorder (the interactive GUI viewer segfaults on this
        # GPU). Frames are captured inside our simulate_at_each_physics_step loop once recording starts.
        import os
        from dataclasses import replace as _replace

        save_dir = os.path.dirname(os.path.abspath(probe_args.out)) or "."
        tyro_config = _replace(
            tyro_config,
            logger=_replace(
                tyro_config.logger,
                headless_recording=True,
                video=_replace(
                    tyro_config.logger.video,
                    enabled=True,
                    save_dir=save_dir,
                    use_recording_thread=False,
                    upload_to_wandb=False,
                ),
            ),
        )

    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)
    env.set_is_evaluating()

    # Boot the sim to a valid initial state before we drive per-mode resets ourselves.
    env.reset_all()

    mc = env.command_manager.get_state("motion_command")
    assert mc.motion.has_object, (
        "probe_grasp_settle requires an object-interaction experiment (e.g. exp:g1-29dof-wbt-w-object)."
    )

    # assign the requested frames round-robin across envs
    frames = probe_args.frames
    frames_tensor = torch.tensor(
        [frames[e % len(frames)] for e in range(env.num_envs)], dtype=torch.long, device=device
    )

    print(f"[probe] frames per env: {frames_tensor.tolist()}")
    print(f"[probe] steps={probe_args.steps} weld={probe_args.weld} drift_tol={probe_args.drift_tol}")

    print(f"[probe] hold={probe_args.hold} squeeze={probe_args.squeeze} video={probe_args.video}")
    print("[probe] rollout 1/2: grasp-settling OFF (baseline = standard RSI)")
    baseline = _run_rollout(
        env, torch, mc, frames_tensor, settle=False, weld=False, steps=probe_args.steps,
        hold=probe_args.hold, squeeze=probe_args.squeeze, record=probe_args.video, follow=probe_args.follow,
    )

    print("[probe] rollout 2/2: grasp-settling ON")
    settle = _run_rollout(
        env, torch, mc, frames_tensor, settle=True, weld=probe_args.weld, steps=probe_args.steps,
        hold=probe_args.hold, squeeze=probe_args.squeeze, follow=probe_args.follow,
    )

    report_text = _report(frames, baseline, settle, probe_args.drift_tol)

    # Persist BEFORE closing: IsaacSim's shutdown hard-exits the process and drops buffered stdout,
    # so write the report + full traces to disk (and flush stdout) to guarantee we keep the results.
    out_path = probe_args.out
    payload = {
        "frames": [frames[e % len(frames)] for e in range(env.num_envs)],
        "steps": probe_args.steps,
        "weld": probe_args.weld,
        "hold": probe_args.hold,
        "squeeze": probe_args.squeeze,
        "drift_tol": probe_args.drift_tol,
        "settle_steps": mc.grasp_settle_cfg.settle_steps,
        "body_names": list(mc.motion_cfg.body_names_to_track),
        "box_half_extents": [0.235, 0.23, 0.204],  # largebox.obj bbox (~0.47x0.46x0.41 m)
        "baseline": baseline,
        "settle": settle,
        "report": report_text,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    with open(out_path + ".txt", "w") as f:
        f.write(report_text + "\n")
    print(report_text, flush=True)
    print(f"[probe] wrote traces -> {out_path} and report -> {out_path}.txt", flush=True)

    # Optional: dump env-0 rollouts as viser-playable qpos NPZs (see --probe-dump-qpos).
    if probe_args.dump_qpos:
        import numpy as np

        fps_out = int(round(1.0 / env.dt))
        for tag, tr in (("baseline", baseline), ("settle", settle)):
            pth = f"{probe_args.dump_qpos}_{tag}.npz"
            np.savez(pth, qpos=np.asarray(tr["qpos"], np.float64), fps=np.int64(fps_out))
            print(f"[probe] qpos dump ({tag}, env 0) -> {pth}", flush=True)
    sys.stdout.flush()

    close_simulation_app(simulation_app)


def main() -> None:
    probe_args, remaining = _pop_probe_args(sys.argv[1:])
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG, args=remaining)
    probe(tyro_cfg, probe_args)


if __name__ == "__main__":
    main()
