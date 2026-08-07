from __future__ import annotations

import json
import sys

import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies import BasePolicy
from holosoma_inference.policies.wbt_utils import MotionClockUtil, PinocchioRobot, TimestepUtil
from holosoma_inference.utils.clock import ClockSub
from holosoma_inference.utils.math.quat import (
    matrix_from_quat,
    quat_mul,
    quat_rotate_inverse,
    quat_to_rpy,
    rpy_to_quat,
    subtract_frame_transforms,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


class WholeBodyTrackingPolicy(BasePolicy):
    def __init__(self, config: InferenceConfig):
        self.config = config

        # initialize motion state
        self.motion_clip_progressing = False
        self.curr_motion_timestep = config.task.motion_start_timestep
        self.motion_command_t = None
        self.ref_quat_xyzw_t = None
        self.motion_command_0 = None
        self.ref_quat_xyzw_0 = None

        # Initialize clock for sim-time synchronization
        clock_sub = ClockSub()
        clock_sub.start()
        clock_util = MotionClockUtil(clock_sub)
        self.timestep_util = TimestepUtil(
            clock=clock_util,
            interval_ms=1000.0 / config.task.rl_rate,
            start_timestep=config.task.motion_start_timestep,
        )

        # Read use_sim_time from config
        self.use_sim_time = config.task.use_sim_time

        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self.per_joint_policy_action_scale: np.ndarray | None = None

        super().__init__(config)
        self._configure_action_scales()

        # Object-carry policies need obj_pos_b/obj_ori_b in the actor obs. The box tracks the
        # reference during contact (kinematic in training), so that box-pose-in-reference-root-frame
        # transform is derived from the clip itself and indexed by the motion timestep.
        self._obj_pos_b_traj = None
        self._obj_ori_b_traj = None
        # Static support table (e.g. the drop-off table): position/orientation relative to the
        # moving torso frame, same clip-derived math as obj_pos_b/obj_ori_b -- only populated when
        # the clip carries support_pos_w/support_quat_w (see wbt_w_object_support obs preset).
        self._support_pos_b_traj = None
        self._support_ori_b_traj = None
        if getattr(config.task, "object_motion_file", None):
            self._load_object_motion(config.task.object_motion_file, config.task.motion_prepend_timesteps)

        # Closed-loop object obs (sim2sim): pose of the REAL simulated box streamed by run_sim's
        # SimulatorBridge. See TaskConfig.live_object_obs. Clip lookup remains the fallback until
        # the first pose message lands.
        self._object_pose_sub = None
        self._live_obs_active = False  # for one-shot logging of the fallback->live transition
        if getattr(config.task, "live_object_obs", False):
            from holosoma_inference.utils.clock import PoseSub  # noqa: PLC0415 -- optional feature

            self._object_pose_sub = PoseSub()
            self._object_pose_sub.start()
            logger.info("Live object obs enabled: subscribing to run_sim box pose (port 5556)")

        # Guard both ways so the two obs shapes never get crossed:
        #  - object config (obj_pos_b in actor_obs) but no clip -> the obs assembly would KeyError on
        #    the missing term; fail early with an actionable message instead.
        #  - a clip was passed but the config has no object terms -> the extra obs would be silently
        #    dropped; warn so the mismatch is visible.
        actor_terms = self.obs_dict.get("actor_obs", [])
        wants_object = "obj_pos_b" in actor_terms
        if wants_object and self._obj_pos_b_traj is None and self._object_pose_sub is None:
            raise ValueError(
                "Inference config expects object observations (obj_pos_b/obj_ori_b) but no "
                "--task.object-motion-file was given. Pass the training clip NPZ, or use "
                "inference:g1-29dof-wbt (non-object / full-loco) instead."
            )
        if self._obj_pos_b_traj is not None and not wants_object:
            logger.warning(
                "object_motion_file was provided but this obs config has no obj_pos_b/obj_ori_b terms "
                "(non-object policy); the object motion will be ignored."
            )

        wants_support = "support_pos_b" in actor_terms
        if wants_support and self._support_pos_b_traj is None:
            raise ValueError(
                "Inference config expects support-table observations (support_pos_b/support_ori_b) "
                "but the clip passed via --task.object-motion-file has no support_pos_w/support_quat_w "
                "fields. Use a clip that was exported with the static support table, or use "
                "inference:g1-29dof-wbt-w-object (no support terms) instead."
            )

        # Biais de couple de prise (60 N/main a l'entrainement). Voir policies/grip_force.py pour
        # le detail : cette force est ajoutee par l'ENVIRONNEMENT a l'entrainement, jamais par le
        # reseau, donc elle doit etre reconstruite ici ou la caisse tombe.
        self._grip_force = None
        self._last_obj_pos_b = None
        if getattr(config.task, "grip_force_enable", False):
            from holosoma_inference.policies.grip_force import (  # noqa: PLC0415 -- optionnel
                GripForceBias,
                GripForceParams,
            )

            urdf = getattr(config.task, "grip_force_urdf", None)
            if not urdf:
                raise ValueError(
                    "--task.grip-force-enable exige --task.grip-force-urdf (l'URDF du run "
                    "d'entrainement). Sans FK du poignet, le couple de prise est incalculable."
                )
            if not getattr(config.task, "object_motion_file", None):
                raise ValueError(
                    "--task.grip-force-enable exige --task.object-motion-file : le flag de contact "
                    "GT (object_ref_contact) sert de porte a la force. Sans lui elle serait "
                    "appliquee en continu, y compris pendant l'approche et apres le depot."
                )
            self._grip_force = GripForceBias(
                urdf,
                tuple(self.robot_config.dof_names),
                GripForceParams(
                    target_force_n=float(config.task.grip_force_target_n),
                    gate=str(getattr(config.task, "grip_force_gate", "clip")),
                    contact_distance_m=float(getattr(config.task, "grip_force_contact_distance_m", 0.35)),
                ),
            )
            self._grip_force.load_contact_flags(
                config.task.object_motion_file, config.task.motion_prepend_timesteps
            )

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            # Fallback to default_dof_angles if not specified
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for WBT policy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for WBT policy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Prompt user before entering stiff mode (only if stdin is available)
        def _show_warning():
            logger.warning(
                colored(
                    "⚠️  Non-interactive mode detected - cannot prompt for stiff mode confirmation!",
                    "red",
                    attrs=["bold"],
                )
            )

        if hasattr(self, "_shared_hardware_source"):
            logger.info(colored("Skipping stiff hold prompt (secondary policy)", "yellow"))
        elif sys.stdin.isatty():
            logger.info(colored("\n⚠️  Ready to enter stiff hold mode", "yellow", attrs=["bold"]))
            logger.info(colored("Press Enter to continue...", "yellow"))
            try:
                input()
                logger.info(colored("✓ Entering stiff hold mode", "green"))
            except EOFError:
                # [drockyd] seems like in some cases, input() will raise EOFError even in interactive mode.
                _show_warning()
        else:
            _show_warning()

    def _load_object_motion(self, npz_path: str, prepend: int) -> None:
        """Precompute obj_pos_b (3) + obj_ori_b (6D) per motion frame from the training clip.

        obj_pos_b/obj_ori_b are the box pose expressed in the REF-BODY frame -- torso_link, matching
        training's obj_pos_b/obj_ori_b terms which use robot_ref_pos_w/robot_ref_quat_w
        (= motion_config.body_name_ref, the torso), NOT the pelvis root. Both the box and the torso
        come from the same clip, so the transform is clip-internal (independent of where the robot
        actually is in the world) and matches what the policy saw in training whenever the robot and
        box tracked the reference (contact frames -- kinematic -- and rest). We hold frame 0 for the
        `prepend` default-pose frames so the index lines up with the ONNX motion timestep.
        """
        data = np.load(npz_path)
        obj_pos = np.asarray(data["object_pos_w"], np.float32)  # (T, 3) world
        obj_quat = np.asarray(data["object_quat_w"], np.float32)  # (T, 4) wxyz
        # Reference body = torso_link (training body_name_ref), taken from the clip's body tracks.
        # Using the pelvis root here instead is ~5 cm off at stand but up to ~0.45 m / a large
        # rotation off mid-clip when the reference leans to pick the box.
        body_names = [str(n) for n in data["body_names"]]
        ref_idx = body_names.index("torso_link")
        root_pos = np.asarray(data["body_pos_w"], np.float32)[:, ref_idx]  # (T, 3)
        root_quat = np.asarray(data["body_quat_w"], np.float32)[:, ref_idx]  # (T, 4) wxyz

        # box pose in the reference torso frame (inference math utils, all wxyz -- same geometry as
        # the training obj_pos_b/obj_ori_b, which used the xyzw sim convention).
        rel_pos = quat_rotate_inverse(root_quat, obj_pos - root_pos)  # (T, 3)
        rel_quat = subtract_frame_transforms(root_quat, obj_quat)  # (T, 4) wxyz
        rel_mat = matrix_from_quat(rel_quat)  # (T, 3, 3)
        ori6 = rel_mat[..., :2].reshape(rel_mat.shape[0], -1)  # (T, 6) first two columns

        if prepend > 0:
            rel_pos = np.concatenate([np.repeat(rel_pos[:1], prepend, axis=0), rel_pos], axis=0)
            ori6 = np.concatenate([np.repeat(ori6[:1], prepend, axis=0), ori6], axis=0)

        self._obj_pos_b_traj = rel_pos.astype(np.float32)
        self._obj_ori_b_traj = ori6.astype(np.float32)
        logger.info(
            f"Loaded object motion from {npz_path}: {self._obj_pos_b_traj.shape[0]} frames "
            f"(prepend={prepend}) -> obj_pos_b(3)+obj_ori_b(6) available in actor obs."
        )

        # Static support table (optional): support_pos_w/support_quat_w are a single world pose
        # (not per-frame), broadcast across T then expressed in the same moving torso frame as
        # obj_pos_b/obj_ori_b above -- matches training's support_pos_b/support_ori_b terms.
        if "support_pos_w" in data:
            support_pos = np.broadcast_to(np.asarray(data["support_pos_w"], np.float32), obj_pos.shape)
            support_quat = np.broadcast_to(np.asarray(data["support_quat_w"], np.float32), obj_quat.shape)

            support_rel_pos = quat_rotate_inverse(root_quat, support_pos - root_pos)  # (T, 3)
            support_rel_quat = subtract_frame_transforms(root_quat, support_quat)  # (T, 4) wxyz
            support_rel_mat = matrix_from_quat(support_rel_quat)  # (T, 3, 3)
            support_ori6 = support_rel_mat[..., :2].reshape(support_rel_mat.shape[0], -1)  # (T, 6)

            if prepend > 0:
                support_rel_pos = np.concatenate(
                    [np.repeat(support_rel_pos[:1], prepend, axis=0), support_rel_pos], axis=0
                )
                support_ori6 = np.concatenate([np.repeat(support_ori6[:1], prepend, axis=0), support_ori6], axis=0)

            self._support_pos_b_traj = support_rel_pos.astype(np.float32)
            self._support_ori_b_traj = support_ori6.astype(np.float32)
            logger.info(
                f"Loaded support table pose from {npz_path} -> support_pos_b(3)+support_ori_b(6) "
                "available in actor obs."
            )

    def _update_feedforward_torque(self, robot_state_data, policy_active: bool) -> None:
        """Rejoue le biais de couple de prise dans ``cmd_tau``.

        Pendant a l'inference de ``JointPositionActionTerm._compute_grip_force_bias``. Valide
        contre la formule d'entrainement sur le clip femto14_box36 : 0.54 % d'ecart median sur le
        couple, 1.42 % au pire, et zero exact sur les 162 frames hors contact.
        """
        if self._grip_force is None:
            super()._update_feedforward_torque(robot_state_data, policy_active)
            return
        if not policy_active or self._last_obj_pos_b is None:
            self.cmd_tau[:] = 0.0
            return

        joint_pos = np.asarray(robot_state_data[0, 7 : 7 + self.num_dofs], dtype=float)
        self.cmd_tau[:] = self._grip_force.compute(
            joint_pos,
            np.asarray(self._last_obj_pos_b, dtype=float).reshape(3),
            self.curr_motion_timestep,
        )

    def _get_ref_body_orientation_in_world(self, robot_state_data):
        # Create configuration for pinocchio robot
        # Note:
        # 1. pinocchio quaternion is in xyzw format, robot_state_data is in wxyz format
        # 2. joint sequences in pinocchio robot and real robot are different

        # free base pos, does not matter
        root_pos = robot_state_data[0, :3]

        # free base ori, wxyz -> xyzw
        root_ori_xyzw = wxyz_to_xyzw(robot_state_data[:, 3:7])[0]

        # dof pos in real robot -> pinocchio robot
        num_dofs = self.num_dofs
        dof_pos_in_real = robot_state_data[0, 7 : 7 + num_dofs]
        dof_pos_in_pinocchio = dof_pos_in_real[self.pinocchio_robot.real2pinocchio_index]

        configuration = np.concatenate([root_pos, root_ori_xyzw, dof_pos_in_pinocchio], axis=0)

        ref_ori_xyzw = self.pinocchio_robot.fk_and_get_ref_body_orientation_in_world(configuration)
        return xyzw_to_wxyz(ref_ori_xyzw)

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        # Extract KP/KD from ONNX metadata (same as base class)
        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            metadata[prop.key] = json.loads(prop.value)

        # Extract URDF text from ONNX metadata
        assert "robot_urdf" in metadata, "Robot urdf text not found in ONNX metadata"
        self.pinocchio_robot = PinocchioRobot(self.config.robot, metadata["robot_urdf"])

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            from pathlib import Path

            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        # get initial command and ref quat xyzw at the configured start timestep
        time_step = np.array([[self.config.task.motion_start_timestep]], dtype=np.float32)

        # Use configured observation dimensions (including history) instead of a hard-coded value.
        actor_obs_template = self.obs_buf_dict.get("actor_obs")
        if actor_obs_template is None:
            raise ValueError("Observation group 'actor_obs' must be configured for WBT policy.")
        obs = actor_obs_template.copy()
        input_feed = {"obs": obs, "time_step": time_step}
        outputs = self.onnx_policy_session.run(["joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)

        # motion_command_t/ref_quat_xyzw_t will be used in get_current_obs_buffer_dict
        self.motion_command_t = np.concatenate(outputs[0:2], axis=1)  # (1, 58)
        self.ref_quat_xyzw_t = outputs[2]
        # duplicate, will be used in _get_init_target and _handle_stop_policy
        self.motion_command_0 = self.motion_command_t.copy()
        self.ref_quat_xyzw_0 = self.ref_quat_xyzw_t.copy()

        def policy_act(input_feed):
            output = self.onnx_policy_session.run(["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"], input_feed)
            action = output[0]
            motion_command = np.concatenate(output[1:3], axis=1)
            ref_quat_xyzw = output[3]
            return action, motion_command, ref_quat_xyzw

        self.policy = policy_act

    def _capture_policy_state(self):
        state = super()._capture_policy_state()
        state.update(
            {
                "motion_command_0": self.motion_command_0.copy(),
                "ref_quat_xyzw_0": self.ref_quat_xyzw_0.copy(),
                "per_joint_policy_action_scale": self.per_joint_policy_action_scale.copy()
                if self.per_joint_policy_action_scale is not None
                else None,
            }
        )
        return state

    def _restore_policy_state(self, state):
        super()._restore_policy_state(state)
        self.motion_command_0 = state["motion_command_0"].copy()
        self.ref_quat_xyzw_0 = state["ref_quat_xyzw_0"].copy()
        saved = state["per_joint_policy_action_scale"]
        self.per_joint_policy_action_scale = saved.copy() if saved is not None else None
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def _on_policy_switched(self, model_path: str):
        super()._on_policy_switched(model_path)
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self._stiff_hold_active = True
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._configure_action_scales()

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to first pose in motion command
            target_dof_pos = self.motion_command_0[:, : self.num_dofs]

            q_target = dof_pos + (target_dof_pos - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # motion_command
        current_obs_buffer_dict["motion_command"] = self.motion_command_t

        # motion_ref_ori_b
        motion_ref_ori = xyzw_to_wxyz(self.ref_quat_xyzw_t)  # wxyz
        motion_ref_ori = self._remove_yaw_offset(motion_ref_ori, self.motion_yaw_offset)

        # robot_ref_ori
        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  #  wxyz
        robot_ref_ori = self._remove_yaw_offset(robot_ref_ori, self.robot_yaw_offset)

        motion_ref_ori_b = matrix_from_quat(subtract_frame_transforms(robot_ref_ori, motion_ref_ori))
        current_obs_buffer_dict["motion_ref_ori_b"] = motion_ref_ori_b[..., :2].reshape(1, -1)

        # base_ang_vel
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]

        # dof_pos
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles

        # dof_vel
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # actions
        current_obs_buffer_dict["actions"] = self.last_policy_action

        # obj_pos_b / obj_ori_b (object-carry policies). Three sources, in priority order:
        #  1. zero_object_obs debug override (OOD for object-actor checkpoints -- see TaskConfig);
        #  2. live_object_obs: the REAL simulated box (run_sim pose stream) relative to the robot's
        #     REAL torso -- same frames as training's obj_pos_b/obj_ori_b terms, closes the loop;
        #  3. clip lookup at the current motion timestep (open loop; also the fallback of 2 until
        #     the first pose message arrives). For real deployment a mocap/RGB-D box pose would
        #     feed source 2's channel.
        if self._obj_pos_b_traj is not None or self._object_pose_sub is not None:
            live_pose = self._object_pose_sub.get_pose() if self._object_pose_sub is not None else None
            if getattr(self.config.task, "zero_object_obs", False):
                current_obs_buffer_dict["obj_pos_b"] = np.zeros((1, 3), dtype=np.float32)
                current_obs_buffer_dict["obj_ori_b"] = np.zeros((1, 6), dtype=np.float32)
            elif live_pose is not None:
                if not self._live_obs_active:
                    self._live_obs_active = True
                    logger.info(colored("Live object obs active: box pose now from run_sim stream", "green"))
                box_pos, box_quat_wxyz, torso_pos, torso_quat_wxyz = live_pose
                torso_quat_wxyz = torso_quat_wxyz[None, :]
                rel_pos = quat_rotate_inverse(torso_quat_wxyz, (box_pos - torso_pos)[None, :])
                rel_quat = subtract_frame_transforms(torso_quat_wxyz, box_quat_wxyz[None, :])
                rel_ori6 = matrix_from_quat(rel_quat)[..., :2].reshape(1, -1)
                current_obs_buffer_dict["obj_pos_b"] = rel_pos.astype(np.float32)
                current_obs_buffer_dict["obj_ori_b"] = rel_ori6.astype(np.float32)
            elif self._obj_pos_b_traj is not None:
                idx = int(np.clip(int(round(self.curr_motion_timestep)), 0, self._obj_pos_b_traj.shape[0] - 1))
                current_obs_buffer_dict["obj_pos_b"] = self._obj_pos_b_traj[idx : idx + 1]
                current_obs_buffer_dict["obj_ori_b"] = self._obj_ori_b_traj[idx : idx + 1]
            else:
                # live mode without clip fallback, before the first message: hold zeros and warn --
                # transient by construction (run_sim publishes every physics step once up).
                logger.warning("live_object_obs: no box pose received yet and no clip fallback; sending zeros")
                current_obs_buffer_dict["obj_pos_b"] = np.zeros((1, 3), dtype=np.float32)
                current_obs_buffer_dict["obj_ori_b"] = np.zeros((1, 6), dtype=np.float32)

        # Le biais de prise consomme la MEME pose de caisse que l'observation, quelle que soit la
        # source retenue ci-dessus (live, clip ou zeros). C'est l'invariant qui garantit que ce que
        # la policy voit et ce que la force fait ne peuvent pas diverger.
        self._last_obj_pos_b = current_obs_buffer_dict.get("obj_pos_b")

        # support_pos_b / support_ori_b (support-table-aware policies only, see
        # wbt_w_object_support obs preset). The table is static in world frame, so unlike the box
        # there's no live/closed-loop source needed -- clip lookup at the current timestep suffices.
        if self._support_pos_b_traj is not None:
            idx = int(np.clip(int(round(self.curr_motion_timestep)), 0, self._support_pos_b_traj.shape[0] - 1))
            current_obs_buffer_dict["support_pos_b"] = self._support_pos_b_traj[idx : idx + 1]
            current_obs_buffer_dict["support_ori_b"] = self._support_ori_b_traj[idx : idx + 1]

        return current_obs_buffer_dict

    def rl_inference(self, robot_state_data):
        # prepare obs, run policy inference
        if not self.motion_clip_progressing:
            # Keep motion index pinned at the configured start while waiting to trigger the clip.
            self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
            self.curr_motion_timestep = self.timestep_util.timestep

        obs = self.prepare_obs_for_rl(robot_state_data)
        if self.config.task.print_observations:
            self._print_observations(obs)

        input_feed = {"time_step": np.array([[self.curr_motion_timestep]], dtype=np.float32), "obs": obs["actor_obs"]}
        policy_action, self.motion_command_t, self.ref_quat_xyzw_t = self.policy(input_feed)

        # clip policy action
        policy_action = np.clip(policy_action, -100, 100)
        # store last policy action
        self.last_policy_action = policy_action.copy()
        # scale policy action
        if self.per_joint_policy_action_scale is None:
            self.scaled_policy_action = policy_action * self.policy_action_scale
        else:
            self.scaled_policy_action = policy_action * self.per_joint_policy_action_scale
        # update motion timestep
        self._set_motion_timestep()

        return self.scaled_policy_action

    def _configure_action_scales(self) -> None:
        """Configure action scales, prioritising ONNX metadata over config fallbacks.

        Resolution order:
        1. ONNX metadata ``action_scale`` (scalar or per-joint list)
        2. ``robot.default_per_joint_action_scale`` when
           ``task.action_scales_by_effort_limit_over_p_gain`` is True
        3. Fall back to the scalar ``task.policy_action_scale``
        """
        raw_metadata = dict(self.onnx_policy_session.get_modelmeta().custom_metadata_map)
        onnx_action_scale = self._parse_action_scale_metadata(raw_metadata.get("action_scale"))

        if onnx_action_scale is not None:
            scales = onnx_action_scale.astype(np.float32, copy=False).reshape(-1)
        elif self.config.task.action_scales_by_effort_limit_over_p_gain:
            fallback = self.config.robot.default_per_joint_action_scale
            if fallback is None:
                raise ValueError(
                    "task.action_scales_by_effort_limit_over_p_gain=True requires ONNX metadata key "
                    "'action_scale' (scalar or per-joint list) or "
                    "robot.default_per_joint_action_scale."
                )
            scales = np.asarray(fallback, dtype=np.float32).reshape(-1)
            logger.warning("ONNX metadata 'action_scale' missing; using robot.default_per_joint_action_scale.")
        else:
            self.per_joint_policy_action_scale = None
            return

        if scales.size == 1:
            scales = np.full(self.num_dofs, scales.item(), dtype=np.float32)
        elif scales.size != self.num_dofs:
            raise ValueError(f"Action scale must contain 1 or {self.num_dofs} values, got {scales.size}.")

        self.per_joint_policy_action_scale = scales.reshape(1, -1)

    @staticmethod
    def _parse_action_scale_metadata(raw_value: str | None) -> np.ndarray | None:
        """Parse action_scale metadata from JSON-serialized or CSV string formats."""
        if raw_value is None:
            return None

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = raw_value

        if isinstance(parsed, (int, float)):
            return np.array([float(parsed)], dtype=np.float32)
        if isinstance(parsed, str):
            values = [float(token.strip()) for token in parsed.split(",") if token.strip()]
            if not values:
                raise ValueError("ONNX metadata action_scale is an empty string.")
            return np.array(values, dtype=np.float32)

        values = np.asarray(parsed, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("ONNX metadata action_scale is empty.")
        return values

    def _get_manual_command(self, robot_state_data):
        # TODO: instead of adding kp/kd_override in def _set_motor_command,
        # just use the motor_kp/motor_kd when calling it in _fill_motor_commands
        if not self._stiff_hold_active:
            return None
        return {
            "q": self._stiff_hold_q.copy(),
            "kp": self._stiff_hold_kp,
            "kd": self._stiff_hold_kd,
        }

    def _handle_start_policy(self):
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._capture_robot_yaw_offset()
        self._capture_motion_yaw_offset(self.ref_quat_xyzw_0)

    def _set_motion_timestep(self):
        if self.motion_clip_progressing:
            prev = self.curr_motion_timestep

            if self.use_sim_time:
                self.curr_motion_timestep = self.timestep_util.get_timestep(log=self.logger)
            else:
                self.curr_motion_timestep += 1

            if self.curr_motion_timestep != prev:
                self.logger.info(f"Motion timestep: {prev} → {self.curr_motion_timestep}")  # noqa: G004

            # Stop motion clip at configured end timestep (keep policy running at final pose)
            if (end := self.config.task.motion_end_timestep) and self.curr_motion_timestep >= end:
                self.logger.info(colored(f"Reached end timestep {end}, stopping motion clip", "yellow"))
                self.motion_clip_progressing = False
                self.curr_motion_timestep = end

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self.logger.info("Actions set to stiff startup command")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

        self.motion_clip_progressing = False
        self.timestep_util.reset(start_timestep=0)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.motion_command_t = self.motion_command_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0

    def _handle_start_motion_clip(self):
        """Handle start motion clip action."""
        self.timestep_util.reset(start_timestep=self.config.task.motion_start_timestep)
        self.curr_motion_timestep = self.timestep_util.timestep
        self.motion_clip_progressing = True

        if self.config.task.motion_start_timestep > 0 or self.config.task.motion_end_timestep is not None:
            start_str = str(self.config.task.motion_start_timestep)
            end_str = str(self.config.task.motion_end_timestep) if self.config.task.motion_end_timestep else "end"
            self.logger.info(colored(f"Starting motion clip from timestep {start_str} to {end_str}", "blue"))
        else:
            self.logger.info(colored("Starting motion clip", "blue"))

    def _dispatch_command(self, cmd):
        from holosoma_inference.inputs.api.commands import StateCommand

        if cmd == StateCommand.START_MOTION_CLIP:
            self._handle_start_motion_clip()
        else:
            super()._dispatch_command(cmd)

    def _capture_robot_yaw_offset(self):
        """Capture robot yaw when policy starts to use as reference offset."""
        robot_state_data = self.interface.get_low_state()
        if robot_state_data is None:
            self.robot_yaw_offset = 0.0
            self.logger.warning("Unable to capture robot yaw offset - missing robot state.")
            return

        robot_ref_ori = self._get_ref_body_orientation_in_world(robot_state_data)  # wxyz
        yaw = self._quat_yaw(robot_ref_ori)
        self.robot_yaw_offset = yaw
        self.logger.info(colored(f"Robot yaw offset captured at {np.degrees(yaw):.1f} deg", "blue"))

    def _capture_motion_yaw_offset(self, ref_quat_xyzw_0: np.ndarray) -> float:
        """Capture motion yaw when policy starts to use as reference offset."""
        self.motion_yaw_offset = self._quat_yaw(xyzw_to_wxyz(ref_quat_xyzw_0))
        self.logger.info(colored(f"Motion yaw offset captured at {np.degrees(self.motion_yaw_offset):.1f} deg", "blue"))

    def _remove_yaw_offset(self, quat_wxyz: np.ndarray, yaw_offset: float) -> np.ndarray:
        """Remove stored yaw offset from robot orientation quaternion."""
        if abs(yaw_offset) < 1e-6:
            return quat_wxyz
        yaw_quat = rpy_to_quat((0.0, 0.0, -yaw_offset)).reshape(1, 4)
        yaw_quat = np.broadcast_to(yaw_quat, quat_wxyz.shape)
        return quat_mul(yaw_quat, quat_wxyz)

    @staticmethod
    def _quat_yaw(quat_wxyz: np.ndarray) -> float:
        """Extract yaw angle from quaternion array of shape (1, 4)."""
        quat_flat = quat_wxyz.reshape(-1, 4)[0]
        _, _, yaw = quat_to_rpy(quat_flat)
        return float(yaw)
