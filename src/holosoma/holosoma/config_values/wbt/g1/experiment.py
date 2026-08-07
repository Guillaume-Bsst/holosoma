from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values.wbt.g1.action import (
    g1_29dof_joint_pos_grip_force,
    g1_29dof_joint_pos_grip_force_dyn,
)
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

g1_29dof_wbt = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.ppo,
        config=replace(
            algo.ppo.config,
            num_learning_iterations=30000,
            num_learning_epochs=5,
            save_interval=4000,
            entropy_coef=0.005,
            init_noise_std=1.0,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            init_at_random_ep_len=True,
            empirical_normalization=True,
            use_symmetry=False,
            actor_optimizer=replace(algo.ppo.config.actor_optimizer, weight_decay=0.000),
            critic_optimizer=replace(algo.ppo.config.critic_optimizer, weight_decay=0.000),
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_reward,
    nightly=NightlyConfig(
        iterations=8000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.3, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.4, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.85, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.7, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.60, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.45, "inf"],
        },
    ),
)

g1_29dof_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="WholeBodyTracking",
        name="g1_29dof_wbt_fast_sac_manager",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            # v_max 20 -> 30 : le critic distributionnel a un SUPPORT BORNE, les atomes du haut
            # saturent si le retour depasse v_max. Somme des poids positifs = 11.5 (corps 5.0 +
            # objet 6.5) -> 0.23/step a dt=0.02 -> retour actualise max 22.9 a gamma=0.99 sur des
            # episodes de 10 s. C'est le max theorique (tous les termes a 1.0 en meme temps, ce qui
            # n'arrive pas), mais la marge etait deja nulle avant (20.9) et on perdrait la
            # resolution du critic exactement dans le regime haute performance. 501 atomes sur
            # +-30 laissent 0.12 de resolution.
            v_max=30.0,
            v_min=-20.0,
            gamma=0.99,  # For motion tracking, high gamma + high num_steps is better
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(
                simulator.isaacsim.config.sim,
                max_episode_length_s=10.0,
            ),
        ),
    ),
    robot=replace(
        robot.g1_29dof,
        control=replace(
            robot.g1_29dof.control,
            action_scale=0.25,
            action_scales_by_effort_limit_over_p_gain=True,
        ),
        asset=replace(robot.g1_29dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_29dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    terrain=terrain.terrain_locomotion_plane,
    observation=observation.g1_29dof_wbt_observation,
    action=action.g1_29dof_joint_pos,
    termination=termination.g1_29dof_wbt_termination,
    randomization=randomization.g1_29dof_wbt_randomization,
    command=command.g1_29dof_wbt_command,
    curriculum=curriculum.g1_29dof_wbt_curriculum,
    reward=reward.g1_29dof_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.40, "inf"],
            "Episode/rew_motion_global_ref_orientation_error_exp": [0.25, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [1.1, "inf"],
            "Episode/rew_motion_relative_body_orientation_error_exp": [0.35, "inf"],
            "Episode/rew_motion_global_body_lin_vel": [0.45, "inf"],
            "Episode/rew_motion_global_body_ang_vel": [0.15, "inf"],
        },
    ),
)

g1_29dof_wbt_w_object = replace(
    g1_29dof_wbt,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(
            robot.g1_29dof_w_object.asset,
            enable_self_collisions=True,
        ),
        # box36 (cube 0.36 m, demi-tailles 0.18) et NON largebox.obj. Les rewards de contact
        # calculent leur SDF contre grasp_settle.box_half_extents = (0.18, 0.18, 0.18) : le mesh
        # spawne doit correspondre. largebox.obj mesure en realite 0.471 x 0.459 x 0.408
        # (demi-tailles 0.236 / 0.229 / 0.204), soit ~5 cm de plus par cote. Avec ce mesh, une
        # prise parfaitement plate lit une distance signee de ~+0.05 m au lieu de 0, donc
        # object_flat_contact_quality_exp plafonne a exp(-0.05^2/0.03^2) = 0.06 -- le terme est
        # mort sans aucune faute du robot, et object_surface_contact_error_exp prend le meme biais
        # sur sa composante profondeur. Les deux URDF ont la meme masse (0.811 kg), le changement
        # est purement geometrique.
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_29dof_wbt_fast_sac_w_object = replace(
    g1_29dof_wbt_fast_sac,
    command=command.g1_29dof_wbt_command_w_object,
    robot=replace(
        robot.g1_29dof_w_object,
        asset=replace(robot.g1_29dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_29dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_29dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_27dof_wbt = replace(
    g1_29dof_wbt,
    training=replace(g1_29dof_wbt.training, name="g1_27dof_wbt_manager"),
    robot=replace(
        robot.g1_27dof,
        control=replace(robot.g1_27dof.control, action_scales_by_effort_limit_over_p_gain=True, action_scale=0.25),
        asset=replace(robot.g1_27dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_27dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    command=command.g1_27dof_wbt_command,
    termination=termination.g1_27dof_wbt_termination,
)

g1_27dof_wbt_w_object = replace(
    g1_27dof_wbt,
    command=command.g1_27dof_wbt_command_w_object,
    robot=replace(
        robot.g1_27dof_w_object,
        asset=replace(robot.g1_27dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_27dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_27dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

g1_27dof_wbt_fast_sac = replace(
    g1_29dof_wbt_fast_sac,
    training=replace(g1_29dof_wbt_fast_sac.training, name="g1_27dof_wbt_fast_sac_manager"),
    robot=replace(
        robot.g1_27dof,
        control=replace(robot.g1_27dof.control, action_scales_by_effort_limit_over_p_gain=True, action_scale=0.25),
        asset=replace(robot.g1_27dof.asset, enable_self_collisions=True),
        init_state=replace(robot.g1_27dof.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    command=command.g1_27dof_wbt_command,
    termination=termination.g1_27dof_wbt_termination,
)

g1_29dof_wbt_w_object_actor = replace(
    g1_29dof_wbt_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object_actor,
)

# Portage caisse + table avec FORCE DE PRISE REELLE (60 N par main) au lieu du curriculum de
# physicalite. La caisse est physique des le step 0 ; chaque main la presse a
# grip_force.target_force_n des que le flag de contact GT du command term est actif (cf.
# GripForceCfg et JointPositionActionTerm._configure_grip_force).
#
# A NE PAS COMBINER avec kinematic-object-during-contact / physicality-curriculum : les deux
# mecanismes repondent au meme probleme (tenir la caisse) et se neutralisent -- une caisse pilotee
# cinematiquement n'a rien a se faire serrer. Les defauts de GraspSettleConfig laissent bien les
# deux a False, donc ce preset se lance SANS aucun flag de curriculum.
#
# L'observation est la variante ACTEUR : la policy voit la pose de la caisse ET de la table
# (obj_pos_b/obj_ori_b + support_pos_b/support_ori_b), en history_length=1 -- impose par le warm
# start depuis le checkpoint full-loco 3ivghz1e (voir la note en tete de observation.py).
g1_29dof_wbt_w_object_actor_grip_force = replace(
    g1_29dof_wbt_w_object_actor,
    training=replace(
        g1_29dof_wbt_w_object_actor.training,
        name="g1_29dof_wbt_w_object_actor_grip_force",
    ),
    action=g1_29dof_joint_pos_grip_force,
    # Clip CABLE lui aussi : le defaut herite est sub3_largebox_003_mj_w_obj.npz, qui n'a ni table
    # ni contact GT. Celui-ci porte object_* ET support_* complets (support_ref_contact,
    # support_half_extents), donc support_surface_contact_error_exp est reellement actif.
    command=command.g1_29dof_wbt_command_w_object_femto14_box36,
    robot=replace(
        g1_29dof_wbt_w_object_actor.robot,
        object=replace(
            g1_29dof_wbt_w_object_actor.robot.object,
            # La table est CABLEE ici, pas laissee a un flag CLI. Sans support_urdf_path,
            # isaacsim.py:453 ne spawne aucun acteur "support" : le clip porte bien la pose de la
            # table, la policy la voit dans son obs, le reward support_surface_contact la note --
            # mais elle n'existe pas physiquement et la caisse traverserait au depot.
            support_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/support_table.urdf",
        ),
    ),
)

# Variante ETAGE 05 du preset ci-dessus : meme tache, meme robot, meme observation, meme warm start
# -- seule la source d'information change. Le clip est la version enrichie (dyn_* fusionnes depuis
# le run SPIDER femto14_box36_halfsphere_torquecap, couples dans les limites de l'URDF), et trois
# choses s'en servent :
#   1. reward : feet_contact_schedule + feet_slip_on_ref_stance + torque_envelope_penalty, et les
#      deux rewards de contact main<->caisse deviennent bimanuels (cf. reward.py).
#   2. action : la force de prise suit le profil mesure par main au lieu du 60 N constant.
#   3. action : le couple de reference est injecte a 50 % en feed-forward dans la loi PD.
#
# L'espace d'observation et le nombre de parametres sont INCHANGES, donc le warm start depuis
# 3ivghz1e reste utilisable tel quel.
#
# Point de vigilance deploiement : (3) suppose que holosoma_inference sache rejouer le meme couple
# de feed-forward, indexe sur la phase du clip (le low-level Unitree a bien un champ tau). Une
# policy entrainee avec et deployee sans commandera un couple faux. Tant que ce plumbing n'est pas
# fait, ce preset est un preset d'ENTRAINEMENT/etude, pas un candidat au deploiement.
g1_29dof_wbt_w_object_actor_grip_force_dyn = replace(
    g1_29dof_wbt_w_object_actor_grip_force,
    training=replace(
        g1_29dof_wbt_w_object_actor_grip_force.training,
        name="g1_29dof_wbt_w_object_actor_grip_force_dyn",
    ),
    command=command.g1_29dof_wbt_command_w_object_femto14_box36_dyn,
    action=g1_29dof_joint_pos_grip_force_dyn,
    reward=reward.g1_29dof_wbt_reward_w_object_dyn,
    # Seuil objet 0.15 -> 0.45 m : mesure a l'appui dans termination.py. A 350 iterations, 2 morts
    # sur 3 sont "la caisse a bouge", et l'episode meurt au lieu de laisser la policy rattraper.
    termination=termination.g1_29dof_wbt_termination_dyn,
    # Critique enrichi (+17 dims), ACTEUR INCHANGE. Le critique n'etant jamais deploye, tout ce qui
    # reduit la variance de son estimation de valeur est gratuit cote robot. Detail dans
    # observation.py -- le gain principal est motion_phase : avec le RSI, le retour atteignable
    # depend du temps restant dans le clip, que rien ne disait au critique.
    observation=observation.g1_29dof_wbt_observation_w_object_actor_dyn,
)

g1_27dof_wbt_w_object_actor = replace(
    g1_27dof_wbt_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object_actor,
)

g1_27dof_wbt_fast_sac_w_object = replace(
    g1_27dof_wbt_fast_sac,
    command=command.g1_27dof_wbt_command_w_object,
    robot=replace(
        robot.g1_27dof_w_object,
        asset=replace(robot.g1_27dof_w_object.asset, enable_self_collisions=True),
        object=replace(
            robot.g1_27dof_w_object.object,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf",
        ),
        init_state=replace(robot.g1_27dof_w_object.init_state, pos=[0.0, 0.0, 0.76]),
    ),
    randomization=randomization.g1_29dof_wbt_randomization_w_object,
    observation=observation.g1_29dof_wbt_observation_w_object,
    reward=reward.g1_29dof_wbt_reward_w_object,
    simulator=replace(
        simulator.isaacsim,
        config=replace(simulator.isaacsim.config, scene=replace(simulator.isaacsim.config.scene, env_spacing=0.0)),
    ),
)

__all__ = [
    "g1_29dof_wbt",
    "g1_29dof_wbt_fast_sac",
    "g1_29dof_wbt_fast_sac_w_object",
    "g1_29dof_wbt_w_object",
    "g1_29dof_wbt_w_object_actor",
    "g1_29dof_wbt_w_object_actor_grip_force",
    "g1_29dof_wbt_w_object_actor_grip_force_dyn",
    "g1_27dof_wbt",
    "g1_27dof_wbt_w_object",
    "g1_27dof_wbt_w_object_actor",
    "g1_27dof_wbt_fast_sac",
    "g1_27dof_wbt_fast_sac_w_object",
]

"""
Example 1: Robot only:
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt

Example 2: Robot+Object:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object

Example 3: Robot+Terrain:
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt \
  terrain:terrain-load-obj \
  --terrain.terrain-term.obj-file-path="holosoma/data/motions/g1_29dof/whole_body_tracking/terrain_slope.obj" \
  --command.setup_terms.motion_command.params.motion_config.motion_file\
="holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz" \
  --simulator.config.scene.env_spacing=0.0
"""
