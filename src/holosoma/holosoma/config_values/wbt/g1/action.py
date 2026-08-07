"""Whole Body Tracking action presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.action import GripForceCfg, TorqueFeedforwardCfg, TorqueReferenceNoiseCfg
from holosoma.config_values.loco.g1.action import g1_29dof_joint_pos

# Object-carry training with real physical grip force instead of the box-physicality curriculum
# (see config_values/wbt/g1/command.py): the box is fully physical from step 0, and each hand
# presses it at grip_force.target_force_n whenever the command term's GT contact flag is active.
#
# 60 N (not 30): with object mass randomized abs [0.8, 1.2] kg and independently-randomized
# box/hand friction (box static/dynamic in [0.1, 0.6], hand in [0.3, 1.6]/[0.3, 1.2]), the worst-case
# combine="average" effective mu is ~0.20 -- 2 * 0.20 * 30N = 12.0N vs a max box weight of
# 1.2*9.81 = 11.77N is only a ~2% margin. 60N/hand gives 2 * 0.20 * 60N = 24N, a ~2x margin even in
# that worst-case draw (still within the wrist torque limits: ~3 N.m at 60N given the ~0.05m lever
# arm, under the 5 N.m pitch/yaw and 25 N.m roll effort limits).
grip_force_cfg = GripForceCfg(
    enable=True,
    target_force_n=60.0,
    force_command_max_n=90.0,
    # Recale sur la main HALF-SPHERE : origine de son hand_palm_joint dans le repere poignet.
    # Le defaut (0.0415, 0.003, 0) est celui de la main RUBBER -- c'est la valeur qu'utilisait la
    # run de reference c4k7xrin, qui tournait pourtant en half-sphere : elle appliquait donc sa
    # force 1.25 cm devant la paume reelle, soit ~25% d'erreur sur le bras de levier du couple
    # poignet. Corrige ici.
    hand_offset_local=(0.029, -0.003, 0.0),
)

g1_29dof_joint_pos_grip_force = replace(
    g1_29dof_joint_pos,
    terms={
        **g1_29dof_joint_pos.terms,
        "joint_control": replace(
            g1_29dof_joint_pos.terms["joint_control"],
            params={"grip_force": grip_force_cfg},
        ),
    },
)

# Variante etage-05 : la prise suit le PROFIL de force mesure par le solve physique (par main, par
# frame) au lieu du 60 N constant, et le couple de reference est injecte en feed-forward dans la loi
# PD (cf. TorqueFeedforwardCfg).
#
# Les deux retombent silencieusement sur le comportement ci-dessus si le clip charge ne porte pas
# les champs dyn_* (has_dyn_grip / has_dyn_tau) -- le feed-forward log un warning en se desactivant.
grip_force_cfg_profile = replace(grip_force_cfg, use_reference_profile=True)

# scale=0.5 : la moitie du couple de reference suffit a retirer l'essentiel de la charge de
# compensation de gravite tout en laissant a la boucle PD l'autorite de contredire le
# feed-forward -- le couple de reference est exact pour l'ETAT de reference, et la policy n'y est
# jamais exactement.
#
# Poignets EXCLUS (cf. TorqueFeedforwardCfg.exclude_joint_names) : sur ce clip, wrist_pitch et
# wrist_yaw sont colles a leur limite de 5 N.m sur 41-50 % des frames. La valeur enregistree y est
# donc la butee du cap de couple, pas la demande du mouvement -- la feed-forwarder reviendrait a
# commander un biais permanent a mi-butee que la policy devrait combattre. Les autres DOF saturent
# 0.9-16 % du temps, ce qui reste un signal exploitable.
torque_ff_cfg = TorqueFeedforwardCfg(
    enable=True,
    scale=0.5,
    exclude_joint_names=("wrist_pitch", "wrist_yaw"),
)

# Bruit de couple proportionnel au couple DEMANDE plutot qu'a la butee de l'actionneur (cf.
# TorqueReferenceNoiseCfg). Le RFI existant (actuator_randomizer_state) est desactive
# (enable_rfi_lim: false) et perturbe de toute facon une fraction fixe de la LIMITE de chaque joint,
# identique a chaque frame -- ce qui donne un bruit negligeable sous charge et dominant a vide.
#
# ref_scale=0.15 : +/-15 % du couple demande. floor_scale=0.01 : plancher a 1 % de la butee, pour
# que l'exploration ne disparaisse pas la ou tau_ref ~ 0 (jambe en vol), ou un bruit purement
# proportionnel s'annulerait exactement la ou la policy a le plus de latitude.
#
# Poignets exclus pour la meme raison que le feed-forward, PLUS une propre au bruit : colles a leur
# butee ~50 % du clip, le clip_torques final rogne toute excursion positive, donc le "bruit" y
# devient un biais unilateral vers le bas -- pire que pas de randomisation du tout.
torque_noise_cfg = TorqueReferenceNoiseCfg(
    enable=True,
    ref_scale=0.15,
    floor_scale=0.01,
    exclude_joint_names=("wrist_pitch", "wrist_yaw"),
)

g1_29dof_joint_pos_grip_force_dyn = replace(
    g1_29dof_joint_pos,
    terms={
        **g1_29dof_joint_pos.terms,
        "joint_control": replace(
            g1_29dof_joint_pos.terms["joint_control"],
            params={
                "grip_force": grip_force_cfg_profile,
                "torque_feedforward": torque_ff_cfg,
                "torque_reference_noise": torque_noise_cfg,
            },
        ),
    },
)

__all__ = ["g1_29dof_joint_pos_grip_force", "g1_29dof_joint_pos_grip_force_dyn"]
