"""Whole Body Tracking termination presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ],
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                # object tracking
                # only triggered when has_object=True
                # Resserre 0.25 -> 0.15 : le succes de tracking est a 0.10 m, donc 0.25 laissait 15 cm
                # de bande morte ou la boite flotte sans tuer l'episode et sans pression a se recaler
                # sous 10 cm. 0.15 supprime l'essentiel de cette bande (marge 5 cm sur le rayon de
                # succes pour ne pas tuer sur du bruit / transitoires de contact).
                "bad_object_pos_threshold": 0.15,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

g1_27dof_wbt_termination = g1_29dof_wbt_termination

# Variante ETAGE 05 : seuil objet desserre 0.15 -> 0.45 m.
#
# MESURE qui motive le changement (run diagnostic, 4096 envs, init aleatoire, ventilation par cause
# de bad_tracking). La cause de mort BASCULE a mesure que la policy apprend :
#
#   iter              1     60    120    180    240    300    350
#   motion_body_pos  0.87   0.93   0.81   0.70   0.50   0.38   0.29   (se resorbe)
#   object_pos       0.08   0.12   0.18   0.26   0.45   0.54   0.64   (devient dominant)
#   ref_pos          0.00   0.00   0.01   0.02   0.01   0.00   0.03   (jamais un facteur)
#
# ref_pos a ~2% du debut a la fin : le robot ne tombe pas, il n'y a pas de probleme de locomotion.
# A 350 iterations, 2 morts sur 3 sont "la caisse a bouge de plus de 15 cm", et la courbe monte
# encore -- la longueur d'episode plafonne a 56-72 pas depuis l'iteration 180. Le seuil objet est
# devenu le facteur limitant, et il tue l'episode a chaque glissement au lieu de laisser la policy
# RATTRAPER : elle n'apprend jamais la recuperation, seulement a ne pas declencher le seuil.
#
# CE QU'ON ABANDONNE. Le resserrage 0.25 -> 0.15 visait une bande morte reelle (voir le commentaire
# ci-dessus) : entre le rayon de succes 0.10 et le seuil, la boite flotte sans pression a se
# recaler. A 0.45 cette bande devient large -- plus large que la caisse elle-meme (cube de 0.36 m),
# donc un episode survit a une caisse posee a cote de sa reference. Le pari est que la pression de
# recalage ne vient PAS de la termination mais des rewards objet, qui sont denses et multi-echelle
# (object_global_ref_position_error_exp sigma=0.3 + le compagnon fine sigma=0.12) et tirent vers 0
# en continu, que l'episode soit menace ou non.
#
# A SURVEILLER : si object_pos redescend au profit de motion_body_pos, le seuil etait bien le
# facteur limitant. Si la longueur d'episode monte SANS que le reward objet suive, c'est le
# scenario inverse -- la policy laisse tomber la caisse et encaisse des episodes longs et vides.
g1_29dof_wbt_termination_dyn = TerminationManagerCfg(
    terms={
        **g1_29dof_wbt_termination.terms,
        "bad_tracking": replace(
            g1_29dof_wbt_termination.terms["bad_tracking"],
            params={
                **g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "bad_object_pos_threshold": 0.45,
            },
        ),
    }
)

__all__ = ["g1_29dof_wbt_termination", "g1_27dof_wbt_termination", "g1_29dof_wbt_termination_dyn"]
