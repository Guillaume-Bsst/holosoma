#!/usr/bin/env bash
# Training WBT loco-manip : box36 portee + table comme VRAI objet statique (pas de table cuite
# dans le terrain -- le terrain reste terrain_locomotion_plane, sol plat nu, et la table est un
# RigidObject spawne par --robot.object.support-urdf-path, cf. isaacsim.py:489).
#
# Branche : test/reward-tuning-h3
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # repo root wbt_rl

source modules/third_party/holosoma_custom/scripts/source_isaacsim_setup.sh

MOTION=holosoma/data/motions/g1_29dof/whole_body_tracking/femto14_box36_gaitfix2_w_obj_gtcontact_nobj.npz
GS=--command.setup-terms.motion-command.params.motion-config.grasp-settle

# exp:g1-29dof-wbt-w-object-actor = le seul preset ou l'ACTEUR voit la box et la table
# (observation.g1_29dof_wbt_observation_w_object_actor). Il derive de g1_29dof_wbt -> PPO.
#
# Le clip : seul femto14_box36_gaitfix2_w_obj_gtcontact_nobj.npz porte les champs de contact
# table complets (support_ref_contact + support_half_extents). Les autres *_gtcontact ont
# support_pos_w mais pas le contact, donc support_surface_contact_error_exp resterait a 0.
#
# Curriculum de physicalite : il faut les TROIS flags. _update_physicality_curriculum sort en
# early-return sur `if not (physicality_curriculum and kinematic_object_during_contact)`, et
# physicality_curriculum vaut False par defaut -- l'oublier cloue alpha a 1.0 pour tout le run
# (box forcee sur la reference en permanence, la policy n'apprend JAMAIS a porter, et les deux
# EMA du curriculum restent a exactement 0.0 : c'est la signature a surveiller dans les courbes).
#   physicality-curriculum          : active la boucle qui fait descendre alpha
#   kinematic-object-during-contact : l'override cinematique que alpha module
#   physicality-force-mode          : wrench PD borne au lieu du blend d'etat (seule branche ou
#                                     le fix kd "damp-to-zero" agit)
# alpha part a 1 (box cinematique, aucune charge) et descend au succes -> la policy reprend
# progressivement le poids.
#
# physicality-success-threshold 0.70 (defaut 0.90) : mesure sur ce clip, la survie monte vite a
# ~0.65 vers l'iteration 1200 puis PLAFONNE (bande 0.59-0.68 sur les ~1900 iterations suivantes,
# max jamais atteint 0.757). A 0.90 le gate ne s'ouvre jamais -> alpha reste cloue a 1.0, la box
# reste cinematique tout le run et la policy n'apprend jamais a PORTER. 0.70 est juste au-dessus
# du plateau, donc le ladder demarre. Pas de risque de cascade : apres chaque descente les deux
# EMA sont remises a zero et doivent etre re-gagnees a la nouvelle physicalite (+ cooldown 2000).
python modules/02_training/holosoma_custom/train_agent.py exp:g1-29dof-wbt-w-object-actor \
    --training.num-envs 4096 \
    --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION}" \
    ${GS}.physicality-curriculum True \
    ${GS}.kinematic-object-during-contact True \
    ${GS}.physicality-force-mode True \
    ${GS}.physicality-success-threshold 0.70 \
    --robot.object.object-urdf-path holosoma/data/motions/g1_29dof/whole_body_tracking/objects_box36.urdf \
    --robot.object.support-urdf-path holosoma/data/motions/g1_29dof/whole_body_tracking/support_table.urdf \
    "$@"
