#!/usr/bin/env bash
# PHASE 1 du bootstrap en deux temps : apprendre le mouvement femto14 avec la caisse et la table
# PRESENTES mais sans effet, sur ~10 000 iterations. Phase 2 = run_train_gripforce_60n.sh reprenant
# le checkpoint produit ici.
#
# "Invisible" = ne perturbe pas, PAS "absent" :
#   - kinematic-object-during-contact True : la caisse est collee a la trajectoire de reference sur
#     les frames de contact. Elle ne tombe pas, ne derive pas, n'impose aucune charge.
#   - grip-force.enable False : rien a serrer tant que la caisse est cinematique, et surtout plus
#     de 60 N injectes dans les poignets pendant que la policy apprend a marcher.
#   - Les COLLISIONS restent actives : desactiver la table apprendrait au robot a la traverser,
#     ce qui casserait en phase 2 au moment du depot.
#
# Pourquoi garder la caisse et la table dans la scene plutot que de les retirer :
#   1. l'observation garde ses 172 dims, donc la phase 2 recharge ce checkpoint DIRECTEMENT, sans
#      la chirurgie de colonnes qu'un passage 154 -> 172 imposerait ;
#   2. le clip declare has_object des qu'il porte object_pos_w, et MotionCommand fait alors
#      get_actor_indices("object") meme si rien n'est spawne -> le registre leve.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

source modules/third_party/holosoma_custom/scripts/source_isaacsim_setup.sh

CKPT=/home/ecarn/Documents/wbt_rl/logs/warmstart/model_29999_box_table.pt
GS=--command.setup-terms.motion-command.params.motion-config.grasp-settle

if [ ! -f "${CKPT}" ]; then
    echo "ERREUR : checkpoint adapte introuvable : ${CKPT}" >&2
    exit 1
fi

# reward:g1-29dof-wbt = reward CORPS SEUL, aucun terme objet.
# Sans ca, les trois rewards de placement de main (grasp_relative, surface_contact,
# flat_contact_quality) donnent a la policy un objectif CONCURRENT du suivi du corps : elle
# optimise ou poser ses mains sur la caisse pendant qu'elle apprend a marcher, ce qui degrade la
# demarche -- exactement le risque a eviter en phase 1.
# Le placement des mains reste appris, mais via motion_relative_body_position_error_exp, qui suit
# deja les poignets parmi les 14 corps trackes : c'est une composante du mouvement, pas une tache
# separee. Les rewards objet reviennent en phase 2.
python modules/02_training/holosoma_custom/train_agent.py exp:g1-29dof-wbt-w-object-actor-grip-force \
    logger:wandb \
    reward:g1-29dof-wbt \
    --training.name g1_29dof_wbt_femto14_phase1_kinematic \
    --training.num-envs 4096 \
    --training.checkpoint "${CKPT}" \
    --algo.config.load-optimizer False \
    --algo.config.num-learning-iterations 10000 \
    --action.terms.joint-control.params.grip-force.enable False \
    ${GS}.kinematic-object-during-contact True \
    "$@"
