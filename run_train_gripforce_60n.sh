#!/usr/bin/env bash
# Entrainement portage caisse 36 cm + table, avec FORCE DE PRISE REELLE de 60 N par main,
# en WARM START depuis la policy full-locomotion 3ivghz1e.
#
# Branche : feat/box-table-gripforce-60n
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # racine du repo wbt_rl

source modules/third_party/holosoma_custom/scripts/source_isaacsim_setup.sh

CKPT=/home/ecarn/Documents/wbt_rl/logs/warmstart/model_29999_box_table.pt

if [ ! -f "${CKPT}" ]; then
    echo "ERREUR : checkpoint adapte introuvable : ${CKPT}" >&2
    echo "Le regenerer avec :" >&2
    echo "  python modules/third_party/holosoma_custom/scripts/expand_checkpoint_for_object.py \\" >&2
    echo "      --src <.../model_29999.pt> --dst ${CKPT}" >&2
    exit 1
fi

# Tout est CABLE DANS LE PRESET exp:g1-29dof-wbt-w-object-actor-grip-force :
#   - obs ACTEUR (la policy voit caisse + table), history_length=1
#   - caisse objects_box36.urdf + table support_table.urdf
#   - clip femto14_box36_w_obj_gtcontact_nobj.npz (object_* et support_* complets)
#   - action = grip force 60 N/main sur contact GT
#   - PAS de curriculum de physicalite (kinematic/curriculum a False par defaut) : la caisse est
#     physique des le step 0, c'est la prise qui la tient. Ne PAS ajouter les flags physicality-*,
#     les deux mecanismes se neutralisent.
# Rien n'est laisse a un flag CLI oubliable -- c'est ce qui avait coute 3h de run la fois d'avant.
#
# --algo.config.load-optimizer False : obligatoire. expand_checkpoint_for_object.py retire l'etat
# d'optimiseur (ses formes sont celles des anciens tenseurs 154/286), le defaut True planterait.
python modules/02_training/holosoma_custom/train_agent.py exp:g1-29dof-wbt-w-object-actor-grip-force \
    logger:wandb \
    --training.num-envs 4096 \
    --training.checkpoint "${CKPT}" \
    --algo.config.load-optimizer False \
    "$@"
