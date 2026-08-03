#!/usr/bin/env bash
# Entrainement portage caisse 36 cm + table, avec FORCE DE PRISE REELLE de 60 N par main,
# en WARM START depuis la policy full-locomotion 3ivghz1e.
#
# Branche : feat/box-table-gripforce-60n
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # racine du repo wbt_rl

source modules/third_party/holosoma_custom/scripts/source_isaacsim_setup.sh

# PHASE 2 : reprend automatiquement le DERNIER checkpoint de la phase 1
# (run_train_phase1_kinematic.sh). Les deux phases partagent exactement la meme observation
# (172 dims acteur / 307 critic) parce que la phase 1 garde la caisse et la table dans la scene :
# le checkpoint se recharge donc DIRECTEMENT, sans la chirurgie de colonnes qu'un 154 -> 172
# imposerait, et load-optimizer peut rester a son defaut True.
#
# Surcharge possible :  ./run_train_gripforce_60n.sh --training.checkpoint /chemin/model_XXXX.pt
CKPT="${PHASE1_CKPT:-}"
if [ -z "${CKPT}" ]; then
    P1_DIR=$(ls -dt logs/WholeBodyTracking/*phase1_kinematic*/ 2>/dev/null | head -1)
    if [ -n "${P1_DIR}" ]; then
        CKPT=$(ls -1 "${P1_DIR}"model_*.pt 2>/dev/null | sort -V | tail -1)
    fi
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
    echo "ERREUR : aucun checkpoint de phase 1 trouve." >&2
    echo "Lancer d'abord run_train_phase1_kinematic.sh, ou pointer explicitement :" >&2
    echo "  PHASE1_CKPT=/chemin/model_XXXX.pt $0" >&2
    echo "Runs de phase 1 presents :" >&2
    ls -dt logs/WholeBodyTracking/*phase1_kinematic*/ 2>/dev/null >&2 || echo "  (aucun)" >&2
    exit 1
fi
echo "Phase 2 : reprise depuis ${CKPT}"

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
# load-optimizer reste a son defaut True : le checkpoint de phase 1 a exactement les memes formes
# (172/307), son etat d'optimiseur est donc rechargeable. C'etait different avec le checkpoint
# corps-seul patche par expand_checkpoint_for_object.py, qui n'en avait plus.
python modules/02_training/holosoma_custom/train_agent.py exp:g1-29dof-wbt-w-object-actor-grip-force \
    logger:wandb \
    --training.num-envs 4096 \
    --training.checkpoint "${CKPT}" \
    "$@"
