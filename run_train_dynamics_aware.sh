#!/usr/bin/env bash
# Entrainement portage caisse 36 cm + table, ENRICHI PAR LE RETARGETING DYNAMIQUE (etage 05).
#
# Identique a run_train_gripforce_60n.sh (meme robot, meme observation, meme warm start, meme
# espace d'action) sauf que le clip porte les champs dyn_* produits en rejouant le retargeting
# dans un vrai solveur de contact MuJoCo (run SPIDER femto14_box36_halfsphere_torquecap, couples
# plafonnes aux limites d'effort de l'URDF), et que trois choses s'en servent :
#
#   1. reward  : feet_contact_schedule (quel pied porte a quel instant) + feet_slip_on_ref_stance
#                (glissement penalise seulement en appui) + torque_envelope_penalty (enveloppe
#                unilaterale contre le couple que le mouvement demande vraiment).
#   2. reward  : object_grasp_relative et object_flat_contact_quality deviennent BIMANUELS -- la
#                reference presse les deux mains sur ~35 % des frames, ce qu'une ancre unique ne
#                pouvait pas exprimer.
#   3. action  : force de prise = profil mesure par main et par frame (au lieu de 60 N constants),
#                et couple de reference injecte a 50 % en feed-forward dans la loi PD (poignets
#                exclus : leur couple de reference est colle a la butee 5 N.m sur ~45 % du clip).
#
# L'observation et le nombre de parametres sont INCHANGES -> le warm start depuis le checkpoint de
# phase 1 se recharge directement, comme pour run_train_gripforce_60n.sh.
#
# ATTENTION DEPLOIEMENT : le point 3 suppose que holosoma_inference sache rejouer le meme couple de
# feed-forward, indexe sur la phase du clip (le low-level Unitree a bien un champ tau, mais il n'est
# pas encore cable). Une policy entrainee ici et deployee sans ce plumbing commandera un couple
# faux. Pour un run comparable SANS feed-forward, ajouter :
#   --action.terms.joint-control.params.torque-feedforward.enable False
#
# Branche : feat/dynamics-aware-training
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # racine du repo wbt_rl

source modules/third_party/holosoma_custom/scripts/source_isaacsim_setup.sh

# Meme resolution de checkpoint que run_train_gripforce_60n.sh : dernier modele de la phase 1.
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
    exit 1
fi
echo "Warm start depuis ${CKPT}"

python modules/02_training/holosoma_custom/train_agent.py exp:g1-29dof-wbt-w-object-actor-grip-force-dyn \
    logger:wandb \
    --training.num-envs 4096 \
    --training.checkpoint "${CKPT}" \
    "$@"
