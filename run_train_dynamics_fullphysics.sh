#!/usr/bin/env bash
# Entrainement portage caisse 36 cm + table, ETAGE 05, EN UN SEUL RUN : physique complete, aucune
# aide, init aleatoire, 30 000 iterations.
#
# Remplace le bootstrap en deux temps (run_train_phase1_kinematic.sh -> run_train_gripforce_60n.sh
# / run_train_dynamics_aware.sh), qui reste disponible si ce run echoue.
#
# POURQUOI LA PHASE 1 N'EST PLUS NECESSAIRE (moitie de l'argument, verifiee) :
#   Elle existait d'abord pour ne pas injecter 60 N constants dans les poignets pendant que la
#   policy apprend a marcher. Le clip enrichi porte maintenant le profil de prise MESURE par main
#   et par frame (dyn_grip_force_lr : 72 N median a gauche, 56 N a droite), et surtout ce profil
#   est NUL avant la frame 128 = 2.6 s, la premiere frame de contact du clip. Aucune force n'est
#   donc injectee pendant toute la phase d'approche : la bequille est devenue sans objet.
#
# CE QUE CE RUN PARIE (l'autre moitie, NON couverte par la force reelle) :
#   La phase 1 collait aussi la caisse a la trajectoire de reference
#   (kinematic_object_during_contact) pour qu'elle ne puisse pas tomber pendant l'apprentissage.
#   Ici elle est dynamique des la premiere iteration, et la termination bad_object_pos coupe
#   l'episode des 15 cm de derive, pendant que les rewards de portage
#   (object_surface_contact, object_flat_contact_quality) valent 0 hors contact par construction.
#   Le risque est donc un episode qui meurt avant d'atteindre les frames ou ces rewards ont
#   quelque chose a dire. Attenue par l'engagement de la force de prise des 0.35 m, pas supprime.
#
#   Signal a surveiller en priorite : Mean episode length. Le clip fait 6.5 s (327 frames a 50 fps).
#   Si elle plafonne autour de 2.6 s, la policy marche jusqu'a la caisse mais n'arrive pas a la
#   prendre -- c'est exactement le scenario que la phase 1 evitait, et il faudra y revenir.
#
# ATTENTION DEPLOIEMENT : le feed-forward de couple (50 % de dyn_tau injecte dans la loi PD) n'est
# pas encore rejouable par holosoma_inference. Une policy entrainee ici et deployee sans ce
# plumbing commandera un couple faux. Pour un run comparable SANS feed-forward, ajouter :
#   --action.terms.joint-control.params.torque-feedforward.enable False
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."   # racine du repo wbt_rl

source modules/third_party/holosoma_custom/scripts/source_isaacsim_setup.sh

# Pas de --training.checkpoint : init aleatoire, volontairement.
# num-learning-iterations vaut deja 30000 par defaut, explicite ici parce que c'est le pari du run.
python modules/02_training/holosoma_custom/train_agent.py exp:g1-29dof-wbt-w-object-actor-grip-force-dyn \
    logger:wandb \
    --training.num-envs 4096 \
    --algo.config.num-learning-iterations 30000 \
    "$@"
