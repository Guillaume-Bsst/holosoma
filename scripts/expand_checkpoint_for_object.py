#!/usr/bin/env python3
"""Adapte un checkpoint WBT corps-seul a une config box+table (warm start).

Pourquoi ce script existe
-------------------------
``PPO.load`` appelle ``load_state_dict`` SANS ``strict=False`` (agents/ppo/ppo.py:654), donc passer
directement un checkpoint corps-seul a ``--training.checkpoint`` plante sur la difference de forme :

    checkpoint 3ivghz1e (full loco)     cible box+table
      actor   154 dims                    172   (+ obj_pos/ori, support_pos/ori)
      critic  286 dims                    307   (+ obj_lin_vel_b en plus)

Le piege
--------
Les groupes d'observation sont concatenes par ORDRE ALPHABETIQUE des noms de termes
(ObservationManager.compute_group). Cote ACTEUR les nouveaux termes (obj_*, support_*) trient tous
apres les anciens : les 154 colonnes existantes forment un prefixe contigu, une simple extension
suffit. Cote CRITIC en revanche, ``obj_lin_vel_b`` / ``obj_ori_b`` / ``obj_pos_b`` trient AVANT
``robot_body_ori_b`` et ``robot_body_pos_b`` : les nouvelles colonnes s'INSERENT AU MILIEU. Une copie
naive des 286 premieres colonnes decalerait tout le bloc robot_body_* de 12 colonnes et corromprait
le critic sans rien casser visiblement -- l'entrainement tournerait, en apprenant sur du bruit.

Ce script mappe donc ancien -> nouveau PAR NOM DE TERME, et verifie que les sommes de dimensions
retombent exactement sur les formes du checkpoint (sinon il refuse de continuer).

Usage
-----
    python scripts/expand_checkpoint_for_object.py \\
        --src  .../model_29999.pt \\
        --dst  .../model_29999_box_table.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Dimensions par terme pour le G1 29 dof. Verifiees contre les formes reelles du checkpoint plus
# bas -- une valeur fausse ici fait echouer l'assertion, elle ne passe pas en silence.
DIMS: dict[str, int] = {
    "actions": 29,
    "base_ang_vel": 3,
    "base_lin_vel": 3,
    "dof_pos": 29,
    "dof_vel": 29,
    "motion_command": 58,
    "motion_ref_ori_b": 6,
    "motion_ref_pos_b": 3,
    "robot_body_ori_b": 84,
    "robot_body_pos_b": 42,
    "obj_pos_b": 3,
    "obj_ori_b": 6,
    "obj_lin_vel_b": 3,
    "support_pos_b": 3,
    "support_ori_b": 6,
}

SRC_ACTOR = ["actions", "base_ang_vel", "dof_pos", "dof_vel", "motion_command", "motion_ref_ori_b"]
SRC_CRITIC = SRC_ACTOR + ["base_lin_vel", "motion_ref_pos_b", "robot_body_ori_b", "robot_body_pos_b"]
TGT_ACTOR = SRC_ACTOR + ["obj_pos_b", "obj_ori_b", "support_pos_b", "support_ori_b"]
TGT_CRITIC = SRC_CRITIC + ["obj_pos_b", "obj_ori_b", "obj_lin_vel_b", "support_pos_b", "support_ori_b"]


def layout(terms: list[str]) -> tuple[dict[str, tuple[int, int]], int]:
    """(nom -> (debut, fin)) dans l'ordre ALPHABETIQUE, + dimension totale."""
    spans, off = {}, 0
    for name in sorted(terms):
        spans[name] = (off, off + DIMS[name])
        off += DIMS[name]
    return spans, off


def column_map(src_terms: list[str], tgt_terms: list[str], expect_src: int, expect_tgt: int) -> torch.Tensor:
    """Indices cible de chaque colonne source, via le nom de terme."""
    src, n_src = layout(src_terms)
    tgt, n_tgt = layout(tgt_terms)
    if n_src != expect_src or n_tgt != expect_tgt:
        raise SystemExit(
            f"Dimensions incoherentes : calcule src={n_src} cible={n_tgt}, "
            f"attendu src={expect_src} cible={expect_tgt}. Le dict DIMS ne correspond pas a ce "
            f"checkpoint -- ne pas forcer, corriger DIMS."
        )
    idx = torch.empty(n_src, dtype=torch.long)
    for name, (s0, s1) in src.items():
        t0, _ = tgt[name]
        idx[s0:s1] = torch.arange(t0, t0 + (s1 - s0))
    return idx


def expand_linear(w: torch.Tensor, idx: torch.Tensor, n_tgt: int) -> torch.Tensor:
    """(out, n_src) -> (out, n_tgt) : colonnes remappees, le reste a zero.

    Zero et non aleatoire : a l'initialisation la policy ignore donc exactement les nouvelles
    entrees et reproduit son comportement d'origine. Le gradient les fera emerger.
    """
    out = torch.zeros(w.shape[0], n_tgt, dtype=w.dtype)
    out[:, idx] = w
    return out


def expand_norm(state: dict, idx: torch.Tensor, n_tgt: int) -> dict:
    """Normalizer empirique : mean->0, var->1, std->1 sur les colonnes neuves (identite)."""
    out = dict(state)
    for key, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        if key not in state:
            continue
        v = state[key]
        new = torch.full((v.shape[0], n_tgt), fill, dtype=v.dtype)
        new[:, idx] = v
        out[key] = new
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep-iter", action="store_true",
                    help="Conserver le compteur d'iterations du checkpoint (defaut : remis a 0).")
    a = ap.parse_args()

    ck = torch.load(a.src, map_location="cpu", weights_only=False)

    actor_key = next(k for k in ck["actor_model_state_dict"] if k.endswith(".0.weight"))
    critic_key = next(k for k in ck["critic_model_state_dict"] if k.endswith(".0.weight"))
    n_actor_src = ck["actor_model_state_dict"][actor_key].shape[1]
    n_critic_src = ck["critic_model_state_dict"][critic_key].shape[1]

    _, n_actor_tgt = layout(TGT_ACTOR)
    _, n_critic_tgt = layout(TGT_CRITIC)
    ia = column_map(SRC_ACTOR, TGT_ACTOR, n_actor_src, n_actor_tgt)
    ic = column_map(SRC_CRITIC, TGT_CRITIC, n_critic_src, n_critic_tgt)

    print(f"actor  {n_actor_src} -> {n_actor_tgt}")
    print(f"critic {n_critic_src} -> {n_critic_tgt}")
    sa, _ = layout(SRC_CRITIC)
    ta, _ = layout(TGT_CRITIC)
    for name in ("robot_body_ori_b", "robot_body_pos_b"):
        print(f"  critic {name}: {sa[name]} -> {ta[name]}   (decalage {ta[name][0] - sa[name][0]} colonnes)")

    ck["actor_model_state_dict"][actor_key] = expand_linear(ck["actor_model_state_dict"][actor_key], ia, n_actor_tgt)
    ck["critic_model_state_dict"][critic_key] = expand_linear(ck["critic_model_state_dict"][critic_key], ic, n_critic_tgt)
    if ck.get("actor_obs_normalizer_state_dict"):
        ck["actor_obs_normalizer_state_dict"] = expand_norm(ck["actor_obs_normalizer_state_dict"], ia, n_actor_tgt)
    if ck.get("critic_obs_normalizer_state_dict"):
        ck["critic_obs_normalizer_state_dict"] = expand_norm(ck["critic_obs_normalizer_state_dict"], ic, n_critic_tgt)

    # L'etat de l'optimiseur porte les formes des parametres d'origine : le garder ferait planter
    # load_state_dict. On repart donc avec un optimiseur neuf -- il faut lancer avec
    # --algo.config.load-optimizer False (le defaut est True, cf. config_types/algo.py:171).
    for k in ("actor_optimizer_state_dict", "critic_optimizer_state_dict"):
        ck.pop(k, None)

    if not a.keep_iter:
        ck["iter"] = 0
        ck["iteration"] = 0

    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.dst)
    print(f"\nEcrit : {a.dst}")
    print("Lancer avec --algo.config.load-optimizer False (l'etat d'optimiseur a ete retire).")


if __name__ == "__main__":
    main()
