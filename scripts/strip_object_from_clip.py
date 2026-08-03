#!/usr/bin/env python3
"""Produit une variante CORPS-SEUL d'un clip d'entrainement, sans caisse ni table.

Pourquoi
--------
``MotionCommand.__init__`` fait ``get_actor_indices("object")`` des que le clip declare
``has_object`` (present des que la cle ``object_pos_w`` existe), sans regarder si la scene a
reellement spawne un objet -- et le registre leve sur un acteur inconnu. Entrainer un clip
"w_obj" avec un preset corps-seul (``exp:g1-29dof-wbt``, sans ``object_urdf_path``) plante donc au
setup. Retirer les cles objet/support du NPZ est la facon propre d'obtenir une phase 1 de pure
locomotion sur EXACTEMENT le meme mouvement.

Usage
-----
    python scripts/strip_object_from_clip.py --src <clip_w_obj.npz> --dst <clip_loco.npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Tout ce qui declenche has_object / has_support cote MotionLoader, plus les champs de contact
# qui n'ont aucun sens sans objet.
DROP_PREFIXES = ("object_", "support_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()

    src = np.load(a.src)
    kept, dropped = {}, []
    for k in src.files:
        if k.startswith(DROP_PREFIXES):
            dropped.append(k)
        else:
            kept[k] = src[k]

    if "object_pos_w" not in dropped:
        print("ATTENTION : le clip source ne portait deja pas d'objet.")

    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.dst, **kept)

    print(f"conserve ({len(kept)}) : {sorted(kept)}")
    print(f"retire   ({len(dropped)}) : {sorted(dropped)}")
    print(f"\nEcrit : {a.dst}")
    print(f"  T = {kept['joint_pos'].shape[0]} frames a {float(kept['fps'])} fps")


if __name__ == "__main__":
    main()
