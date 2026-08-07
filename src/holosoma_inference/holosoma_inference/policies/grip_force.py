"""Portage a l'inference du biais de couple de prise applique a l'entrainement.

Pourquoi ce fichier existe
--------------------------
A l'entrainement, les 60 N par main ne sortent PAS du reseau : ils sont ajoutes par
l'environnement dans ``JointPositionActionTerm._compute_torques`` ::

    if self._grip_enabled:
        torques = torques + self._compute_grip_force_bias()

La policy apprend donc a porter la caisse *en presence* de cette force, sans jamais apprendre a
la produire. L'ONNX exporte n'en contient rien, et ``cmd_tau`` cote inference restait un vecteur
de zeros -- la caisse tombait des la premiere frame de portage en sim2sim comme sur le robot.
Ce module reconstruit exactement le meme biais a partir de la seule proprioception.

Ce qui rend le portage exact
----------------------------
1. ``tau = J^T @ F`` est INVARIANT par changement de repere tant que ``J`` et ``F`` sont exprimes
   dans le meme repere. On travaille donc entierement dans le repere ``torso_link``, jamais dans le
   monde : aucune odometrie, aucune estimation de pose globale n'est necessaire.
2. La chaine du poignet (elbow -> wrist_roll -> wrist_pitch -> wrist_yaw) se deduit des seuls
   angles articulaires mesures, par FK pinocchio. Verifie contre les poses de liens que le
   simulateur fournit a l'entrainement : ecart median 1.3 mm, max 10 mm, et le couple resultant
   reproduit la formule d'entrainement a 0.65 % median / 4.2 % max sur les frames de contact.
3. La pose de la caisse est prise a la MEME source que l'observation (``obj_pos_b``, repere torse).
   C'est l'invariant important : ce que la policy voit et ce que la force fait ne peuvent pas
   diverger, quelle que soit la source active (flux live sim2sim, ou lecture du clip en boucle
   ouverte).

Limite connue
-------------
En boucle ouverte (pas de flux live, pose caisse lue du clip), la direction de serrage se degrade
proportionnellement a la derive de la caisse par rapport a sa trajectoire de reference. Cette
degradation n'a pas encore ete chiffree sur un rollout de phase 2 -- les dumps d'eval disponibles
proviennent d'un clip anterieur et ne sont pas comparables. A mesurer avant tout passage au reel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

_AXIS_ROLL = np.array([1.0, 0.0, 0.0])
_AXIS_PITCH = np.array([0.0, 1.0, 0.0])
_AXIS_YAW = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class GripForceParams:
    """Miroir de ``holosoma.config_types.action.GripForceCfg``.

    Les valeurs par defaut reprennent celles du preset d'entrainement
    ``g1_29dof_joint_pos_grip_force`` : toute divergence ici se traduit par une force differente
    de celle vue a l'entrainement, donc par un ecart sim2real silencieux.
    """

    target_force_n: float = 60.0
    force_command_max_n: float = 90.0
    hand_offset_local: tuple[float, float, float] = (0.029, -0.003, 0.0)
    ref_body_name: str = "torso_link"
    left_chain_body_names: tuple[str, ...] = (
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
    )
    right_chain_body_names: tuple[str, ...] = (
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    )
    left_wrist_joint_names: tuple[str, ...] = (
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    )
    right_wrist_joint_names: tuple[str, ...] = (
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    max_wrist_torque_frac: float = 0.75
    """Fraction de la limite d'effort URDF allouee au feedforward, PAR ARTICULATION.

    Les poignets du G1 n'ont PAS le meme budget : roll 25 N.m, pitch 5 N.m, yaw 5 N.m. Une borne
    scalaire unique laisserait passer sur pitch/yaw des couples largement au-dessus de la limite
    physique de l'actionneur.

    Pourquoi une fraction et pas la limite entiere : a l'entrainement le clip final porte sur le
    couple TOTAL (PD + biais) contre ``torque_limits``, donc le simulateur arbitre lui-meme entre
    les deux. A l'inference le PD est calcule dans le driver moteur et reste invisible ici : on ne
    peut borner que le feedforward, il faut donc lui reserver moins que le budget complet pour que
    le PD garde de quoi asservir. Mesure sur femto14_box36 : le biais culmine a 3.64 N.m sur le
    pitch, soit 73 % d'un budget de 5 N.m -- la marge est etroite.

    0.75 est le reglage retenu parce qu'il est le plus bas qui n'ecrete AUCUNE composante sur ce
    clip (plafond pitch 3.75 N.m > pic 3.64), donc le couple reproduit l'entrainement partout, tout
    en laissant 1.25 N.m au PD. A 0.5 on ecrete 32 composantes et l'erreur monte a 30 % au pire ;
    a 0.65, 7 composantes. Si un autre clip ou une autre consigne de force fait remonter le pic,
    ``saturation_count`` le signalera."""

    gate: str = "clip"
    """Quand les mains serrent. Trois modes :

    ``clip``     -- porte de l'ENTRAINEMENT : le flag GT du clip, indexe par le temps
                    (``MotionCommand.grip_active = ref_contact``). Reproduit exactement ce que la
                    policy a vu, mais serre selon l'horloge du clip : si le robot est en avance ou
                    en retard sur la reference, il serre au mauvais moment.
    ``distance`` -- porte MESUREE : serre des que la caisse observee est a moins de
                    ``contact_distance_m`` des mains. Demande une pose de caisse fiable (mocap ou
                    RGB-D) ; c'est le mode qui suit la realite plutot que le chronometre.
    ``both``     -- ET logique des deux. Le plus conservateur : ne serre que sur les frames ou la
                    reference prevoit un portage ET ou les mains sont effectivement sur la caisse.

    Attention au decalage entrainement/deploiement : la policy a ete entrainee avec ``clip``.
    ``distance`` et ``both`` sont des ameliorations de robustesse, mais elles divergent de ce
    qu'elle a vu des que les deux portes ne coincident pas. Pour supprimer l'ecart, il faut passer
    la MEME porte cote entrainement (``managers/command/terms/wbt.py``, ligne ~1203)."""

    contact_distance_m: float = 0.35
    """Seuil main<->caisse du mode ``distance``. Reprend
    ``grasp_settle.contact_distance_threshold`` de l'entrainement, pour que les deux portes
    s'ouvrent sur le meme critere geometrique."""

    ref_body_offset: tuple[float, float, float] = (-0.001674, -0.000272, 0.009878)
    """Recalage de l'origine ``torso_link`` entre l'URDF et le simulateur, exprime dans le repere
    torse. Mesure sur femto14_box36 : l'origine du torse cote sim est ~9.9 mm au-dessus de celle
    que donne la FK URDF, orientation strictement identique (0.000 deg sur tout le clip).

    Sans ce recalage, ``obj_pos_b`` (calcule cote observation dans le repere torse du SIMULATEUR)
    et les poses de poignet issues de la FK ne sont pas dans le meme repere : la direction de
    serrage est biaisee et le couple derive de 7.6 % en median. Avec, l'erreur retombe au niveau
    du bruit de FK. La faible dispersion (ecart-type 0.5 mm sur z) confirme un offset de modele
    constant, pas une erreur dependant de la posture."""


class GripForceBias:
    """Reconstruit ``_compute_grip_force_bias`` a partir des angles articulaires mesures."""

    def __init__(self, urdf_path: str, dof_names: tuple[str, ...], params: GripForceParams) -> None:
        import pinocchio as pin  # noqa: PLC0415 -- dependance optionnelle, seulement si grip actif

        self._pin = pin
        self.params = params
        self.dof_names = tuple(dof_names)

        # Base FIXE : on ne veut jamais dependre de la pose monde du robot. Les poses sortent donc
        # dans le repere du pelvis, puis sont ramenees dans le repere torse -- le meme que celui
        # dans lequel obj_pos_b est exprime.
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        pin_joints = {self.model.names[i] for i in range(1, self.model.njoints)}
        missing = [n for n in self.dof_names if n not in pin_joints]
        if missing:
            raise ValueError(
                f"URDF {urdf_path} ne contient pas les articulations {missing} attendues par "
                f"dof_names. Le biais de prise ne peut pas etre calcule."
            )
        self._qidx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in self.dof_names], dtype=int
        )

        self._ref_fid = self._frame_id(params.ref_body_name)
        self._chain_fids = {
            "left": [self._frame_id(n) for n in params.left_chain_body_names],
            "right": [self._frame_id(n) for n in params.right_chain_body_names],
        }
        self._wrist_dof_idx = {
            "left": np.array([self.dof_names.index(n) for n in params.left_wrist_joint_names], dtype=int),
            "right": np.array([self.dof_names.index(n) for n in params.right_wrist_joint_names], dtype=int),
        }
        # Budget de couple par articulation, lu dans l'URDF (roll 25 N.m, pitch/yaw 5 N.m sur le
        # G1) plutot qu'une borne scalaire commune qui serait 5x trop permissive sur pitch/yaw.
        self._wrist_tau_cap = {}
        for side, names in (
            ("left", params.left_wrist_joint_names),
            ("right", params.right_wrist_joint_names),
        ):
            caps = []
            for n in names:
                jid = self.model.getJointId(n)
                caps.append(float(self.model.effortLimit[self.model.joints[jid].idx_v]))
            self._wrist_tau_cap[side] = np.asarray(caps) * float(params.max_wrist_torque_frac)
        self._hand_offset = np.asarray(params.hand_offset_local, dtype=float)
        self._ref_offset = np.asarray(params.ref_body_offset, dtype=float)
        self._target_n = float(min(params.target_force_n, params.force_command_max_n))

        # Flag de contact GT du clip, meme indexation que obj_pos_b.
        self._contact: np.ndarray | None = None

        if params.gate not in ("clip", "distance", "both"):
            raise ValueError(f"gate='{params.gate}' inconnu ; attendu clip | distance | both.")

        self.last_force_n: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.last_active: bool = False
        self.last_hand_distance: float = float("nan")
        self.saturation_count: int = 0
        """Nombre de composantes ecretees depuis le demarrage. Non nul = le couple demande depasse
        le budget alloue : soit la caisse est mal localisee (bras de levier aberrant), soit
        max_wrist_torque_frac est trop bas. A surveiller au premier run."""

        logger.info(
            f"GripForceBias pret : {self._target_n:.1f} N/main, repere {params.ref_body_name}, "
            f"URDF {urdf_path.split('/')[-1]}"
        )
        for side, cap in self._wrist_tau_cap.items():
            logger.info(f"  plafond feedforward {side:5s} (roll/pitch/yaw) : {np.round(cap, 2)} N.m")

    def _frame_id(self, name: str) -> int:
        if not self.model.existFrame(name):
            raise ValueError(f"Frame '{name}' absente de l'URDF : biais de prise impossible.")
        return self.model.getFrameId(name)

    def load_contact_flags(self, npz_path: str, prepend: int = 0) -> None:
        """Charge ``object_ref_contact`` du clip -- la meme porte que ``MotionCommand.grip_active``.

        Sans ce flag la force serait appliquee en permanence, y compris pendant l'approche et
        apres le depot, ou elle pousserait les poignets vers une caisse qu'ils ne touchent pas.
        """
        if self.params.gate == "distance":
            logger.info("gate='distance' : le flag de contact GT du clip n'est pas utilise.")
            return
        data = np.load(npz_path, allow_pickle=True)
        if "object_ref_contact" not in data.files:
            raise ValueError(
                f"{npz_path} ne porte pas object_ref_contact : ce clip n'a pas de contact GT, "
                f"le biais de prise n'a pas de porte et serait applique en continu. "
                f"Utiliser gate='distance' si la pose de caisse est mesuree (mocap)."
            )
        contact = np.asarray(data["object_ref_contact"], dtype=bool)
        if prepend:
            contact = np.concatenate([np.repeat(contact[:1], prepend, axis=0), contact], axis=0)
        self._contact = contact
        logger.info(
            f"Contact GT charge depuis {npz_path.split('/')[-1]} : {contact.shape[0]} frames "
            f"(prepend={prepend}), contact actif sur {100 * contact.mean():.1f} % des frames."
        )

    def is_gripping(self, motion_timestep: float) -> bool:
        if self._contact is None:
            return False
        idx = int(np.clip(int(round(motion_timestep)), 0, self._contact.shape[0] - 1))
        return bool(self._contact[idx])

    def compute(self, joint_pos: np.ndarray, obj_pos_b: np.ndarray, motion_timestep: float) -> np.ndarray:
        """Couple feedforward (num_dofs,) a additionner a ``cmd_tau``.

        Args:
            joint_pos: positions articulaires MESUREES, dans l'ordre ``dof_names``.
            obj_pos_b: position de la caisse dans le repere ``ref_body_name`` -- exactement le
                vecteur envoye a l'observation, quelle que soit sa source (live ou clip).
            motion_timestep: index de frame courant du clip, pour la porte de contact.
        """
        tau = np.zeros(len(self.dof_names), dtype=np.float64)
        self.last_force_n = {"left": 0.0, "right": 0.0}
        self.last_active = False

        gate = self.params.gate
        clip_open = self.is_gripping(motion_timestep) if gate in ("clip", "both") else True
        if gate == "clip" and not clip_open:
            # Mode clip pur : rien a calculer, la FK serait du travail perdu.
            return tau

        pin = self._pin
        q = pin.neutral(self.model)
        q[self._qidx] = np.asarray(joint_pos, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        M_ref = self.data.oMf[self._ref_fid]
        # obj_pos_b arrive dans le repere torse du SIMULATEUR ; la FK travaille dans celui de
        # l'URDF. Les deux ne different que d'une translation constante (cf. ref_body_offset).
        box = np.asarray(obj_pos_b, dtype=float).reshape(3) + self._ref_offset

        # Porte mesuree : distance main<->caisse reellement observee. Symetrique comme a
        # l'entrainement (les deux mains serrent ensemble), donc on retient la main la plus proche.
        if gate in ("distance", "both"):
            hand_dist = min(
                float(np.linalg.norm(box - M_ref.actInv(self.data.oMf[fids[-1]]).translation))
                for fids in self._chain_fids.values()
            )
            self.last_hand_distance = hand_dist
            dist_open = hand_dist <= self.params.contact_distance_m
        else:
            self.last_hand_distance = float("nan")
            dist_open = True

        if not (clip_open and dist_open):
            return tau
        self.last_active = True

        for side, fids in self._chain_fids.items():
            # Poses de la chaine ramenees dans le repere torse : meme repere que obj_pos_b.
            rel = [M_ref.actInv(self.data.oMf[f]) for f in fids]
            (p_elbow, R_elbow), (p_roll, R_roll), (p_pitch, R_pitch), (p_yaw, R_yaw) = [
                (m.translation, m.rotation) for m in rel
            ]
            del p_elbow  # l'elbow n'intervient que par son orientation (axe du roll)

            p_hand = p_yaw + R_yaw @ self._hand_offset
            axes = (R_elbow @ _AXIS_ROLL, R_roll @ _AXIS_PITCH, R_pitch @ _AXIS_YAW)
            pivots = (p_roll, p_pitch, p_yaw)
            jacobian = np.stack([np.cross(axes[k], p_hand - pivots[k]) for k in range(3)], axis=1)

            squeeze = box - p_yaw
            norm = float(np.linalg.norm(squeeze))
            if norm < 1e-6:
                continue
            force = (self._target_n / norm) * squeeze

            wrist_tau = jacobian.T @ force
            cap = self._wrist_tau_cap[side]
            saturated = np.abs(wrist_tau) > cap
            if saturated.any():
                self.saturation_count += int(saturated.sum())
            wrist_tau = np.clip(wrist_tau, -cap, cap)
            tau[self._wrist_dof_idx[side]] += wrist_tau
            self.last_force_n[side] = self._target_n

        return tau
