"""Whole Body Tracking reward presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        # Meme fonction que le terme "coarse" ci-dessus, sigma resserre : reward MULTI-ECHELLE sur
        # la position de la box. Le terme large (sigma=0.3) est plat la ou ca compte -- la
        # termination bad_object_pos coupe a 0.15 m, et exp(-0.15^2/0.3^2)=0.78 : sur TOUTE la bande
        # ou l'episode est encore vivant le coarse ne varie que de 1.00 -> 0.78, soit ~0.02 par cm.
        # Resultat observe : la box stagne quelques cm sous la hauteur de portage, le gradient ne
        # paie pas l'effort des derniers cm. sigma=0.12 redonne de la dynamique sur cette bande
        # (1.00 -> 0.21 entre 0 et 15 cm, ~3.5x le gradient du coarse a 10 cm). Le coarse reste pour
        # le champ lointain (reapproche apres une derive / phase assistee a alpha eleve).
        "object_global_ref_position_error_fine_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.12},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Dense grasp signal: hand<->object relative position on contact frames (neutral on free
        # frames). Complements the global object terms, which say nothing about WHERE in the hand
        # frame the object should be while carried.
        # Poids 0.5 (et non 1.0) : trois termes notent la geometrie main<->box (celui-ci,
        # object_surface_contact et object_flat_contact_quality). Celui-ci est le moins informatif
        # des trois -- une pose relative RIGIDE ne distingue pas "bon endroit sur la box" de
        # "mauvais endroit au bon offset", ce que surface_contact tranche via le witness. Il reste
        # utile comme signal dense/lisse, mais il ne merite pas le meme poids que les deux autres.
        "object_grasp_relative_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_grasp_relative_error_exp",
            params={"sigma": 0.1},
            weight=0.5,
        ),
        # WHERE on the box surface + how deep the current contact is, vs the retargeting reference
        # (HoloV2's own witness/distance point-cloud contact fields, baked per-frame -- see
        # gvhmr-fp-pipeline/contact_from_retarget.py). Neutral automatically if the loaded motion
        # doesn't carry a reference witness (older/synthetic clips).
        # sigma_dist 0.05 -> 0.08 : ce terme est un PRODUIT de deux exp (geodesique x profondeur),
        # donc les deux pentes se cumulent. Et d_ref vient du witness FoundationPose du retargeting,
        # lui-meme bruite : a 0.05 la tolerance en profondeur etait plus serree que le bruit de sa
        # propre reference, ce qui note du bruit plutot que la policy.
        "object_surface_contact_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_surface_contact_error_exp",
            params={"sigma_geodesic": 0.1, "sigma_dist": 0.08},
            weight=1.0,
        ),
        # Contact QUALITY (option 2): reward the hand's flat-face keypoints to be flush against the
        # box (patch contact that resists rotation). Independent of the reference witness -- teaches
        # HOW to grip, pairs with the physicality curriculum. sigma=0.03 m (flush within a few cm).
        "object_flat_contact_quality_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_flat_contact_quality_exp",
            params={"sigma": 0.03},
            weight=1.0,
        ),
        # Compagnon LARGE du terme ci-dessus (meme func, sigma 0.10, poids moitie). sigma=0.03 seul
        # est inapprenable : une policy qui n'a pas encore trouve le patch plat est a >=5 cm de rms,
        # ou exp(-0.05^2/0.03^2)=0.06, et a 8 cm 0.0008 -- numeriquement nul, donc aucun gradient
        # pour DECOUVRIR le patch (et le terme vaut deja 0 hors contact, il n'y a pas d'autre
        # source de signal). Le sigma=0.10 porte de 8 cm a 2 cm (0.53 -> 0.96), le sigma=0.03 prend
        # le relais sur les derniers cm. Meme construction multi-echelle que
        # object_global_ref_position_error_fine_exp.
        "object_flat_contact_quality_coarse_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_flat_contact_quality_exp",
            params={"sigma": 0.10},
            weight=0.5,
        ),
        # robot<->TABLE : même reward de contact de surface (witness + SDF) que pour la box, mais
        # sur l'objet statique table. Apprend au robot à placer sa main au bon endroit près de la
        # table (approche/dépôt) et à ne pas foncer dedans. Neutre si le clip ne porte pas de table.
        # Poids plus faible que la box (signal secondaire de placement, pas la tâche principale).
        "support_surface_contact_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:support_surface_contact_error_exp",
            params={"sigma_geodesic": 0.1, "sigma_dist": 0.05},
            weight=0.5,
        ),
        # Override du terme herite : un contact est "indesirable" s'il n'est PAS dans le
        # retargeting a cette frame, pas s'il est dans une liste de noms de corps. Le regex seul
        # n'excluait que pieds/chevilles/wrist_yaw ; or en portant un cube de 0.36 m la REFERENCE
        # elle-meme colle l'avant-bras (elbow_link) et le poignet (wrist_roll/pitch) contre la box,
        # et le terme renvoie un COMPTE de corps en contact -> ~-0.4/s en permanence sur exactement
        # le comportement qu'on entraine. Le mask (cf. UndesiredContacts) exempte par frame tout
        # corps que la reference met au contact du sol / de la box / de la table, et continue de
        # punir un corps loin de tout dans la reference qui touche quand meme en sim (genou au sol,
        # torse dans la table) -- quel que soit son nom. Le regex reste comme plancher inconditionnel.
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                **g1_29dof_wbt_reward.terms["undesired_contacts"].params,
                "use_reference_mask": True,
                # Marges generreuses : body_pos_w donne l'ORIGINE du link, pas sa surface de
                # collision, donc la marge doit absorber le rayon du link. L'asymetrie justifie de
                # viser large -- une exemption en trop ne fait que ne pas punir un contact, une
                # punition en trop combat activement la tache.
                "ref_contact_margin": 0.15,
                "ground_margin": 0.10,
            },
            weight=-0.1,
        ),
    }
)

# Portage caisse + table, avec les signaux de l'etage 05 (SPIDER) fusionnes dans le clip par
# wbt_rl/scripts/merge_dynamics.py (cf. motion_config_w_object_femto14_box36_dyn).
#
# Les termes herites du preset w_object ne changent PAS de poids : object_grasp_relative_error_exp
# et object_flat_contact_quality_exp deviennent simplement bimanuels d'eux-memes des que le clip
# porte dyn_obj_contact_lr (cf. _bimanual_available), en gardant leur echelle 0..1 -- la moyenne sur
# les mains en contact, pas la somme. Un clip sans sidecar retombe automatiquement sur l'ancre
# unique. Seuls les trois termes ci-dessous sont nouveaux, et chacun renvoie 0 sans sidecar, donc ce
# preset est utilisable tel quel sur un clip non enrichi (il vaut alors exactement w_object).
_GATE = "holosoma.managers.reward.terms.wbt:"

# FUSION DES PAIRES COARSE/FINE. Le preset w_object configure DEUX fois la meme fonction a deux
# sigmas et laisse la somme ponderee faire le melange multi-echelle. Ca marche, mais ca depense deux
# poids pleins pour UNE grandeur physique : 2.0 pour la position de la caisse (autant que tout le
# tracking relatif du corps) et 1.5 pour la qualite de contact plat. Cumule sur 21 termes, ca a fait
# passer le budget "caisse" (6.5) devant le budget "corps" (5.0) sans que ce soit un choix.
#
# _multiscale_exp fait la moyenne des noyaux A L'INTERIEUR d'un terme : profil de gradient
# identique, borne a 0..1, un seul poids. sigma_weights conserve l'emphase que la version eclatee
# encodait dans ses deux poids (2:1 pour fin:large sur le contact plat, ou 1.0 et 0.5).
# Economie : 2.0 -> 1.0 et 1.5 -> 1.0, soit 1.5 de budget rendu, zero gradient perdu.
_merged = {
    k: v
    for k, v in g1_29dof_wbt_reward_w_object.terms.items()
    if k not in ("object_global_ref_position_error_fine_exp", "object_flat_contact_quality_coarse_exp")
}

g1_29dof_wbt_reward_w_object_dyn = RewardManagerCfg(
    terms={
        **_merged,
        # Ex object_global_ref_position_error_exp (sigma 0.3, w 1.0) + _fine_exp (sigma 0.12, w 1.0).
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigmas": (0.3, 0.12)},
            weight=1.0,
        ),
        # Ex object_flat_contact_quality_exp (sigma 0.03, w 1.0) + _coarse_exp (sigma 0.10, w 0.5).
        # Le 2:1 reproduit exactement le rapport des deux anciens poids : le sigma large sert a
        # DECOUVRIR le patch (a 8 cm le sigma=0.03 vaut 0.0008, numeriquement nul), le sigma fin
        # prend le relais sur les derniers centimetres.
        "object_flat_contact_quality_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_flat_contact_quality_exp",
            params={"sigmas": (0.03, 0.10), "sigma_weights": (2.0, 1.0)},
            weight=1.0,
            achievable_gate=f"{_GATE}gate_object_contact",
        ),
        # limits_dof_pos : -10.0 representait 93 % du budget negatif, deux ordres de grandeur
        # au-dessus de tout le reste, sur une somme brute de RADIANS non normalisee (0.1 rad de
        # depassement ne veut pas dire la meme chose sur un poignet a 0.5 rad de course que sur une
        # epaule a 5 rad). normalize=True divise par la marge butee souple -> butee dure : 1.0 = "ce
        # joint a consomme toute sa marge", identiquement pour tous. Le terme se lit alors comme un
        # NOMBRE de DOF satures, et -1.0 est interpretable seul.
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9, "normalize": True},
            weight=-1.0,
        ),
        # Gates de reward atteignable (diagnostic pur, cf. RewardTermCfg.achievable_gate) : ces
        # termes valent 0 quand la reference n'a pas de contact a noter, donc le budget disponible
        # varie le long du clip et le scalaire de reward n'est pas comparable d'une phase a l'autre.
        "object_grasp_relative_error_exp": replace(
            _merged["object_grasp_relative_error_exp"], achievable_gate=f"{_GATE}gate_object_contact"
        ),
        "object_surface_contact_error_exp": replace(
            _merged["object_surface_contact_error_exp"], achievable_gate=f"{_GATE}gate_object_contact"
        ),
        "support_surface_contact_error_exp": replace(
            _merged["support_surface_contact_error_exp"], achievable_gate=f"{_GATE}gate_support_contact"
        ),
        # Calendrier d'appui par pied. Rien dans les termes de tracking ne dit QUEL pied doit porter
        # a QUEL instant -- ils contraignent ou sont les liens, ce qui laisse la policy libre
        # d'inventer sa propre sequence d'appuis (trainage, double appui permanent) tant que les
        # positions y sont approximativement. Poids 0.5 : c'est un terme dense, borne a 1, qui doit
        # informer la demarche sans concurrencer le tracking de corps (poids 1.0 chacun).
        "feet_contact_schedule": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FeetContactSchedule",
            params={"threshold": 1.0},
            weight=0.5,
            achievable_gate="holosoma.managers.reward.terms.wbt:gate_dyn_sidecar",
        ),
        # Repartition de charge gauche/droite en double appui. Complement direct du terme ci-dessus,
        # qui ne lit que le booleen : sur femto14_box36 la reference est en double appui sur 257 des
        # 327 frames (79 %), donc 4 frames sur 5 le calendrier vaut 1.0 quoi que fasse la policy de
        # son poids. C'est la GRF (dyn_foot_grf_lr) qui informe ces frames-la, sous forme de RATIO
        # et non de force absolue -- la reference pique a 2427 N sur un pied (~7x le poids) aux
        # transitions de contact, meme artefact kp=500 que dyn_tau, et le ratio l'annule.
        # Poids 0.5, aligne sur feet_contact_schedule : meme famille, meme echelle 0..1, et ne doit
        # pas concurrencer le tracking de corps (1.0).
        # sigma=0.15 mesure sur le clip : la reference balaie une part gauche de 0.02 a 1.00, et le
        # calibrage se fait contre la policy PARESSEUSE (poids fige a 50/50, l'optimum trivial de ce
        # terme). Elle marque 0.79 a sigma=0.25 -- seulement 0.21 d'ecart avec la policy parfaite,
        # trop peu pour payer un transfert de poids actif. sigma=0.15 la ramene a 0.63 (0.37
        # d'ecart) tout en laissant 0.64 a 10 % d'erreur de repartition, donc le quasi-juste reste
        # paye. sigma=0.10 serait le piege inverse : 0.002 a 25 % d'erreur, numeriquement nul, donc
        # aucun gradient pour DECOUVRIR la bonne repartition depuis un depart quelconque -- exactement
        # l'ecueil documente sur object_flat_contact_quality_exp.
        "feet_load_share": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FeetLoadShare",
            params={"sigma": 0.15},
            weight=0.5,
            achievable_gate="holosoma.managers.reward.terms.wbt:gate_feet_double_support",
        ),
        # Anti-glissement, actif uniquement quand la REFERENCE a le pied plante (une jambe en vol
        # doit aller vite). Le clip femto14_box36 glisse a 0.020 m/s median en appui -- soit un cout
        # de 0.5 * 0.0004 = 2e-4/pas au niveau observe, negligeable, et qui ne mord que si la policy
        # se met a patiner franchement (0.2 m/s -> 0.02). C'est l'intention : une barriere, pas une
        # taxe permanente.
        "feet_slip_on_ref_stance": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:FeetSlipOnRefStance",
            weight=-0.5,
        ),
        # Enveloppe de couple UNILATERALE contre le couple que le mouvement demande reellement
        # (dyn_tau). Ne punit jamais le sous-couple -- cf. le docstring de torque_envelope_penalty :
        # dyn_tau vient d'un solve a kp=500 qui peut suivre des transitions de contact plus
        # sechement que les vrais gains, donc ses pics ne sont pas atteignables et un terme de
        # tracking punirait la policy de ne pas faire l'impossible.
        # margin=1.5 : 50 % de marge au-dessus de la reference avant toute penalite. Poids -0.05
        # sur une somme normalisee de 29 relu^2 -- volontairement petit, c'est un regularisateur
        # (remplacer un proxy d'action-rate par la vraie grandeur physique), pas un objectif.
        "torque_envelope_penalty": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:torque_envelope_penalty",
            params={"margin": 1.5},
            weight=-0.05,
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_dyn",
]
