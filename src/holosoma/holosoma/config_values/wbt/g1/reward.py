"""Whole Body Tracking reward presets for the G1 robot."""

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
        # "Si ca ne tient pas, tu peux toujours serrer plus fort" -- donne en SHAPING
        # POTENTIEL (Ng, Harada & Russell 1999) : gamma*Phi(s') - Phi(s) laisse la politique
        # optimale strictement inchangee quel que soit Phi. C'est donc une intuition (la direction
        # du progres), jamais une consigne (combien serrer) : contrairement a tous les autres termes
        # de contact ici, celui-ci ne PEUT pas deplacer l'optimum, donc pas d'attracteur parasite.
        # Phi porte sur la FORCE de contact aux paumes et non sur la penetration : en contact rigide
        # serrer plus fort n'enfonce pas la main dans la box (penetration PhysX sub-millimetrique),
        # donc les distances signees de object_flat_contact_quality_exp saturent des le contact et
        # sont aveugles a l'effort de prise.
        # gamma DOIT suivre celui de l'algo (0.99 ici, cf. experiment.py:98) sinon l'invariance
        # tombe et le terme redevient un reward ordinaire qui deplace l'optimum.
        "object_grip_force_potential": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:ObjectGripForcePotential",
            # force_ref 40 N (et non 20) : 20 N venait du mu=0.9 de l'URDF, mais la DR de friction
            # descend bien plus bas (objet [0.1, 0.6] combine au robot [0.3, 1.6]), donc les envs
            # difficiles ont besoin de bien plus de force normale. Saturer trop tot n'est pas FAUX
            # (l'invariance PBRS tient quel que soit Phi), juste moins informatif precisement la ou
            # le signal sert le plus.
            params={"gamma": 0.99, "force_ref": 40.0},
            weight=1.0,
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

__all__ = ["g1_29dof_wbt_fast_sac_reward", "g1_29dof_wbt_reward", "g1_29dof_wbt_reward_w_object"]
