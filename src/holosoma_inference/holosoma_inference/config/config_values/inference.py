"""Default inference configurations for holosoma_inference."""

from dataclasses import replace

import tyro
from typing_extensions import Annotated

from holosoma_inference.compat import entry_points
from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_values import observation, robot, task

# Shared safety secondary for all G1 configs — FastSAC locomotion.
# Each config references the same object; users can override any field
# with --secondary.task.model-path etc., or disable with --secondary none.
_g1_safety_secondary = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=task.safety_locomotion_g1,
)

g1_29dof_loco = InferenceConfig(
    robot=robot.g1_29dof,
    observation=observation.loco_g1_29dof,
    task=replace(task.locomotion, model_path=task.safety_locomotion_g1.model_path),
    secondary=_g1_safety_secondary,
)

t1_29dof_loco = InferenceConfig(
    robot=robot.t1_29dof,
    observation=observation.loco_t1_29dof,
    task=task.locomotion,
)

# fmt: off
_g1_29dof_wbt_robot = replace(
    robot.g1_29dof,
    stiff_startup_pos=(
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
        0.0, 0.0, 0.0,                          # waist
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
    ),
    stiff_startup_kp=(
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
        200.0, 200.0, 200.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    ),
    stiff_startup_kd=(
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
        5.0, 5.0, 5.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
    ),
)

g1_29dof_wbt = InferenceConfig(
    robot=_g1_29dof_wbt_robot,
# fmt: on
    observation=observation.wbt,
    task=task.wbt,
    secondary=_g1_safety_secondary,
)

# Object-carry policy (trained with exp:g1-29dof-wbt-w-object-actor): 163-dim actor obs = wbt 154 +
# box pose (obj_pos_b 3 + obj_ori_b 6). Pass --task.object-motion-file <clip>.npz so the box pose can
# be derived from the reference clip (see WholeBodyTrackingPolicy._load_object_motion).
g1_29dof_wbt_w_object = InferenceConfig(
    robot=_g1_29dof_wbt_robot,
    observation=observation.wbt_w_object,
    task=task.wbt,
    secondary=_g1_safety_secondary,
)

# fmt: off
g1_27dof_wbt = InferenceConfig(
    robot=replace(
        robot.g1_27dof,
        stiff_startup_pos=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left leg
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg
            0.0,                                      # waist (yaw only)
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,      # left arm
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # right arm
        ),
        stiff_startup_kp=(
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            350.0, 200.0, 200.0, 300.0, 300.0, 150.0,
            200.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
        ),
        stiff_startup_kd=(
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 10.0, 5.0, 5.0,
            5.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        ),
    ),
# fmt: on
    observation=observation.wbt_g1_27dof,
    task=task.wbt,
)

# Core defaults - no extension imports at module load time
DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
    "g1-29dof-wbt-w-object": g1_29dof_wbt_w_object,
    "g1-27dof-wbt": g1_27dof_wbt,
}

# Track whether extensions have been loaded
_extensions_loaded = False


def _load_extensions() -> None:
    """Lazily load extension configs from entry points.

    This is deferred to avoid circular imports when extensions import
    from holosoma_inference.config at module load time.
    """
    global _extensions_loaded  # noqa: PLW0603
    if _extensions_loaded:
        return
    _extensions_loaded = True
    for ep in entry_points(group="holosoma.config.inference"):
        DEFAULTS[ep.name] = ep.load()


def get_annotated_inference_config() -> type:
    """Build the annotated InferenceConfig type with all discovered configs.

    This function loads extension configs lazily and returns a tyro-compatible
    annotated type for CLI subcommand generation.

    Returns:
        Annotated type suitable for use with tyro.cli()
    """
    _load_extensions()
    return Annotated[
        InferenceConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"inference:{k}": v for k, v in DEFAULTS.items()}
            )
        ),
    ]


def get_defaults() -> dict:
    """Get all inference config defaults, including extensions.

    Returns:
        Dictionary mapping config names to InferenceConfig instances.
    """
    _load_extensions()
    return DEFAULTS
