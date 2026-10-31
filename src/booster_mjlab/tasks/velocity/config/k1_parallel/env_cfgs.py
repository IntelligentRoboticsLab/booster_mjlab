"""Booster K1 velocity configs for the parallel (closed-loop) ankle model."""

from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    K1_PARALLEL_ACTION_SCALE,
    K1_PARALLEL_ACTUATED_JOINTS,
    K1_PARALLEL_FREE_RESET_JOINTS,
    get_k1_parallel_robot_cfg,
)
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import (
    booster_k1_flat_env_cfg,
    booster_k1_rough_env_cfg,
)
from booster_mjlab.tasks.velocity.config.parallel_ankle import (
    convert_to_parallel_ankle,
)
from mjlab.envs import ManagerBasedRlEnvCfg

# Joints the AMP discriminator scores. The passive ankle DOFs are kept (rather
# than the cranks) so the style features stay identical to the serial task.
AMP_JOINT_NAMES = (
    r".*_Hip_.*",
    r".*_Knee_.*",
    r".*_Ankle_Pitch",
    r".*_Ankle_Roll",
)


def make_parallel_ankle_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Convert a serial-ankle K1 velocity config to the parallel-ankle model."""
    return convert_to_parallel_ankle(
        cfg,
        get_k1_parallel_robot_cfg(),
        K1_PARALLEL_ACTION_SCALE,
        K1_PARALLEL_ACTUATED_JOINTS,
        K1_PARALLEL_FREE_RESET_JOINTS,
    )


def booster_k1_parallel_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Parallel-ankle K1 rough terrain velocity tracking configuration."""
    cfg = booster_k1_rough_env_cfg(play=play)
    return make_parallel_ankle_cfg(cfg)


def booster_k1_parallel_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Parallel-ankle K1 flat terrain velocity tracking configuration."""
    cfg = booster_k1_flat_env_cfg(play=play)
    return make_parallel_ankle_cfg(cfg)
