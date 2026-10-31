"""Booster T1 velocity configs for the parallel (closed-loop) ankle model."""

from mjlab.envs import ManagerBasedRlEnvCfg

from booster_mjlab.robots.booster_t1.t1_parallel_constants import (
    T1_PARALLEL_ACTION_SCALE,
    T1_PARALLEL_ACTUATED_JOINTS,
    T1_PARALLEL_FREE_RESET_JOINTS,
    get_t1_parallel_robot_cfg,
)
from booster_mjlab.tasks.velocity.config.parallel_ankle import (
    convert_to_parallel_ankle,
)
from booster_mjlab.tasks.velocity.config.t1.env_cfgs import (
    booster_t1_flat_env_cfg,
    booster_t1_rough_env_cfg,
    with_actor_history,
)

# Joints the AMP discriminator scores: `T1_AMP_JOINT_NAMES` with the ankle narrowed
# to the passive pitch/roll DOFs, so the style features stay identical to the
# serial task (`.*_Ankle_.*` would also match the cranks and rods).
T1_PARALLEL_AMP_JOINT_NAMES = (
    r"Waist",
    r".*_Hip_.*",
    r".*_Knee_.*",
    r".*_Ankle_Pitch",
    r".*_Ankle_Roll",
    r".*_Shoulder_.*",
    r".*_Elbow_.*",
)


def make_t1_parallel_cfg(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Convert a serial-ankle T1 velocity config to the parallel-ankle model."""
    return convert_to_parallel_ankle(
        cfg,
        get_t1_parallel_robot_cfg(),
        T1_PARALLEL_ACTION_SCALE,
        T1_PARALLEL_ACTUATED_JOINTS,
        T1_PARALLEL_FREE_RESET_JOINTS,
    )


def booster_t1_parallel_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Parallel-ankle T1 rough terrain velocity tracking configuration."""
    return make_t1_parallel_cfg(booster_t1_rough_env_cfg(play=play))


def booster_t1_parallel_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Parallel-ankle T1 flat terrain velocity tracking configuration."""
    return make_t1_parallel_cfg(booster_t1_flat_env_cfg(play=play))


def booster_t1_parallel_flat_history_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Flat parallel-ankle T1 task whose actor sees the last ``T1_ACTOR_HISTORY_LEN`` observation frames."""
    return with_actor_history(booster_t1_parallel_flat_env_cfg(play=play))
