"""RL configuration for parallel-ankle Booster T1 velocity tasks."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.tasks.velocity.config.t1.rl_cfg import (
    booster_t1_amp_ppo_runner_cfg,
    booster_t1_amp_ppo_symmetric_history_runner_cfg,
    booster_t1_amp_ppo_symmetric_runner_cfg,
    booster_t1_ppo_runner_cfg,
    booster_t1_symmetric_ppo_runner_cfg,
)

_AUGMENTATION_FUNC = (
    "booster_mjlab.tasks.velocity.mdp.observations:augment_symmetries_t1_parallel"
)

PARALLEL_MOTION_TRANSFORM = (
    "booster_mjlab.robots.booster_t1.parallel_retarget:to_parallel"
)


def _to_parallel(cfg, experiment_prefix: str = "t1_"):
    """Point a serial T1 runner cfg at the parallel layout's symmetry and experiment name."""
    if cfg.algorithm.symmetry_cfg is not None:
        cfg.algorithm.symmetry_cfg.data_augmentation_func = _AUGMENTATION_FUNC
    cfg.experiment_name = cfg.experiment_name.replace(
        experiment_prefix, "t1_parallel_", 1
    )
    if isinstance(cfg, AmpOnPolicyRunnerCfg):
        # The serial dataset is retargeted onto the linkage at load time, after
        # augmentations (so mirroring still sees the serial layout).
        cfg.dataset_transform = PARALLEL_MOTION_TRANSFORM
    return cfg


def booster_t1_parallel_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """Create the base parallel-ankle PPO runner configuration."""
    return _to_parallel(booster_t1_ppo_runner_cfg(optimizer=optimizer))


def booster_t1_parallel_symmetric_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """Create the parallel-ankle PPO config with symmetry augmentation."""
    return _to_parallel(booster_t1_symmetric_ppo_runner_cfg(optimizer=optimizer))


def booster_t1_parallel_amp_ppo_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create the AMP runner using motion retargeted to the parallel linkage."""
    return _to_parallel(booster_t1_amp_ppo_runner_cfg(optimizer=optimizer))


def booster_t1_parallel_amp_ppo_symmetric_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create parallel AMP PPO with policy symmetry augmentation."""
    return _to_parallel(booster_t1_amp_ppo_symmetric_runner_cfg(optimizer=optimizer))


def booster_t1_parallel_amp_ppo_symmetric_history_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create parallel symmetric AMP PPO for the stacked-history actor."""
    return _to_parallel(
        booster_t1_amp_ppo_symmetric_history_runner_cfg(optimizer=optimizer)
    )
