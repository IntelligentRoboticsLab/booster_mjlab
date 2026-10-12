"""RL configuration for Booster T1 velocity tasks."""

from booster_mjlab.amp.config import T1_AMP_DATASET
from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import (
    RslRlOnPolicyRunnerCfg,
    RslRlSymmetryCfg,
)
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import (
    booster_k1_amp_ppo_runner_cfg,
    booster_k1_ppo_runner_cfg,
)

_AUGMENTATION_FUNC = (
    "booster_mjlab.tasks.velocity.mdp.observations:augment_symmetries_t1"
)


def _use_t1_symmetry(
    cfg: RslRlOnPolicyRunnerCfg, *, use_data_augmentation: bool
) -> None:
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=use_data_augmentation,
        use_mirror_loss=False,
        data_augmentation_func=_AUGMENTATION_FUNC,
    )


def booster_t1_ppo_runner_cfg(optimizer: str = "adam") -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 velocity task."""
    cfg = booster_k1_ppo_runner_cfg(optimizer=optimizer)
    _use_t1_symmetry(cfg, use_data_augmentation=False)
    cfg.experiment_name = cfg.experiment_name.replace("k1_velocity", "t1_velocity")
    return cfg


def booster_t1_symmetric_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner configuration with symmetry data augmentation."""
    cfg = booster_t1_ppo_runner_cfg(optimizer=optimizer)
    _use_t1_symmetry(cfg, use_data_augmentation=True)
    cfg.experiment_name = cfg.experiment_name.replace(
        "t1_velocity", "t1_velocity_symmetric"
    )
    return cfg


def booster_t1_amp_ppo_runner_cfg(optimizer: str = "adam") -> AmpOnPolicyRunnerCfg:
    """Create AMP runner configuration for Booster T1 velocity task."""
    cfg = booster_k1_amp_ppo_runner_cfg(optimizer=optimizer)
    cfg.dataset_root = T1_AMP_DATASET
    # 0.1 s of history: with 10 frames the discriminator separated expert and policy at 95-98%
    # accuracy (runs 9dzo12dm, u5golm2n).
    cfg.num_amp_obs_steps = 5
    cfg.experiment_name = cfg.experiment_name.replace(
        "k1_velocity_amp", "t1_velocity_amp"
    )
    return cfg


def booster_t1_amp_ppo_symmetric_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """AMP runner configuration with symmetry data augmentation."""
    cfg = booster_t1_amp_ppo_runner_cfg(optimizer=optimizer)
    _use_t1_symmetry(cfg, use_data_augmentation=True)
    cfg.experiment_name = cfg.experiment_name.replace(
        "t1_velocity_amp", "t1_velocity_amp_symmetric"
    )
    return cfg


def booster_t1_amp_ppo_symmetric_history_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Symmetric AMP PPO for the stacked-history actor."""
    cfg = booster_t1_amp_ppo_symmetric_runner_cfg(optimizer=optimizer)
    cfg.experiment_name = cfg.experiment_name.replace(
        "t1_velocity_amp_symmetric", "t1_velocity_amp_symmetric_history"
    )
    return cfg
