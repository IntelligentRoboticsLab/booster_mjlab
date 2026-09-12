"""RL configuration for parallel-ankle Booster K1 velocity tasks."""

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
    "booster_mjlab.tasks.velocity.mdp.observations:augment_symmetries_parallel"
)

PARALLEL_MOTION_TRANSFORM = (
    "booster_mjlab.robots.booster_k1.parallel_retarget:to_parallel"
)


def _use_parallel_symmetry(
    cfg: RslRlOnPolicyRunnerCfg, *, use_data_augmentation: bool
) -> None:
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=use_data_augmentation,
        use_mirror_loss=False,
        data_augmentation_func=_AUGMENTATION_FUNC,
    )


def booster_k1_parallel_ppo_runner_cfg(
    use_muon: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """Create the base parallel-ankle PPO runner configuration."""
    cfg = booster_k1_ppo_runner_cfg(use_muon=use_muon)
    _use_parallel_symmetry(cfg, use_data_augmentation=False)
    cfg.experiment_name = cfg.experiment_name.replace(
        "k1_velocity", "k1_parallel_velocity"
    )
    return cfg


def booster_k1_parallel_symmetric_ppo_runner_cfg(
    use_muon: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """Create the parallel-ankle PPO config with symmetry augmentation."""
    cfg = booster_k1_parallel_ppo_runner_cfg(use_muon=use_muon)
    _use_parallel_symmetry(cfg, use_data_augmentation=True)
    cfg.experiment_name = cfg.experiment_name.replace(
        "k1_parallel_velocity", "k1_parallel_velocity_symmetric"
    )
    return cfg


def booster_k1_parallel_amp_ppo_runner_cfg(
    use_muon: bool = False,
) -> AmpOnPolicyRunnerCfg:
    """Create the AMP runner using motion retargeted to the parallel linkage."""
    cfg = booster_k1_amp_ppo_runner_cfg(use_muon=use_muon)
    # The serial dataset is retargeted onto the linkage at load time, after
    # augmentations (so mirroring still sees the serial layout); clips that
    # already carry the parallel layout pass through untouched.
    cfg.dataset_transform = PARALLEL_MOTION_TRANSFORM
    cfg.experiment_name = cfg.experiment_name.replace(
        "k1_velocity_amp", "k1_parallel_velocity_amp"
    )
    return cfg


def booster_k1_parallel_amp_ppo_symmetric_runner_cfg(
    use_muon: bool = False,
) -> AmpOnPolicyRunnerCfg:
    """Create parallel AMP PPO with policy symmetry augmentation."""
    cfg = booster_k1_parallel_amp_ppo_runner_cfg(use_muon=use_muon)
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=_AUGMENTATION_FUNC,
    )
    cfg.experiment_name = cfg.experiment_name.replace(
        "k1_parallel_velocity_amp", "k1_parallel_velocity_amp_symmetric"
    )
    return cfg
