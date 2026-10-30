"""RL configuration for Booster T1 velocity tasks."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.robots.booster_t1.t1_constants import T1_MIRROR
from booster_mjlab.tasks.velocity.config.rl_cfg import (
    booster_amp_ppo_runner_cfg,
    booster_ppo_runner_cfg,
)
from booster_mjlab.tasks.velocity.mdp.observations import make_augment_symmetries

T1_AMP_DATASET = "whirlwind-ams/amass_booster_t1"
"""AMASS (ACCAD + CMU) locomotion clips of booster-lab's velocity AMP set, retargeted to the Booster T1."""

T1_MOTION_MIRROR = "booster_mjlab.robots.booster_t1.t1_constants:T1_MIRROR"

augment_symmetries = make_augment_symmetries(T1_MIRROR.joint_mirror())
"""Left/right symmetry of the serial-ankle T1's observations and actions."""

_AUGMENTATION_FUNC = f"{__name__}:augment_symmetries"


def t1_amp_ppo_runner_cfg(
    experiment_name: str,
    *,
    data_augmentation_func: str | None = None,
    dataset_transform: str | None = None,
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """AMP runner on the T1 motion dataset; see ``booster_amp_ppo_runner_cfg``."""
    cfg = booster_amp_ppo_runner_cfg(
        experiment_name,
        T1_AMP_DATASET,
        T1_MOTION_MIRROR,
        data_augmentation_func=data_augmentation_func,
        dataset_transform=dataset_transform,
        optimizer=optimizer,
    )
    # 0.1 s of history: with 10 frames the discriminator separated expert and policy at 95-98%
    # accuracy (runs 9dzo12dm, u5golm2n).
    cfg.num_amp_obs_steps = 5
    return cfg


def booster_t1_ppo_runner_cfg(optimizer: str = "adam") -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 velocity task."""
    return booster_ppo_runner_cfg(
        "t1_velocity", _AUGMENTATION_FUNC, optimizer=optimizer
    )


def booster_t1_symmetric_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner configuration with symmetry data augmentation."""
    return booster_ppo_runner_cfg(
        "t1_velocity_symmetric",
        _AUGMENTATION_FUNC,
        use_data_augmentation=True,
        optimizer=optimizer,
    )


def booster_t1_amp_ppo_runner_cfg(optimizer: str = "adam") -> AmpOnPolicyRunnerCfg:
    """Create AMP runner configuration for Booster T1 velocity task."""
    return t1_amp_ppo_runner_cfg("t1_velocity_amp", optimizer=optimizer)


def booster_t1_amp_ppo_symmetric_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """AMP runner configuration with symmetry data augmentation."""
    return t1_amp_ppo_runner_cfg(
        "t1_velocity_amp_symmetric",
        data_augmentation_func=_AUGMENTATION_FUNC,
        optimizer=optimizer,
    )


def booster_t1_amp_ppo_symmetric_history_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Symmetric AMP PPO for the stacked-history actor."""
    return t1_amp_ppo_runner_cfg(
        "t1_velocity_amp_symmetric_history",
        data_augmentation_func=_AUGMENTATION_FUNC,
        optimizer=optimizer,
    )
