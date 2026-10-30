"""RL configuration for Booster K1 velocity task."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.robots.booster_k1.k1_constants import K1_MIRROR
from booster_mjlab.tasks.velocity.config.rl_cfg import (
    booster_amp_ppo_runner_cfg,
    booster_ppo_runner_cfg,
)
from booster_mjlab.tasks.velocity.mdp.observations import make_augment_symmetries

K1_AMP_DATASET = "whirlwind-ams/lafan_locomotion_k1"
"""LAFAN1 locomotion clips retargeted to the K1."""

K1_MOTION_MIRROR = "booster_mjlab.robots.booster_k1.k1_constants:K1_MIRROR"

augment_symmetries = make_augment_symmetries(K1_MIRROR.joint_mirror())
"""Left/right symmetry of the serial-ankle K1's observations and actions."""

_AUGMENTATION_FUNC = f"{__name__}:augment_symmetries"


def booster_k1_ppo_runner_cfg(optimizer: str = "adam") -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster K1 velocity task.

    ``optimizer`` selects the actor/critic optimizer (e.g. ``"muon"``); non-Adam
    optimizers are appended to the experiment name.
    """
    return booster_ppo_runner_cfg(
        "k1_velocity", _AUGMENTATION_FUNC, optimizer=optimizer
    )


def booster_k1_symmetric_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner configuration with symmetry data augmentation."""
    return booster_ppo_runner_cfg(
        "k1_velocity_symmetric",
        _AUGMENTATION_FUNC,
        use_data_augmentation=True,
        optimizer=optimizer,
    )


def booster_k1_amp_ppo_runner_cfg(optimizer: str = "adam") -> AmpOnPolicyRunnerCfg:
    """Create AMP runner configuration for Booster K1 velocity task.

    ``optimizer`` selects the actor/critic optimizer (e.g. ``"muon"``); the
    discriminator keeps its own optimizer.
    """
    return booster_amp_ppo_runner_cfg(
        "k1_velocity_amp", K1_AMP_DATASET, K1_MOTION_MIRROR, optimizer=optimizer
    )


def booster_k1_amp_ppo_symmetric_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """AMP runner configuration with symmetry data augmentation."""
    return booster_amp_ppo_runner_cfg(
        "k1_velocity_amp_symmetric",
        K1_AMP_DATASET,
        K1_MOTION_MIRROR,
        data_augmentation_func=_AUGMENTATION_FUNC,
        optimizer=optimizer,
    )
