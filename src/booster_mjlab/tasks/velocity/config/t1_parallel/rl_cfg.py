"""RL configuration for parallel-ankle Booster T1 velocity tasks."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.robots.booster_t1.t1_constants import T1_MIRROR
from booster_mjlab.robots.booster_t1.t1_parallel_constants import (
    T1_PARALLEL_ACTUATED_JOINTS,
)
from booster_mjlab.tasks.velocity.config.rl_cfg import booster_ppo_runner_cfg
from booster_mjlab.tasks.velocity.config.t1.rl_cfg import t1_amp_ppo_runner_cfg
from booster_mjlab.tasks.velocity.mdp.observations import make_augment_symmetries

augment_symmetries = make_augment_symmetries(
    T1_MIRROR.joint_mirror(T1_PARALLEL_ACTUATED_JOINTS)
)
"""Left/right symmetry over the actuated joints. The A/B crank axes both point along
+y and map onto the same crank of the other leg, so neither changes sign."""

_AUGMENTATION_FUNC = f"{__name__}:augment_symmetries"

# The serial dataset is retargeted onto the linkage at load time, after
# augmentations (so mirroring still sees the serial layout).
PARALLEL_MOTION_TRANSFORM = (
    "booster_mjlab.robots.booster_t1.parallel_retarget:to_parallel"
)


def booster_t1_parallel_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """Create the base parallel-ankle PPO runner configuration."""
    return booster_ppo_runner_cfg(
        "t1_parallel_velocity", _AUGMENTATION_FUNC, optimizer=optimizer
    )


def booster_t1_parallel_symmetric_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """Create the parallel-ankle PPO config with symmetry augmentation."""
    return booster_ppo_runner_cfg(
        "t1_parallel_velocity_symmetric",
        _AUGMENTATION_FUNC,
        use_data_augmentation=True,
        optimizer=optimizer,
    )


def booster_t1_parallel_amp_ppo_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create the AMP runner using motion retargeted to the parallel linkage."""
    return t1_amp_ppo_runner_cfg(
        "t1_parallel_velocity_amp",
        dataset_transform=PARALLEL_MOTION_TRANSFORM,
        optimizer=optimizer,
    )


def booster_t1_parallel_amp_ppo_symmetric_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create parallel AMP PPO with policy symmetry augmentation."""
    return t1_amp_ppo_runner_cfg(
        "t1_parallel_velocity_amp_symmetric",
        data_augmentation_func=_AUGMENTATION_FUNC,
        dataset_transform=PARALLEL_MOTION_TRANSFORM,
        optimizer=optimizer,
    )


def booster_t1_parallel_amp_ppo_symmetric_history_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create parallel symmetric AMP PPO for the stacked-history actor."""
    return t1_amp_ppo_runner_cfg(
        "t1_parallel_velocity_amp_symmetric_history",
        data_augmentation_func=_AUGMENTATION_FUNC,
        dataset_transform=PARALLEL_MOTION_TRANSFORM,
        optimizer=optimizer,
    )
