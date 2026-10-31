"""RL configuration for parallel-ankle Booster K1 velocity tasks."""

from booster_mjlab.amp.runners import AmpOnPolicyRunnerCfg
from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.robots.booster_k1.k1_constants import K1_MIRROR
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    K1_PARALLEL_ACTUATED_JOINTS,
)
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import (
    K1_AMP_DATASET,
    K1_MOTION_MIRROR,
)
from booster_mjlab.tasks.velocity.config.rl_cfg import (
    booster_amp_ppo_runner_cfg,
    booster_ppo_runner_cfg,
)
from booster_mjlab.tasks.velocity.mdp.observations import make_augment_symmetries

augment_symmetries = make_augment_symmetries(
    K1_MIRROR.joint_mirror(K1_PARALLEL_ACTUATED_JOINTS)
)
"""Left/right symmetry over the actuated joints. The A/B crank axes both point along
+y, so unlike serial ankle roll neither crank changes sign under a mirror."""

_AUGMENTATION_FUNC = f"{__name__}:augment_symmetries"

# The serial dataset is retargeted onto the linkage at load time, after
# augmentations (so mirroring still sees the serial layout); clips that
# already carry the parallel layout pass through untouched.
PARALLEL_MOTION_TRANSFORM = (
    "booster_mjlab.robots.booster_k1.parallel_retarget:to_parallel"
)


def booster_k1_parallel_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """Create the base parallel-ankle PPO runner configuration."""
    return booster_ppo_runner_cfg(
        "k1_parallel_velocity", _AUGMENTATION_FUNC, optimizer=optimizer
    )


def booster_k1_parallel_symmetric_ppo_runner_cfg(
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """Create the parallel-ankle PPO config with symmetry augmentation."""
    return booster_ppo_runner_cfg(
        "k1_parallel_velocity_symmetric",
        _AUGMENTATION_FUNC,
        use_data_augmentation=True,
        optimizer=optimizer,
    )


def booster_k1_parallel_amp_ppo_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create the AMP runner using motion retargeted to the parallel linkage."""
    return booster_amp_ppo_runner_cfg(
        "k1_parallel_velocity_amp",
        K1_AMP_DATASET,
        K1_MOTION_MIRROR,
        dataset_transform=PARALLEL_MOTION_TRANSFORM,
        optimizer=optimizer,
    )


def booster_k1_parallel_amp_ppo_symmetric_runner_cfg(
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create parallel AMP PPO with policy symmetry augmentation."""
    return booster_amp_ppo_runner_cfg(
        "k1_parallel_velocity_amp_symmetric",
        K1_AMP_DATASET,
        K1_MOTION_MIRROR,
        data_augmentation_func=_AUGMENTATION_FUNC,
        dataset_transform=PARALLEL_MOTION_TRANSFORM,
        optimizer=optimizer,
    )
