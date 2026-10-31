"""RL runner configurations shared by the Booster velocity tasks."""

from mjlab.rl import RslRlModelCfg
from booster_mjlab.rl.config import (
    RslRlPpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlSymmetryCfg,
)
from booster_mjlab.amp.runners import (
    AmpOnPolicyRunnerCfg,
    AmpDiscriminatorCfg,
    AmpRslRlPpoAlgorithmCfg,
)


def _experiment_suffix(optimizer: str) -> str:
    return "" if optimizer == "adam" else f"_{optimizer}"


def booster_ppo_runner_cfg(
    experiment_name: str,
    data_augmentation_func: str,
    *,
    use_data_augmentation: bool = False,
    optimizer: str = "adam",
) -> RslRlOnPolicyRunnerCfg:
    """Create the PPO runner configuration for a Booster velocity task.

    Args:
      experiment_name: Experiment name; non-Adam optimizers are appended to it.
      data_augmentation_func: ``module:attr`` of the robot's symmetry augmentation.
        Without ``use_data_augmentation`` it only computes symmetry metrics.
      use_data_augmentation: Train on mirrored samples as well.
      optimizer: The actor/critic optimizer (e.g. ``"muon"``).
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            optimizer=optimizer,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=use_data_augmentation,
                use_mirror_loss=False,
                data_augmentation_func=data_augmentation_func,
            ),
        ),
        experiment_name=experiment_name + _experiment_suffix(optimizer),
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )


def booster_amp_ppo_runner_cfg(
    experiment_name: str,
    dataset_root: str,
    mirror: str,
    *,
    data_augmentation_func: str | None = None,
    dataset_transform: str | None = None,
    optimizer: str = "adam",
) -> AmpOnPolicyRunnerCfg:
    """Create the AMP runner configuration for a Booster velocity task.

    Args:
      experiment_name: Experiment name; non-Adam optimizers are appended to it.
      dataset_root: The robot's motion dataset.
      mirror: ``module:attr`` of the robot's ``Mirror``, for the mirrored motions.
      data_augmentation_func: ``module:attr`` of the robot's symmetry augmentation,
        to also train the policy on mirrored samples.
      dataset_transform: ``module:attr`` remapping each clip onto the robot's joint
        layout, applied after the motion augmentations.
      optimizer: The actor/critic optimizer (e.g. ``"muon"``); the discriminator
        keeps its own optimizer.
    """
    return AmpOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": 1.0,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        discriminator=AmpDiscriminatorCfg(
            hidden_layer_sizes=(256, 128),
            reward_scale=1.0,
            loss_type="bce",
            loss_fn_kwargs={},
        ),
        algorithm=AmpRslRlPpoAlgorithmCfg(
            optimizer=optimizer,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=None
            if data_augmentation_func is None
            else RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=False,
                data_augmentation_func=data_augmentation_func,
            ),
        ),
        experiment_name=experiment_name + _experiment_suffix(optimizer),
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
        dataset_root=dataset_root,
        dataset_transform=dataset_transform,
        speed_factor=1.0,
        dataset_augmentations=[
            {"name": "mirror", "mirror": mirror},
            {"name": "speed", "percent": 10.0},
            {"name": "speed", "percent": -10.0},
            {"name": "speed", "percent": 20.0},
            {"name": "speed", "percent": -20.0},
        ],
        style_reward_weight=0.3,
    )
