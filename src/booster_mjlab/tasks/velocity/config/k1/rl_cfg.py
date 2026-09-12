"""RL configuration for Booster K1 velocity task."""

from mjlab.rl import RslRlModelCfg
from booster_mjlab.rl.config import (
    RslRlMuonPpoAlgorithmCfg,
    RslRlPpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlSymmetryCfg,
)
from booster_mjlab.amp.runners import (
    AmpOnPolicyRunnerCfg,
    AmpDiscriminatorCfg,
    AmpRslRlMuonPpoAlgorithmCfg,
    AmpRslRlPpoAlgorithmCfg,
)


def booster_k1_ppo_runner_cfg(use_muon: bool = False) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster K1 velocity task.

    With ``use_muon`` the actor/critic matrices are optimized with Muon instead of Adam.
    """
    algorithm_cls = RslRlMuonPpoAlgorithmCfg if use_muon else RslRlPpoAlgorithmCfg
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
        algorithm=algorithm_cls(
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
            # only used to calculate metrics
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=False,
                use_mirror_loss=False,
                data_augmentation_func="booster_mjlab.tasks.velocity.mdp.observations:augment_symmetries",
            ),
        ),
        experiment_name="k1_velocity" + ("_muon" if use_muon else ""),
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )


def booster_k1_symmetric_ppo_runner_cfg(
    use_muon: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner configuration with symmetry data augmentation."""
    cfg = booster_k1_ppo_runner_cfg(use_muon=use_muon)
    cfg.algorithm.symmetry_cfg.use_data_augmentation = True
    cfg.experiment_name = cfg.experiment_name.replace(
        "k1_velocity", "k1_velocity_symmetric"
    )
    return cfg


def booster_k1_amp_ppo_runner_cfg(use_muon: bool = False) -> AmpOnPolicyRunnerCfg:
    """Create AMP runner configuration for Booster K1 velocity task.

    With ``use_muon`` the actor/critic matrices are optimized with Muon; the
    discriminator stays on Adam.
    """
    algorithm_cls = AmpRslRlMuonPpoAlgorithmCfg if use_muon else AmpRslRlPpoAlgorithmCfg
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
            reward_clamp_epsilon=1.0e-4,
            loss_type="bce",
            loss_fn_kwargs={},
        ),
        algorithm=algorithm_cls(
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
        ),
        experiment_name="k1_velocity_amp" + ("_muon" if use_muon else ""),
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
        speed_factor=1.0,
        dataset_augmentations=[
            {"name": "mirror"},
            {"name": "speed", "percent": 10.0},
            {"name": "speed", "percent": -10.0},
            {"name": "speed", "percent": 20.0},
            {"name": "speed", "percent": -20.0},
        ],
        style_reward_weight=0.3,
    )


def booster_k1_amp_ppo_symmetric_runner_cfg(
    use_muon: bool = False,
) -> AmpOnPolicyRunnerCfg:
    """AMP runner configuration with symmetry data augmentation."""
    cfg = booster_k1_amp_ppo_runner_cfg(use_muon=use_muon)
    cfg.algorithm.symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func="booster_mjlab.tasks.velocity.mdp.observations:augment_symmetries",
    )
    cfg.experiment_name = cfg.experiment_name.replace(
        "k1_velocity_amp", "k1_velocity_amp_symmetric"
    )
    return cfg
