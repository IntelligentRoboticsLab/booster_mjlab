"""RL runner configuration shared by the Booster tracking tasks."""

from mjlab.rl import RslRlModelCfg
from booster_mjlab.rl.config import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def booster_tracking_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
    """Create the RL runner configuration for a Booster tracking task."""
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
            optimizer="muon",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
