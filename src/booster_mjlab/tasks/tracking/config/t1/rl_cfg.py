"""RL configuration for Booster T1 tracking task."""

from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.tasks.tracking.config.k1.rl_cfg import (
    booster_k1_tracking_ppo_runner_cfg,
)


def booster_t1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 tracking task."""
    cfg = booster_k1_tracking_ppo_runner_cfg()
    cfg.experiment_name = "t1_tracking"
    return cfg
