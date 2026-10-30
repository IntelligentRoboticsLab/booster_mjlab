"""RL configuration for Booster T1 tracking task."""

from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.tasks.tracking.config.rl_cfg import booster_tracking_ppo_runner_cfg


def booster_t1_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Booster T1 tracking task."""
    return booster_tracking_ppo_runner_cfg("t1_tracking")
