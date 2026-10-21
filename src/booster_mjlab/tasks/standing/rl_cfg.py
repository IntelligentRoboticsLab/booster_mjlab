"""RL configuration for the Booster K1 standing task: the K1 velocity PPO setup."""

from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_ppo_runner_cfg


def booster_k1_standing_ppo_runner_cfg(
    use_muon: bool = False,
) -> RslRlOnPolicyRunnerCfg:
    cfg = booster_k1_ppo_runner_cfg(use_muon=use_muon)
    # the velocity task's symmetry metrics mirror the velocity observation layout
    cfg.algorithm.symmetry_cfg = None
    cfg.experiment_name = "k1_standing" + ("_muon" if use_muon else "")
    cfg.max_iterations = 10_000
    return cfg
