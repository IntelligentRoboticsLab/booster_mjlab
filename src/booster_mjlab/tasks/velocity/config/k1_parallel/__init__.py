"""Velocity tasks for the parallel-linkage Booster K1."""

from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.config import with_amp_obs_group
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
    booster_k1_parallel_flat_env_cfg,
    booster_k1_parallel_rough_env_cfg,
    with_parallel_amp_joints,
)
from .rl_cfg import (
    booster_k1_parallel_amp_ppo_runner_cfg,
    booster_k1_parallel_amp_ppo_symmetric_runner_cfg,
    booster_k1_parallel_ppo_runner_cfg,
    booster_k1_parallel_symmetric_ppo_runner_cfg,
)


def _with_amp_reset_cfg(env_cfg, runner_cfg):
    return with_parallel_amp_joints(
        with_amp_obs_group(
            env_cfg,
            dataset_root=runner_cfg.dataset_root,
            speed_factor=runner_cfg.speed_factor,
            dataset_weights=runner_cfg.dataset_weights,
            augmentations=runner_cfg.dataset_augmentations,
            dataset_transform=runner_cfg.dataset_transform,
        )
    )


# Task name -> (env cfg factory, runner cfg factory). Every entry is registered
# twice: once with Adam and once with Muon (``-Muon`` inserted before ``-Booster``).
_PPO_TASKS = {
    "Rough": (booster_k1_parallel_rough_env_cfg, booster_k1_parallel_ppo_runner_cfg),
    "Flat": (booster_k1_parallel_flat_env_cfg, booster_k1_parallel_ppo_runner_cfg),
    "Rough-DA": (
        booster_k1_parallel_rough_env_cfg,
        booster_k1_parallel_symmetric_ppo_runner_cfg,
    ),
    "Flat-DA": (
        booster_k1_parallel_flat_env_cfg,
        booster_k1_parallel_symmetric_ppo_runner_cfg,
    ),
}
_AMP_TASKS = {
    "Rough-Amp": (
        booster_k1_parallel_rough_env_cfg,
        booster_k1_parallel_amp_ppo_runner_cfg,
    ),
    "Flat-Amp": (
        booster_k1_parallel_flat_env_cfg,
        booster_k1_parallel_amp_ppo_runner_cfg,
    ),
    "Flat-Amp-DA": (
        booster_k1_parallel_flat_env_cfg,
        booster_k1_parallel_amp_ppo_symmetric_runner_cfg,
    ),
}

for use_muon in (False, True):
    optimizer = "-Muon" if use_muon else ""
    for name, (env_cfg_fn, rl_cfg_fn) in _PPO_TASKS.items():
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-{name}{optimizer}-Booster-K1-Parallel",
            env_cfg=env_cfg_fn(),
            play_env_cfg=env_cfg_fn(play=True),
            rl_cfg=rl_cfg_fn(use_muon=use_muon),
            runner_cls=BoosterOnPolicyRunner,
        )
    for name, (env_cfg_fn, rl_cfg_fn) in _AMP_TASKS.items():
        rl_cfg = rl_cfg_fn(use_muon=use_muon)
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-{name}{optimizer}-Booster-K1-Parallel",
            env_cfg=_with_amp_reset_cfg(env_cfg_fn(), rl_cfg),
            play_env_cfg=_with_amp_reset_cfg(env_cfg_fn(play=True), rl_cfg),
            rl_cfg=rl_cfg,
            runner_cls=VelocityAmpOnPolicyRunner,
        )
