"""Velocity tasks for the Booster T1."""

from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.config._amp_wrapper import with_amp_obs_group
from booster_mjlab.tasks.velocity.rl.runner import VelocityAmpOnPolicyRunner

from .env_cfgs import (
    T1_AMP_JOINT_NAMES,
    booster_t1_flat_env_cfg,
    booster_t1_flat_history_env_cfg,
    booster_t1_rough_env_cfg,
)
from .rl_cfg import (
    booster_t1_amp_ppo_runner_cfg,
    booster_t1_amp_ppo_symmetric_history_runner_cfg,
    booster_t1_amp_ppo_symmetric_runner_cfg,
    booster_t1_ppo_runner_cfg,
    booster_t1_symmetric_ppo_runner_cfg,
)


def _with_amp_reset_cfg(env_cfg, runner_cfg):
    return with_amp_obs_group(
        env_cfg,
        dataset_root=runner_cfg.dataset_root,
        speed_factor=runner_cfg.speed_factor,
        dataset_weights=runner_cfg.dataset_weights,
        augmentations=runner_cfg.dataset_augmentations,
        include_base_lin_vel=True,
        amp_joint_names=T1_AMP_JOINT_NAMES,
    )


# Task name -> (env cfg factory, runner cfg factory). Every entry is registered
# twice: once with Adam and once with Muon (``-Muon`` inserted before ``-Booster``).
_PPO_TASKS = {
    "Rough": (booster_t1_rough_env_cfg, booster_t1_ppo_runner_cfg),
    "Flat": (booster_t1_flat_env_cfg, booster_t1_ppo_runner_cfg),
    "Rough-DA": (booster_t1_rough_env_cfg, booster_t1_symmetric_ppo_runner_cfg),
    "Flat-DA": (booster_t1_flat_env_cfg, booster_t1_symmetric_ppo_runner_cfg),
}
_AMP_TASKS = {
    "Rough-Amp": (booster_t1_rough_env_cfg, booster_t1_amp_ppo_runner_cfg),
    "Flat-Amp": (booster_t1_flat_env_cfg, booster_t1_amp_ppo_runner_cfg),
    "Flat-Amp-DA": (booster_t1_flat_env_cfg, booster_t1_amp_ppo_symmetric_runner_cfg),
    "Flat-Amp-DA-History": (
        booster_t1_flat_history_env_cfg,
        booster_t1_amp_ppo_symmetric_history_runner_cfg,
    ),
}

for optimizer, task_suffix in (("adam", ""), ("muon", "-Muon")):
    for name, (env_cfg_fn, rl_cfg_fn) in _PPO_TASKS.items():
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-{name}{task_suffix}-Booster-T1",
            env_cfg=env_cfg_fn(),
            play_env_cfg=env_cfg_fn(play=True),
            rl_cfg=rl_cfg_fn(optimizer=optimizer),
            runner_cls=BoosterOnPolicyRunner,
        )
    for name, (env_cfg_fn, rl_cfg_fn) in _AMP_TASKS.items():
        rl_cfg = rl_cfg_fn(optimizer=optimizer)
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-{name}{task_suffix}-Booster-T1",
            env_cfg=_with_amp_reset_cfg(env_cfg_fn(), rl_cfg),
            play_env_cfg=_with_amp_reset_cfg(env_cfg_fn(play=True), rl_cfg),
            rl_cfg=rl_cfg,
            runner_cls=VelocityAmpOnPolicyRunner,
        )
