from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.rl.runner import BoosterOnPolicyRunner

from .env_cfgs import booster_k1_standing_env_cfg
from .rl_cfg import booster_k1_standing_ppo_runner_cfg

for use_muon in (False, True):
    optimizer = "-Muon" if use_muon else ""
    register_mjlab_task(
        task_id=f"Mjlab-Standing-Flat{optimizer}-Booster-K1",
        env_cfg=booster_k1_standing_env_cfg(),
        play_env_cfg=booster_k1_standing_env_cfg(play=True),
        rl_cfg=booster_k1_standing_ppo_runner_cfg(use_muon=use_muon),
        runner_cls=BoosterOnPolicyRunner,
    )
