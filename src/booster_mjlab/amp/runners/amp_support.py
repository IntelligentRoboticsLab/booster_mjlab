from __future__ import annotations

import torch
from tensordict import TensorDict

from booster_mjlab.amp.curriculums import AMP_STYLE_WEIGHT_ATTR

AMP_STYLE_REWARD_KEY = "Episode_Metrics/Amp/style_reward"
AMP_TASK_REWARD_KEY = "Episode_Metrics/Amp/task_reward"
AMP_STYLE_TASK_RATIO_KEY = "Episode_Metrics/Amp/style_task_ratio"


class AmpRunner:
    """Shared AMP reward/metrics utilities for runners."""

    _amp_enabled: bool = False

    def _setup_amp_support(self, train_cfg: dict, *, require_amp_group: bool) -> bool:
        amp_groups = train_cfg["obs_groups"].get("amp")
        if not amp_groups:
            if require_amp_group:
                raise ValueError("AMP runner requires an 'amp' observation group.")
            self._amp_enabled = False
            return False

        self.style_reward_weight = train_cfg["style_reward_weight"]
        self.amp_group_names = list(amp_groups)

        # Per-environment episode sums for cumulative reward tracking.
        self._amp_style_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.device
        )
        self._amp_task_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float32, device=self.device
        )
        self._amp_metric_log_keys = (
            AMP_STYLE_REWARD_KEY,
            AMP_TASK_REWARD_KEY,
            AMP_STYLE_TASK_RATIO_KEY,
        )
        self._amp_enabled = True
        return True

    def _compute_style_weight(self) -> float:
        env = self.env.unwrapped
        return getattr(env, AMP_STYLE_WEIGHT_ATTR, self.style_reward_weight)

    def _compute_style_reward_scale(self) -> float:
        """Match the environment's reward-rate scaling convention."""
        env = self.env.unwrapped
        return env.step_dt if env.cfg.scale_rewards_by_dt else 1.0

    def _get_amp_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.amp_group_names], dim=-1)

    def _reset_amp_metrics(self) -> None:
        if not self._amp_enabled or self.logger.log_dir is None:
            return
        self._amp_style_reward_sum.zero_()
        self._amp_task_reward_sum.zero_()

    def _amp_collect_step(
        self,
        obs: TensorDict,
        amp_obs: torch.Tensor | None,
    ) -> tuple[TensorDict, torch.Tensor | None, torch.Tensor, torch.Tensor, dict]:
        """One env step with optional AMP reward shaping.

        Returns (obs, amp_obs, rewards, dones, extras) where rewards are
        the final (potentially AMP-combined) rewards.
        """
        actions = self.alg.act(obs)
        if self._amp_enabled:
            self.alg.act_amp(amp_obs)

        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
        obs = obs.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        if self._amp_enabled:
            next_amp_obs = self._get_amp_obs(obs)
            next_amp_history = self.alg.process_amp_step(next_amp_obs, dones)
            style_weight = self._compute_style_weight()
            task_rewards = (1.0 - style_weight) * rewards
            style_rewards = (
                style_weight
                * self._compute_style_reward_scale()
                * self.alg.predict_style_reward(next_amp_history).view_as(rewards)
            )
            rewards = task_rewards + style_rewards
            self._update_amp_metrics_log(
                extras=extras,
                dones=dones,
                task_rewards=task_rewards,
                style_rewards=style_rewards,
            )
            amp_obs = next_amp_obs

        self.alg.process_env_step(obs, rewards, dones, extras)
        return obs, amp_obs, rewards, dones, extras

    def _update_amp_metrics_log(
        self,
        extras: dict,
        dones: torch.Tensor,
        task_rewards: torch.Tensor,
        style_rewards: torch.Tensor,
    ) -> None:
        if not self._amp_enabled or self.logger.log_dir is None:
            return

        # Accumulate per-environment episode sums.
        self._amp_task_reward_sum.add_(task_rewards.view(-1))
        self._amp_style_reward_sum.add_(style_rewards.view(-1))

        log_data = extras.get("log")
        if isinstance(log_data, dict):
            log_data = dict(log_data)
        else:
            log_data = {}

        # Keep AMP metrics write-once per reset event.
        for key in self._amp_metric_log_keys:
            log_data.pop(key, None)

        done_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if done_env_ids.numel() == 0:
            return

        style_sums = self._amp_style_reward_sum[done_env_ids]
        task_sums = self._amp_task_reward_sum[done_env_ids]
        mean_style = style_sums.mean()
        mean_task = task_sums.mean()

        log_data[AMP_STYLE_REWARD_KEY] = mean_style
        log_data[AMP_TASK_REWARD_KEY] = mean_task
        log_data[AMP_STYLE_TASK_RATIO_KEY] = mean_style.item() / (
            abs(mean_task.item()) + 1.0e-8
        )

        # Reset accumulators for done environments.
        self._amp_style_reward_sum[done_env_ids] = 0.0
        self._amp_task_reward_sum[done_env_ids] = 0.0

        extras["log"] = log_data
