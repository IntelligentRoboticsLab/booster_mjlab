"""Velocity curricula — extends the default mjlab implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, cast

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.velocity.mdp.curriculums import (
    VelocityStage as _BaseVelocityStage,
    commands_vel as _base_commands_vel,
)

from booster_mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class VelocityStage(_BaseVelocityStage):
    turning_ang_vel_z: NotRequired[tuple[float, float] | None]


def commands_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
    """mjlab's command velocity curriculum, also staging the turn-in-place yaw range."""
    log = _base_commands_vel(env, env_ids, command_name, velocity_stages)  # type: ignore[arg-type]
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
    for stage in velocity_stages:
        if env.common_step_counter >= stage["step"]:
            turning = stage.get("turning_ang_vel_z")
            if turning is not None:
                cfg.turning_ang_vel_z = turning
    turning_range = cfg.turning_ang_vel_z or cfg.ranges.ang_vel_z
    log["turning_ang_vel_z_min"] = torch.tensor(turning_range[0])
    log["turning_ang_vel_z_max"] = torch.tensor(turning_range[1])
    return log


class commands_vel_by_tracking(ManagerTermBase):
    """Velocity command curriculum that moves to the next stage once the current one is tracked, not at fixed steps.

    As legged_gym's command curriculum, an ended episode scores its summed ``reward_name`` over the most that term
    can earn in a full episode, so falls and early terminations count as poor tracking. Scores are averaged with a
    per-step decay; the next stage starts when that average reaches ``threshold`` and the current stage has run for
    ``min_stage_steps`` policy steps, or regardless of the score after ``max_stage_steps``, so a stage whose tracking
    plateaus below the threshold cannot stall training. Stages are taken in order and their ``step`` keys are ignored.
    The stage is not checkpointed: a resumed run climbs again from the first stage.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        super().__init__(env)
        self.stage = 0
        self.stage_start_step = env.common_step_counter
        self._last_check_step = env.common_step_counter
        self._score_sum = torch.zeros((), device=env.device)
        self._score_count = torch.zeros((), device=env.device)
        _apply_velocity_stage(
            env, cfg.params["command_name"], cfg.params["velocity_stages"][0]
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        command_name: str,
        velocity_stages: list[VelocityStage],
        reward_name: str,
        threshold: float,
        min_stage_steps: int,
        max_stage_steps: int,
        score_decay: float = 0.995,
        check_interval: int = 24,
    ) -> dict[str, torch.Tensor]:
        if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
            weight = env.reward_manager.get_term_cfg(reward_name).weight
            episode_sum = env.reward_manager._episode_sums[reward_name][env_ids]
            score = episode_sum / (weight * env.max_episode_length_s)
            self._score_sum = score_decay * self._score_sum + score.sum()
            self._score_count = score_decay * self._score_count + len(env_ids)
        mean_score = self._score_sum / self._score_count.clamp_min(1.0)

        # Checking the score syncs with the host, so do it at most once per check_interval steps. The term only runs on
        # steps with resets, so this counts from the last check rather than testing step % check_interval.
        step = env.common_step_counter
        dwell = step - self.stage_start_step
        check = step - self._last_check_step >= check_interval
        if check:
            self._last_check_step = step
        if (
            self.stage + 1 < len(velocity_stages)
            and check
            and (
                dwell >= max_stage_steps
                or (dwell >= min_stage_steps and mean_score.item() >= threshold)
            )
        ):
            self.stage += 1
            self.stage_start_step = step
            # Scores from here on come from episodes that end in the new stage.
            self._score_sum.zero_()
            self._score_count.zero_()
            _apply_velocity_stage(env, command_name, velocity_stages[self.stage])

        cfg = cast(
            UniformVelocityCommandCfg, env.command_manager.get_term(command_name).cfg
        )
        turning_range = cfg.turning_ang_vel_z or cfg.ranges.ang_vel_z
        return {
            "stage": torch.tensor(float(self.stage)),
            "tracking_score": mean_score,
            "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
            "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
            "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
            "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
            "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
            "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
            "turning_ang_vel_z_min": torch.tensor(turning_range[0]),
            "turning_ang_vel_z_max": torch.tensor(turning_range[1]),
        }


def _apply_velocity_stage(
    env: ManagerBasedRlEnv, command_name: str, stage: VelocityStage
) -> None:
    cfg = cast(
        UniformVelocityCommandCfg, env.command_manager.get_term(command_name).cfg
    )
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        value = stage.get(key)
        if value is not None:
            setattr(cfg.ranges, key, value)
    turning = stage.get("turning_ang_vel_z")
    if turning is not None:
        cfg.turning_ang_vel_z = turning
