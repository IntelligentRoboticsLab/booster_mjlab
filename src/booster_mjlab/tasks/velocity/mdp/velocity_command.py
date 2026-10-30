"""Velocity command — extends the default mjlab implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand as _BaseUniformVelocityCommand,
    UniformVelocityCommandCfg as _BaseUniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformVelocityCommand(_BaseUniformVelocityCommand):
    """mjlab velocity command with motion-aligned reset sampling.

    Motion resets write the sampled pose and velocity before the command manager is reset. On that
    first command sample, a configurable fraction of environments therefore use the freshly written
    root velocity as their command instead of an unrelated uniform sample. Periodic command
    resampling later in the episode remains unchanged.

    The velocity is read directly from qpos/qvel because MuJoCo's derived body velocity buffers are
    not refreshed until the forward pass after reset.
    """

    cfg: UniformVelocityCommandCfg

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._yaw_body_id: int | None = None
        if cfg.yaw_body_name is not None:
            body_ids, _ = self.robot.find_bodies(cfg.yaw_body_name)
            assert len(body_ids) == 1, cfg.yaw_body_name
            self._yaw_body_id = body_ids[0]
        # Low-passed yaw rate of the yaw body for error_vel_yaw, seeded on the first update after a reset.
        self._yaw_rate_filtered = torch.zeros(self.num_envs, device=self.device)
        self._yaw_rate_seeded = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        self._yaw_rate_seeded[slice(None) if env_ids is None else env_ids] = False
        return extras

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        self._align_initial_commands(env_ids)
        self._cap_high_speed_samples(env_ids)
        self._sample_slow_walk_commands(env_ids)
        self._sample_turn_in_place_commands(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if self.cfg.heading_command and self._yaw_body_id is not None:
            # mjlab steers toward the heading target with the root's heading. Use the yaw body's instead, so twisting
            # a body above it (the T1 trunk on its waist) toward the target does not zero the yaw command.
            data = self.robot.data
            forward_w = quat_apply(
                data.body_link_quat_w[:, self._yaw_body_id], data.forward_vec_b
            )
            heading_w = torch.atan2(forward_w[:, 1], forward_w[:, 0])
            self.heading_error = wrap_to_pi(self.heading_target - heading_w)
            heading_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            self.vel_command_b[heading_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * self.heading_error[heading_ids],
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Heading envs recompute their yaw rate every step, clipped only to ranges.ang_vel_z.
        if self.cfg.high_speed_lin_vel_x is None or not self.cfg.heading_command:
            return
        cap = self.cfg.high_speed_ang_vel_z
        fast_heading = self.is_heading_env & (
            self.vel_command_b[:, 0].abs() >= self.cfg.high_speed_lin_vel_x
        )
        self.vel_command_b[fast_heading, 2] = self.vel_command_b[fast_heading, 2].clamp(
            -cap, cap
        )

    def _update_metrics(self) -> None:
        if self._yaw_body_id is None:
            super()._update_metrics()
            return
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        data = self.robot.data
        body_ang_vel_b = quat_apply_inverse(
            data.body_link_quat_w[:, self._yaw_body_id],
            data.body_link_ang_vel_w[:, self._yaw_body_id],
        )
        yaw_rate = body_ang_vel_b[:, 2]
        if self.cfg.yaw_rate_time_constant > 0.0:
            alpha = self._env.step_dt / (
                self.cfg.yaw_rate_time_constant + self._env.step_dt
            )
            self._yaw_rate_filtered = torch.where(
                self._yaw_rate_seeded,
                self._yaw_rate_filtered + alpha * (yaw_rate - self._yaw_rate_filtered),
                yaw_rate,
            )
            self._yaw_rate_seeded[:] = True
            yaw_rate = self._yaw_rate_filtered
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - data.root_link_lin_vel_b[:, :2], dim=-1
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - yaw_rate) / max_command_step
        )

    def _align_initial_commands(self, env_ids: torch.Tensor) -> None:
        if self.cfg.motion_aligned_command_prob <= 0.0 or len(env_ids) == 0:
            return

        # CommandTerm.reset() clears this counter immediately before calling _resample(). A zero
        # counter therefore identifies reset-time samples; periodic resamples have a positive one.
        reset_ids = env_ids[self.command_counter[env_ids] == 0]
        if len(reset_ids) == 0:
            return

        align_mask = (
            torch.rand(len(reset_ids), device=self.device)
            < self.cfg.motion_aligned_command_prob
        )
        align_ids = reset_ids[align_mask]
        if len(align_ids) == 0:
            return

        robot_data = self.robot.data
        qvel_indices = robot_data.indexing.free_joint_v_adr
        qpos_indices = robot_data.indexing.free_joint_q_adr
        root_qvel = robot_data.data.qvel[align_ids][:, qvel_indices]
        root_quat_w = robot_data.data.qpos[align_ids][:, qpos_indices[3:7]]

        # MuJoCo free-joint qvel stores linear velocity in world coordinates and angular velocity
        # in body coordinates.
        lin_vel_b = quat_apply_inverse(root_quat_w, root_qvel[:, :3])
        ang_vel_b = root_qvel[:, 3:6]

        self.vel_command_b[align_ids, 0] = lin_vel_b[:, 0].clamp(
            *self.cfg.ranges.lin_vel_x
        )
        self.vel_command_b[align_ids, 1] = lin_vel_b[:, 1].clamp(
            *self.cfg.ranges.lin_vel_y
        )
        self.vel_command_b[align_ids, 2] = ang_vel_b[:, 2].clamp(
            *self.cfg.ranges.ang_vel_z
        )

        # Prevent the base class's other command modes from overriding the aligned sample in
        # _update_command().
        self.vel_command_w[align_ids] = self.vel_command_b[align_ids]
        self.is_standing_env[align_ids] = False
        self.is_world_env[align_ids] = False
        self.is_forward_env[align_ids] = False
        if self.cfg.heading_command:
            self.is_heading_env[align_ids] = False

    def _cap_high_speed_samples(self, env_ids: torch.Tensor) -> None:
        """Shrink the lateral and yaw samples of fast forward/backward commands to within high_speed_lin_vel_y and
        high_speed_ang_vel_z.

        Scaling instead of clamping keeps both uniform over the capped range rather than piling samples onto its bounds.
        """
        if self.cfg.high_speed_lin_vel_x is None or len(env_ids) == 0:
            return
        fast_ids = env_ids[
            self.vel_command_b[env_ids, 0].abs() >= self.cfg.high_speed_lin_vel_x
        ]
        if len(fast_ids) == 0:
            return
        for axis, cap, sample_range in (
            (1, self.cfg.high_speed_lin_vel_y, self.cfg.ranges.lin_vel_y),
            (2, self.cfg.high_speed_ang_vel_z, self.cfg.ranges.ang_vel_z),
        ):
            bound = max(abs(v) for v in sample_range)
            scale = min(1.0, cap / max(bound, 1e-6))
            self.vel_command_b[fast_ids, axis] *= scale
            self.vel_command_w[fast_ids, axis] = self.vel_command_b[fast_ids, axis]

    def _sample_slow_walk_commands(self, env_ids: torch.Tensor) -> None:
        """Replace a fraction of moving samples with slow walking commands.

        Uniform per-axis ranges almost never produce a planar speed below ~0.4 m/s with little yaw, and the
        nearest commands the policy does see are standing ones, so it learns to stand and lean rather than step
        slowly. These samples draw the planar speed uniformly from slow_lin_vel in a random direction.
        """
        if self.cfg.rel_slow_envs <= 0.0 or len(env_ids) == 0:
            return

        # Standing samples are left alone so rel_standing_envs keeps its meaning.
        candidate_ids = env_ids[~self.is_standing_env[env_ids]]
        slow_mask = (
            torch.rand(len(candidate_ids), device=self.device) < self.cfg.rel_slow_envs
        )
        slow_ids = candidate_ids[slow_mask]
        if len(slow_ids) == 0:
            return

        speed = torch.empty(len(slow_ids), device=self.device).uniform_(
            *self.cfg.slow_lin_vel
        )
        direction = torch.empty(len(slow_ids), device=self.device).uniform_(
            -torch.pi, torch.pi
        )
        self.vel_command_b[slow_ids, 0] = speed * torch.cos(direction)
        self.vel_command_b[slow_ids, 1] = speed * torch.sin(direction)
        # Scale rather than clamp the yaw sample, as in _cap_high_speed_samples.
        yaw_bound = max(abs(v) for v in self.cfg.ranges.ang_vel_z)
        self.vel_command_b[slow_ids, 2] *= min(
            1.0, self.cfg.slow_ang_vel_z / max(yaw_bound, 1e-6)
        )

        self.vel_command_w[slow_ids] = self.vel_command_b[slow_ids]
        self.is_world_env[slow_ids] = False
        self.is_forward_env[slow_ids] = False
        if self.cfg.heading_command:
            self.is_heading_env[slow_ids] = False

    def _sample_turn_in_place_commands(self, env_ids: torch.Tensor) -> None:
        """Replace a fraction of samples with pure yaw commands."""
        if self.cfg.rel_turning_envs <= 0.0 or len(env_ids) == 0:
            return

        turn_mask = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.rel_turning_envs
        )
        turn_ids = env_ids[turn_mask]
        if len(turn_ids) == 0:
            return

        yaw_range = self.cfg.turning_ang_vel_z or self.cfg.ranges.ang_vel_z
        yaw_commands = torch.empty(len(turn_ids), device=self.device).uniform_(
            *yaw_range
        )
        self.vel_command_b[turn_ids, :2] = 0.0
        self.vel_command_b[turn_ids, 2] = yaw_commands

        # Turning in place is a mutually exclusive command mode, applied after reset-velocity
        # alignment so its configured fraction is preserved.
        self.vel_command_w[turn_ids] = self.vel_command_b[turn_ids]
        self.is_standing_env[turn_ids] = False
        self.is_world_env[turn_ids] = False
        self.is_forward_env[turn_ids] = False
        if self.cfg.heading_command:
            self.is_heading_env[turn_ids] = False


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(_BaseUniformVelocityCommandCfg):
    """mjlab velocity command config with reset-time motion alignment."""

    motion_aligned_command_prob: float = 0.0
    """Probability of aligning an initial command with the reset root velocity."""

    rel_turning_envs: float = 0.0
    """Fraction of command samples that request pure turning in place."""

    rel_slow_envs: float = 0.0
    """Fraction of non-standing command samples that request slow walking. Turn-in-place sampling runs
    afterwards and may override some of them."""

    slow_lin_vel: tuple[float, float] = (0.1, 0.5)
    """Planar speed range of slow-walk samples, in a uniformly random direction."""

    slow_ang_vel_z: float = 0.3
    """Maximum |yaw rate| commanded together with a slow-walk sample."""

    turning_ang_vel_z: tuple[float, float] | None = None
    """Yaw-rate range for turn-in-place samples. None samples from ranges.ang_vel_z."""

    high_speed_lin_vel_x: float | None = None
    """Commanded |lin_vel_x| from which lateral and yaw commands are capped to high_speed_lin_vel_y and
    high_speed_ang_vel_z. None disables the cap."""

    high_speed_lin_vel_y: float = 0.5
    """Maximum |lin_vel_y| commanded together with a forward/backward speed of at least high_speed_lin_vel_x."""

    high_speed_ang_vel_z: float = 0.5
    """Maximum |yaw rate| commanded together with a forward/backward speed of at least high_speed_lin_vel_x."""

    yaw_body_name: str | None = None
    """Body that turns for the yaw command: heading commands steer by its heading and the error_vel_yaw metric
    compares its yaw rate to the command. None uses the root, as mjlab does; set it to the body the yaw tracking
    reward measures."""

    yaw_rate_time_constant: float = 0.0
    """Time constant (s) of the first-order low-pass on the yaw body's yaw rate in error_vel_yaw; 0 reports it
    unfiltered. Match the yaw tracking reward's filter. Only used with yaw_body_name."""

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
        return UniformVelocityCommand(self, env)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.motion_aligned_command_prob <= 1.0:
            raise ValueError("motion_aligned_command_prob must be between 0 and 1")
        if not 0.0 <= self.rel_turning_envs <= 1.0:
            raise ValueError("rel_turning_envs must be between 0 and 1")
        if not 0.0 <= self.rel_slow_envs <= 1.0:
            raise ValueError("rel_slow_envs must be between 0 and 1")
