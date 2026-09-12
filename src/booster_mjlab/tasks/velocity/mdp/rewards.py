from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    std_at_rest: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    relative_std: float = 0.75,
    progress_weight: float = 0.0,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero. The progress term, when
    enabled, still scores the xy error only.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error_sq = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error_sq = torch.square(actual[:, 2])

    cmd_norm = torch.norm(command[:, :2], dim=1)
    sigma = torch.clamp(relative_std * cmd_norm, min=std_at_rest, max=std)
    tracking = torch.exp(-(xy_error_sq + z_error_sq) / torch.square(sigma))
    if progress_weight <= 0.0:
        return tracking

    xy_error = torch.sqrt(xy_error_sq)
    progress = torch.clamp(
        1.0 - xy_error / torch.clamp(cmd_norm, min=std_at_rest), 0.0, 1.0
    )
    return (1.0 - progress_weight) * tracking + progress_weight * progress


def standing_pose_deviation_l1(
    env: "ManagerBasedRlEnv",
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """L1 joint-position deviation from default, active only when commanded to stand."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    deviation = torch.mean(torch.abs(current - default), dim=1)  # [B]

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed
    standing_scale = 1.0 - (total_speed / command_threshold).clamp(max=1.0)

    return deviation * standing_scale


def variable_upright(
    env: "ManagerBasedRlEnv",
    command_name: str,
    std_standing: float,
    std_walking: float | None = None,
    std_running: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    walking_threshold: float = 0.05,
    running_threshold: float = 1.5,
) -> torch.Tensor:
    """Reward for keeping the base upright, with a velocity-command-dependent tolerance.

    Same tilt measure as mjlab's ``upright`` but the std follows the three speed regimes.
    """
    if std_walking is None:
        std_walking = std_standing
    if std_running is None:
        std_running = std_walking

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
        (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()
    std = (
        std_standing * standing_mask
        + std_walking * walking_mask
        + std_running * running_mask
    )

    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
        body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
        body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / (std**2))


class upper_body_posture_penalty:
    """Penalize upper-body joint deviation from the default pose."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        self.joint_names = joint_names

        stds = {}
        for regime in ("std_standing", "std_walking", "std_running"):
            _, _, values = resolve_matching_names_values(
                data=cfg.params[regime],
                list_of_strings=joint_names,
            )
            stds[regime] = torch.tensor(values, device=env.device, dtype=torch.float32)
        self.std_standing = stds["std_standing"]
        self.std_walking = stds["std_walking"]
        self.std_running = stds["std_running"]

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command '{command_name}' not found."

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current - desired)
        return torch.mean(error_squared / (std**2), dim=1)
