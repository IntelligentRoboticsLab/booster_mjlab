"""Rewards, observations and curriculum of the K1 standing task (BDX-style imitation of a reference)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from booster_mjlab.tasks.standing.commands import StandCommand, quat_error

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _cmd(env: ManagerBasedRlEnv, command_name: str) -> StandCommand:
    return env.command_manager.get_term(command_name)


def _joint_ids(env: ManagerBasedRlEnv, pattern_ids: str) -> torch.Tensor:
    return torch.tensor(
        env.scene["robot"].find_joints(pattern_ids)[0], device=env.device
    )


# ---- rewards ----


def torso_height(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    err = env.scene["robot"].data.root_link_pos_w[:, 2] - cmd.ref_torso_height
    return torch.exp(-err.square() / std**2)


def torso_orientation(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    err = quat_error(env.scene["robot"].data.root_link_quat_w, cmd.ref_torso_quat)
    return torch.exp(-err.square() / std**2)


def joint_pos_tracking(
    env: ManagerBasedRlEnv, command_name: str, joint_names: str, std: float
) -> torch.Tensor:
    """exp(-mean squared error / std^2) between the joints and the reference pose."""
    cmd = _cmd(env, command_name)
    ids = _joint_ids(env, joint_names)
    err = env.scene["robot"].data.joint_pos[:, ids] - cmd.ref_joint_pos[:, ids]
    return torch.exp(-err.square().mean(-1) / std**2)


def feet_planted(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Squared linear speed of both feet: standing means the feet don't move."""
    cmd = _cmd(env, command_name)
    v = env.scene["robot"].data.body_link_lin_vel_w[:, cmd.foot_ids]
    return v.square().sum(dim=(-1, -2))


def both_feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    found = env.scene[sensor_name].data.found
    return (found > 0).float().mean(-1)


# ---- observations (critic) ----


def ref_joint_pos_rel(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    return cmd.ref_joint_pos - env.scene["robot"].data.default_joint_pos


def torso_tracking_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = _cmd(env, command_name)
    data = env.scene["robot"].data
    return torch.stack(
        [
            data.root_link_pos_w[:, 2] - cmd.ref_torso_height,
            quat_error(data.root_link_quat_w, cmd.ref_torso_quat),
        ],
        dim=-1,
    )


# ---- curriculum ----


def stand_range_ramp(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start: float,
    ramp_steps: int,
) -> dict[str, torch.Tensor]:
    """Command ranges grow from `start` to the full range over ramp_steps env steps: the policy learns to
    stand before it learns to bow 54 degrees."""
    del env_ids
    a = min(1.0, start + (1.0 - start) * env.common_step_counter / max(ramp_steps, 1))
    _cmd(env, command_name).range_scale = a
    return {"range_scale": torch.tensor(a)}
