"""booster_mjlab shared MDP observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_vel_filtered(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Joint velocity estimated via low-pass filtered position differences.

    Matches the real robot's velocity sensor behavior:
      raw     = (current_pos - prev_pos) / step_dt
      filtered = (1 - alpha) * prev_filtered + alpha * raw

    On episode reset the filter state is re-initialized to zero velocity,
    so the first observation of each episode is always zero.

    Args:
        env: The RL environment.
        asset_cfg: Configuration identifying the robot entity and joints.
        alpha: Low-pass filter weight for the new sample.  Default 0.25
            corresponds to the observed real-robot sensor behavior
            (75 % old value, 25 % new value).

    Returns:
        Filtered joint velocity relative to default, shape (num_envs, num_joints).
    """
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    current_pos = asset.data.joint_pos[:, jnt_ids]  # (num_envs, num_joints)

    state_key = f"_jvf_{asset_cfg.name}"
    if not hasattr(env, state_key):
        setattr(
            env,
            state_key,
            {
                "prev_pos": current_pos.clone(),
                "filtered": torch.zeros_like(current_pos),
            },
        )

    state = getattr(env, state_key)

    # Re-initialise state for environments that just reset.
    # reset_buf is not set during the initial _prepare_terms() probe call.
    reset_mask = getattr(env, "reset_buf", None)
    if reset_mask is not None and reset_mask.any():
        state["prev_pos"][reset_mask] = current_pos[reset_mask]
        state["filtered"][reset_mask] = 0.0

    # Estimate raw velocity from finite difference.
    raw_vel = (current_pos - state["prev_pos"]) / env.step_dt

    # Apply the low-pass filter in-place.
    state["filtered"].mul_(1.0 - alpha).add_(alpha * raw_vel)

    # Advance the position buffer.
    state["prev_pos"].copy_(current_pos)

    default_joint_vel = asset.data.default_joint_vel
    assert default_joint_vel is not None
    return state["filtered"] - default_joint_vel[:, jnt_ids]
