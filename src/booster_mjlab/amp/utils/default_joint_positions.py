from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from booster_mjlab.rl import RslRlVecEnvWrapper


def _resolve_default_joint_positions(
    env: RslRlVecEnvWrapper,
) -> tuple[torch.Tensor, list[int] | None]:
    """Resolve the default joint positions from the environment.

    Returns a tuple of ``(default_joint_positions, joint_indices)`` where
    *joint_indices* are the resolved joint ids used by the AMP ``joint_pos``
    observation term (``None`` when all joints are selected).
    """
    obs_manager = env.unwrapped.observation_manager

    # Assume that the "amp" group is present at this point.
    term_names = obs_manager._group_obs_term_names["amp"]
    term_cfgs = obs_manager._group_obs_term_cfgs["amp"]

    joint_pos_cfg = None
    for name, term_cfg in zip(term_names, term_cfgs):
        if name == "joint_pos":
            joint_pos_cfg = term_cfg
            break

    if joint_pos_cfg is None:
        raise ValueError(
            "Cannot infer default joint positions: 'joint_pos' observation term not found in 'amp' observation group"
        )

    asset_cfg = joint_pos_cfg.params.get("asset_cfg", None)
    if asset_cfg is None:
        raise ValueError(
            "Cannot infer default joint positions: `joint_pos` observation term does not have an `asset_cfg` parameter"
        )

    robot = env.unwrapped.scene.entities["robot"]
    default_joint_pos = robot.data.default_joint_pos[0]
    joint_ids = asset_cfg.joint_ids

    if isinstance(joint_ids, list):
        return default_joint_pos[joint_ids], joint_ids
    return default_joint_pos, None
