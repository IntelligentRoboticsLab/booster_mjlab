"""Reusable termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.terminations import bad_orientation
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def stochastic_bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float,
    probability: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate independently with the given probability while tilted too far.

    ``limit_angle`` is in radians from upright.
    Sample once per environment per control step, only terminating if its current tilt exceeds the limit.
    At probability 0.01, a persistently tilted robot survives an expected 100 steps until termination (including the terminating step).
    Recovering below the limit immediately removes this termination risk; no countdown is retained.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    tilted = bad_orientation(env, limit_angle, asset_cfg)
    return tilted & (torch.rand(tilted.shape, device=tilted.device) < probability)
