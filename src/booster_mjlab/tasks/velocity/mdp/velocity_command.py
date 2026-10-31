"""Velocity command — extends the default mjlab implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand as _BaseUniformVelocityCommand,
    UniformVelocityCommandCfg as _BaseUniformVelocityCommandCfg,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformVelocityCommand(_BaseUniformVelocityCommand):
    """Default mjlab velocity command."""

    cfg: UniformVelocityCommandCfg


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(_BaseUniformVelocityCommandCfg):
    """Default mjlab velocity command config."""

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
        return UniformVelocityCommand(self, env)
