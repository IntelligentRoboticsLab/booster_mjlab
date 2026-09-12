"""Booster base on-policy (PPO) runner.

Thin :class:`~mjlab.tasks.velocity.rl.VelocityOnPolicyRunner` subclass that all
booster PPO tasks build on. After ONNX export it replaces mjlab's base metadata
with booster_mjlab's passive-joint-aware version, then appends joint/EE exclusion
metadata used by reduced-action policies.

This is a no-op for tasks that don't exclude anything (see
:func:`~booster_mjlab.rl.exporter_utils.attach_exclusion_metadata`), so it is safe
to use as the universal base for every booster PPO task.
"""

from __future__ import annotations

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from booster_mjlab.rl.exporter_utils import (
    attach_base_metadata,
    attach_exclusion_metadata,
)


class BoosterOnPolicyRunner(VelocityOnPolicyRunner):
    """Base PPO runner with corrected booster_mjlab ONNX metadata."""

    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        _, _, onnx_path = self._get_export_paths(path)
        if onnx_path.exists():
            attach_base_metadata(self.env.unwrapped, str(onnx_path))
            attach_exclusion_metadata(self.env.unwrapped, onnx_path)
