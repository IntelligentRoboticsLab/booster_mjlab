"""Batched physics-step torque limits with MuJoCo's implicit position control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import Entity, EntityCfg
from mjlab.managers.event_manager import requires_model_fields

from booster_mjlab.robots.booster_k1.actuators import MotorPositionActuatorCfg

if TYPE_CHECKING:
    from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sim.sim import ModelBridge


@dataclass(kw_only=True)
class BoosterPositionActuatorCfg(MotorPositionActuatorCfg):
    """Builtin position actuator whose motor limits are applied by BoosterEntity."""

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> BuiltinPositionActuator:
        if not isinstance(entity, BoosterEntity):
            raise TypeError("BoosterPositionActuatorCfg requires BoosterEntityCfg")
        return super().build(entity, target_ids, target_names)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transmission_type != TransmissionType.JOINT:
            raise ValueError("Booster torque-speed limits require joint transmission")
        if self.effort_limit is None:
            raise ValueError("Booster torque-speed limits require effort_limit")


class TorqueSpeedLimits:
    """Precompute per-joint constants and update all motor groups together."""

    def __init__(self, actuators: list[BuiltinPositionActuator]) -> None:
        self.target_ids = torch.cat([a.target_ids for a in actuators])
        self.ctrl_ids = torch.cat([a.global_ctrl_ids for a in actuators])
        peaks, speeds, widths = [], [], []
        for actuator in actuators:
            cfg = actuator.cfg
            assert isinstance(cfg, BoosterPositionActuatorCfg)
            assert cfg.effort_limit is not None
            speed = cfg.motor.velocity_limit
            peak = cfg.effort_limit
            if speed is not None and speed <= 0:
                peak, speed, width = 0.0, 0.0, 1.0
            elif speed is None or not math.isfinite(speed):
                speed, width = float("inf"), 1.0
            else:
                knee = cfg.motor.effective_knee_point_velocity
                assert knee is not None
                width = max(speed - knee, 1e-6)
            count = actuator.target_ids.numel()
            peaks.extend([peak] * count)
            speeds.extend([speed] * count)
            widths.extend([width] * count)
        device = self.target_ids.device
        self.peak = torch.tensor(peaks, device=device)
        self.speed = torch.tensor(speeds, device=device)
        self.width = torch.tensor(widths, device=device)

    def max_effort(self, velocity: torch.Tensor) -> torch.Tensor:
        # Divide before multiplying so disabled motors cannot produce 0 * infinity.
        fraction = ((self.speed - velocity.abs()) / self.width).clamp(0.0, 1.0)
        return self.peak * fraction

    def update(self, entity: Entity, model: ModelBridge) -> None:
        limit = self.max_effort(entity.data.joint_vel[:, self.target_ids])
        # Read through the bridge: model expansion may replace the backing array.
        model.actuator_forcerange[:, self.ctrl_ids, :] = torch.stack(
            (-limit, limit), -1
        )


@dataclass
class BoosterEntityCfg(EntityCfg):
    """Entity with a batched torque-speed update before every physics step."""

    def build(self) -> BoosterEntity:
        return BoosterEntity(self)


class BoosterEntity(Entity):
    _torque_speed_limits: TorqueSpeedLimits | None = None
    _torque_speed_model: ModelBridge | None = None

    def write_data_to_sim(self) -> None:
        if self._torque_speed_limits is None or self._torque_speed_model is None:
            raise RuntimeError(
                "BoosterEntity requires initialize_torque_speed_limits as a startup event"
            )
        super().write_data_to_sim()
        self._torque_speed_limits.update(self, self._torque_speed_model)


@requires_model_fields("actuator_forcerange")
def initialize_torque_speed_limits(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
    """Bind one updater per entity after mjlab expands model force ranges."""
    del env_ids
    for entity in env.scene.entities.values():
        actuators = [
            a for a in entity.actuators if isinstance(a.cfg, BoosterPositionActuatorCfg)
        ]
        if not actuators:
            continue
        if not isinstance(entity, BoosterEntity):
            raise TypeError("BoosterPositionActuatorCfg requires BoosterEntityCfg")
        entity._torque_speed_limits = TorqueSpeedLimits(actuators)
        entity._torque_speed_model = env.sim.model
