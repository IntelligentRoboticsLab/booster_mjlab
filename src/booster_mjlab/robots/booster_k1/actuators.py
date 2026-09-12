"""Shared per-motor configuration."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.actuator import BuiltinPositionActuatorCfg

##
# Per-motor configuration.
##


@dataclass(frozen=True, kw_only=True)
class ActuatorConfig:
    """Properties of a single actuator/motor kind."""

    armature: float
    """Motor reflected inertia / armature (kg*m^2)."""

    effort_limit: float
    """Peak torque (N*m). Available in full only below ``knee_point_velocity``."""

    velocity_limit: float | None = None
    """No-load speed (rad/s), or None for a constant torque limit."""

    knee_point_velocity: float | None = None
    """Start of linear torque derating (rad/s); defaults to no-load speed."""

    stiffness: float
    """PD position gain (N*m/rad)."""

    damping: float
    """PD velocity gain (N*m*s/rad)."""

    @property
    def effective_knee_point_velocity(self) -> float | None:
        """Clamp the knee speed to the motor speed range."""
        if self.velocity_limit is None:
            return None
        if self.knee_point_velocity is None:
            return self.velocity_limit
        return min(max(self.knee_point_velocity, 0.0), self.velocity_limit)


@dataclass(kw_only=True)
class MotorPositionActuatorCfg(BuiltinPositionActuatorCfg):
    """Implicit position actuator with motor torque-speed metadata."""

    motor: ActuatorConfig
