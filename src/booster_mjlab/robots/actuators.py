"""Shared per-motor configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity.entity import EntityArticulationInfoCfg

##
# Per-motor configuration.
##


def pd_gains(
    inertia: float, natural_freq: float, damping_ratio: float
) -> dict[str, float]:
    """booster_train's PD gains for a joint moving ``inertia`` (kg*m^2).

    k = I w^2 and d = 2 zeta w I, with w = 2 pi ``natural_freq``.
    """
    omega = 2.0 * math.pi * natural_freq
    return dict(
        stiffness=inertia * omega**2, damping=2.0 * damping_ratio * omega * inertia
    )


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


##
# Joint actuators.
##

# Every Booster articulation gets the same scale rule for the joint position
# action: 0.25 * effort_limit / stiffness per joint group.
ACTION_SCALE_FACTOR = 0.25

# Shared delay parameters for all joint groups.
ACTUATOR_DELAY = dict(delay_min_lag=2, delay_max_lag=8, delay_hold_prob=0.3)


def actuator_cfg(
    target_names_expr: tuple[str, ...], a: ActuatorConfig
) -> MotorPositionActuatorCfg:
    """Position actuator driving the joints matched by ``target_names_expr`` with motor ``a``."""
    return MotorPositionActuatorCfg(
        motor=a,
        target_names_expr=target_names_expr,
        effort_limit=a.effort_limit,
        armature=a.armature,
        stiffness=a.stiffness,
        damping=a.damping,
        **ACTUATOR_DELAY,
    )


def build_action_scale(
    articulation: EntityArticulationInfoCfg, factor: float = ACTION_SCALE_FACTOR
) -> dict[str, float]:
    """Build joint-name -> scale dict from an articulation tuple.

    Each actuator group must expose ``effort_limit`` and ``stiffness``. The same
    scale (factor * effort_limit / stiffness) is applied to every joint matched
    by that group's regex.
    """
    out: dict[str, float] = {}
    for a in articulation.actuators:
        e = a.effort_limit
        s = a.stiffness
        assert e is not None
        for n in a.target_names_expr:
            out[n] = factor * e / s
    return out
