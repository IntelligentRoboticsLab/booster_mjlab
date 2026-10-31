"""Booster K1 flat tracking environment configurations."""

import dataclasses as _dc
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg

from booster_mjlab.robots import (
    get_k1_robot_cfg,
)
from booster_mjlab.robots.actuators import build_action_scale
from booster_mjlab.robots.booster_k1.k1_constants import (
    K1_ACTUATOR_R14,
    K1_FOOT_SOLE_SIZE,
)
from booster_mjlab.robots.torque_speed import with_torque_speed_limits
from booster_mjlab.tasks.tracking.config.env_cfgs import booster_flat_tracking_env_cfg

_TRACKING_ARMS = _dc.replace(K1_ACTUATOR_R14, stiffness=20.0, damping=2.0)
_TRACKING_ARM_ACTION_SCALE = {
    expr: 0.5 * _TRACKING_ARMS.effort_limit / _TRACKING_ARMS.stiffness
    for expr in _TRACKING_ARMS.target_names_expr
}

K1_TRACKING_BODY_NAMES = (
    "Trunk",
    "Head_1",
    "Head_2",
    "Left_Arm_1",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "left_hand_end_ball",
    "Right_Arm_1",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
    "right_hand_end_ball",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Shank",
    "Left_Ankle_Cross",
    "left_foot_link",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Shank",
    "Right_Ankle_Cross",
    "right_foot_link",
)


def booster_k1_flat_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Booster K1 flat terrain tracking configuration."""
    base_robot_cfg = get_k1_robot_cfg()
    articulation = deepcopy(base_robot_cfg.articulation)
    assert articulation is not None
    articulation.actuators = tuple(
        _TRACKING_ARMS if actuator == K1_ACTUATOR_R14 else actuator
        for actuator in articulation.actuators
    )
    base_robot_cfg.articulation = articulation
    robot_cfg = with_torque_speed_limits(base_robot_cfg)

    assert robot_cfg.articulation is not None
    action_scale = build_action_scale(robot_cfg.articulation)
    action_scale.update(_TRACKING_ARM_ACTION_SCALE)

    return booster_flat_tracking_env_cfg(
        robot_cfg,
        action_scale,
        K1_FOOT_SOLE_SIZE,
        body_names=K1_TRACKING_BODY_NAMES,
        ee_body_names=(
            "left_foot_link",
            "right_foot_link",
            "left_hand_end_ball",
            "right_hand_end_ball",
        ),
        play=play,
    )
