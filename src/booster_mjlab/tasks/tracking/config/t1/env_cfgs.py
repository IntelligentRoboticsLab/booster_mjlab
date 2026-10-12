"""Booster T1 flat tracking environment configurations.

The T1 shares the K1's foot, trunk and collision geom names, so this config
starts from the K1 one and swaps in the robot, its bodies and its foot size.
"""

import dataclasses as _dc
from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from booster_mjlab.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from booster_mjlab.robots.booster_k1.sensors import (
    FootClearanceSensorCfg,
    FootSoleGridPatternCfg,
)
from booster_mjlab.robots.booster_k1.torque_speed import (
    BoosterEntityCfg,
    BoosterPositionActuatorCfg,
)
from booster_mjlab.tasks.tracking.config.k1.env_cfgs import (
    booster_k1_flat_tracking_env_cfg,
)
from booster_mjlab.tasks.velocity.config.t1.env_cfgs import T1_FOOT_SOLE_SIZE

# All T1 bodies, in MJCF order (= the link_body_list of the T1 motion datasets).
T1_TRACKING_BODY_NAMES = (
    "Trunk",
    "H1",
    "H2",
    "AL1",
    "AL2",
    "AL3",
    "left_hand_link",
    "AR1",
    "AR2",
    "AR3",
    "right_hand_link",
    "Waist",
    "Hip_Pitch_Left",
    "Hip_Roll_Left",
    "Hip_Yaw_Left",
    "Shank_Left",
    "Ankle_Cross_Left",
    "left_foot_link",
    "Hip_Pitch_Right",
    "Hip_Roll_Right",
    "Hip_Yaw_Right",
    "Shank_Right",
    "Ankle_Cross_Right",
    "right_foot_link",
)


def _with_torque_speed_limits(robot_cfg: EntityCfg) -> BoosterEntityCfg:
    """Rebuild a robot config with batched motor torque-speed limits."""
    cfg = BoosterEntityCfg(
        **{f.name: getattr(robot_cfg, f.name) for f in _dc.fields(robot_cfg)}
    )
    cfg.articulation = deepcopy(cfg.articulation)
    assert cfg.articulation is not None
    cfg.articulation.actuators = tuple(
        BoosterPositionActuatorCfg(
            **{f.name: getattr(actuator, f.name) for f in _dc.fields(actuator)}
        )
        for actuator in cfg.articulation.actuators
    )
    return cfg


def booster_t1_flat_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 flat terrain tracking configuration."""
    cfg = booster_k1_flat_tracking_env_cfg(play=play)

    cfg.scene.entities = {"robot": _with_torque_speed_limits(get_t1_robot_cfg())}

    for sensor in cfg.scene.sensors:
        if isinstance(sensor, FootClearanceSensorCfg):
            sensor.pattern = FootSoleGridPatternCfg(size=T1_FOOT_SOLE_SIZE)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = dict(T1_ACTION_SCALE)

    assert cfg.commands is not None
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.body_names = T1_TRACKING_BODY_NAMES

    # The T1 has no hand end-effector bodies; the forearm links end at the hands.
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_foot_link",
        "right_foot_link",
        "left_hand_link",
        "right_hand_link",
    )

    return cfg
