"""Booster T1 constants."""

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from booster_mjlab.robots.actuators import (
    ActuatorConfig,
    actuator_cfg,
    build_action_scale,
    pd_gains,
)
from booster_mjlab.robots.collisions import FULL_COLLISION
from booster_mjlab.robots.mirror import Mirror

if TYPE_CHECKING:
    import torch

##
# MJCF and assets.
##

# Link frames, visual meshes and collision capsules follow Booster's T1 MJCF.
# Visual meshes (incl. the chest logo and the 43.5 mm deep feet) are booster_assets'
# robots/T1/meshes, inertials are from its T1_23dof.urdf, joint ranges are those of
# the real robot, and each foot collides with a 48-vertex hull of its visual mesh.
T1_XML: Path = BOOSTER_MJLAB_SRC_PATH / "robots" / "booster_t1" / "xmls" / "t1.xml"
assert T1_XML.exists()


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(T1_XML))

    # Keep all robot collision geoms in the collision visualization group.
    for geom in spec.geoms:
        if geom.name and "_collision" in geom.name.lower():
            geom.group = 3

    return spec


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.6775),
    joint_pos={
        "Left_Shoulder_Roll": -1.4,
        "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Yaw": 0.4,
        "Left_Hip_Pitch": -0.2,
        "Left_Knee_Pitch": 0.4,
        "Left_Ankle_Pitch": -0.2,
        "Right_Hip_Pitch": -0.2,
        "Right_Knee_Pitch": 0.4,
        "Right_Ankle_Pitch": -0.2,
    },
    joint_vel={".*": 0.0},
)


##
# T1 motor constants (one per physical motor kind).
##
# Motor limits and armature match booster_train/assets/robots/actuator.py. Default
# gains follow mjlab_playground's T1: k = I w_n^2 and d = 2 zeta w_n I with
# w_n = 2 pi f, at 5 Hz and damping ratio 2.0 against each motor's armature. The
# joints below override gains and torque limits; only the parallel T1's ankle
# cranks use the defaults.

NATURAL_FREQ = 5.0
DAMPING_RATIO = 2.0


def _motor(
    armature: float,
    effort_limit: float,
    velocity_limit: float,
    knee_point_velocity: float,
) -> ActuatorConfig:
    return ActuatorConfig(
        armature=armature,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        knee_point_velocity=knee_point_velocity,
        **pd_gains(armature, NATURAL_FREQ, DAMPING_RATIO),
    )


# knee
ACTUATOR_E8116 = _motor(0.0636012, 130.5, 14.76, 6.71)
# hip pitch
ACTUATOR_E8112 = _motor(0.0523908, 98.8, 15.29, 7.53)
# waist, hip roll, hip yaw
ACTUATOR_E6408 = _motor(0.0478125, 68.0, 14.66, 1.88)
# arm (shoulder + elbow)
ACTUATOR_E4310 = _motor(0.0282528, 38.3, 17.59, 7.85)
# head / neck. Gains against booster_train's 0.0018 armature, as in
# mjlab_playground, but simulated at twice that.
ACTUATOR_DM4310 = dataclasses.replace(
    _motor(0.0018, 7.0, 41.89, 26.18), armature=2 * 0.0018
)
# ankle motor
ACTUATOR_E4315 = _motor(0.0339552, 43.0, 12.15, 8.38)


##
# T1 joint actuators.
##

# Torque limits and damping per joint are from the B-Human SimRobot T1 config
# ("regular" motor set) the policy is deployed with. The stiffness is picked so the
# action scale, 0.25 * effort_limit / stiffness, is the motor's default scale (peak
# torque over the default gain) shrunk by T1_ACTION_SCALE_SHRINK. Armature and the
# torque-speed metadata stay those of each joint's motor.
T1_ACTION_SCALE_SHRINK = 1.75


def _joint(
    m: ActuatorConfig,
    damping: float,
    effort_limit: float,
    stiffness: float | None = None,
) -> ActuatorConfig:
    if stiffness is None:
        stiffness = T1_ACTION_SCALE_SHRINK * m.stiffness * effort_limit / m.effort_limit
    return dataclasses.replace(
        m, stiffness=stiffness, damping=damping, effort_limit=effort_limit
    )


T1_KNEE = _joint(ACTUATOR_E8116, 5.0, 110.0)
T1_HIP_PITCH = _joint(ACTUATOR_E8112, 5.0, 90.0)
T1_HIP_ROLL = _joint(ACTUATOR_E6408, 5.0, 105.0)
T1_HIP_YAW = _joint(ACTUATOR_E6408, 5.0, 40.0)
T1_WAIST = _joint(ACTUATOR_E6408, 5.0, 40.0)
# ankle: parallel linkage of two E4315 motors, modeled as serial pitch and roll
# joints with one E4315's armature each. Fixed stiffness rather than the shrunk
# scale.
T1_ANKLE = _joint(ACTUATOR_E4315, 3.0, 57.0, stiffness=50.0)
T1_ARM = _joint(ACTUATOR_E4310, 1.0, 18.0)
# head / neck
T1_HEAD = _joint(ACTUATOR_DM4310, 1.2, 7.0)


##
# T1 articulation.
##

# Shared with the parallel-ankle T1.
T1_LEG_WAIST_ACTUATORS = (
    actuator_cfg((".*_Knee_Pitch",), T1_KNEE),
    actuator_cfg((".*_Hip_Pitch",), T1_HIP_PITCH),
    actuator_cfg((".*_Hip_Roll",), T1_HIP_ROLL),
    actuator_cfg((".*_Hip_Yaw",), T1_HIP_YAW),
    actuator_cfg(("Waist",), T1_WAIST),
)
T1_ARM_HEAD_ACTUATORS = (
    actuator_cfg(
        (
            ".*_Shoulder_Pitch",
            ".*_Shoulder_Roll",
            ".*_Elbow_Pitch",
            ".*_Elbow_Yaw",
        ),
        T1_ARM,
    ),
    actuator_cfg(("AAHead_yaw", "Head_pitch"), T1_HEAD),
)


T1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        *T1_LEG_WAIST_ACTUATORS,
        actuator_cfg((".*_Ankle_Pitch", ".*_Ankle_Roll"), T1_ANKLE),
        *T1_ARM_HEAD_ACTUATORS,
    ),
    soft_joint_pos_limit_factor=0.9,
)


T1_ACTION_SCALE = build_action_scale(T1_ARTICULATION)


def get_t1_robot_cfg(
    default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
    default_collisions: tuple[CollisionCfg, ...] = (FULL_COLLISION,),
) -> EntityCfg:
    """Get a fresh T1 robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.

    Args:
      default_keyframe: Initial state keyframe to seed each environment.
      default_collisions: Collision config tuple.
    """
    return EntityCfg(
        init_state=default_keyframe,
        collisions=default_collisions,
        spec_fn=get_spec,
        articulation=T1_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_t1_robot_cfg())

    viewer.launch(robot.spec.compile())


# All 23 joints in MJCF definition order.
T1_JOINT_ORDER: tuple[str, ...] = (
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Waist",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)


# Joints whose axes change sign under a left/right mirror (roll and yaw axes). The arm
# links are named AL1..AL3 / AR1..AR3 rather than Left/Right.
T1_MIRROR = Mirror(
    joint_names=T1_JOINT_ORDER,
    inverted_joints=frozenset(
        (
            "AAHead_yaw",
            "Waist",
            *(
                f"{side}_{joint}"
                for side in ("Left", "Right")
                for joint in (
                    "Shoulder_Roll",
                    "Elbow_Yaw",
                    "Hip_Roll",
                    "Hip_Yaw",
                    "Ankle_Roll",
                )
            ),
        )
    ),
    name_pairs=tuple((f"AL{i}", f"AR{i}") for i in (1, 2, 3)),
)

# Inner sole contact area sampled by the foot clearance scan. The T1 foot is
# 0.225 x 0.10 m; this keeps the K1 grid's margin to the foot edge.
T1_FOOT_SOLE_SIZE = (0.16, 0.075)


def get_t1_default_joint_pos(
    keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
) -> "torch.Tensor":
    """Return the T1 default joint positions as a 1D tensor in XML joint order."""
    import torch

    return torch.tensor(
        [keyframe.joint_pos.get(name, 0.0) for name in T1_JOINT_ORDER],
        dtype=torch.float32,
    )
