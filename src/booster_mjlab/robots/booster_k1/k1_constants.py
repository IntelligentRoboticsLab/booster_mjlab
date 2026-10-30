"""Booster K1 constants."""

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from booster_mjlab.robots.booster_k1.actuators import (
    ActuatorConfig,
    MotorPositionActuatorCfg,
)

if TYPE_CHECKING:
    import torch

##
# MJCF and assets.
##

K1_XML: Path = BOOSTER_MJLAB_SRC_PATH / "robots" / "booster_k1" / "xmls" / "k1.xml"
assert K1_XML.exists()


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(K1_XML))

    # Keep all robot collision geoms in the collision visualization group.
    for geom in spec.geoms:
        if geom.name and "_collision" in geom.name.lower():
            geom.group = 3

    return spec


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.5125),
    joint_pos={
        "Left_Shoulder_Roll": -1.4,
        "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Yaw": 0.4,
        "Left_Hip_Pitch": -0.4,
        "Left_Knee_Pitch": 0.8,
        "Left_Ankle_Pitch": -0.4,
        "Right_Hip_Pitch": -0.4,
        "Right_Knee_Pitch": 0.8,
        "Right_Ankle_Pitch": -0.4,
    },
    joint_vel={".*": 0.0},
)

OLD_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.518),
    joint_pos={
        "Left_Shoulder_Roll": -1.4,
        "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Yaw": 0.4,
        "Left_Hip_Pitch": -0.4,
        "Left_Knee_Pitch": 0.8,
        "Left_Ankle_Pitch": -0.4,
        "Right_Hip_Pitch": -0.4,
        "Right_Knee_Pitch": 0.8,
        "Right_Ankle_Pitch": -0.4,
    },
    joint_vel={".*": 0.0},
)

KICK_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.495),
    joint_pos={
        "Head_Pitch": 0.7,
        "Left_Shoulder_Roll": -1.4,
        "Left_Elbow_Yaw": -0.4,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Yaw": 0.4,
        "Left_Hip_Pitch": -0.525,
        "Left_Knee_Pitch": 1.05,
        "Left_Ankle_Pitch": -0.525,
        "Right_Hip_Pitch": -0.525,
        "Right_Knee_Pitch": 1.05,
        "Right_Ankle_Pitch": -0.525,
    },
    joint_vel={".*": 0.0},
)

CRAWL_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.353),
    rot=(0, 0, -0.707, 0.707),
    joint_pos={
        "Left_Shoulder_Pitch": -1.4,
        "Left_Shoulder_Roll": 0.6,
        "Left_Elbow_Pitch": -1.7,
        "Left_Elbow_Yaw": -1.0,
        "Right_Shoulder_Pitch": -1.4,
        "Right_Shoulder_Roll": -0.6,
        "Right_Elbow_Pitch": -1.7,
        "Right_Elbow_Yaw": 1.0,
        "Left_Hip_Pitch": 0.1,
        "Left_Hip_Roll": 0.7,
        "Left_Knee_Pitch": 1.5,
        "Left_Ankle_Pitch": 0.03,
        "Right_Hip_Pitch": 0.1,
        "Right_Hip_Roll": -0.7,
        "Right_Knee_Pitch": 1.5,
        "Right_Ankle_Pitch": 0.03,
    },
    joint_vel={".*": 0.0},
)


##
# Collision config.
##

# Pair with terrain_collisions(). Equal priority with the terrain, so contact
# compliance mixes with the (randomized) terrain solref/solimp; the terrain's
# zero friction keeps the robot's friction coefficients in effect.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    solref=(0.01, 1),
    condim=3,
    priority=0,
    solmix=0.001,
    friction={r"^(left|right)_foot_collision$": (1.0,), ".*": (0.45,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    # Harden all collision geoms.
    solref=(0.01, 1),
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1, ".*": 0},
    friction={r"^(left|right)_foot_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_foot_collision$",),
    contype=0,
    conaffinity=1,
    solref=(0.01, 1),
    condim=3,
    priority=1,
    friction=(0.6,),
)


# All K1 articulations get the same scale rule for the joint position action:
# 0.25 * effort_limit / stiffness per joint group.
_ACTION_SCALE_FACTOR = 0.25

# Shared delay parameters for all joint groups.
ACTUATOR_DELAY = dict(delay_min_lag=2, delay_max_lag=8, delay_hold_prob=0.3)


def _build_action_scale(
    articulation: EntityArticulationInfoCfg, factor: float = _ACTION_SCALE_FACTOR
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


##
# K1 motor constants (one per physical motor kind).
##
# Motor limits match booster_train/assets/robots/booster.py; PD gains match
# booster-lab.

# knee
ACTUATOR_E6416 = ActuatorConfig(
    armature=0.095625,
    effort_limit=112.0,
    velocity_limit=12.57,
    knee_point_velocity=2.09,
    stiffness=80.0,
    damping=4.0,
)

# hip yaw
ACTUATOR_E4310 = ActuatorConfig(
    armature=0.0282528,
    effort_limit=38.3,
    velocity_limit=17.59,
    knee_point_velocity=7.85,
    stiffness=80.0,
    damping=4.0,
)

# hip pitch
ACTUATOR_E6408 = ActuatorConfig(
    armature=0.0478125,
    effort_limit=68.0,
    velocity_limit=14.66,
    knee_point_velocity=1.88,
    stiffness=80.0,
    damping=4.0,
)

# hip roll
ACTUATOR_E4315 = ActuatorConfig(
    armature=0.0339552,
    effort_limit=76.0,
    velocity_limit=12.57,
    knee_point_velocity=2.62,
    stiffness=80.0,
    damping=4.0,
)

# arm (shoulder + elbow)
ACTUATOR_R14 = ActuatorConfig(
    armature=0.001,
    effort_limit=14.0,
    velocity_limit=33.51,
    knee_point_velocity=5.24,
    stiffness=10.0,
    damping=1.0,
)

# head / neck
ACTUATOR_HT4438 = ActuatorConfig(
    armature=0.001,
    effort_limit=6.0,
    # The source knee speed exceeds the no-load speed; clamp when used.
    velocity_limit=7.85,
    knee_point_velocity=10.47,
    stiffness=4.0,
    damping=0.25,
)

# ankle: parallel linkage of two E4310 motors.
# TODO: These values could use some extra investigation due to their importance for
# locomotion and these being the only ones with remote-actuated parallel linkage.
ACTUATOR_K1_ANKLE = ActuatorConfig(
    armature=ACTUATOR_E4310.armature * 2,
    effort_limit=ACTUATOR_E4310.effort_limit,
    velocity_limit=ACTUATOR_E4310.velocity_limit,
    knee_point_velocity=ACTUATOR_E4310.knee_point_velocity,
    stiffness=50.0,
    damping=2.0,
)


##
# K1 articulation.
##


def _actuator_cfg(
    target_names_expr: tuple[str, ...], a: ActuatorConfig
) -> MotorPositionActuatorCfg:
    return MotorPositionActuatorCfg(
        motor=a,
        target_names_expr=target_names_expr,
        effort_limit=a.effort_limit,
        armature=a.armature,
        stiffness=a.stiffness,
        damping=a.damping,
        **ACTUATOR_DELAY,
    )


K1_ACTUATOR_E6416 = _actuator_cfg((".*_Knee_Pitch",), ACTUATOR_E6416)
K1_ACTUATOR_E4310 = _actuator_cfg((".*_Hip_Yaw",), ACTUATOR_E4310)
K1_ACTUATOR_ANKLE = _actuator_cfg((".*_Ankle_.*",), ACTUATOR_K1_ANKLE)
K1_ACTUATOR_E6408 = _actuator_cfg((".*_Hip_Pitch",), ACTUATOR_E6408)
K1_ACTUATOR_E4315 = _actuator_cfg((".*_Hip_Roll",), ACTUATOR_E4315)
K1_ACTUATOR_R14 = _actuator_cfg((".*_Shoulder_.*", ".*_Elbow_.*"), ACTUATOR_R14)
K1_ACTUATOR_HT4438 = _actuator_cfg(("Head_.*",), ACTUATOR_HT4438)


K1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        K1_ACTUATOR_E6416,
        K1_ACTUATOR_E4310,
        K1_ACTUATOR_ANKLE,
        K1_ACTUATOR_E6408,
        K1_ACTUATOR_E4315,
        K1_ACTUATOR_R14,
        K1_ACTUATOR_HT4438,
    ),
    soft_joint_pos_limit_factor=0.9,
)


K1_ACTION_SCALE = _build_action_scale(K1_ARTICULATION)


def get_k1_robot_cfg(
    default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
    default_collisions: tuple[CollisionCfg, ...] = (FULL_COLLISION,),
) -> EntityCfg:
    """Get a fresh K1 robot configuration instance.

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
        articulation=K1_ARTICULATION,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_k1_robot_cfg())

    viewer.launch(robot.spec.compile())


K1_JOINT_ORDER: tuple[str, ...] = (
    "Head_Yaw",
    "Head_Pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
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


def get_k1_default_joint_pos(
    keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
) -> "torch.Tensor":
    """Return the K1 default joint positions as a 1D tensor.

    Order matches the XML joint order (= the order used in motion .pkl
    ``dof_pos`` arrays and in ``robot.data.default_joint_pos`` at runtime).
    Missing joints in ``keyframe.joint_pos`` default to 0.
    """
    import torch

    return torch.tensor(
        [keyframe.joint_pos.get(name, 0.0) for name in K1_JOINT_ORDER],
        dtype=torch.float32,
    )
