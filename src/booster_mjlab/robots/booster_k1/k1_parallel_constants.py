"""Booster K1 with the physical parallel (closed-loop) ankle linkage.

Same robot as :mod:`k1_constants`, except each ankle is driven by two remote
cranks pushing rods onto the foot instead of by direct pitch/roll motors. The
ankle pitch/roll joints become passive and two ``*_Ankle_A`` / ``*_Ankle_B``
drive joints take their slots in the actuated-joint ordering, so the 22-wide
action and joint-observation layout is unchanged from the serial model.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from booster_mjlab.robots.booster_k1.actuators import ActuatorConfig
from booster_mjlab.robots.booster_k1.k1_constants import (
    _ACTION_SCALE_FACTOR,
    ACTUATOR_E4310,
    FULL_COLLISION,
    K1_ACTUATOR_E4310,
    K1_ACTUATOR_E4315,
    K1_ACTUATOR_E6408,
    K1_ACTUATOR_E6416,
    K1_ACTUATOR_HT4438,
    K1_ACTUATOR_R14,
    _actuator_cfg,
    _build_action_scale,
)

if TYPE_CHECKING:
    import torch

##
# MJCF and assets.
##

K1_PARALLEL_XML: Path = (
    BOOSTER_MJLAB_SRC_PATH / "robots" / "booster_k1" / "xmls" / "k1_parallel.xml"
)
assert K1_PARALLEL_XML.exists()


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(K1_PARALLEL_XML))

    # Keep all robot collision geoms in the collision visualization group.
    for geom in spec.geoms:
        if geom.name and "_collision" in geom.name.lower():
            geom.group = 3

    return spec


##
# Joint layout.
##

# Passive joints of the closed loop: the two ankle DOFs the linkage drives, and
# each rod's universal joint.
_PASSIVE_ANKLE_SUFFIXES = (
    "Ankle_Rod_A_Pitch",
    "Ankle_Rod_A_Roll",
    "Ankle_Rod_B_Pitch",
    "Ankle_Rod_B_Roll",
    "Ankle_Pitch",
    "Ankle_Roll",
)


def _leg_joints(side: str) -> tuple[str, ...]:
    return (
        f"{side}_Hip_Pitch",
        f"{side}_Hip_Roll",
        f"{side}_Hip_Yaw",
        f"{side}_Knee_Pitch",
        f"{side}_Ankle_A",
        f"{side}_Ankle_Rod_A_Pitch",
        f"{side}_Ankle_Rod_A_Roll",
        f"{side}_Ankle_B",
        f"{side}_Ankle_Rod_B_Pitch",
        f"{side}_Ankle_Rod_B_Roll",
        f"{side}_Ankle_Pitch",
        f"{side}_Ankle_Roll",
    )


# All 34 joints in MJCF definition order. This is the column order of
# `Entity.joint_names`, `data.joint_pos`, and retargeted AMP motion files.
K1_PARALLEL_JOINT_ORDER: tuple[str, ...] = (
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
    *_leg_joints("Left"),
    *_leg_joints("Right"),
)

# Passive joints of the closed loop, excluded from observations and rewards.
K1_PARALLEL_PASSIVE_JOINTS: tuple[str, ...] = tuple(
    f"{side}_{suffix}"
    for side in ("Left", "Right")
    for suffix in _PASSIVE_ANKLE_SUFFIXES
)

# The 22 actuated joints, in MJCF definition order. Identical to the serial K1
# ordering except that the ankle pitch/roll slots hold the two crank drives.
K1_PARALLEL_ACTUATED_JOINTS: tuple[str, ...] = tuple(
    name for name in K1_PARALLEL_JOINT_ORDER if name not in K1_PARALLEL_PASSIVE_JOINTS
)

# Actuated joints outside the closed loop. Reset randomization is limited to
# these so the linkage starts exactly closed.
K1_PARALLEL_FREE_RESET_JOINTS: tuple[str, ...] = tuple(
    name for name in K1_PARALLEL_ACTUATED_JOINTS if "Ankle" not in name
)


##
# Keyframe config.
##


def _linkage_pose(
    drive_a: float, rod_a_pitch: float, drive_b: float, rod_b_pitch: float
) -> dict[str, float]:
    """Both legs' linkage angles for a symmetric ankle pose (roll = 0)."""
    return {
        f"{side}_Ankle_{joint}": value
        for side in ("Left", "Right")
        for joint, value in (
            ("A", drive_a),
            ("Rod_A_Pitch", rod_a_pitch),
            ("B", drive_b),
            ("Rod_B_Pitch", rod_b_pitch),
        )
    }


# Solved with the closed-form linkage IK so each keyframe closes the loop to
# machine precision at the serial model's ankle pitch.
_HOME_LINKAGE = _linkage_pose(0.334363, -0.326563, 0.317093, -0.389355)  # pitch -0.4
_KICK_LINKAGE = _linkage_pose(0.430693, -0.433025, 0.415008, -0.532013)  # pitch -0.525
_CRAWL_LINKAGE = _linkage_pose(-0.026431, 0.023211, -0.024624, 0.025432)  # pitch 0.03

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
        **_HOME_LINKAGE,
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
        **_KICK_LINKAGE,
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
        **_CRAWL_LINKAGE,
    },
    joint_vel={".*": 0.0},
)


##
# Ankle drive actuator.
##

# Each ankle crank uses one E4310, with half the serial ankle armature.
ACTUATOR_K1_ANKLE_DRIVE = ActuatorConfig(
    armature=ACTUATOR_E4310.armature,
    effort_limit=ACTUATOR_E4310.effort_limit,
    velocity_limit=ACTUATOR_E4310.velocity_limit,
    knee_point_velocity=ACTUATOR_E4310.knee_point_velocity,
    stiffness=25.0,
    damping=1.0,
)

_ANKLE_DRIVE_NAMES = (r".*_Ankle_A", r".*_Ankle_B")

K1_ACTUATOR_ANKLE_DRIVE = _actuator_cfg(_ANKLE_DRIVE_NAMES, ACTUATOR_K1_ANKLE_DRIVE)

K1_PARALLEL_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        K1_ACTUATOR_E6416,
        K1_ACTUATOR_E4310,
        K1_ACTUATOR_ANKLE_DRIVE,
        K1_ACTUATOR_E6408,
        K1_ACTUATOR_E4315,
        K1_ACTUATOR_R14,
        K1_ACTUATOR_HT4438,
    ),
    soft_joint_pos_limit_factor=0.9,
)

K1_PARALLEL_ACTION_SCALE = _build_action_scale(
    K1_PARALLEL_ARTICULATION, _ACTION_SCALE_FACTOR
)


##
# Top-level config.
##


def get_k1_parallel_robot_cfg(
    default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
    default_collisions: tuple[CollisionCfg, ...] = (FULL_COLLISION,),
) -> EntityCfg:
    """Get a fresh parallel-ankle K1 robot configuration instance.

    Mirrors :func:`booster_mjlab.robots.booster_k1.k1_constants.get_k1_robot_cfg`,
    but builds the closed-loop ankle model.
    """
    return EntityCfg(
        init_state=default_keyframe,
        collisions=default_collisions,
        spec_fn=get_spec,
        articulation=K1_PARALLEL_ARTICULATION,
    )


def get_k1_parallel_default_joint_pos(
    keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
) -> "torch.Tensor":
    """Return the 22 actuated default joint positions as a 1D tensor."""
    import torch

    assert keyframe.joint_pos is not None
    return torch.tensor(
        [keyframe.joint_pos.get(name, 0.0) for name in K1_PARALLEL_ACTUATED_JOINTS],
        dtype=torch.float32,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = get_k1_parallel_robot_cfg().build()

    viewer.launch(robot.spec.compile())
