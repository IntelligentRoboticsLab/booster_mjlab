"""Booster T1 with the physical parallel (closed-loop) ankle linkage.

Same robot as :mod:`t1_constants`, except each ankle is driven by two cranks
pushing rods onto the foot instead of by direct pitch/roll motors. The ankle
pitch/roll joints become passive and two ``*_Ankle_A`` / ``*_Ankle_B`` drive joints
take their slots in the actuated-joint ordering, so the 23-wide action and
joint-observation layout is unchanged from the serial model.

The linkage geometry is booster_assets' ``robots/T1/T1_23dof_parallel.xml``. It is
added to the serial T1 spec here instead of living in a second MJCF, so both
models share one set of meshes, colliders and joint limits.
"""

from typing import TYPE_CHECKING

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.entity.entity import EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg

from booster_mjlab.robots.booster_k1.k1_constants import (
    FULL_COLLISION,
    _actuator_cfg,
    _build_action_scale,
)
from booster_mjlab.robots.booster_k1.parallel_ankle_ik import (
    LINKAGE_JOINT_SUFFIXES,
    AnkleGeometry,
    _Linkage,
    solve_leg,
)
from booster_mjlab.robots.booster_t1 import t1_constants as serial
from booster_mjlab.robots.booster_t1.t1_constants import (
    ACTUATOR_E4315,
    T1_ARM_HEAD_ACTUATORS,
    T1_LEG_WAIST_ACTUATORS,
)

if TYPE_CHECKING:
    import torch

##
# Linkage geometry (left leg, shank frame; the right leg is its mirror image).
##

_ANKLE_ORIGIN = np.array([0.0, 0.0, -0.28])
_FOOT_OFFSET = np.array([0.0, 0.00025, -0.012])

_CRANK_A_PIVOT = np.array([-0.014, 0.02775, -0.096])
_CRANK_A_ARM = np.array([0.04293593596233938, 0.0, 0.002346359528693609])
_ANCHOR_A = np.array([-0.0415, 0.0275, 0.0327])

# booster_assets puts crank B at y = -0.02775, which leaves its rod 0.5 mm out of
# the crank plane at the zero pose (the foot sits 0.25 mm off-center). Centering
# the crank on its anchor keeps both rods in-plane, as the closed-form IK assumes.
_CRANK_B_PIVOT = np.array([-0.014, -0.02725, -0.156])
_CRANK_B_ARM = np.array([0.04259337754140823, 0.0, -0.0058995075739472075])
_ANCHOR_B = np.array([-0.0415, -0.0275, 0.0327])


def _rod(pivot: np.ndarray, arm: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Rod vector that closes the loop exactly at the zero ankle pose."""
    return _ANKLE_ORIGIN + _FOOT_OFFSET + anchor - (pivot + arm)


T1_ANKLE = AnkleGeometry(
    origin=_ANKLE_ORIGIN,
    foot_offset=_FOOT_OFFSET,
    linkage_a=_Linkage(
        _CRANK_A_PIVOT,
        _CRANK_A_ARM,
        _rod(_CRANK_A_PIVOT, _CRANK_A_ARM, _ANCHOR_A),
        _ANCHOR_A,
    ),
    linkage_b=_Linkage(
        _CRANK_B_PIVOT,
        _CRANK_B_ARM,
        _rod(_CRANK_B_PIVOT, _CRANK_B_ARM, _ANCHOR_B),
        _ANCHOR_B,
    ),
)

# Crank hard stops, from booster_assets.
T1_DRIVE_RANGE = {
    "Ankle_A": (-0.8187464729740654, 1.0227913151109491),
    "Ankle_B": (-0.6454320082084025, 1.097331001571644),
}


##
# MJCF.
##

_SHANK = {"Left": "Shank_Left", "Right": "Shank_Right"}

# Cranks and rods are drawn orange like the K1 parallel model's, to set the linkage
# apart from the robot's own meshes. Their geoms clear the `medium_gray` material
# they inherit from the T1 default class, since viewers that color by material
# (e.g. viser) would otherwise ignore the rgba.
_LINKAGE_RGBA = [0.85, 0.35, 0.25, 1.0]
_FOOT = {"Left": "left_foot_link", "Right": "right_foot_link"}


def _add_linkage(spec: mujoco.MjSpec, side: str) -> None:
    _, _, linkages = T1_ANKLE.side(side)
    shank = spec.body(_SHANK[side])
    foot = spec.body(_FOOT[side])
    for tag, linkage in linkages.items():
        low, high = T1_DRIVE_RANGE[f"Ankle_{tag}"]
        crank = shank.add_body(
            name=f"{side}_Ankle_Crank_{tag}",
            pos=linkage.pivot,
            mass=0.03,
            ipos=linkage.arm / 2,
            inertia=[2e-5, 2e-5, 2e-5],
        )
        crank.add_joint(
            name=f"{side}_Ankle_{tag}",
            axis=[0, 1, 0],
            range=[low, high],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            damping=[0.1, 0, 0],
        )
        crank.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, *linkage.arm],
            size=[0.008, 0, 0],
            rgba=_LINKAGE_RGBA,
            material="",
            contype=0,
            conaffinity=0,
            group=2,
            density=0,
        )
        rod = crank.add_body(
            name=f"{side}_Ankle_Rod_{tag}",
            pos=linkage.arm,
            mass=0.02,
            ipos=linkage.rod / 2,
            inertia=[1e-5, 1e-5, 1e-5],
        )
        for axis_name, axis in (("Pitch", [0, 1, 0]), ("Roll", [1, 0, 0])):
            rod.add_joint(
                name=f"{side}_Ankle_Rod_{tag}_{axis_name}",
                axis=axis,
                limited=mujoco.mjtLimited.mjLIMITED_FALSE,
                armature=0.0001,
                damping=[0.01, 0, 0],
            )
        rod.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, *linkage.rod],
            size=[0.005, 0, 0],
            rgba=_LINKAGE_RGBA,
            material="",
            contype=0,
            conaffinity=0,
            group=2,
            density=0,
        )
        lower = side.lower()
        rod.add_site(name=f"{lower}_ankle_rod_{tag.lower()}_end", pos=linkage.rod)
        foot.add_site(name=f"{lower}_ankle_{tag.lower()}_anchor", pos=linkage.anchor)
        spec.add_equality(
            name=f"{lower}_ankle_{tag.lower()}_loop",
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"{lower}_ankle_rod_{tag.lower()}_end",
            name2=f"{lower}_ankle_{tag.lower()}_anchor",
            solref=[0.01, 1],
            solimp=[0.95, 0.99, 0.001, 0.5, 2],
        )
    # The ankle pitch/roll joints are passive now, damped like booster_assets'.
    for joint in ("Ankle_Pitch", "Ankle_Roll"):
        spec.joint(f"{side}_{joint}").damping = [0.2, 0, 0]


def get_spec() -> mujoco.MjSpec:
    spec = serial.get_spec()
    for side in ("Left", "Right"):
        _add_linkage(spec, side)
    return spec


##
# Joint layout.
##

# Passive joints of the closed loop: the two ankle DOFs the linkage drives, and
# each rod's universal joint.
_PASSIVE_ANKLE_SUFFIXES = (
    "Ankle_Pitch",
    "Ankle_Roll",
    "Ankle_Rod_A_Pitch",
    "Ankle_Rod_A_Roll",
    "Ankle_Rod_B_Pitch",
    "Ankle_Rod_B_Roll",
)


def _leg_joints(side: str) -> tuple[str, ...]:
    # The linkage bodies are appended to the shank after the ankle cross, so the
    # ankle joints come first in MJCF order.
    return (
        f"{side}_Hip_Pitch",
        f"{side}_Hip_Roll",
        f"{side}_Hip_Yaw",
        f"{side}_Knee_Pitch",
        f"{side}_Ankle_Pitch",
        f"{side}_Ankle_Roll",
        f"{side}_Ankle_A",
        f"{side}_Ankle_Rod_A_Pitch",
        f"{side}_Ankle_Rod_A_Roll",
        f"{side}_Ankle_B",
        f"{side}_Ankle_Rod_B_Pitch",
        f"{side}_Ankle_Rod_B_Roll",
    )


# All 35 joints in MJCF definition order. This is the column order of
# `Entity.joint_names`, `data.joint_pos`, and retargeted AMP motion files.
T1_PARALLEL_JOINT_ORDER: tuple[str, ...] = (
    *serial.T1_JOINT_ORDER[:11],  # head, arms, waist
    *_leg_joints("Left"),
    *_leg_joints("Right"),
)

# Passive joints of the closed loop, excluded from observations and rewards.
T1_PARALLEL_PASSIVE_JOINTS: tuple[str, ...] = tuple(
    f"{side}_{suffix}"
    for side in ("Left", "Right")
    for suffix in _PASSIVE_ANKLE_SUFFIXES
)

# The 23 actuated joints, in MJCF definition order. Identical to the serial T1
# ordering except that the ankle pitch/roll slots hold the two crank drives.
T1_PARALLEL_ACTUATED_JOINTS: tuple[str, ...] = tuple(
    name for name in T1_PARALLEL_JOINT_ORDER if name not in T1_PARALLEL_PASSIVE_JOINTS
)

# Actuated joints outside the closed loop. Reset randomization is limited to
# these so the linkage starts exactly closed.
T1_PARALLEL_FREE_RESET_JOINTS: tuple[str, ...] = tuple(
    name for name in T1_PARALLEL_ACTUATED_JOINTS if "Ankle" not in name
)


##
# Keyframe config.
##


def linkage_pose(ankle_pitch: float, ankle_roll: float = 0.0) -> dict[str, float]:
    """Both legs' crank and rod angles that close the loop at an ankle pose."""
    pose = {}
    for side in ("Left", "Right"):
        roll = ankle_roll if side == "Left" else -ankle_roll
        angles = solve_leg(side, np.array(ankle_pitch), np.array(roll), T1_ANKLE)
        for suffix, value in zip(LINKAGE_JOINT_SUFFIXES, angles):
            pose[f"{side}_{suffix}"] = float(value)
    return pose


_HOME_ANKLE_PITCH = serial.HOME_KEYFRAME.joint_pos["Left_Ankle_Pitch"]

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=serial.HOME_KEYFRAME.pos,
    joint_pos={
        **serial.HOME_KEYFRAME.joint_pos,
        **linkage_pose(_HOME_ANKLE_PITCH),
    },
    joint_vel={".*": 0.0},
)


##
# Ankle drive actuator.
##

# Each crank is one E4315 motor with its own armature and the motor's default
# gains (mjlab_playground's 5 Hz, damping ratio 2.0 against that armature).
ACTUATOR_T1_ANKLE_DRIVE = ACTUATOR_E4315

_ANKLE_DRIVE_NAMES = (r".*_Ankle_A", r".*_Ankle_B")

T1_ACTUATOR_ANKLE_DRIVE = _actuator_cfg(_ANKLE_DRIVE_NAMES, ACTUATOR_T1_ANKLE_DRIVE)

T1_PARALLEL_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        *T1_LEG_WAIST_ACTUATORS,
        T1_ACTUATOR_ANKLE_DRIVE,
        *T1_ARM_HEAD_ACTUATORS,
    ),
    soft_joint_pos_limit_factor=0.9,
)

T1_PARALLEL_ACTION_SCALE = _build_action_scale(T1_PARALLEL_ARTICULATION)


##
# Top-level config.
##


def get_t1_parallel_robot_cfg(
    default_keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
    default_collisions: tuple[CollisionCfg, ...] = (FULL_COLLISION,),
) -> EntityCfg:
    """Get a fresh parallel-ankle T1 robot configuration instance.

    Mirrors :func:`booster_mjlab.robots.booster_t1.t1_constants.get_t1_robot_cfg`,
    but builds the closed-loop ankle model.
    """
    return EntityCfg(
        init_state=default_keyframe,
        collisions=default_collisions,
        spec_fn=get_spec,
        articulation=T1_PARALLEL_ARTICULATION,
    )


def get_t1_parallel_default_joint_pos(
    keyframe: EntityCfg.InitialStateCfg = HOME_KEYFRAME,
) -> "torch.Tensor":
    """Return the 23 actuated default joint positions as a 1D tensor."""
    import torch

    assert keyframe.joint_pos is not None
    return torch.tensor(
        [keyframe.joint_pos.get(name, 0.0) for name in T1_PARALLEL_ACTUATED_JOINTS],
        dtype=torch.float32,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = get_t1_parallel_robot_cfg().build()

    viewer.launch(robot.spec.compile())
