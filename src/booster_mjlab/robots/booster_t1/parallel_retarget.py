"""Retarget serial-T1 motion clips onto the parallel-ankle joint layout.

Serial clips store ``dof_pos`` as (frames, 23); the parallel model has 35 joints.
See :mod:`booster_mjlab.robots.parallel_retarget` for the method.
"""

from booster_mjlab.motion import MotionFile
from booster_mjlab.robots import parallel_retarget
from booster_mjlab.robots.booster_t1.t1_constants import T1_JOINT_ORDER
from booster_mjlab.robots.booster_t1.t1_parallel_constants import (
    HOME_KEYFRAME,
    T1_ANKLE,
    T1_DRIVE_RANGE,
    T1_PARALLEL_JOINT_ORDER,
)
from booster_mjlab.robots.parallel_retarget import ParallelLayout

T1_LAYOUT = ParallelLayout(
    serial_joints=T1_JOINT_ORDER,
    parallel_joints=T1_PARALLEL_JOINT_ORDER,
    drive_range=T1_DRIVE_RANGE,
    home_ankle_pitch=HOME_KEYFRAME.joint_pos["Left_Ankle_Pitch"],
    geometry=T1_ANKLE,
)


def to_parallel(motion: MotionFile) -> MotionFile:
    """Return ``motion`` in the parallel T1 joint layout, retargeting if it is serial."""
    return parallel_retarget.to_parallel(motion, T1_LAYOUT)
