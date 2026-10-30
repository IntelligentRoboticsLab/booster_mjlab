"""Retarget serial-K1 motion clips onto the parallel-ankle joint layout.

Serial clips store ``dof_pos`` as (frames, 22); the parallel model has 34 joints.
See :mod:`booster_mjlab.robots.parallel_retarget` for the method.
"""

from booster_mjlab.motion import MotionFile
from booster_mjlab.robots import parallel_retarget
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    HOME_KEYFRAME,
    K1_ANKLE,
    K1_DRIVE_RANGE,
    K1_PARALLEL_JOINT_ORDER,
)
from booster_mjlab.robots.parallel_retarget import ParallelLayout

K1_LAYOUT = ParallelLayout(
    serial_joints=K1_JOINT_ORDER,
    parallel_joints=K1_PARALLEL_JOINT_ORDER,
    drive_range=K1_DRIVE_RANGE,
    home_ankle_pitch=HOME_KEYFRAME.joint_pos["Left_Ankle_Pitch"],
    geometry=K1_ANKLE,
)


def to_parallel(motion: MotionFile) -> MotionFile:
    """Return ``motion`` in the parallel K1 joint layout, retargeting if it is serial."""
    return parallel_retarget.to_parallel(motion, K1_LAYOUT)
