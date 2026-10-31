from booster_mjlab.robots.booster_k1.k1_constants import (
    K1_ACTION_SCALE,
    K1_MIRROR,
    get_k1_robot_cfg,
)
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    K1_PARALLEL_ACTION_SCALE,
    get_k1_parallel_robot_cfg,
)
from booster_mjlab.robots.booster_k1.parallel_retarget import K1_LAYOUT
from booster_mjlab.robots.booster_t1.parallel_retarget import T1_LAYOUT
from booster_mjlab.robots.booster_t1.t1_constants import (
    T1_ACTION_SCALE,
    T1_MIRROR,
    get_t1_robot_cfg,
)
from booster_mjlab.robots.booster_t1.t1_parallel_constants import (
    T1_PARALLEL_ACTION_SCALE,
    get_t1_parallel_robot_cfg,
)
from booster_mjlab.robots.mirror import Mirror
from booster_mjlab.robots.parallel_retarget import ParallelLayout

# Every robot's mirror and parallel-ankle layout, for tools that work out the robot
# from a compiled model's joint names.
MIRRORS: tuple[Mirror, ...] = (K1_MIRROR, T1_MIRROR)
PARALLEL_LAYOUTS: tuple[ParallelLayout, ...] = (K1_LAYOUT, T1_LAYOUT)

__all__ = (
    "K1_ACTION_SCALE",
    "K1_PARALLEL_ACTION_SCALE",
    "MIRRORS",
    "PARALLEL_LAYOUTS",
    "T1_ACTION_SCALE",
    "T1_PARALLEL_ACTION_SCALE",
    "get_k1_robot_cfg",
    "get_k1_parallel_robot_cfg",
    "get_t1_robot_cfg",
    "get_t1_parallel_robot_cfg",
)
