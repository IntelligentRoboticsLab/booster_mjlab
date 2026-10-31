"""Collision configs shared by the Booster robots.

They rely on the MJCF naming the Booster models share: every collision geom ends
in ``_collision`` and the soles are ``left_foot_collision`` / ``right_foot_collision``.
"""

from mjlab.utils.spec_config import CollisionCfg

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
