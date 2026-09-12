"""Soccer ball constants."""

from pathlib import Path
from typing import Literal

import mujoco

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH
from mjlab.entity import EntityCfg


##
# Ball size scaling factors (relative to size 5).
##

# Base size 5 properties
BALL_SIZE_5_RADIUS: float = 0.11  # meters
BALL_SIZE_5_MASS: float = 0.430  # kg
BALL_FREEJOINT_DAMPING: float = 0.0
BALL_COLLISION_CONDIM: int = 6
BALL_COLLISION_SOLREF: tuple[float, float] = (0.05, 0.15)
BALL_SHELL_INERTIA: bool = True
BALL_INERTIA_RATIO: float = 2 / 3 if BALL_SHELL_INERTIA else 2 / 5
BALL_SLIDING_FRICTION: float = 0.5
BALL_SLIDING_FRICTION_RANGE: tuple[float, float] = (0.4, 0.6)
BALL_TORSIONAL_FRICTION: float = 0.002
BALL_TORSIONAL_FRICTION_RANGE: tuple[float, float] = (0.001, 0.004)
BALL_ROLLING_RESISTANCE: float = 0.060
BALL_ROLLING_RESISTANCE_RANGE: tuple[float, float] = (0.046, 0.070)

# Ball mass spread for DR, as a multiple of nominal. FIFA size 3 allows 300-320 g;
# the wider band also covers a worn or damp ball.
BALL_MASS_SCALE_RANGE: tuple[float, float] = (0.94, 1.07)

# 1.0 under the elliptic friction cone, which reproduces the rigid-body prediction
# exactly. The pyramidal cone needs ~0.957 to match at 1.5 m/s and is still ~5x off
# above 2 m/s, so the tasks set cone="elliptic".
_ROLLING_FRICTION_SOLVER_GAIN: float = 1.0

# Scale factors for each FIFA ball size (relative to size 5)
BALL_SIZE_SCALES: dict[int, dict[str, float]] = {
    1: {"size": 0.6304, "weight": 0.4419},
    2: {"size": 0.8188, "weight": 0.6163},
    3: {"size": 0.8551, "weight": 0.7209},
    4: {"size": 0.9384, "weight": 0.8605},
    5: {"size": 1.0, "weight": 1.0},
}

BallSize = Literal[1, 2, 3, 4, 5]

##
# MJCF and assets.
##

BALL_XML: Path = BOOSTER_MJLAB_SRC_PATH / "environment" / "ball" / "xmls" / "ball.xml"
assert BALL_XML.exists()


def get_ball_radius(size: BallSize = 4) -> float:
    """Get the ball radius for a given FIFA size."""
    return BALL_SIZE_5_RADIUS * BALL_SIZE_SCALES[size]["size"]


def get_ball_mass(size: BallSize = 4) -> float:
    """Get the ball mass for a given FIFA size."""
    return BALL_SIZE_5_MASS * BALL_SIZE_SCALES[size]["weight"]


def get_rolling_friction(
    size: BallSize = 4,
    rolling_resistance: float = BALL_ROLLING_RESISTANCE,
) -> float:
    """Get the MuJoCo rolling-friction coefficient for a given rolling resistance.

    Args:
        size: FIFA ball size (1-5). Default is 4.
        rolling_resistance: Dimensionless C_r = |a| / g the ball should roll out with.

    Returns:
        Rolling-friction coefficient (metres) for the ball's collision geom.
    """
    radius = get_ball_radius(size)
    return (
        (1.0 + BALL_INERTIA_RATIO)
        * rolling_resistance
        * radius
        / _ROLLING_FRICTION_SOLVER_GAIN
    )


def get_ball_friction(
    size: BallSize = 4,
    rolling_resistance: float = BALL_ROLLING_RESISTANCE,
) -> tuple[float, float, float]:
    """Get the (sliding, torsional, rolling) friction triple for the ball."""
    return (
        BALL_SLIDING_FRICTION,
        BALL_TORSIONAL_FRICTION,
        get_rolling_friction(size, rolling_resistance),
    )


def get_ball_friction_dr_ranges() -> dict[int, tuple[float, float]]:
    """Per-axis DR ranges for ``geom_friction``, as scales on the nominal values.

    Scales rather than absolutes so the ranges hold for any ball size: each axis is
    proportional to its nominal, and mu_roll is proportional to the rolling
    resistance. Use with ``operation="scale"`` and ``axes=[0, 1, 2]``.
    """

    def _scale(rng: tuple[float, float], nominal: float) -> tuple[float, float]:
        return (rng[0] / nominal, rng[1] / nominal)

    return {
        0: _scale(BALL_SLIDING_FRICTION_RANGE, BALL_SLIDING_FRICTION),
        1: _scale(BALL_TORSIONAL_FRICTION_RANGE, BALL_TORSIONAL_FRICTION),
        2: _scale(BALL_ROLLING_RESISTANCE_RANGE, BALL_ROLLING_RESISTANCE),
    }


def get_spec(size: BallSize = 4) -> mujoco.MjSpec:
    """Get the MuJoCo spec for the soccer ball.

    Args:
        size: FIFA ball size (1-5). Default is 4.

    Returns:
        MuJoCo spec with the ball model scaled for the specified size.
    """
    spec = mujoco.MjSpec.from_file(str(BALL_XML))

    friction = get_ball_friction(size)

    # Get scale factors
    scale_factor = BALL_SIZE_SCALES[size]["size"]
    mass = get_ball_mass(size)
    radius = get_ball_radius(size)

    # Scale the meshes
    if scale_factor != 1.0:
        for mesh in spec.meshes:
            mesh.scale = (scale_factor, scale_factor, scale_factor)

    # Update ball body properties
    ball_body = spec.worldbody.bodies[0]

    # Ensure visual meshes are massless. Otherwise MuJoCo computes mesh mass from
    # default density and silently inflates total ball mass.
    for geom in ball_body.geoms:
        if geom.name.startswith("ball_visual"):
            geom.mass = 0.0
            geom.density = 0.0
            geom.friction = friction

    # Update collision sphere size and explicit physical mass.
    for geom in ball_body.geoms:
        if geom.name == "ball_collision":
            geom.group = 3
            geom.size: tuple[float, Literal[0], Literal[0]] = (radius, 0, 0)
            if BALL_SHELL_INERTIA:
                geom.typeinertia = mujoco.mjtGeomInertia.mjINERTIA_SHELL
            geom.priority = 1
            geom.condim = BALL_COLLISION_CONDIM
            geom.mass = mass
            geom.solref = BALL_COLLISION_SOLREF
            geom.friction = friction

    return spec


def get_ball_cfg(
    size: BallSize = 4,
    ball_pos: tuple[float, float, float | None] = (2.0, 0.0, None),
) -> EntityCfg:
    """Create a soccer ball entity in front of the robot.

    Args:
        size: FIFA ball size (1-5). Default is 4.
        ball_pos: Initial ball position (x, y, z). If z is None, ball sits on ground
                  (z = ball radius). Default is 2m in front of robot.

    Returns:
        EntityCfg for the ball.
    """
    radius = get_ball_radius(size)

    # Z position: if None or 0, place ball on ground (z = radius)
    z_pos = ball_pos[2] if ball_pos[2] is not None and ball_pos[2] > 0 else radius
    final_pos = (ball_pos[0], ball_pos[1], z_pos)

    def _make_ball_spec():
        spec = get_spec(size)
        # Set the ball body position directly in the spec
        # This is required for free bodies (with freejoint) to spawn at the correct location
        ball_body = spec.worldbody.bodies[0]
        ball_body.pos = final_pos
        return spec

    init_state = EntityCfg.InitialStateCfg(
        pos=final_pos,
        rot=(1.0, 0.0, 0.0, 0.0),
    )

    return EntityCfg(
        init_state=init_state,
        spec_fn=_make_ball_spec,
    )


##
# Convenience constants (Size 4 default for humanoid robotics).
##

BALL_RADIUS: float = get_ball_radius(4)
BALL_MASS: float = get_ball_mass(4)
