"""Soccer goal constants."""

import math
from enum import StrEnum
from pathlib import Path
from typing import Literal

import mujoco

from booster_mjlab import BOOSTER_MJLAB_SRC_PATH


##
# Division enum and scale factors.
##


class Division(StrEnum):
    """Soccer goal division sizes."""

    SMALL = "small"
    MID = "mid"
    LARGE = "large"


# Scale factors relative to small division (base mesh size)
DIVISION_SCALE: dict[Division, float] = {
    Division.SMALL: 1.0,
    Division.MID: 240 / 180,  # ~1.333
    Division.LARGE: 300 / 180,  # ~1.667
}


##
# MJCF and assets.
##

GOAL_XML: Path = BOOSTER_MJLAB_SRC_PATH / "environment" / "goal" / "xmls" / "goal.xml"
assert GOAL_XML.exists()


def get_spec(
    division: Division | Literal["small", "mid", "large"] = Division.SMALL,
) -> mujoco.MjSpec:
    """Get the MuJoCo spec for the soccer goal.

    Args:
        division: The goal division size. The mesh is scaled accordingly.
                  Options: Division.SMALL, Division.MID, Division.LARGE
                  or string literals "small", "mid", "large".

    Returns:
        MuJoCo spec with the goal model scaled for the specified division.
    """
    # Convert string to enum if needed
    if isinstance(division, str):
        division = Division(division)

    spec = mujoco.MjSpec.from_file(str(GOAL_XML))

    for mesh in spec.meshes:
        # Separate meshes for each division by appending division name to mesh name
        mesh.name = f"{mesh.name}_{division.value}"

    for geom in spec.geoms:
        # Update only mesh-backed geoms to match the new mesh names.
        if geom.meshname:
            geom.meshname = f"{geom.meshname}_{division.value}"

        # Ensure all collidable geoms are consistently tagged as collision group.
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.group = 3

    # Apply scale factor for the division
    scale = DIVISION_SCALE[division]
    if scale != 1.0:
        # Scale the mesh
        mesh = spec.meshes[0]
        mesh.scale = (scale, scale, scale)

        # Scale primitive collision geoms to stay aligned with the visual mesh.
        for geom in spec.geoms:
            if geom.group != 3:
                continue

            geom.pos = tuple(v * scale for v in geom.pos)
            geom.size = tuple(v * scale for v in geom.size)

            # fromto geoms use explicit endpoints instead of pos/quat.
            if not math.isnan(geom.fromto[0]):
                geom.fromto = tuple(v * scale for v in geom.fromto)

    return spec


##
# Goal dimensions by division.
##

# Small Division: 180cm × 120cm (6ft × 4ft)
SMALL_GOAL_WIDTH: float = 1.80  # meters
SMALL_GOAL_HEIGHT: float = 1.20  # meters

# Mid Division: 240cm × 160cm (8ft × 5ft)
MID_GOAL_WIDTH: float = 2.40  # meters
MID_GOAL_HEIGHT: float = 1.60  # meters

# Large Division: 300cm × 200cm (10ft × 6.5ft)
LARGE_GOAL_WIDTH: float = 3.00  # meters
LARGE_GOAL_HEIGHT: float = 2.00  # meters
