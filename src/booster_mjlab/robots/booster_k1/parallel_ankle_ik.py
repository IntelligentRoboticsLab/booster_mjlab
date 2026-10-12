"""Closed-form inverse kinematics for a Booster parallel ankle linkage.

Maps a serial ankle pose ``(pitch, roll)`` onto the six joint angles the
closed-loop model needs per leg: the two crank drives and the two rod universal
joints. Geometry is expressed in the shank frame; :data:`K1_ANKLE` is taken from
``xmls/k1_parallel.xml`` and is the default, other robots pass their own
:class:`AnkleGeometry`.

Each linkage reduces to one scalar equation, ``A cos(theta) + B sin(theta) = C``,
because the rod length is fixed and the crank pivots about a single axis. The
rod's own two hinges then follow from the resulting rod direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ankle cross origin in the shank frame; the foot link sits at the same point.
_ANKLE_ORIGIN = np.array([0.00019706, 0.0002, -0.24519])

# Rod anchors in the foot frame.
_ANCHOR_A = np.array([-0.029, 0.024, 0.0175])
_ANCHOR_B = np.array([-0.029, -0.024, 0.0175])

# Crank pivots in the shank frame, crank arm vectors, and rod vectors, all at
# the zero pose (where the loop closes exactly).
_CRANK_A_PIVOT = np.array([-0.00148194, 0.0242, -0.081637428])
_CRANK_A_ARM = np.array([0.0408490153, 0.0, -0.00351538739])
_ROD_A = np.array([-0.0681700153, 0.0, -0.142537185])

_CRANK_B_PIVOT = np.array([-0.00967794, -0.0238, -0.140057167])
_CRANK_B_ARM = np.array([0.032999524, 0.0, -0.0243316957])
_ROD_B = np.array([-0.052124524, 0.0, -0.0633011373])


@dataclass(frozen=True)
class _Linkage:
    pivot: np.ndarray
    arm: np.ndarray
    rod: np.ndarray
    anchor: np.ndarray

    def mirrored(self) -> "_Linkage":
        return _Linkage(
            pivot=self.pivot * _FLIP_Y,
            arm=self.arm * _FLIP_Y,
            rod=self.rod * _FLIP_Y,
            anchor=self.anchor * _FLIP_Y,
        )


_FLIP_Y = np.array([1.0, -1.0, 1.0])


@dataclass(frozen=True)
class AnkleGeometry:
    """Left-leg linkage geometry; the right leg is its mirror image in y."""

    origin: np.ndarray
    """Ankle pitch axis origin (ankle cross body) in the shank frame."""

    foot_offset: np.ndarray
    """Ankle roll axis origin (foot body) in the ankle cross frame."""

    linkage_a: _Linkage
    linkage_b: _Linkage

    def side(self, side: str) -> tuple[np.ndarray, np.ndarray, dict[str, _Linkage]]:
        linkages = {"A": self.linkage_a, "B": self.linkage_b}
        if side == "Left":
            return self.origin, self.foot_offset, linkages
        return (
            self.origin * _FLIP_Y,
            self.foot_offset * _FLIP_Y,
            {tag: linkage.mirrored() for tag, linkage in linkages.items()},
        )


K1_ANKLE = AnkleGeometry(
    origin=_ANKLE_ORIGIN,
    # The K1 foot link sits on the ankle cross origin.
    foot_offset=np.zeros(3),
    linkage_a=_Linkage(_CRANK_A_PIVOT, _CRANK_A_ARM, _ROD_A, _ANCHOR_A),
    linkage_b=_Linkage(_CRANK_B_PIVOT, _CRANK_B_ARM, _ROD_B, _ANCHOR_B),
)

# Joint order returned by `solve_leg`, matching the MJCF definition order.
LINKAGE_JOINT_SUFFIXES: tuple[str, ...] = (
    "Ankle_A",
    "Ankle_Rod_A_Pitch",
    "Ankle_Rod_A_Roll",
    "Ankle_B",
    "Ankle_Rod_B_Pitch",
    "Ankle_Rod_B_Roll",
)


def _anchor_in_shank(
    origin: np.ndarray,
    foot_offset: np.ndarray,
    anchor: np.ndarray,
    pitch: np.ndarray,
    roll: np.ndarray,
) -> np.ndarray:
    """Anchor position in the shank frame for a batch of ankle poses."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ax, ay, az = anchor

    # Rx(roll) on the anchor offset, shifted to the roll axis, then Ry(pitch).
    x_r = np.broadcast_to(ax + foot_offset[0], pitch.shape)
    y_r = cr * ay - sr * az + foot_offset[1]
    z_r = sr * ay + cr * az + foot_offset[2]

    x = cp * x_r + sp * z_r
    y = y_r
    z = -sp * x_r + cp * z_r

    return origin + np.stack([x, y, z], axis=-1)


def _solve_crank(linkage: _Linkage, anchor_w: np.ndarray) -> np.ndarray:
    """Crank angle placing the rod end exactly on the anchor."""
    tx, _, tz = linkage.arm
    d = anchor_w - linkage.pivot
    a = d[..., 0] * tx + d[..., 2] * tz
    b = d[..., 0] * tz - d[..., 2] * tx
    cos_arg = np.clip(_crank_cos_arg(linkage, anchor_w), -1.0, 1.0)
    # The branch is fixed by requiring theta == 0 at the zero ankle pose.
    return np.arctan2(b, a) - np.arccos(cos_arg)


def _crank_cos_arg(linkage: _Linkage, anchor_w: np.ndarray) -> np.ndarray:
    """The arccos argument of the crank solve; |value| > 1 means unreachable."""
    rod_length_sq = float(linkage.rod @ linkage.rod)
    tx, _, tz = linkage.arm
    d = anchor_w - linkage.pivot
    dx, dz = d[..., 0], d[..., 2]
    a = dx * tx + dz * tz
    b = dx * tz - dz * tx
    c = 0.5 * (tx * tx + tz * tz + np.sum(d * d, axis=-1) - rod_length_sq)
    return c / np.hypot(a, b)


def reachable(
    side: str,
    ankle_pitch: np.ndarray,
    ankle_roll: np.ndarray,
    geometry: AnkleGeometry = K1_ANKLE,
) -> np.ndarray:
    """Whether both linkages can close the loop at the given ankle pose."""
    pitch, roll = np.broadcast_arrays(
        np.asarray(ankle_pitch, dtype=np.float64),
        np.asarray(ankle_roll, dtype=np.float64),
    )
    origin, foot_offset, linkages = geometry.side(side)
    ok = np.ones(pitch.shape, dtype=bool)
    for linkage in linkages.values():
        anchor_w = _anchor_in_shank(origin, foot_offset, linkage.anchor, pitch, roll)
        ok &= np.abs(_crank_cos_arg(linkage, anchor_w)) <= 1.0
    return ok


def _solve_rod(linkage: _Linkage, theta: np.ndarray, anchor_w: np.ndarray):
    """Rod pitch/roll hinge angles given the crank angle."""
    ct, st = np.cos(theta), np.sin(theta)

    arm = linkage.arm
    tip = linkage.pivot + np.stack(
        [
            ct * arm[0] + st * arm[2],
            np.broadcast_to(arm[1], theta.shape),
            -st * arm[0] + ct * arm[2],
        ],
        axis=-1,
    )

    # Rod target direction, expressed in the crank frame (undo Ry(theta)).
    delta = anchor_w - tip
    vx = ct * delta[..., 0] - st * delta[..., 2]
    vy = delta[..., 1]
    vz = st * delta[..., 0] + ct * delta[..., 2]

    rx, _, rz = linkage.rod
    roll = np.arcsin(np.clip(-vy / rz, -1.0, 1.0))
    k = rz * np.cos(roll)
    denom = rx * rx + k * k
    pitch = np.arctan2((k * vx - rx * vz) / denom, (rx * vx + k * vz) / denom)
    return pitch, roll


def solve_leg(
    side: str,
    ankle_pitch: np.ndarray,
    ankle_roll: np.ndarray,
    geometry: AnkleGeometry = K1_ANKLE,
) -> np.ndarray:
    """Solve one leg's linkage angles.

    Args:
      side: ``"Left"`` or ``"Right"``.
      ankle_pitch: Ankle pitch angles, any shape.
      ankle_roll: Ankle roll angles, broadcastable against ``ankle_pitch``.
      geometry: Linkage geometry of the robot.

    Returns:
      Array of shape ``(..., 6)`` ordered as :data:`LINKAGE_JOINT_SUFFIXES`.
    """
    if side not in ("Left", "Right"):
        raise ValueError(f"side must be 'Left' or 'Right', got {side!r}")

    pitch, roll = np.broadcast_arrays(
        np.asarray(ankle_pitch, dtype=np.float64),
        np.asarray(ankle_roll, dtype=np.float64),
    )
    origin, foot_offset, linkages = geometry.side(side)

    columns = []
    for tag in ("A", "B"):
        linkage = linkages[tag]
        anchor_w = _anchor_in_shank(origin, foot_offset, linkage.anchor, pitch, roll)
        theta = _solve_crank(linkage, anchor_w)
        rod_pitch, rod_roll = _solve_rod(linkage, theta, anchor_w)
        columns.extend([theta, rod_pitch, rod_roll])

    return np.stack(columns, axis=-1)
