"""Retarget serial motion clips onto a parallel-ankle joint layout.

Serial K1 clips store ``dof_pos`` as (frames, 22). The parallel model has 34 joints,
so every frame's ankle pitch/roll is solved through the linkage (see
``parallel_ankle_ik``) to fill in the crank and rod columns. All other channels
are copied straight across. :data:`K1_LAYOUT` is the default; other robots pass
their own :class:`ParallelLayout`.

Ankle poses outside the linkage's reachable workspace, or that need a crank angle
past its hard stop, are pulled toward the home ankle pose until they fit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from booster_mjlab.motion import MotionFile
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    HOME_KEYFRAME,
    K1_PARALLEL_JOINT_ORDER,
)
from booster_mjlab.robots.booster_k1.parallel_ankle_ik import (
    K1_ANKLE,
    LINKAGE_JOINT_SUFFIXES,
    AnkleGeometry,
    reachable,
    solve_leg,
)


@dataclass(frozen=True)
class ParallelLayout:
    """Serial and parallel joint layouts of one robot, plus its linkage."""

    serial_joints: tuple[str, ...]
    parallel_joints: tuple[str, ...]
    drive_range: dict[str, tuple[float, float]]
    """Crank hard stops, from the MJCF."""
    home_ankle_pitch: float
    geometry: AnkleGeometry


K1_LAYOUT = ParallelLayout(
    serial_joints=K1_JOINT_ORDER,
    parallel_joints=K1_PARALLEL_JOINT_ORDER,
    drive_range={"Ankle_A": (-0.55, 0.8), "Ankle_B": (-0.5, 1.1)},
    home_ankle_pitch=HOME_KEYFRAME.joint_pos["Left_Ankle_Pitch"],
    geometry=K1_ANKLE,
)

_HOME_ANKLE_ROLL = 0.0


def _feasible(
    layout: ParallelLayout, side: str, pitch: np.ndarray, roll: np.ndarray
) -> np.ndarray:
    """Frames whose ankle pose both closes the loop and respects the stops."""
    ok = reachable(side, pitch, roll, layout.geometry)
    angles = solve_leg(side, pitch, roll, layout.geometry)
    for index, suffix in enumerate(LINKAGE_JOINT_SUFFIXES):
        limits = layout.drive_range.get(suffix)
        if limits is None:
            continue
        column = angles[..., index]
        ok &= np.isfinite(column) & (column >= limits[0]) & (column <= limits[1])
    return ok


def _project_to_workspace(
    layout: ParallelLayout,
    side: str,
    pitch: np.ndarray,
    roll: np.ndarray,
    steps: int = 24,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Blend infeasible ankle poses toward the home pose until they fit."""
    home_pitch = layout.home_ankle_pitch
    pitch = pitch.copy()
    roll = roll.copy()
    bad = ~_feasible(layout, side, pitch, roll)
    num_adjusted = int(bad.sum())
    if num_adjusted == 0:
        return pitch, roll, 0

    # Bisect the blend factor per frame: alpha=1 keeps the original pose,
    # alpha=0 collapses to home.
    lo = np.zeros_like(pitch)
    hi = np.ones_like(pitch)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        trial_pitch = home_pitch + mid * (pitch - home_pitch)
        trial_roll = _HOME_ANKLE_ROLL + mid * (roll - _HOME_ANKLE_ROLL)
        ok = _feasible(layout, side, trial_pitch, trial_roll)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    pitch = np.where(bad, home_pitch + lo * (pitch - home_pitch), pitch)
    roll = np.where(bad, _HOME_ANKLE_ROLL + lo * (roll - _HOME_ANKLE_ROLL), roll)
    return pitch, roll, num_adjusted


def retarget_dof_pos(
    dof_pos: np.ndarray, layout: ParallelLayout = K1_LAYOUT
) -> tuple[np.ndarray, dict[str, int]]:
    """Expand serial dof positions (e.g. K1: 22 columns) to the parallel layout (34)."""
    serial_joints, parallel_joints = layout.serial_joints, layout.parallel_joints
    if dof_pos.shape[1] != len(serial_joints):
        raise ValueError(
            f"Expected {len(serial_joints)} serial dof columns, got {dof_pos.shape[1]}"
        )

    serial_index = {name: i for i, name in enumerate(serial_joints)}
    out = np.zeros((dof_pos.shape[0], len(parallel_joints)), dtype=np.float64)
    parallel_index = {name: i for i, name in enumerate(parallel_joints)}

    for name, column in serial_index.items():
        out[:, parallel_index[name]] = dof_pos[:, column]

    stats = {}
    for side in ("Left", "Right"):
        pitch = dof_pos[:, serial_index[f"{side}_Ankle_Pitch"]].astype(np.float64)
        roll = dof_pos[:, serial_index[f"{side}_Ankle_Roll"]].astype(np.float64)
        pitch, roll, num_adjusted = _project_to_workspace(layout, side, pitch, roll)
        stats[side] = num_adjusted

        # Keep the ankle columns consistent with the linkage that was solved.
        out[:, parallel_index[f"{side}_Ankle_Pitch"]] = pitch
        out[:, parallel_index[f"{side}_Ankle_Roll"]] = roll

        angles = solve_leg(side, pitch, roll, layout.geometry)
        for index, suffix in enumerate(LINKAGE_JOINT_SUFFIXES):
            out[:, parallel_index[f"{side}_{suffix}"]] = angles[:, index]

    return out, stats


def to_parallel(motion: MotionFile, layout: ParallelLayout = K1_LAYOUT) -> MotionFile:
    """Return ``motion`` in the parallel joint layout, retargeting if it is serial."""
    dof_pos = np.asarray(motion.dof_pos)
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected 2D dof_pos, got shape {dof_pos.shape}")
    if dof_pos.shape[1] == len(layout.parallel_joints):
        return motion

    retargeted, _ = retarget_dof_pos(dof_pos, layout)
    return dataclasses.replace(
        motion, dof_pos=retargeted.astype(dof_pos.dtype, copy=False)
    )
