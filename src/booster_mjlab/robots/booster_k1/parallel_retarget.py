"""Retarget serial-K1 motion clips onto the parallel-ankle joint layout.

Serial clips store ``dof_pos`` as (frames, 22). The parallel model has 34 joints,
so every frame's ankle pitch/roll is solved through the linkage (see
``parallel_ankle_ik``) to fill in the crank and rod columns. All other channels
are copied straight across.

Ankle poses outside the linkage's reachable workspace, or that need a crank angle
past its hard stop, are pulled toward the home ankle pose until they fit.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from booster_mjlab.motion import MotionFile
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    HOME_KEYFRAME,
    K1_PARALLEL_JOINT_ORDER,
)
from booster_mjlab.robots.booster_k1.parallel_ankle_ik import (
    LINKAGE_JOINT_SUFFIXES,
    reachable,
    solve_leg,
)

# Crank hard stops, from the MJCF.
_DRIVE_RANGE = {"Ankle_A": (-0.55, 0.8), "Ankle_B": (-0.5, 1.1)}

_HOME_ANKLE_PITCH = HOME_KEYFRAME.joint_pos["Left_Ankle_Pitch"]
_HOME_ANKLE_ROLL = 0.0


def _feasible(side: str, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """Frames whose ankle pose both closes the loop and respects the stops."""
    ok = reachable(side, pitch, roll)
    angles = solve_leg(side, pitch, roll)
    for index, suffix in enumerate(LINKAGE_JOINT_SUFFIXES):
        limits = _DRIVE_RANGE.get(suffix)
        if limits is None:
            continue
        column = angles[..., index]
        ok &= np.isfinite(column) & (column >= limits[0]) & (column <= limits[1])
    return ok


def _project_to_workspace(
    side: str, pitch: np.ndarray, roll: np.ndarray, steps: int = 24
) -> tuple[np.ndarray, np.ndarray, int]:
    """Blend infeasible ankle poses toward the home pose until they fit."""
    pitch = pitch.copy()
    roll = roll.copy()
    bad = ~_feasible(side, pitch, roll)
    num_adjusted = int(bad.sum())
    if num_adjusted == 0:
        return pitch, roll, 0

    # Bisect the blend factor per frame: alpha=1 keeps the original pose,
    # alpha=0 collapses to home.
    lo = np.zeros_like(pitch)
    hi = np.ones_like(pitch)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        trial_pitch = _HOME_ANKLE_PITCH + mid * (pitch - _HOME_ANKLE_PITCH)
        trial_roll = _HOME_ANKLE_ROLL + mid * (roll - _HOME_ANKLE_ROLL)
        ok = _feasible(side, trial_pitch, trial_roll)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    pitch = np.where(bad, _HOME_ANKLE_PITCH + lo * (pitch - _HOME_ANKLE_PITCH), pitch)
    roll = np.where(bad, _HOME_ANKLE_ROLL + lo * (roll - _HOME_ANKLE_ROLL), roll)
    return pitch, roll, num_adjusted


def retarget_dof_pos(dof_pos: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Expand (frames, 22) serial dof positions to (frames, 34) parallel ones."""
    if dof_pos.shape[1] != len(K1_JOINT_ORDER):
        raise ValueError(
            f"Expected {len(K1_JOINT_ORDER)} serial dof columns, got {dof_pos.shape[1]}"
        )

    serial_index = {name: i for i, name in enumerate(K1_JOINT_ORDER)}
    out = np.zeros((dof_pos.shape[0], len(K1_PARALLEL_JOINT_ORDER)), dtype=np.float64)
    parallel_index = {name: i for i, name in enumerate(K1_PARALLEL_JOINT_ORDER)}

    for name, column in serial_index.items():
        out[:, parallel_index[name]] = dof_pos[:, column]

    stats = {}
    for side in ("Left", "Right"):
        pitch = dof_pos[:, serial_index[f"{side}_Ankle_Pitch"]].astype(np.float64)
        roll = dof_pos[:, serial_index[f"{side}_Ankle_Roll"]].astype(np.float64)
        pitch, roll, num_adjusted = _project_to_workspace(side, pitch, roll)
        stats[side] = num_adjusted

        # Keep the ankle columns consistent with the linkage that was solved.
        out[:, parallel_index[f"{side}_Ankle_Pitch"]] = pitch
        out[:, parallel_index[f"{side}_Ankle_Roll"]] = roll

        angles = solve_leg(side, pitch, roll)
        for index, suffix in enumerate(LINKAGE_JOINT_SUFFIXES):
            out[:, parallel_index[f"{side}_{suffix}"]] = angles[:, index]

    return out, stats


def to_parallel(motion: MotionFile) -> MotionFile:
    """Return ``motion`` in the parallel joint layout, retargeting if it is serial."""
    dof_pos = np.asarray(motion.dof_pos)
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected 2D dof_pos, got shape {dof_pos.shape}")
    if dof_pos.shape[1] == len(K1_PARALLEL_JOINT_ORDER):
        return motion

    retargeted, _ = retarget_dof_pos(dof_pos)
    return dataclasses.replace(
        motion, dof_pos=retargeted.astype(dof_pos.dtype, copy=False)
    )
