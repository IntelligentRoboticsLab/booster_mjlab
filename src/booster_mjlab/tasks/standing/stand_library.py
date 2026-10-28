"""Standing reference library for the K1: leg joint angles for every torso command, feet planted.

The reference of Disney's BDX "perpetual standing" policy: the operator commands the torso's height
offset, roll, pitch and yaw, and the reference pose keeps both feet exactly where they are in the
default stance (HOME_KEYFRAME). For each point of a (height, roll, pitch, yaw) grid the legs are
solved with damped least-squares IK on both feet's full 6D pose; arms stay at the keyframe and the
head is commanded separately.

The grid spans each axis's full single-axis range (every leg joint within 95% of its limit). Not
every combination of extremes is reachable (e.g. deepest crouch + full lean + full twist runs out of
ankle roll / hip range), so each grid point also stores whether the IK held both feet; the command
term scales unreachable combinations back toward neutral (StandCommand).

  uv run python -m booster_mjlab.tasks.standing.stand_library     # writes data/k1_stand_library.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from booster_mjlab.robots.booster_k1.k1_constants import (
    HOME_KEYFRAME,
    K1_JOINT_ORDER,
    get_spec,
)

DATA = Path(__file__).parent / "data" / "k1_stand_library.npz"
FEET = ("left_foot_link", "right_foot_link")
LEGS = tuple(
    f"{side}_{joint}"
    for side in ("Left", "Right")
    for joint in (
        "Hip_Pitch",
        "Hip_Roll",
        "Hip_Yaw",
        "Knee_Pitch",
        "Ankle_Pitch",
        "Ankle_Roll",
    )
)
# Single-axis maxima (dheight m, roll / pitch / yaw rad) with both feet planted and every leg joint
# inside 95% of its range: crouch 13.5 cm / rise 3 cm, roll 17 deg, lean 54 deg forward / 48 deg
# back, twist 41 deg (python -m booster_mjlab.tasks.standing.stand_library --probe).
RANGES = {
    "height": (-0.135, 0.03),
    "roll": (-0.29, 0.29),
    "pitch": (-0.845, 0.94),
    "yaw": (-0.72, 0.72),
}
GRID = (12, 9, 17, 13)  # ~1.5 cm, ~4 deg, ~6 deg, ~7 deg spacing
LIMIT_USE = 0.95
FOOT_POS_TOL = 1e-3  # m
FOOT_ROT_TOL = np.radians(0.5)


def torso_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(w, x, y, z) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    out = np.array([1.0, 0.0, 0.0, 0.0])
    for angle, axis in ((yaw, (0, 0, 1)), (pitch, (0, 1, 0)), (roll, (1, 0, 0))):
        q, res = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array(axis, float), angle)
        mujoco.mju_mulQuat(res, out, q)
        out = res
    return out


class StandIK:
    """Damped least-squares IK of both legs for a commanded torso pose, feet fixed in the world."""

    def __init__(self):
        self.model = get_spec().compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = {n: m.jnt_qposadr[m.joint(n).id] for n in K1_JOINT_ORDER}
        self.leg_q = np.array([self.qadr[n] for n in LEGS])
        self.leg_d = np.array([m.jnt_dofadr[m.joint(n).id] for n in LEGS])
        lo = np.array([m.jnt_range[m.joint(n).id][0] for n in LEGS])
        hi = np.array([m.jnt_range[m.joint(n).id][1] for n in LEGS])
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        self.lo, self.hi = mid - LIMIT_USE * half, mid + LIMIT_USE * half
        self.feet = [m.body(f).id for f in FEET]
        self.home = m.qpos0.copy()
        self.home[0:3] = HOME_KEYFRAME.pos
        self.home[3:7] = (1, 0, 0, 0)
        for name, value in HOME_KEYFRAME.joint_pos.items():
            self.home[self.qadr[name]] = value
        self.data.qpos[:] = self.home
        mujoco.mj_kinematics(m, self.data)
        self.foot_pos = [self.data.xpos[b].copy() for b in self.feet]
        self.foot_quat = [self.data.xquat[b].copy() for b in self.feet]
        self.base_pos0 = self.home[0:3].copy()

    def _foot_errors(self):
        errs = []
        for body, p0, q0 in zip(self.feet, self.foot_pos, self.foot_quat):
            rot, q_err, neg = np.zeros(3), np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(neg, self.data.xquat[body])
            mujoco.mju_mulQuat(q_err, q0, neg)
            mujoco.mju_quat2Vel(rot, q_err, 1.0)
            errs.append((p0 - self.data.xpos[body], rot))
        return errs

    def solve(self, dh: float, roll: float, pitch: float, yaw: float, iters: int = 200):
        """Leg angles (LEGS order); returns (angles, worst foot position error m, rotation error rad)."""
        m, d = self.model, self.data
        d.qpos[:] = self.home
        d.qpos[0:3] = self.base_pos0 + (0, 0, dh)
        d.qpos[3:7] = torso_quat(roll, pitch, yaw)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e = np.concatenate([np.concatenate(pr) for pr in self._foot_errors()])
            if np.linalg.norm(e) < 1e-7:
                break
            jac = []
            for body in self.feet:
                mujoco.mj_jacBody(m, d, jp, jr, body)
                jac.append(np.vstack([jp[:, self.leg_d], jr[:, self.leg_d]]))
            J = np.vstack(jac)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            d.qpos[self.leg_q] = np.clip(d.qpos[self.leg_q] + dq, self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        errs = self._foot_errors()
        pos_err = max(np.linalg.norm(p) for p, _ in errs)
        rot_err = max(np.linalg.norm(r) for _, r in errs)
        return d.qpos[self.leg_q].copy(), float(pos_err), float(rot_err)

    def reachable(self, *cmd) -> bool:
        _, pos_err, rot_err = self.solve(*cmd)
        return pos_err < FOOT_POS_TOL and rot_err < FOOT_ROT_TOL


def build(ranges=RANGES, grid=GRID) -> dict:
    ik = StandIK()
    keys = ("height", "roll", "pitch", "yaw")
    axes = [np.linspace(*ranges[k], n) for k, n in zip(keys, grid)]
    legs = np.zeros((*grid, len(LEGS)))
    reachable = np.zeros(grid, dtype=bool)
    for idx in np.ndindex(*grid):
        q, pos_err, rot_err = ik.solve(*(ax[i] for ax, i in zip(axes, idx)))
        legs[idx] = q
        reachable[idx] = pos_err < FOOT_POS_TOL and rot_err < FOOT_ROT_TOL
    home = np.array([ik.home[ik.qadr[n]] for n in K1_JOINT_ORDER])
    return dict(
        height=axes[0],
        roll=axes[1],
        pitch=axes[2],
        yaw=axes[3],
        leg_joint_pos=legs.astype(np.float32),
        reachable=reachable,
        leg_joint_names=np.array(LEGS),
        joint_names=np.array(K1_JOINT_ORDER),
        home_joint_pos=home.astype(np.float32),
        base_height=np.float32(ik.base_pos0[2]),
    )


def probe(step: float = 0.005) -> None:
    """Print each axis's single-axis reach (the others at 0)."""
    ik = StandIK()
    for i, name in enumerate(("height", "roll", "pitch", "yaw")):
        reach = []
        for sign in (-1, 1):
            best, c = 0.0, step
            while c < 1.5:
                cmd = [0.0] * 4
                cmd[i] = sign * c
                if not ik.reachable(*cmd):
                    break
                best, c = c, c + step
            reach.append(sign * best)
        print(f"{name:7s} {reach[0]:+.3f} .. {reach[1]:+.3f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=DATA)
    ap.add_argument(
        "--probe", action="store_true", help="print each axis's reach and exit"
    )
    args = ap.parse_args()
    if args.probe:
        probe()
        return
    lib = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **lib)
    r = lib["reachable"]
    print(
        f"{args.out}: grid {r.shape}, {r.mean():.0%} of the combinations reachable "
        f"(feet within {1000 * FOOT_POS_TOL:.0f} mm / {np.degrees(FOOT_ROT_TOL):.1f} deg)"
    )


if __name__ == "__main__":
    main()
