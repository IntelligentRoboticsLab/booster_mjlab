"""Stand command: the operator's torso (height, roll, pitch, yaw) + head (yaw, pitch) command and the
reference pose it maps to (k1_stand_library.npz).

Every few seconds a new target is drawn (sometimes neutral, sometimes a single axis, otherwise all
axes at once). The command moves toward it at a speed drawn per target, from slow to the K1's fast
moves, so the policy follows commands at any speed in that range -- a joystick or a scripted motion
can then drive it slowly or snappily. Torso combinations the legs can't reach with both feet planted
are scaled back toward neutral (the library's reachability mask).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from booster_mjlab.tasks.standing.stand_library import DATA

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

AXES = ("height", "roll", "pitch", "yaw", "head_yaw", "head_pitch")


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class StandLibrary:
    """The grid on the GPU, with multilinear interpolation of leg angles and reachability."""

    def __init__(self, path: str | Path, device: str):
        d = np.load(path)
        self.axes = [
            torch.tensor(d[k], dtype=torch.float32, device=device) for k in AXES[:4]
        ]
        self.legs = torch.tensor(d["leg_joint_pos"], dtype=torch.float32, device=device)
        self.reach = torch.tensor(d["reachable"], dtype=torch.float32, device=device)
        self.leg_names = [str(n) for n in d["leg_joint_names"]]
        self.base_height = float(d["base_height"])
        self.lo = torch.stack([a[0] for a in self.axes])
        self.hi = torch.stack([a[-1] for a in self.axes])

    def _corners(self, cmd: torch.Tensor):
        """Lower grid index and weight per axis for a (N, 4) torso command."""
        idx, w = [], []
        for k, ax in enumerate(self.axes):
            x = cmd[:, k].clamp(ax[0], ax[-1])
            i = (torch.searchsorted(ax, x.contiguous(), right=True) - 1).clamp(
                0, len(ax) - 2
            )
            idx.append(i)
            w.append((x - ax[i]) / (ax[i + 1] - ax[i]))
        return idx, w

    def interp(self, table: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        idx, w = self._corners(cmd)
        out = 0.0
        for corner in range(16):
            bits = [(corner >> b) & 1 for b in range(4)]
            wt = torch.ones_like(w[0])
            for wk, bk in zip(w, bits):
                wt = wt * (wk if bk else 1 - wk)
            val = table[
                idx[0] + bits[0], idx[1] + bits[1], idx[2] + bits[2], idx[3] + bits[3]
            ]
            out = out + (wt[:, None] * val if val.dim() == 2 else wt * val)
        return out

    def leg_angles(self, cmd: torch.Tensor) -> torch.Tensor:
        return self.interp(self.legs, cmd)

    def reachable(self, cmd: torch.Tensor) -> torch.Tensor:
        """True where every grid point around the command is reachable (conservative)."""
        return self.interp(self.reach, cmd) > 0.999

    def fit(self, cmd: torch.Tensor, iters: int = 8) -> torch.Tensor:
        """Scale unreachable torso commands toward neutral until reachable (bisection)."""
        ok = self.reachable(cmd)
        if bool(ok.all()):
            return cmd
        lo = torch.zeros(cmd.shape[0], device=cmd.device)
        hi = torch.ones(cmd.shape[0], device=cmd.device)
        for _ in range(iters):
            mid = (lo + hi) / 2
            good = self.reachable(cmd * mid[:, None])
            lo = torch.where(good, mid, lo)
            hi = torch.where(good, hi, mid)
        return torch.where(ok[:, None], cmd, cmd * lo[:, None])


class StandCommand(CommandTerm):
    cfg: StandCommandCfg

    def __init__(self, cfg: StandCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.lib = StandLibrary(cfg.library_path, self.device)
        names = list(self.robot.joint_names)
        self.leg_ids = torch.tensor(
            [names.index(n) for n in self.lib.leg_names], device=self.device
        )
        self.head_ids = torch.tensor(
            [names.index(n) for n in ("Head_Yaw", "Head_Pitch")], device=self.device
        )
        self.foot_ids = torch.tensor(
            [self.robot.body_names.index(b) for b in cfg.foot_body_names],
            device=self.device,
        )
        n = self.num_envs
        head_lo, head_hi = [], []
        for j in ("Head_Yaw", "Head_Pitch"):
            lim = self.robot.data.soft_joint_pos_limits[0, names.index(j)]
            head_lo.append(float(lim[0]))
            head_hi.append(float(lim[1]))
        self.cmd_lo = torch.cat(
            [self.lib.lo, torch.tensor(head_lo, device=self.device)]
        )
        self.cmd_hi = torch.cat(
            [self.lib.hi, torch.tensor(head_hi, device=self.device)]
        )
        self.max_rate = torch.tensor(cfg.max_rate, device=self.device)
        self.range_scale = 1.0  # curriculum (stand_range_ramp)
        self.target = torch.zeros(n, 6, device=self.device)
        self.cmd = torch.zeros(n, 6, device=self.device)
        self.speed = torch.ones(
            n, device=self.device
        )  # share of max_rate for the current target
        self.ref_joint_pos = self.robot.data.default_joint_pos.clone()
        self.ref_torso_height = torch.full(
            (n,), self.lib.base_height, device=self.device
        )
        self.ref_torso_quat = torch.zeros(n, 4, device=self.device)
        self.ref_torso_quat[:, 0] = 1.0
        self.metrics["torso_height_error"] = torch.zeros(n, device=self.device)
        self.metrics["torso_orientation_error"] = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """(height, roll, pitch, yaw, head yaw, head pitch) offsets from the default stance."""
        return self.cmd

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        r = torch.rand(n, 6, device=self.device)
        lo, hi = self.cmd_lo * self.range_scale, self.cmd_hi * self.range_scale
        target = lo + r * (hi - lo)
        u = torch.rand(n, device=self.device)
        zero = u < self.cfg.rel_neutral_envs
        single = (u >= self.cfg.rel_neutral_envs) & (
            u < self.cfg.rel_neutral_envs + self.cfg.rel_single_axis_envs
        )
        axis = torch.randint(0, 6, (n,), device=self.device)
        keep = torch.nn.functional.one_hot(axis, 6).bool()
        target = torch.where(single[:, None] & ~keep, torch.zeros_like(target), target)
        target[zero] = 0.0
        target[:, :4] = self.lib.fit(target[:, :4])
        self.target[env_ids] = target
        lo_s, hi_s = self.cfg.speed_range
        self.speed[env_ids] = lo_s + torch.rand(n, device=self.device) * (hi_s - lo_s)
        first = (
            self.command_counter[env_ids] == 0
        )  # an episode reset: start at the target
        self.cmd[env_ids[first]] = target[first]

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        """Per step (env_ids None): move every command toward its target. On reset: only refresh the
        reference of the reset envs (their command was set to the new target)."""
        ids = slice(None) if env_ids is None else env_ids
        if env_ids is None:
            step = self.max_rate * self.speed[:, None] * self._env.step_dt
            self.cmd = self.cmd + (self.target - self.cmd).clamp(-step, step)
        cmd = self.cmd[ids]
        cmd[:, :4] = self.lib.fit(cmd[:, :4])
        self.cmd[ids] = cmd
        # reference: library legs, commanded head, keyframe arms
        ref = self.robot.data.default_joint_pos[ids].clone()
        ref[:, self.leg_ids] = self.lib.leg_angles(cmd[:, :4])
        ref[:, self.head_ids] = cmd[:, 4:]
        self.ref_joint_pos[ids] = ref
        self.ref_torso_height[ids] = self.lib.base_height + cmd[:, 0]
        # torso orientation: commanded roll/pitch w.r.t. gravity, yaw w.r.t. the feet's heading
        feet = yaw_from_quat(self.robot.data.body_link_quat_w[ids][:, self.foot_ids])
        feet_yaw = torch.atan2(torch.sin(feet).mean(-1), torch.cos(feet).mean(-1))
        zeros = torch.zeros_like(feet_yaw)
        q_yaw = quat_from_euler_xyz(zeros, zeros, feet_yaw + cmd[:, 3])
        q_rp = quat_from_euler_xyz(cmd[:, 1], cmd[:, 2], zeros)
        self.ref_torso_quat[ids] = quat_mul(q_yaw, q_rp)

    def _update_metrics(self) -> None:
        self.metrics["torso_height_error"] = (
            self.robot.data.root_link_pos_w[:, 2] - self.ref_torso_height
        ).abs()
        self.metrics["torso_orientation_error"] = quat_error(
            self.robot.data.root_link_quat_w, self.ref_torso_quat
        )


def quat_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation angle between two (w, x, y, z) quaternions."""
    dot = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(dot)


@dataclass(kw_only=True)
class StandCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    library_path: str = str(DATA)
    foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link")
    resampling_time_range: tuple[float, float] = (1.5, 5.0)
    rel_neutral_envs: float = 0.15
    rel_single_axis_envs: float = 0.25
    # fastest the command moves (height m/s, roll/pitch/yaw rad/s, head yaw/pitch rad/s) -- about the
    # fast moves of the reference preview, within the motors' speed limits
    max_rate: tuple[float, ...] = (0.5, 2.0, 3.0, 3.0, 6.0, 6.0)
    # each target is approached at a random share of max_rate: slow and snappy moves both trained
    speed_range: tuple[float, float] = (0.1, 1.0)

    def build(self, env: ManagerBasedRlEnv) -> StandCommand:
        return StandCommand(self, env)
