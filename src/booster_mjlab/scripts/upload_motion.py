"""Convert a GMR motion .pkl to a tracking .npz and upload it to W&B as `motion.npz`.

The .pkl is expected to contain ``fps``, ``root_pos`` (N, 3), ``root_rot``
(N, 4, scipy xyzw) and ``dof_pos`` (N, 22, K1 joint order). It is resampled to
the tracking env's control rate and replayed through forward kinematics to
produce the ``joint_pos``/``joint_vel``/``body_*_w`` arrays consumed by mjlab's
tracking ``MotionLoader``.

Defaults to entity `ww-booster-lab` and project `motion_upload`. The artifact
name defaults to `<pkl-stem>-tracking` (e.g. `subject3.pkl` -> `subject3-tracking`).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro
import wandb
from scipy.spatial.transform import Rotation

from booster_mjlab.motion import MotionFile
from booster_mjlab.tasks.tracking.config.k1.env_cfgs import (
    booster_k1_flat_tracking_env_cfg,
)
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader as TrackingMotionLoader


@dataclass(frozen=True)
class Config:
    input_file: Path
    artifact_name: str | None = None
    entity: str = "ww-booster-lab"
    project: str = "motion_upload"
    artifact_type: str = "motions"
    output_fps: float | None = None
    speed_factor: float = 1.0
    device: str = "cpu"
    validate: bool = True


@dataclass
class ResampledMotion:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor
    root_rot: Rotation
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    output_fps: float

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]


def _default_output_fps() -> float:
    cfg = booster_k1_flat_tracking_env_cfg()
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    return 1.0 / step_dt


def _resample_motion(
    motion_file: MotionFile,
    output_fps: float,
    device: torch.device,
    speed_factor: float = 1.0,
) -> ResampledMotion:
    if motion_file.num_frames < 2:
        raise ValueError("Input motion must contain at least 2 frames.")
    if output_fps <= 0.0:
        raise ValueError(f"output_fps must be > 0. Got: {output_fps}")
    if speed_factor <= 0.0:
        raise ValueError(f"speed_factor must be > 0. Got: {speed_factor}")

    output_dt = 1.0 / output_fps
    input_fps = motion_file.fps if motion_file.fps > 0 else 30.0
    input_dt = 1.0 / input_fps / speed_factor

    num_frames = motion_file.num_frames
    original_duration = num_frames * input_dt
    # Keep in sync with existing AMP prep logic.
    target_frames = max(2, int(original_duration / output_dt))

    original_keyframes = torch.linspace(
        0.0, original_duration, steps=num_frames, device=device
    )
    resampled_keyframes = torch.linspace(
        0.0, original_duration, steps=target_frames, device=device
    )

    joint_pos = torch.as_tensor(motion_file.dof_pos, dtype=torch.float32, device=device)
    root_pos = torch.as_tensor(motion_file.root_pos, dtype=torch.float32, device=device)

    resampled_joint_pos = motion_file._resample_Rn(
        joint_pos, original_keyframes, resampled_keyframes
    )
    resampled_joint_vel = motion_file._compute_linear_velocities(
        resampled_joint_pos, output_dt
    )
    resampled_root_pos = motion_file._resample_Rn(
        root_pos, original_keyframes, resampled_keyframes
    )
    resampled_root_rot = motion_file._resample_SO3(
        motion_file.root_rot, original_keyframes, resampled_keyframes
    )
    resampled_root_lin_vel = motion_file._compute_linear_velocities(
        resampled_root_pos, output_dt
    )
    resampled_root_ang_vel = motion_file._compute_angular_velocities(
        resampled_root_rot, output_dt, local=False, device=device
    )

    return ResampledMotion(
        joint_pos=resampled_joint_pos,
        joint_vel=resampled_joint_vel,
        root_pos=resampled_root_pos,
        root_rot=resampled_root_rot,
        root_lin_vel=resampled_root_lin_vel,
        root_ang_vel=resampled_root_ang_vel,
        output_fps=output_fps,
    )


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray, device: torch.device) -> torch.Tensor:
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32
    )
    return torch.as_tensor(quat_wxyz, dtype=torch.float32, device=device).unsqueeze(0)


def _build_scene_and_sim(device: str) -> tuple[Scene, Simulation, Entity]:
    cfg = booster_k1_flat_tracking_env_cfg(play=True)
    cfg.scene.num_envs = 1
    scene = Scene(cfg.scene, device=device)
    sim = Simulation(num_envs=1, cfg=cfg.sim, model=scene.compile(), device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]
    return scene, sim, robot


def _run_forward_kinematics(
    resampled: ResampledMotion,
    scene: Scene,
    sim: Simulation,
    robot: Entity,
) -> dict[str, np.ndarray]:
    log: dict[str, list[np.ndarray] | np.ndarray] = {
        "fps": np.array([resampled.output_fps], dtype=np.float32),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    scene.reset()
    for frame_idx in range(resampled.num_frames):
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = resampled.root_pos[frame_idx : frame_idx + 1]
        root_quat_xyzw = resampled.root_rot[frame_idx].as_quat()
        root_state[:, 3:7] = _quat_xyzw_to_wxyz(root_quat_xyzw, root_state.device)
        root_state[:, 7:10] = resampled.root_lin_vel[frame_idx : frame_idx + 1]
        root_state[:, 10:13] = resampled.root_ang_vel[frame_idx : frame_idx + 1]
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:] = resampled.joint_pos[frame_idx : frame_idx + 1]
        joint_vel[:] = resampled.joint_vel[frame_idx : frame_idx + 1]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        joint_pos_arr = robot.data.joint_pos[0].cpu().numpy().copy()
        joint_vel_arr = robot.data.joint_vel[0].cpu().numpy().copy()
        body_pos_arr = robot.data.body_link_pos_w[0].cpu().numpy().copy()
        body_quat_arr = robot.data.body_link_quat_w[0].cpu().numpy().copy()
        body_lin_arr = robot.data.body_link_lin_vel_w[0].cpu().numpy().copy()
        body_ang_arr = robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()

        cast(list[np.ndarray], log["joint_pos"]).append(joint_pos_arr)
        cast(list[np.ndarray], log["joint_vel"]).append(joint_vel_arr)
        cast(list[np.ndarray], log["body_pos_w"]).append(body_pos_arr)
        cast(list[np.ndarray], log["body_quat_w"]).append(body_quat_arr)
        cast(list[np.ndarray], log["body_lin_vel_w"]).append(body_lin_arr)
        cast(list[np.ndarray], log["body_ang_vel_w"]).append(body_ang_arr)

    output: dict[str, np.ndarray] = {"fps": cast(np.ndarray, log["fps"])}
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        output[key] = np.stack(cast(list[np.ndarray], log[key]), axis=0)
    return output


def _validate_output(output_file: Path, robot: Entity, device: str) -> None:
    cfg = booster_k1_flat_tracking_env_cfg(play=True)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    body_indexes = torch.tensor(
        robot.find_bodies(motion_cmd.body_names, preserve_order=True)[0],
        dtype=torch.long,
        device=device,
    )
    loader = TrackingMotionLoader(
        str(output_file), body_indexes=body_indexes, device=device
    )
    print(
        "[validate] Loaded with tracking MotionLoader: "
        f"{loader.time_step_total} frames."
    )


def convert(cfg: Config, output_file: Path) -> None:
    output_fps = cfg.output_fps if cfg.output_fps is not None else _default_output_fps()
    device = torch.device(cfg.device)

    print(f"[info] Loading motion file: {cfg.input_file}")
    motion_file = MotionFile.load(cfg.input_file)
    print(
        f"[info] Input frames={motion_file.num_frames}, input_fps={motion_file.fps:.6f}, "
        f"output_fps={output_fps:.6f}, speed_factor={cfg.speed_factor:.6f}"
    )

    resampled = _resample_motion(
        motion_file,
        output_fps=output_fps,
        device=device,
        speed_factor=cfg.speed_factor,
    )
    print(f"[info] Resampled frames={resampled.num_frames}")

    scene, sim, robot = _build_scene_and_sim(device=cfg.device)
    motion_npz = _run_forward_kinematics(resampled, scene, sim, robot)
    np.savez(output_file, **motion_npz)

    if cfg.validate:
        _validate_output(output_file, robot=robot, device=cfg.device)


def run(cfg: Config) -> str:
    if not cfg.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_file}")

    artifact_name = (
        cfg.artifact_name
        if cfg.artifact_name is not None
        else f"{cfg.input_file.stem}-tracking"
    )

    with tempfile.TemporaryDirectory() as tmp:
        motion_path = Path(tmp) / "motion.npz"
        convert(cfg, motion_path)

        run = wandb.init(entity=cfg.entity, project=cfg.project, name=artifact_name)
        artifact = run.log_artifact(
            artifact_or_path=str(motion_path),
            name=artifact_name,
            type=cfg.artifact_type,
        )
        artifact.wait()
        qualified = artifact.qualified_name
        print(f"[info] Uploaded artifact: {qualified}")
        wandb.finish()

    return qualified


def main() -> None:
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
