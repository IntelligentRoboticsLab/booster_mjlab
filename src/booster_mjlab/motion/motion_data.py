from __future__ import annotations
from os import PathLike
from typing import Optional, List

from dataclasses import dataclass
import pickle
from pathlib import Path
import numpy as np
import numpy.typing as npt
import torch

from scipy.spatial.transform import Rotation

from .resampling import (
    finite_diff_angular_velocity,
    finite_diff_linear_velocity,
    resample_linear,
    resample_rotations,
)

BOOSTER_K1_CSV_FPS = 30.0
BOOSTER_K1_CSV_HEADER = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    "Head_Yaw_dof",
    "Head_Pitch_dof",
    "Left_Shoulder_Pitch_dof",
    "Left_Shoulder_Roll_dof",
    "Left_Elbow_Pitch_dof",
    "Left_Elbow_Yaw_dof",
    "Right_Shoulder_Pitch_dof",
    "Right_Shoulder_Roll_dof",
    "Right_Elbow_Pitch_dof",
    "Right_Elbow_Yaw_dof",
    "Left_Hip_Pitch_dof",
    "Left_Hip_Roll_dof",
    "Left_Hip_Yaw_dof",
    "Left_Knee_Pitch_dof",
    "Left_Ankle_Pitch_dof",
    "Left_Ankle_Roll_dof",
    "Right_Hip_Pitch_dof",
    "Right_Hip_Roll_dof",
    "Right_Hip_Yaw_dof",
    "Right_Knee_Pitch_dof",
    "Right_Ankle_Pitch_dof",
    "Right_Ankle_Roll_dof",
]


@dataclass
class MotionFile:
    """
    A motion file for use in AMP training.
    This file follows the same structure as the motion files in GMR once loaded.
    """

    fps: float
    root_pos: npt.NDArray
    root_rot: List[npt.NDArray]
    dof_pos: npt.NDArray
    local_body_pos: Optional[npt.NDArray] = None
    link_body_list: Optional[npt.NDArray] = None

    @property
    def num_frames(self) -> int:
        return self.root_pos.shape[0]

    @property
    def dt(self) -> float:
        """The delta time between each frame. This will default to 1/30 if fps is 0.

        Returns:
            float: The delta time between each frame.
        """
        return 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

    @classmethod
    def load(cls, path: PathLike[str]) -> MotionFile:
        motion_path = Path(path)
        suffix = motion_path.suffix.lower()

        if suffix == ".pkl":
            return cls._load_pickle(motion_path)
        if suffix == ".csv":
            return cls._load_booster_k1_csv(motion_path)

        raise ValueError(
            f"Unsupported motion file format '{motion_path.suffix}' for {motion_path}."
        )

    @classmethod
    def _load_pickle(cls, path: Path) -> MotionFile:
        with path.open("rb") as f:
            data = pickle.load(f)

        return cls(
            fps=data["fps"],
            root_pos=data["root_pos"],
            root_rot=data["root_rot"],
            dof_pos=data["dof_pos"],
            local_body_pos=data.get("local_body_pos"),
            link_body_list=data.get("link_body_list"),
        )

    @classmethod
    def _load_booster_k1_csv(cls, path: Path) -> MotionFile:
        with path.open(encoding="utf-8-sig") as f:
            header = [column.strip() for column in f.readline().strip().split(",")]

        if header != BOOSTER_K1_CSV_HEADER:
            expected = ", ".join(BOOSTER_K1_CSV_HEADER)
            found = ", ".join(header)
            raise ValueError(
                "Unsupported Booster K1 CSV motion layout in "
                f"{path}. Expected header [{expected}], found [{found}]."
            )

        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        data = np.atleast_2d(data)
        if data.shape[1] != len(BOOSTER_K1_CSV_HEADER):
            raise ValueError(
                f"Expected {len(BOOSTER_K1_CSV_HEADER)} CSV columns in {path}, "
                f"but found {data.shape[1]}."
            )

        root_pos = data[:, 1:4] * 0.01
        root_rot = Rotation.from_euler("xyz", data[:, 4:7], degrees=True).as_quat()
        root_rot = root_rot.astype(np.float32, copy=False)
        dof_pos = np.deg2rad(data[:, 7:]).astype(np.float32, copy=False)

        return cls(
            fps=BOOSTER_K1_CSV_FPS,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
        )

    def _resample_Rn(
        self,
        data: torch.Tensor,
        t_original: torch.Tensor,
        t_resampled: torch.Tensor,
    ) -> torch.Tensor:
        """Resample a tensor [B, N] to [R, N] by linear interpolation along axis 0."""
        return resample_linear(data, t_original, t_resampled)

    def _resample_SO3(
        self,
        raw_quaternions: List[npt.NDArray],
        original_keyframes: torch.Tensor,
        target_keyframes: torch.Tensor,
    ) -> Rotation:
        """Resample a list of (xyzw) quaternions by slerping."""
        return resample_rotations(raw_quaternions, original_keyframes, target_keyframes)

    def _compute_linear_velocities(self, data: torch.Tensor, dt: float) -> torch.Tensor:
        """Forward finite-difference velocity of ``data`` [N, D] at step ``dt``."""
        return finite_diff_linear_velocity(data, dt)

    def _compute_angular_velocities(
        self,
        data: List[Rotation],
        dt: float,
        local: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Angular velocity [N, 3] (rad/s) from a sequence of Rotations at step ``dt``."""
        return finite_diff_angular_velocity(data, dt, local=local, device=device)

    def prepare(
        self,
        simulation_dt: float,
        speed_factor: float = 1,
        device: torch.device = torch.device("cpu"),
    ) -> MotionData:
        """Prepare this `MotionFile` by resampling it to the desired simulation delta time,
        and computing velocities. Used by both the AMP and SMP motion pipelines.

        Args:
            simulation_dt (float): The desired simulation delta time to resample the motion to.
            speed_factor (float, optional): Positive playback speed multiplier applied as
                `dt = base_dt / speed_factor` (so `>1` speeds up motion and `<1` slows
                it down). Defaults to 1.
            device (torch.device, optional): Device on which to place the output tensors. Defaults to torch.device("cpu").

        Returns:
            MotionData: Prepared motion data for AMP training.
        """
        joint_positions = torch.as_tensor(
            self.dof_pos, dtype=torch.float32, device=device
        )

        fps = self.fps if self.fps > 0 else 30.0
        dt = 1.0 / fps / speed_factor

        num_frames = self.num_frames
        original_duration = num_frames * dt
        original_keyframes = torch.linspace(
            0, original_duration, steps=num_frames, device=device
        )
        resampled_keyframes = torch.linspace(
            0,
            num_frames * dt,
            steps=int(original_duration / simulation_dt),
            device=device,
        )

        resampled_joint_positions = self._resample_Rn(
            joint_positions, original_keyframes, resampled_keyframes
        )
        resampled_joint_velocities = self._compute_linear_velocities(
            resampled_joint_positions, simulation_dt
        )
        root_pos = torch.as_tensor(self.root_pos, dtype=torch.float32, device=device)
        resampled_root_pos = self._resample_Rn(
            root_pos, original_keyframes, resampled_keyframes
        )
        root_rot = self.root_rot
        resampled_root_rot = self._resample_SO3(
            root_rot, original_keyframes, resampled_keyframes
        )

        resampled_base_lin_velocities_mixed = self._compute_linear_velocities(
            resampled_root_pos, simulation_dt
        )
        resampled_base_ang_velocities_mixed = self._compute_angular_velocities(
            resampled_root_rot, simulation_dt, local=False, device=device
        )

        resampled_base_lin_velocities_local = torch.as_tensor(
            np.asarray(
                [
                    rot.as_matrix().T @ v.cpu().numpy()
                    for (rot, v) in zip(
                        resampled_root_rot, resampled_base_lin_velocities_mixed
                    )
                ]
            ),
            dtype=torch.float32,
            device=device,
        )
        resampled_base_ang_velocities_local = self._compute_angular_velocities(
            resampled_root_rot, simulation_dt, local=True, device=device
        )

        # Root height (z-coordinate)
        resampled_root_height = resampled_root_pos[:, 2:3]

        # Resample local_body_pos onto the same time grid if it's present on
        # the source MotionFile. SMP needs this; AMP ignores it.
        resampled_local_body_pos: Optional[torch.Tensor] = None
        if self.local_body_pos is not None:
            lbp = torch.as_tensor(
                self.local_body_pos, dtype=torch.float32, device=device
            )
            T0, B, _ = lbp.shape
            flat = lbp.reshape(T0, B * 3)
            flat_resampled = self._resample_Rn(
                flat, original_keyframes, resampled_keyframes
            ).to(dtype=torch.float32)
            resampled_local_body_pos = flat_resampled.reshape(-1, B, 3)
        link_body_list_local: Optional[List[str]] = (
            list(self.link_body_list)
            if isinstance(self.link_body_list, (list, tuple))
            else (
                [str(n) for n in self.link_body_list]
                if self.link_body_list is not None
                else None
            )
        )

        # Projected gravity: gravity vector [0,0,-1] rotated into body frame
        n_frames = len(resampled_root_rot)
        gravity_world = np.tile([0.0, 0.0, -1.0], (n_frames, 1))
        resampled_projected_gravity = torch.as_tensor(
            resampled_root_rot.inv().apply(gravity_world),
            dtype=torch.float32,
            device=device,
        )

        # Heading-relative linear velocity: world velocity with only yaw removed
        euler_zyx = resampled_root_rot.as_euler("ZYX")
        # Keep the yaw column as (N, 1): scipy >=1.17 reads the trailing dim of a
        # 1-D array as the axis dim, so a bare (N,) raises a shape mismatch.
        yaw_rot = Rotation.from_euler("Z", euler_zyx[:, [0]])
        resampled_lin_vel_heading = torch.as_tensor(
            yaw_rot.inv().apply(resampled_base_lin_velocities_mixed.cpu().numpy()),
            dtype=torch.float32,
            device=device,
        )

        # Root rotation in heading frame as 6D (tan-norm): yaw cancelled, then
        # [x-axis, z-axis] of the remaining pitch+roll rotation matrix.
        heading_rot = yaw_rot.inv() * resampled_root_rot
        heading_rot_mat = heading_rot.as_matrix()  # (N, 3, 3)
        resampled_rot_6d = torch.as_tensor(
            np.concatenate(
                [heading_rot_mat[:, :, 0], heading_rot_mat[:, :, 2]], axis=-1
            ),
            dtype=torch.float32,
            device=device,
        )

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_velocities_mixed,
            base_ang_velocities_mixed=resampled_base_ang_velocities_mixed,
            base_lin_velocities_local=resampled_base_lin_velocities_local,
            base_ang_velocities_local=resampled_base_ang_velocities_local,
            base_quat=torch.as_tensor(
                resampled_root_rot.as_quat(), dtype=torch.float32, device=device
            ),
            base_root_height=resampled_root_height,
            base_projected_gravity=resampled_projected_gravity,
            base_lin_velocities_heading=resampled_lin_vel_heading,
            base_rot_6d=resampled_rot_6d,
            device=device,
            root_pos=resampled_root_pos.to(dtype=torch.float32),
            local_body_pos=resampled_local_body_pos,
            link_body_list=link_body_list_local,
        )


@dataclass
class MotionData:
    """Motion data, prepared for use in AMP training and stored as Tensors on a specific device."""

    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    base_lin_velocities_mixed: torch.Tensor
    base_ang_velocities_mixed: torch.Tensor
    base_lin_velocities_local: torch.Tensor
    base_ang_velocities_local: torch.Tensor
    base_quat: torch.Tensor
    base_root_height: torch.Tensor
    base_projected_gravity: torch.Tensor
    base_lin_velocities_heading: torch.Tensor
    base_rot_6d: torch.Tensor
    device: torch.device = torch.device("cpu")
    # Resampled root position in world frame (T, 3). AMP only needs height (
    # ``base_root_height``) but SMP needs full XY too for feature anchoring.
    root_pos: Optional[torch.Tensor] = None
    # Per-frame per-body positions in the trunk's body-local frame (T, B, 3),
    # together with the body-name list. Surfaced for the SMP prior's
    # end-effector feature; ``None`` when the source .pkl didn't include
    # ``local_body_pos``.
    local_body_pos: Optional[torch.Tensor] = None
    link_body_list: Optional[List[str]] = None

    @property
    def num_frames(self) -> int:
        return self.joint_positions.shape[0]

    def to(self, device: torch.device) -> MotionData:
        """Moves the motion data to a specific device.

        Args:
            device (torch.device): The device to move the data to.

        Returns:
            MotionData: The motion data on the specified device.
        """
        return MotionData(
            joint_positions=self.joint_positions.to(device),
            joint_velocities=self.joint_velocities.to(device),
            base_lin_velocities_mixed=self.base_lin_velocities_mixed.to(device),
            base_ang_velocities_mixed=self.base_ang_velocities_mixed.to(device),
            base_lin_velocities_local=self.base_lin_velocities_local.to(device),
            base_ang_velocities_local=self.base_ang_velocities_local.to(device),
            base_quat=self.base_quat.to(device),
            base_root_height=self.base_root_height.to(device),
            base_projected_gravity=self.base_projected_gravity.to(device),
            base_lin_velocities_heading=self.base_lin_velocities_heading.to(device),
            base_rot_6d=self.base_rot_6d.to(device),
            device=device,
            root_pos=self.root_pos.to(device) if self.root_pos is not None else None,
            local_body_pos=(
                self.local_body_pos.to(device)
                if self.local_body_pos is not None
                else None
            ),
            link_body_list=self.link_body_list,
        )

    def get_obs(self, indices: torch.Tensor) -> torch.Tensor:
        """Get the observation at the specified indices.

        Args:
            indices (torch.Tensor): The indices to get the observations for.

        Returns:
            torch.Tensor: The observations at the specified indices.
        """
        obs = torch.cat(
            [
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_local[indices],
                self.base_ang_velocities_local[indices],
                self.base_projected_gravity[indices],
            ],
            dim=-1,
        )
        return obs

    def get_reset_state(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Get the reset state at the specified indices.

        Args:
            indices (torch.Tensor): The indices to get the reset states for.

        Returns:
            tuple[torch.Tensor, ...]: The reset states at the specified indices.
        """
        return (
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
            self.base_quat[indices],
        )
