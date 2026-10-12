from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .motion_data import MotionFile


# K1 joint order (22 DOF total, order matches MuJoCo qpos after floating base)
_K1_JOINT_NAMES = [
    "Head_Yaw",
    "Head_Pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# T1 joint order (23 DOF total): the K1 layout plus a yaw-only waist.
_T1_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    *_K1_JOINT_NAMES[2:10],
    "Waist",
    *_K1_JOINT_NAMES[10:],
]

# Joints whose axes change sign under a left/right mirror (roll and yaw axes).
_INVERTED_JOINTS = {
    "Head_Yaw",
    "AAHead_yaw",
    "Waist",
    *(
        f"{side}_{joint}"
        for side in ("Left", "Right")
        for joint in ("Shoulder_Roll", "Elbow_Yaw", "Hip_Roll", "Hip_Yaw", "Ankle_Roll")
    ),
}


def _mirror_mapping(joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Index and sign maps that swap each Left_/Right_ joint pair."""
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    index_map = np.array(
        [name_to_idx[_mirror_body_name(name)] for name in joint_names]
    )
    sign_map = np.array(
        [-1.0 if name in _INVERTED_JOINTS else 1.0 for name in joint_names],
        dtype=np.float32,
    )
    return index_map, sign_map


_MIRROR_MATRIX = np.diag([1.0, -1.0, 1.0])


def _mirror_body_name(name: str) -> str:
    """Return the left/right counterpart of a body name (identity if central)."""
    # T1 arm links are named AL1..AL3 / AR1..AR3.
    if re.fullmatch(r"A[LR]\d+", name):
        return name.translate(str.maketrans("LR", "RL"))
    for left, right in (("Left", "Right"), ("left", "right")):
        if left in name:
            return name.replace(left, right)
        if right in name:
            return name.replace(right, left)
    return name


def _body_mirror_index_map(link_body_list: Sequence[Any]) -> np.ndarray:
    """Permutation over body indices that swaps each left body with its right
    counterpart (central bodies map to themselves)."""
    names = [str(n) for n in link_body_list]
    name_to_idx = {name: i for i, name in enumerate(names)}
    index_map = np.arange(len(names))
    for i, name in enumerate(names):
        partner = _mirror_body_name(name)
        if partner in name_to_idx:
            index_map[i] = name_to_idx[partner]
    return index_map


# Motion files carry no joint names, so the robot is identified by its DOF count.
_MIRROR_MAPS = {
    len(names): _mirror_mapping(names) for names in (_K1_JOINT_NAMES, _T1_JOINT_NAMES)
}


@runtime_checkable
class MotionAugmentation(Protocol):
    name: str

    def apply(self, motion_file: MotionFile) -> MotionFile: ...


MotionAugmentationSpec = str | Mapping[str, Any] | MotionAugmentation


@dataclass(frozen=True)
class MirrorAugmentation:
    name: str = "mirror"

    def apply(self, motion_file: MotionFile) -> MotionFile:
        num_dof = motion_file.dof_pos.shape[1]
        if num_dof not in _MIRROR_MAPS:
            raise ValueError(
                "Mirror augmentation expects K1 or T1 DOF ordering with "
                f"{sorted(_MIRROR_MAPS)} joints, but got {num_dof} joints."
            )
        index_map, sign_map = _MIRROR_MAPS[num_dof]

        flipped_dof_pos = motion_file.dof_pos[:, index_map] * sign_map

        flipped_root_pos = motion_file.root_pos.copy()
        flipped_root_pos[:, 1] = -flipped_root_pos[:, 1]

        flipped_root_rot: list[np.ndarray] = []
        for quat in motion_file.root_rot:
            rot = Rotation.from_quat(quat)
            mirrored = _MIRROR_MATRIX @ rot.as_matrix() @ _MIRROR_MATRIX
            flipped_root_rot.append(Rotation.from_matrix(mirrored).as_quat())

        # Mirror local_body_pos (T, B, 3): swap left/right bodies and negate y,
        # matching the root mirror. Needs body names to pair them; without a
        # matching link_body_list the L/R swap can't be done correctly, so leave
        # it untouched rather than produce a half-mirrored clip.
        flipped_local_body_pos = motion_file.local_body_pos
        lbp = motion_file.local_body_pos
        names = motion_file.link_body_list
        if lbp is not None and names is not None and len(names) == lbp.shape[1]:
            body_map = _body_mirror_index_map(names)
            flipped_local_body_pos = lbp[:, body_map, :].copy()
            flipped_local_body_pos[:, :, 1] *= -1.0

        return MotionFile(
            fps=motion_file.fps,
            root_pos=flipped_root_pos,
            root_rot=flipped_root_rot,
            dof_pos=flipped_dof_pos,
            local_body_pos=flipped_local_body_pos,
            link_body_list=motion_file.link_body_list,
        )


@dataclass(frozen=True)
class SpeedModifierAugmentation:
    factor: float
    name: str = "speed"

    def apply(self, motion_file: MotionFile) -> MotionFile:
        if self.factor <= 0.0:
            raise ValueError("Speed modifier factor must be > 0.")

        fps = motion_file.fps if motion_file.fps > 0 else 30.0
        if motion_file.num_frames < 2:
            return MotionFile(
                fps=motion_file.fps,
                root_pos=motion_file.root_pos.copy(),
                root_rot=[np.array(quat, copy=True) for quat in motion_file.root_rot],
                dof_pos=motion_file.dof_pos.copy(),
                local_body_pos=motion_file.local_body_pos,
                link_body_list=motion_file.link_body_list,
            )

        dt = 1.0 / fps
        original_duration = motion_file.num_frames * dt
        new_duration = original_duration / self.factor
        new_num_frames = max(2, int(new_duration * fps))

        original_keyframes = torch.linspace(
            0.0, original_duration, steps=motion_file.num_frames, device="cpu"
        )
        new_keyframes = torch.linspace(
            0.0, original_duration, steps=new_num_frames, device="cpu"
        )

        root_pos = torch.as_tensor(motion_file.root_pos, dtype=torch.float32)
        dof_pos = torch.as_tensor(motion_file.dof_pos, dtype=torch.float32)

        new_root_pos = motion_file._resample_Rn(
            root_pos, original_keyframes, new_keyframes
        ).numpy()
        new_dof_pos = motion_file._resample_Rn(
            dof_pos, original_keyframes, new_keyframes
        ).numpy()
        new_root_rot = motion_file._resample_SO3(
            motion_file.root_rot, original_keyframes, new_keyframes
        ).as_quat()

        # Resample local_body_pos (T, B, 3) onto the same new time grid so it
        # keeps the new frame count; otherwise prepare() later resamples
        # mismatched lengths.
        new_local_body_pos = motion_file.local_body_pos
        if new_local_body_pos is not None:
            lbp = torch.as_tensor(new_local_body_pos, dtype=torch.float32)
            T0, B, _ = lbp.shape
            resampled = motion_file._resample_Rn(
                lbp.reshape(T0, B * 3), original_keyframes, new_keyframes
            )
            new_local_body_pos = resampled.reshape(-1, B, 3).numpy()

        return MotionFile(
            fps=motion_file.fps,
            root_pos=new_root_pos,
            root_rot=[np.array(quat, copy=True) for quat in new_root_rot],
            dof_pos=new_dof_pos,
            local_body_pos=new_local_body_pos,
            link_body_list=motion_file.link_body_list,
        )


def build_motion_augmentations(
    specs: Sequence[MotionAugmentationSpec] | None,
) -> list[MotionAugmentation]:
    if specs is None:
        return []

    augmentations: list[MotionAugmentation] = []
    for spec in specs:
        if isinstance(spec, MotionAugmentation):
            augmentations.append(spec)
            continue

        if isinstance(spec, str):
            name = spec
            kwargs: Mapping[str, Any] = {}
        elif isinstance(spec, Mapping):
            kwargs = spec
            raw_name = (
                kwargs.get("name") or kwargs.get("type") or kwargs.get("class_name")
            )
            if raw_name is None:
                raise ValueError(
                    "Augmentation mapping entries must include one of: "
                    "'name', 'type', or 'class_name'."
                )
            name = str(raw_name)
        else:
            raise TypeError(
                "Augmentation entries must be a string, mapping, or MotionAugmentation "
                f"instance. Got: {type(spec).__name__}"
            )

        normalized_name = name.strip().lower().replace("-", "_")
        if normalized_name.endswith("augmentation"):
            normalized_name = normalized_name[: -len("augmentation")]

        if normalized_name in {"mirror", "mirror_lr", "left_right_mirror"}:
            augmentations.append(MirrorAugmentation())
            continue

        if normalized_name == "speed":
            factor = kwargs.get("factor")
            if factor is None:
                percent = kwargs.get("percent")
                if percent is not None:
                    factor = 1.0 + float(percent) / 100.0
            if factor is None:
                raise ValueError(
                    "Speed augmentation requires either 'factor' or 'percent'."
                )
            augmentations.append(SpeedModifierAugmentation(factor=float(factor)))
            continue

        raise ValueError(f"Unknown AMP motion augmentation '{name}'.")

    return augmentations
