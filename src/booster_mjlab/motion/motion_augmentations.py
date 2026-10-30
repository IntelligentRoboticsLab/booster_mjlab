from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .motion_data import MotionFile

if TYPE_CHECKING:
    from booster_mjlab.robots.mirror import Mirror


_MIRROR_MATRIX = np.diag([1.0, -1.0, 1.0])


def _resolve_mirror(mirror: Mirror | str) -> Mirror:
    """Resolve a ``module.path:ATTRIBUTE`` spec into the robot mirror it names."""
    if not isinstance(mirror, str):
        return mirror
    module_path, separator, attribute = mirror.rpartition(":")
    if not separator:
        raise ValueError(f"Mirror '{mirror}' must be given as 'module.path:ATTRIBUTE'.")
    return getattr(importlib.import_module(module_path), attribute)


@runtime_checkable
class MotionAugmentation(Protocol):
    name: str

    def apply(self, motion_file: MotionFile) -> MotionFile: ...


MotionAugmentationSpec = str | Mapping[str, Any] | MotionAugmentation


@dataclass(frozen=True)
class MirrorAugmentation:
    mirror: Mirror
    """Left/right symmetry of the robot the clips are for; its ``joint_names`` give the
    ``dof_pos`` column order."""
    name: str = "mirror"

    def apply(self, motion_file: MotionFile) -> MotionFile:
        num_dof = motion_file.dof_pos.shape[1]
        if num_dof != len(self.mirror.joint_names):
            raise ValueError(
                f"Mirror augmentation expects {len(self.mirror.joint_names)} joints "
                f"({self.mirror.joint_names[0]}, ...), but got {num_dof} joints."
            )
        joint_mirror = self.mirror.joint_mirror()
        sign_map = np.ones(num_dof, dtype=np.float32)
        sign_map[list(joint_mirror.inverted)] = -1.0

        flipped_dof_pos = motion_file.dof_pos[:, list(joint_mirror.perm)] * sign_map

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
            body_map = self.mirror.body_index_map(names)
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
            mirror = kwargs.get("mirror")
            if mirror is None:
                raise ValueError(
                    "Mirror augmentation requires 'mirror': the robot's Mirror or a "
                    "'module.path:ATTRIBUTE' naming it."
                )
            augmentations.append(MirrorAugmentation(mirror=_resolve_mirror(mirror)))
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
