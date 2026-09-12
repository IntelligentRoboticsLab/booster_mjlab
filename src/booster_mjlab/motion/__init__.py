"""Standalone motion-processing infrastructure shared across booster_mjlab.

Frame-rate resampling (linear + SO(3) slerp) and finite-difference velocity
helpers, plus the motion-clip data model (``MotionFile`` / ``MotionData``),
the dataset loader (``MotionLoader``) and clip augmentations. Kept free of any
task- or algorithm-specific imports so the AMP motion pipeline and other motion
consumers can build on the same primitives.
"""

from __future__ import annotations

from booster_mjlab.motion.hf_motion_dataset import (
    HfMotionClip,
    HfMotionDataset,
)
from booster_mjlab.motion.motion_augmentations import (
    MirrorAugmentation,
    MotionAugmentation,
    MotionAugmentationSpec,
    SpeedModifierAugmentation,
    build_motion_augmentations,
)
from booster_mjlab.motion.motion_data import MotionData, MotionFile
from booster_mjlab.motion.motion_loader import (
    SUPPORTED_MOTION_FILE_EXTENSIONS,
    MotionLoader,
    MotionTransform,
    is_hf_dataset_id,
    resolve_motion_transform,
)
from booster_mjlab.motion.resampling import (
    finite_diff_angular_velocity,
    finite_diff_linear_velocity,
    resample_linear,
    resample_rotations,
)

__all__ = [
    "SUPPORTED_MOTION_FILE_EXTENSIONS",
    "HfMotionClip",
    "HfMotionDataset",
    "MirrorAugmentation",
    "MotionAugmentation",
    "MotionAugmentationSpec",
    "MotionData",
    "MotionFile",
    "MotionLoader",
    "MotionTransform",
    "SpeedModifierAugmentation",
    "build_motion_augmentations",
    "finite_diff_angular_velocity",
    "finite_diff_linear_velocity",
    "is_hf_dataset_id",
    "resample_linear",
    "resample_rotations",
    "resolve_motion_transform",
]
