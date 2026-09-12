from __future__ import annotations

import importlib
import itertools
from typing import Callable, Generator, List, Sequence, Tuple
from os import PathLike

import torch
from pathlib import Path

from .motion_data import MotionData, MotionFile
from .motion_augmentations import (
    MotionAugmentationSpec,
    build_motion_augmentations,
)

SUPPORTED_MOTION_FILE_EXTENSIONS = (".pkl", ".csv")

MotionTransform = Callable[[MotionFile], MotionFile]


def resolve_motion_transform(
    transform: MotionTransform | str | None,
) -> MotionTransform | None:
    """Resolve a ``module.path:function`` spec into the callable it names."""
    if transform is None or callable(transform):
        return transform

    module_path, separator, attribute = transform.rpartition(":")
    if not separator:
        raise ValueError(
            f"Motion transform '{transform}' must be given as 'module.path:function'."
        )
    return getattr(importlib.import_module(module_path), attribute)


def is_hf_dataset_id(value: str) -> bool:
    """Return whether ``value`` is an unambiguous ``namespace/repo`` Hub ID."""
    if value.count("/") != 1:
        return False

    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(value)
    except HFValidationError:
        return False
    return True


def _filter_joint_obs(
    obs: torch.Tensor,
    full_joint_dim: int,
    joint_indices: List[int] | None,
    include_base_lin_vel: bool,
    include_base_ang_vel: bool,
    include_projected_gravity: bool = False,
) -> torch.Tensor:
    """Project AMP observations onto the configured discriminator features.

    Motion clips always store observations as
    ``[joint_pos (J), joint_vel (J), base_lin_vel (3), base_ang_vel (3), projected_gravity (3)]``.
    The AMP env config may use all joints or a subset, may drop either base-velocity block independently, and may
    include the body-frame projected gravity. The output order matches the env's AMP observation group:
    ``[joint_pos, joint_vel, (base_lin_vel), (base_ang_vel), (projected_gravity)]``.
    """
    joint_pos = obs[:, :full_joint_dim]
    joint_vel = obs[:, full_joint_dim : 2 * full_joint_dim]
    if joint_indices is not None:
        joint_pos = joint_pos[:, joint_indices]
        joint_vel = joint_vel[:, joint_indices]

    base_start = 2 * full_joint_dim
    obs_terms = [joint_pos, joint_vel]
    if include_base_lin_vel:
        obs_terms.append(obs[:, base_start : base_start + 3])
    if include_base_ang_vel:
        obs_terms.append(obs[:, base_start + 3 : base_start + 6])
    if include_projected_gravity:
        obs_terms.append(obs[:, base_start + 6 : base_start + 9])

    return torch.cat(obs_terms, dim=-1)


class MotionLoader:
    """Load AMP motion clips from a local path or Hugging Face dataset repo.

    ``dataset_root`` may be a local motion file, a local directory, or a Hub
    dataset ID in ``namespace/repo`` form. Hub datasets must use the Parquet
    layout produced by ``motions-to-hf``.

    ``dataset_transform`` optionally remaps every clip onto another joint layout
    (a callable, or a ``module.path:function`` spec). It runs after augmentation,
    so augmentations still see the dataset's own layout.
    """

    def __init__(
        self,
        dataset_root: PathLike[str],
        simulation_dt: float,
        speed_factor: float,
        num_amp_obs_steps: int = 1,
        dataset_weights: List[float] | None = None,
        augmentations: Sequence[MotionAugmentationSpec] | None = None,
        dataset_transform: MotionTransform | str | None = None,
        default_pose: torch.Tensor | None = None,
        joint_indices: List[int] | None = None,
        include_base_lin_vel: bool = True,
        include_base_ang_vel: bool = True,
        include_projected_gravity: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.dataset_source = str(dataset_root)
        self.dataset_root = Path(dataset_root)
        self.dataset_weights = dataset_weights
        self.simulation_dt = simulation_dt
        self.speed_factor = speed_factor
        if num_amp_obs_steps < 1:
            raise ValueError(
                f"num_amp_obs_steps must be at least 1, got {num_amp_obs_steps}"
            )
        self.num_amp_obs_steps = num_amp_obs_steps
        self.augmentations = build_motion_augmentations(augmentations)
        self.dataset_transform = resolve_motion_transform(dataset_transform)
        self.default_pose = default_pose
        self.joint_indices = joint_indices
        self.include_base_lin_vel = include_base_lin_vel
        self.include_base_ang_vel = include_base_ang_vel
        self.include_projected_gravity = include_projected_gravity

        self.motion_data: List[MotionData] = []
        self.motion_weights: List[float] = []
        motion_files = self._resolve_motion_files()
        if self.dataset_weights is None:
            self.dataset_weights = [1.0] * len(motion_files)

        assert len(self.dataset_weights) == len(motion_files), (
            "Number of dataset weights must match number of motion files"
        )

        obs_list: List[torch.Tensor] = []
        reset_states: List[torch.Tensor] = []
        num_retargeted = 0
        retargeted_dims: tuple[int, int] | None = None

        for motion_file, weight in zip(motion_files, self.dataset_weights):
            original = (
                motion_file
                if isinstance(motion_file, MotionFile)
                else MotionFile.load(motion_file)
            )
            variants = self._build_motion_variants(original)

            for variant in variants:
                if self.dataset_transform is not None:
                    source_dim = variant.dof_pos.shape[1]
                    variant = self.dataset_transform(variant)
                    if variant.dof_pos.shape[1] != source_dim:
                        num_retargeted += 1
                        retargeted_dims = (source_dim, variant.dof_pos.shape[1])

                data = variant.prepare(
                    simulation_dt=self.simulation_dt,
                    speed_factor=self.speed_factor,
                    device=self.device,
                )

                self.motion_data.append(data)
                self.motion_weights.append(weight)

                num_frames = data.num_frames
                indices = torch.arange(0, num_frames, device=self.device)
                history_offsets = torch.arange(
                    1 - self.num_amp_obs_steps, 1, device=self.device
                )
                history_indices = (indices[:, None] + history_offsets).clamp_min_(0)
                obs = data.get_obs(history_indices.reshape(-1))

                # Match the discriminator feature layout configured by the AMP observation group: selected joints
                # only, optionally with base linear and/or angular velocity, optionally with body-frame projected gravity.
                obs = _filter_joint_obs(
                    obs,
                    data.joint_positions.shape[1],
                    self.joint_indices,
                    self.include_base_lin_vel,
                    self.include_base_ang_vel,
                    self.include_projected_gravity,
                )

                if self.default_pose is not None:
                    joint_dim = (
                        len(self.joint_indices)
                        if self.joint_indices is not None
                        else data.joint_positions.shape[1]
                    )
                    obs = obs.clone()
                    obs[:, :joint_dim] -= self.default_pose

                # Each discriminator sample is one chronological, frame-major
                # history: [obs(t-K+1), ..., obs(t)]. Early clip frames are
                # backfilled with the first frame, matching the policy history
                # initialization at episode boundaries.
                obs_list.append(obs.reshape(num_frames, -1))
                joint_pos, joint_vel, base_linvel, base_angvel, quat = (
                    data.get_reset_state(indices)
                )
                reset_states.append(
                    torch.cat(
                        [quat, joint_pos, joint_vel, base_linvel, base_angvel], dim=-1
                    )
                )

        if retargeted_dims is not None:
            transform_name = getattr(
                self.dataset_transform, "__name__", repr(self.dataset_transform)
            )
            print(
                f"[MotionLoader] {self.dataset_source}: retargeted {num_retargeted} "
                f"motion(s) from {retargeted_dims[0]} to {retargeted_dims[1]} joints "
                f"via {transform_name}"
            )

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        lengths = [data.num_frames for data in self.motion_data]
        per_frame = torch.cat(
            [
                # Give the clip the configured weight, then distribute that
                # probability mass uniformly across its transitions.
                torch.full((length,), weight / length, device=self.device)
                for length, weight in zip(lengths, self.motion_weights)
            ]
        )

        self.per_frame_weights = per_frame / per_frame.sum()

    def _resolve_motion_files(self) -> list[Path | MotionFile]:
        if self.dataset_root.is_file():
            if self.dataset_root.suffix.lower() not in SUPPORTED_MOTION_FILE_EXTENSIONS:
                supported = ", ".join(SUPPORTED_MOTION_FILE_EXTENSIONS)
                raise ValueError(
                    f"Unsupported motion file '{self.dataset_root}'. "
                    f"Expected one of: {supported}."
                )
            return [self.dataset_root]

        if not self.dataset_root.exists():
            if is_hf_dataset_id(self.dataset_source):
                from .hf_motion_dataset import HfMotionDataset

                motions = HfMotionDataset(self.dataset_source).motion_files()
                if not motions:
                    raise FileNotFoundError(
                        f"Hugging Face dataset contains no motion clips: "
                        f"{self.dataset_source}"
                    )
                return motions

            raise FileNotFoundError(
                f"Motion dataset path does not exist: {self.dataset_root}. "
                "Hugging Face dataset IDs must use 'namespace/repo' form."
            )

        motion_files = sorted(
            path
            for suffix in SUPPORTED_MOTION_FILE_EXTENSIONS
            for path in self.dataset_root.glob(f"*{suffix}")
        )
        if not motion_files:
            supported = ", ".join(SUPPORTED_MOTION_FILE_EXTENSIONS)
            raise FileNotFoundError(
                f"No motion files with extensions {supported} found in {self.dataset_root}"
            )
        return motion_files

    def _build_motion_variants(self, original: MotionFile) -> list[MotionFile]:
        """Build compositional motion variants from configured augmentations.

        Augmentations are grouped by their `name`, and we select at most one
        augmentation per group (plus an identity/no-op option). This yields a
        Cartesian product over augmentation groups.

        Example:
          [mirror, speed(+10), speed(-10)] ->
          (identity|mirror) x (identity|speed(+10)|speed(-10)) = 6 variants.
        """
        if not self.augmentations:
            return [original]

        grouped_augmentations: dict[str, list] = {}
        group_order: list[str] = []
        for augmentation in self.augmentations:
            if augmentation.name not in grouped_augmentations:
                grouped_augmentations[augmentation.name] = []
                group_order.append(augmentation.name)
            grouped_augmentations[augmentation.name].append(augmentation)

        group_choices: list[list] = []
        for group_name in group_order:
            group_choices.append([None, *grouped_augmentations[group_name]])

        variants: list[MotionFile] = []
        for selected_augmentations in itertools.product(*group_choices):
            variant = original
            for augmentation in selected_augmentations:
                if augmentation is None:
                    continue
                variant = augmentation.apply(variant)
            variants.append(variant)

        return variants

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[torch.Tensor, None, None]:
        """Sample motion histories in big chunks to reduce multinomial overhead."""
        total = num_mini_batch * mini_batch_size
        if total <= 0:
            return

        indices = torch.multinomial(self.per_frame_weights, total, replacement=True)
        for i in range(num_mini_batch):
            batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
            yield self.all_obs[batch_idx]

    def get_state_for_reset(self, number_of_samples: int) -> Tuple[torch.Tensor, ...]:
        idx = torch.multinomial(
            self.per_frame_weights, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]
        dims = [4, joint_dim, joint_dim, 3, 3]
        return torch.split(full, dims, dim=1)
