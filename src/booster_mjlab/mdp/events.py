"""booster_mjlab event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)
from mjlab.utils.logging import print_info

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class reset_from_pose_pool:
    """Pre-collect valid random reset poses, then sample from pool on each reset.

    On first call, generates random poses across all environments for N iterations,
    validates each against self-collision, non-foot terrain contact, and foot
    penetration below ground, then stores valid poses in a pool. Subsequent calls
    sample from this pool.

    When ``dataset_root`` is provided, base poses (orientation, joint positions,
    velocities) are sampled from a motion dataset. Position and orientation offsets
    from ``pose_range`` are applied on top of the motion frames. When no dataset is
    provided, poses are generated from the default standing pose with random joint
    offsets.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        params = cfg.params
        self._asset_cfg: SceneEntityCfg = params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        self._self_collision_sensor: str = params["self_collision_sensor"]
        self._nonfoot_sensor: str = params["nonfoot_ground_sensor"]
        self._num_iterations: int = params.get("num_iterations", 3)
        self._pose_range: dict[str, tuple[float, float]] = params.get("pose_range", {})
        self._joint_position_range: tuple[float, float] = params.get(
            "joint_position_range", (-0.5, 0.5)
        )
        self._foot_sites: tuple[str, ...] = tuple(
            params.get("foot_sites", ("left_foot", "right_foot"))
        )
        self._min_foot_height: float = params.get("min_foot_height", -0.005)
        self._z_lift_factor: float = params.get("z_lift_factor", 0.0)

        # Motion dataset params (optional).
        self._dataset_root: str | None = params.get("dataset_root")
        self._speed_factor: float = params.get("speed_factor", 1.0)
        self._dataset_weights: list[float] | None = params.get("dataset_weights")
        self._augmentations: list[dict] | None = params.get("augmentations")
        self._dataset_transform: str | None = params.get("dataset_transform")

        self._device = env.device
        self._pool: torch.Tensor | None = None
        self._pool_size: int = 0
        self._num_joints: int = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        **params,
    ) -> None:
        if self._pool is None:
            self._pre_collect(env)
        self._sample_from_pool(env, env_ids)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        pass

    def _pre_collect(self, env: ManagerBasedRlEnv) -> None:
        """Run N iterations of random pose generation and validation."""
        asset = env.scene[self._asset_cfg.name]
        self_collision_sensor = env.scene.sensors[self._self_collision_sensor]
        nonfoot_sensor = env.scene.sensors[self._nonfoot_sensor]

        num_envs = env.num_envs
        device = self._device
        all_env_ids = torch.arange(num_envs, device=device, dtype=torch.int)

        default_root_state = asset.data.default_root_state
        assert default_root_state is not None
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        soft_limits = asset.data.soft_joint_pos_limits
        assert soft_limits is not None

        # Resolve foot site local IDs (indices into asset.data.site_pos_w).
        foot_local_ids, _ = asset.find_sites(self._foot_sites)
        env_origins_z = env.scene.env_origins[:num_envs, 2:3]  # (num_envs, 1)

        # Build pose sampling ranges.
        range_list = [
            self._pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=device)

        # Optionally load motion dataset.
        loader = None
        if self._dataset_root is not None:
            from booster_mjlab.motion import MotionLoader

            simulation_dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
            loader = MotionLoader(
                dataset_root=self._dataset_root,
                simulation_dt=simulation_dt,
                speed_factor=self._speed_factor,
                dataset_weights=self._dataset_weights,
                augmentations=self._augmentations,
                dataset_transform=self._dataset_transform,
                device=device,
            )
            print_info(
                f"[reset_from_pose_pool] Using motion dataset: {self._dataset_root}"
            )

        # Compute per-joint offset limits (used only when no motion dataset).
        lo_frac, hi_frac = self._joint_position_range
        dist_to_lower = default_joint_pos - soft_limits[..., 0]
        dist_to_upper = soft_limits[..., 1] - default_joint_pos
        joint_offset_lo = lo_frac * dist_to_lower
        joint_offset_hi = hi_frac * dist_to_upper

        all_valid_poses: list[torch.Tensor] = []
        total_self_collision = 0
        total_terrain_contact = 0
        total_feet_below = 0
        total_valid = 0

        source = self._dataset_root or "default pose"
        print_info(
            f"[reset_from_pose_pool] Collecting poses from {source}: "
            f"{self._num_iterations} iterations x {num_envs} envs "
            f"({self._num_iterations * num_envs} candidates)"
        )

        for i in range(self._num_iterations):
            # 1. Get base state: either from motion data or default pose.
            if loader is not None:
                (
                    base_quat_xyzw,
                    motion_joint_pos,
                    motion_joint_vel,
                    base_lin_vel_local,
                    base_ang_vel_local,
                ) = loader.get_state_for_reset(num_envs)
                # scipy (x,y,z,w) -> mjlab (w,x,y,z)
                base_quat = base_quat_xyzw[:, [3, 0, 1, 2]]
                joint_pos = motion_joint_pos
                joint_vel = motion_joint_vel
            else:
                base_quat = default_root_state[:num_envs, 3:7].clone()
                joint_offsets = sample_uniform(
                    joint_offset_lo,
                    joint_offset_hi,
                    default_joint_pos[:num_envs].shape,
                    device,
                )
                joint_pos = default_joint_pos[:num_envs].clone() + joint_offsets
                joint_pos = joint_pos.clamp_(
                    soft_limits[:num_envs, :, 0], soft_limits[:num_envs, :, 1]
                )
                joint_vel = torch.zeros_like(joint_pos)

            # 2. Sample position and orientation offsets.
            pose_samples = sample_uniform(
                ranges[:, 0], ranges[:, 1], (num_envs, 6), device=device
            )

            # Boost z based on tilt magnitude to prevent ground collision
            # for heavily tilted poses.
            if self._z_lift_factor > 0.0:
                tilt = torch.max(pose_samples[:, 3].abs(), pose_samples[:, 4].abs())
                pose_samples[:, 2] += tilt * self._z_lift_factor

            root_states = default_root_state[:num_envs].clone()
            positions = (
                root_states[:, 0:3]
                + pose_samples[:, 0:3]
                + env.scene.env_origins[:num_envs]
            )

            # Apply orientation offsets on top of base orientation.
            orientations_delta = quat_from_euler_xyz(
                pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
            )
            orientations = quat_mul(orientations_delta, base_quat)

            # 3. Compute world-frame velocities.
            if loader is not None:
                root_lin_vel = quat_apply(orientations, base_lin_vel_local)
                root_ang_vel = quat_apply(orientations, base_ang_vel_local)
            else:
                root_lin_vel = torch.zeros((num_envs, 3), device=device)
                root_ang_vel = torch.zeros((num_envs, 3), device=device)
            root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)

            # 4. Write state to sim.
            asset.write_root_link_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=all_env_ids
            )
            asset.write_root_link_velocity_to_sim(root_vel, env_ids=all_env_ids)
            asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=all_env_ids)

            # 5. Forward pass to compute contacts and kinematics.
            env.sim.forward()

            # 6. Read contact sensor data.
            self_collision_sensor._invalidate_cache()
            nonfoot_sensor._invalidate_cache()

            self_coll_found = self_collision_sensor.data.found
            nonfoot_found = nonfoot_sensor.data.found

            has_self_collision = self_coll_found.sum(dim=-1) > 0
            has_terrain_contact = nonfoot_found.sum(dim=-1) > 0

            # 7. Check foot site heights above ground.
            foot_z = asset.data.site_pos_w[:num_envs, foot_local_ids, 2]
            foot_z_relative = foot_z - env_origins_z
            has_feet_below = (foot_z_relative < self._min_foot_height).any(dim=-1)

            is_valid = ~has_self_collision & ~has_terrain_contact & ~has_feet_below

            n_self = int(has_self_collision.sum().item())
            n_terrain = int((~has_self_collision & has_terrain_contact).sum().item())
            n_feet = int(
                (~has_self_collision & ~has_terrain_contact & has_feet_below)
                .sum()
                .item()
            )
            n_valid = int(is_valid.sum().item())

            total_self_collision += n_self
            total_terrain_contact += n_terrain
            total_feet_below += n_feet
            total_valid += n_valid

            print_info(
                f"  iter {i + 1}/{self._num_iterations}: "
                f"{n_valid} valid, "
                f"{n_self} self-collision, "
                f"{n_terrain} terrain-contact, "
                f"{n_feet} feet-below-ground"
            )

            # 8. Collect valid poses (without env_origins for portability).
            valid_ids = is_valid.nonzero(as_tuple=False).squeeze(-1)
            if len(valid_ids) > 0:
                relative_pos = (
                    root_states[valid_ids, 0:3] + pose_samples[valid_ids, 0:3]
                )
                all_valid_poses.append(
                    torch.cat(
                        [
                            relative_pos,
                            orientations[valid_ids],
                            joint_pos[valid_ids],
                            root_vel[valid_ids],
                            joint_vel[valid_ids],
                        ],
                        dim=-1,
                    )
                )

        total_candidates = self._num_iterations * num_envs
        if not all_valid_poses:
            raise RuntimeError(
                f"reset_from_pose_pool: No valid poses found after "
                f"{self._num_iterations} iterations x {num_envs} envs. "
                f"Rejected: {total_self_collision} self-collision, "
                f"{total_terrain_contact} terrain-contact, "
                f"{total_feet_below} feet-below-ground. "
                f"Consider relaxing pose_range or joint_position_range."
            )

        self._pool = torch.cat(all_valid_poses, dim=0)
        self._pool_size = self._pool.shape[0]
        self._num_joints = joint_pos.shape[-1]
        print_info(
            f"[reset_from_pose_pool] Done: {self._pool_size}/{total_candidates} "
            f"valid ({100 * self._pool_size / total_candidates:.1f}%). "
            f"Rejected: {total_self_collision} self-collision, "
            f"{total_terrain_contact} terrain-contact, "
            f"{total_feet_below} feet-below-ground."
        )
        if self._pool_size < num_envs:
            print_info(
                f"[reset_from_pose_pool] WARNING: Pool size ({self._pool_size}) < "
                f"num_envs ({num_envs}). Poses will be sampled with replacement."
            )

    def _sample_from_pool(
        self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
    ) -> None:
        """Sample random poses from the pre-collected pool and write to sim."""
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self._device, dtype=torch.int)

        asset = env.scene[self._asset_cfg.name]
        num_reset = len(env_ids)

        assert self._pool is not None
        pool_indices = torch.randint(
            0, self._pool_size, (num_reset,), device=self._device
        )
        sampled = self._pool[pool_indices]

        # Split: pos_rel(3) | quat(4) | joint_pos(nj) | root_vel(6) | joint_vel(nj)
        nj = self._num_joints
        root_pos_rel = sampled[:, 0:3]
        root_quat = sampled[:, 3:7]
        joint_pos = sampled[:, 7 : 7 + nj]
        root_vel = sampled[:, 7 + nj : 7 + nj + 6]
        joint_vel = sampled[:, 7 + nj + 6 : 7 + nj + 6 + nj]

        positions = root_pos_rel + env.scene.env_origins[env_ids]

        asset.write_root_link_pose_to_sim(
            torch.cat([positions, root_quat], dim=-1), env_ids=env_ids
        )
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
