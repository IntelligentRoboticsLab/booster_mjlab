"""Booster K1-specific sensor patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.utils.lab_api.math import quat_from_matrix

from mjlab.sensor import (
    RayCastSensor,
    TerrainHeightData,
    TerrainHeightSensor,
    TerrainHeightSensorCfg,
)

if TYPE_CHECKING:
    import mujoco_warp as mjwarp

    from mjlab.entity import Entity

    from mjlab.viewer.debug_visualizer import DebugVisualizer


RayAlignment = Literal["base", "yaw", "world"]


@dataclass
class FootSoleGridPatternCfg:
    """Sparse rectangular grid matching the K1 foot sole."""

    size: tuple[float, float] = (0.13, 0.06)
    num_samples: tuple[int, int] = (5, 3)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def generate_rays(
        self, mj_model: mujoco.MjModel | None, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mj_model
        size_x, size_y = self.size
        samples_x, samples_y = self.num_samples
        x = torch.linspace(-size_x / 2, size_x / 2, samples_x, device=device)
        y = torch.linspace(-size_y / 2, size_y / 2, samples_y, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

        offsets = torch.zeros((grid_x.numel(), 3), device=device)
        offsets[:, 0] = grid_x.flatten()
        offsets[:, 1] = grid_y.flatten()

        direction = torch.tensor(self.direction, device=device)
        direction = direction / direction.norm()
        directions = direction.unsqueeze(0).expand(len(offsets), 3).clone()
        return offsets, directions


@dataclass
class FootClearanceSensorCfg(TerrainHeightSensorCfg):
    """Measure clamped clearance over static, single-surface terrain.

    ``max_distance`` caps observations, not ray length. Queries are always world
    vertical; ``ray_alignment`` and pattern directions do not orient them.
    Terrain collision groups are isolated automatically during scene construction.
    """

    offset_alignment: RayAlignment = "base"
    """Alignment applied to pattern offsets independently of ray directions."""

    def build(self) -> FootClearanceSensor:
        return FootClearanceSensor(self)


@dataclass
class FootClearanceData(TerrainHeightData):
    """Clamped heights with per-sample signed clearance and query validity [B,F,N]."""

    signed_clearances: torch.Tensor
    hit_valid: torch.Tensor


class FootClearanceSensor(TerrainHeightSensor):
    """Query terrain from above its bounds, measuring from the actual sole grid."""

    cfg: FootClearanceSensorCfg

    def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
        terrain = entities.get("terrain")
        if terrain is None:
            raise ValueError("Foot clearance requires a static terrain entity")
        names = {g.name for g in terrain.spec.geoms if g.contype or g.conaffinity}
        geoms = [g for g in scene_spec.geoms if g.name in names]
        if not geoms:
            raise ValueError("Foot clearance requires terrain collision geometry")
        self._terrain_geom_names = tuple(g.name for g in geoms)
        # Reuse terrain groups if exclusive; otherwise reserve an unused group.
        groups = {g.group for g in geoms}
        others = {g.group for g in scene_spec.geoms if g.name not in names}
        if groups & others:
            available = set(range(6)) - groups - others
            if not available:
                raise ValueError("Foot clearance needs an exclusive terrain geom group")
            groups = {min(available)}
            for geom in geoms:
                geom.group = next(iter(groups))
        self._terrain_groups = tuple(sorted(groups))

    @property
    def include_geom_groups(self) -> tuple[int, ...]:
        return self._terrain_groups

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        if not np.isfinite(self.cfg.max_distance) or self.cfg.max_distance <= 0:
            raise ValueError("Foot clearance max_distance must be finite and positive")
        super().initialize(mj_model, model, data, device)
        from mjlab.sensor.raycast_sensor import _geom_groups_to_vec6

        self._geomgroup = _geom_groups_to_vec6(self._terrain_groups)
        selected = {mj_model.geom(name).id for name in self._terrain_geom_names}
        included = {
            i
            for i in range(mj_model.ngeom)
            if mj_model.geom_group[i] in self._terrain_groups
        }
        if included != selected:
            raise ValueError("Terrain query groups changed after sensor construction")
        reference = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, reference)
        bounds = []
        for name in self._terrain_geom_names:
            gid = mj_model.geom(name).id
            body = int(mj_model.geom_bodyid[gid])
            while body:
                if mj_model.body_jntnum[body] or mj_model.body_mocapid[body] >= 0:
                    raise ValueError("Foot clearance only supports static terrain")
                body = int(mj_model.body_parentid[body])
            pos = reference.geom_xpos[gid]
            rot = reference.geom_xmat[gid].reshape(3, 3)
            if mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
                if not np.allclose(rot[:, 2], (0, 0, 1), atol=1e-6):
                    raise ValueError(
                        "Foot clearance does not support tilted infinite planes"
                    )
                bounds.append((pos[2], pos[2]))
            else:
                # MuJoCo's local AABB includes mesh and heightfield extents.
                center, half = mj_model.geom_aabb[gid].reshape(2, 3)
                z = (pos + rot @ center)[2]
                extent = np.abs(rot[2]) @ half
                bounds.append((z - extent, z + extent))
        elevations = np.asarray(bounds)
        low, high = elevations[:, 0].min(), elevations[:, 1].max()
        if not np.isfinite([low, high]).all():
            raise ValueError("Terrain elevation bounds must be finite")
        margin = max(0.01, 1e-5 * max(abs(low), abs(high), 1.0))
        self._query_z = float(high + margin)
        self._query_distance = float(high - low + 2 * margin)
        self._sole_points = torch.zeros((data.nworld, self._num_rays, 3), device=device)
        self._distances.fill_(-1)

    @property
    def data(self) -> FootClearanceData:
        return super().data

    def _compute_offset_rotation(self, frame_mat: torch.Tensor) -> torch.Tensor:
        """Compute the rotation used for pattern offsets."""
        if self.cfg.offset_alignment == "base":
            return frame_mat
        if self.cfg.offset_alignment == "yaw":
            return self._extract_yaw_rotation(frame_mat)
        if self.cfg.offset_alignment == "world":
            return (
                torch.eye(3, device=frame_mat.device, dtype=frame_mat.dtype)
                .unsqueeze(0)
                .expand(frame_mat.shape[0], -1, -1)
            )
        raise ValueError(f"Unknown offset_alignment: {self.cfg.offset_alignment}")

    def prepare_rays(self) -> None:
        """Rotate sole samples with the foot, then query their XY from above terrain."""
        assert self._data is not None and self._model is not None
        assert self._local_offsets is not None
        assert self._local_directions is not None

        pos_list: list[torch.Tensor] = []
        mat_list: list[torch.Tensor] = []
        for frame_type, obj_id, _ in self._frame_infos:
            if frame_type == "body":
                pos_list.append(self._data.xpos[:, obj_id])
                mat_list.append(self._data.xmat[:, obj_id].view(-1, 3, 3))
            elif frame_type == "site":
                pos_list.append(self._data.site_xpos[:, obj_id])
                mat_list.append(self._data.site_xmat[:, obj_id].view(-1, 3, 3))
            else:
                pos_list.append(self._data.geom_xpos[:, obj_id])
                mat_list.append(self._data.geom_xmat[:, obj_id].view(-1, 3, 3))

        frame_pos = torch.stack(pos_list, dim=1)
        frame_mat = torch.stack(mat_list, dim=1)
        batch_size, num_frames = frame_pos.shape[:2]

        flat_frame_mat = frame_mat.reshape(batch_size * num_frames, 3, 3)
        offset_rot = self._compute_offset_rotation(flat_frame_mat).reshape(
            batch_size, num_frames, 3, 3
        )
        world_offsets = torch.einsum("bfij,nj->bfni", offset_rot, self._local_offsets)
        sole_points = frame_pos[:, :, None, :] + world_offsets
        self._sole_points = sole_points.reshape(batch_size, self._num_rays, 3)
        world_origins = sole_points.clone()
        world_origins[..., 2] = self._query_z
        world_rays = torch.zeros_like(world_origins)
        world_rays[..., 2] = -1

        world_origins_flat = world_origins.reshape(batch_size, self._num_rays, 3)
        world_rays_flat = world_rays.reshape(batch_size, self._num_rays, 3)

        assert self._ray_pnt is not None and self._ray_vec is not None
        ray_points = wp.to_torch(self._ray_pnt).view(batch_size, self._num_rays, 3)
        ray_vectors = wp.to_torch(self._ray_vec).view(batch_size, self._num_rays, 3)
        ray_points.copy_(world_origins_flat)
        ray_vectors.copy_(world_rays_flat)

        self._cached_world_origins = world_origins_flat
        self._cached_world_rays = world_rays_flat
        self._cached_frame_pos = frame_pos
        self._cached_frame_mat = frame_mat

    def postprocess_rays(self) -> None:
        """Use the terrain query range, independently of the observation cap."""
        distances = wp.to_torch(self._ray_dist)
        normals = wp.to_torch(self._ray_normal).view_as(self._cached_world_rays)
        distances.masked_fill_(distances > self._query_distance, -1)
        self._hit_pos_w = (
            self._cached_world_origins
            + self._cached_world_rays * distances.clamp_min(0).unsqueeze(-1)
        )
        normals.masked_fill_((distances < 0).unsqueeze(-1), 0)
        self._distances, self._normals_w = distances, normals
        self._frame_pos_w = self._cached_frame_pos
        batch = self._frame_pos_w.shape[0]
        self._frame_quat_w = quat_from_matrix(
            self._cached_frame_mat.reshape(-1, 3, 3)
        ).reshape(batch, self._num_frames, 4)
        self._pos_w = self._frame_pos_w[:, 0]
        self._quat_w = self._frame_quat_w[:, 0]
        self._invalidate_cache()

    def _compute_data(self) -> FootClearanceData:
        raw = RayCastSensor._compute_data(self)
        shape = (raw.distances.shape[0], self._num_frames, self._num_rays_per_frame)
        valid = (raw.distances >= 0).view(shape)
        signed = (self._sole_points[..., 2] - raw.hit_pos_w[..., 2]).view(shape)
        signed = signed.masked_fill(~valid, float("nan"))
        clearances = torch.where(
            valid, signed.clamp(0, self.cfg.max_distance), self.cfg.max_distance
        )

        if self.cfg.reduction == "min":
            heights = clearances.min(dim=-1).values
        elif self.cfg.reduction == "max":
            heights = clearances.max(dim=-1).values
        elif self.cfg.reduction == "mean":
            heights = clearances.mean(dim=-1)
        elif self.cfg.reduction == "none":
            heights = clearances
        else:
            raise ValueError(f"Unknown reduction: {self.cfg.reduction!r}")

        return FootClearanceData(
            **vars(raw), heights=heights, signed_clearances=signed, hit_valid=valid
        )

    def debug_vis(self, visualizer: DebugVisualizer) -> None:
        """Draw clearance arrows from sole samples to terrain, including penetration."""
        if not self.cfg.debug_vis or not self._debug_vis_enabled:
            return
        assert self._local_offsets is not None
        assert self._local_directions is not None
        assert self._cached_world_origins is not None
        assert self._cached_world_rays is not None
        assert self._cached_frame_pos is not None
        assert self._cached_frame_mat is not None

        data = self.data
        env_indices = list(visualizer.get_env_indices(data.distances.shape[0]))
        if not env_indices:
            return

        num_frames = self._num_frames
        num_rays = self._num_rays_per_frame
        num_envs = len(env_indices)
        ray_directions = (
            self._cached_world_rays[env_indices]
            .view(num_envs, num_frames, num_rays, 3)
            .cpu()
            .numpy()
        )
        sole_points = self._sole_points[env_indices].cpu().numpy()
        hit_positions = data.hit_pos_w[env_indices].cpu().numpy()
        distances = data.distances[env_indices].cpu().numpy()
        normals = data.normals_w[env_indices].cpu().numpy()

        meansize = visualizer.meansize
        ray_width = 0.065 * meansize
        sphere_radius = 0.65 * self.cfg.viz.hit_sphere_radius * meansize
        normal_length = self.cfg.viz.normal_length * meansize
        normal_width = 0.065 * meansize
        miss_extent = min(0.5, self.cfg.max_distance * 0.05)

        for env_idx in range(num_envs):
            for frame_idx in range(num_frames):
                for ray_in_frame in range(num_rays):
                    ray_idx = frame_idx * num_rays + ray_in_frame
                    origin = sole_points[env_idx, ray_idx]
                    hit = distances[env_idx, ray_idx] >= 0
                    if hit:
                        end = hit_positions[env_idx, ray_idx]
                        color = self.cfg.viz.hit_color
                    else:
                        end = (
                            origin
                            + ray_directions[env_idx, frame_idx, ray_in_frame]
                            * miss_extent
                        )
                        color = self.cfg.viz.miss_color

                    if self.cfg.viz.show_rays:
                        visualizer.add_arrow(
                            start=origin,
                            end=end,
                            color=color,
                            width=ray_width,
                            label=f"{self.cfg.name}_ray_{ray_idx}",
                        )

                    if hit:
                        visualizer.add_sphere(
                            center=end,
                            radius=sphere_radius,
                            color=self.cfg.viz.hit_sphere_color,
                            label=f"{self.cfg.name}_hit_{ray_idx}",
                        )
                        if self.cfg.viz.show_normals:
                            visualizer.add_arrow(
                                start=end,
                                end=end + normals[env_idx, ray_idx] * normal_length,
                                color=self.cfg.viz.normal_color,
                                width=normal_width,
                                label=f"{self.cfg.name}_normal_{ray_idx}",
                            )
