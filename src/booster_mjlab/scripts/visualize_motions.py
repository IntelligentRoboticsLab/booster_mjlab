"""Browse, play, and non-destructively edit robot motion clips in viser."""

from __future__ import annotations

import argparse
import pickle
import threading
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np
import torch
import viser
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

import booster_mjlab.tasks  # noqa: F401, ensure task registry is populated.

from booster_mjlab.motion import (
    SUPPORTED_MOTION_FILE_EXTENSIONS,
    HfMotionDataset,
    MirrorAugmentation,
    MotionData,
    MotionFile,
    MotionTransform,
    SpeedModifierAugmentation,
    is_hf_dataset_id,
)
from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
    K1_PARALLEL_JOINT_ORDER,
)
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer.viser.scene import ViserMujocoScene

# Speed sparkline geometry: a 100 x SPARKLINE_HEIGHT viewBox stretched to the
# panel width, so x is a percentage of the clip and y is speed.
SPARKLINE_BUCKETS = 160
SPARKLINE_HEIGHT = 26.0

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MotionEntry:
    """One selectable clip: a local motion file or an in-memory Hub clip."""

    name: str
    stem: str
    path: Path | None = None
    motion: MotionFile | None = None

    def load(self) -> MotionFile:
        if self.motion is not None:
            return self.motion
        assert self.path is not None
        return MotionFile.load(self.path)

    @property
    def default_save_path(self) -> Path:
        # Hub clips have no source directory, so edits land in the cwd.
        parent = self.path.parent if self.path is not None else Path.cwd()
        return parent / f"{self.stem}_edited.pkl"


@dataclass(slots=True)
class PlayEnvContext:
    env_cfg: ManagerBasedRlEnvCfg
    scene: Scene
    sim: Simulation


@dataclass(slots=True)
class PlaybackState:
    frame: int
    paused: bool
    speed: float
    loop: bool
    total_frames: int
    seek_to: int | None = None
    pending_motion: MotionEntry | None = None
    pending_step: bool = False


@dataclass(slots=True)
class PlaybackUI:
    status_html: Any
    speed_html: Any
    diagnostics_html: Any
    transport_buttons: Any
    frame_slider: Any
    speed_input: Any
    loop_checkbox: Any
    slider_lock: threading.Lock
    motion_select: Any
    slider_internal: bool = False


@dataclass(slots=True)
class ContactViewState:
    """Settings for the orthographic foot-contact view."""

    enabled: bool = True
    view: str = "Side"
    view_height: float = 0.6
    dirty: bool = True


@dataclass(slots=True)
class ViewerContext:
    server: viser.ViserServer
    scene: ViserMujocoScene
    ui: PlaybackUI
    update_status: Callable[..., None]
    update_transport_buttons: Callable[[], None]
    set_slider_value: Callable[[int], None]
    contact_image: Any
    contact_readout: Any


@dataclass(slots=True)
class EditState:
    """State for non-destructive motion editing."""

    # Trim settings
    trim_start: int = 0
    trim_end: int | None = None  # None = end of clip

    # Speed modification
    speed_factor: float = 1.0  # >1 = faster, <1 = slower

    # Mirror settings
    mirrored: bool = False

    # Root offset (initialized in __post_init__)
    root_position_offset: np.ndarray | None = None  # XYZ
    root_rotation_offset: np.ndarray | None = None  # Euler angles (degrees)

    # Ground contact fix
    fix_ground: bool = False

    # Dirty flag
    needs_recompile: bool = False

    def __post_init__(self) -> None:
        if self.root_position_offset is None:
            self.root_position_offset = np.zeros(3)
        if self.root_rotation_offset is None:
            self.root_rotation_offset = np.zeros(3)


@dataclass(slots=True)
class EditUI:
    """UI controls for the Edit tab."""

    # Trim controls
    trim_start_input: Any
    trim_end_input: Any
    trim_reset_button: Any

    # Speed controls
    speed_factor_input: Any
    speed_info: Any

    # Mirror controls
    mirror_checkbox: Any
    mirror_info: Any

    # Root offset controls
    pos_x: Any
    pos_y: Any
    pos_z: Any
    rot_x: Any
    rot_y: Any
    rot_z: Any
    offset_reset_button: Any

    # Ground contact fix
    fix_ground_checkbox: Any

    # Apply/Save controls
    apply_button: Any
    save_button: Any
    save_path_input: Any
    edit_info: Any


class MotionClip:
    """Wrapper around MotionData for visualization."""

    def __init__(
        self,
        motion_data: MotionData,
        root_positions: torch.Tensor,
        source_fps: float,
        timestep: float,
    ) -> None:
        """Initialize MotionClip with resampled motion data.

        Args:
            motion_data: Prepared MotionData from prepare()
            root_positions: Resampled root positions (N, 3) as torch tensor
            source_fps: Original FPS of the motion file
            timestep: Resampled playback timestep in seconds
        """
        self.motion_data = motion_data
        self.root_positions = root_positions
        self.source_fps = float(source_fps)
        self.timestep = float(timestep)
        self._planar_speeds: np.ndarray | None = None
        self._sparkline_points: str | None = None

    @property
    def num_frames(self) -> int:
        return self.motion_data.num_frames

    @property
    def planar_speeds(self) -> np.ndarray:
        """World-frame horizontal speed |v_xy| per frame, in m/s."""
        if self._planar_speeds is None:
            vel = self.motion_data.base_lin_velocities_mixed[:, :2].cpu().numpy()
            self._planar_speeds = np.linalg.norm(vel, axis=1)
        return self._planar_speeds

    @cached_property
    def max_speed(self) -> float:
        speeds = self.planar_speeds
        return float(speeds.max()) if speeds.size else 0.0

    @cached_property
    def max_speed_frame(self) -> int:
        speeds = self.planar_speeds
        return int(speeds.argmax()) if speeds.size else 0

    @property
    def speed_sparkline_points(self) -> str:
        """Speed profile as SVG polyline points in the sparkline viewBox."""
        if self._sparkline_points is None:
            speeds = self.planar_speeds
            if speeds.size == 0:
                self._sparkline_points = ""
            else:
                buckets = min(SPARKLINE_BUCKETS, speeds.size)
                edges = np.linspace(0, speeds.size, buckets + 1).astype(int)
                # Max-pool so brief speed peaks survive the downsample.
                peaks = np.array(
                    [speeds[a:b].max() for a, b in zip(edges[:-1], edges[1:])]
                )
                xs = np.linspace(0.0, 100.0, buckets)
                ys = SPARKLINE_HEIGHT - self.speed_to_y(peaks)
                self._sparkline_points = " ".join(
                    f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys)
                )
        return self._sparkline_points

    def speed_to_y(self, speed: np.ndarray | float) -> np.ndarray | float:
        """Map a speed to a height above the sparkline baseline."""
        denom = self.max_speed or 1.0
        return (SPARKLINE_HEIGHT - 1.0) * speed / denom

    def frame_to_x(self, frame: int) -> float:
        """Map a frame index to a sparkline x coordinate."""
        return 100.0 * frame / max(1, self.num_frames - 1)

    @property
    def dt(self) -> float:
        return self.timestep

    @property
    def duration(self) -> float:
        return max(0, self.num_frames - 1) * self.timestep

    def get_frame_data(
        self, frame_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get motion data for a single frame.

        Args:
            frame_id: Frame index to retrieve

        Returns:
            Tuple of (root_pos, root_quat_wxyz, joint_positions) as numpy arrays
        """
        # Extract joint positions
        joint_pos = self.motion_data.joint_positions[frame_id].cpu().numpy()

        # Extract root position
        root_pos = self.root_positions[frame_id].cpu().numpy()

        # Convert scipy quaternion (xyzw) to MuJoCo (wxyz)
        quat_xyzw = self.motion_data.base_quat[frame_id].cpu().numpy()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        return root_pos, quat_wxyz, joint_pos


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def resolve_device(device_arg: str) -> str:
    if device_arg.lower() != "auto":
        return device_arg
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _hf_motion_entries(dataset: HfMotionDataset) -> list[MotionEntry]:
    """Wrap the clips of a Hugging Face dataset as selectable entries."""
    if not len(dataset):
        raise FileNotFoundError(f"Hugging Face dataset contains no clips: {dataset!r}")

    entries: list[MotionEntry] = []
    seen: dict[str, int] = {}
    for clip in sorted(dataset.clips, key=lambda c: c.name):
        # Dropdown options must be unique, so disambiguate repeated clip names.
        count = seen.get(clip.name, 0)
        seen[clip.name] = count + 1
        name = clip.name if count == 0 else f"{clip.name} ({count + 1})"
        entries.append(MotionEntry(name=name, stem=name, motion=clip.motion))
    return entries


def scan_motion_entries(dataset: str, revision: str | None = None) -> list[MotionEntry]:
    """Return selectable clips from a local path or a Hugging Face dataset repo.

    ``dataset`` may be a motion file, a directory of motion files, a staged
    parquet export from ``motions-to-hf``, or a Hub ID in ``namespace/repo`` form.
    """
    path = Path(dataset)
    supported = ", ".join(SUPPORTED_MOTION_FILE_EXTENSIONS)

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_MOTION_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported motion file '{path}'. Expected one of: {supported}."
            )
        return [MotionEntry(name=path.name, stem=path.stem, path=path)]

    if path.is_dir():
        files = sorted(
            candidate
            for suffix in SUPPORTED_MOTION_FILE_EXTENSIONS
            for candidate in path.glob(f"*{suffix}")
        )
        if files:
            return [
                MotionEntry(name=file.name, stem=file.stem, path=file) for file in files
            ]
        if any(path.glob("*.parquet")) or any((path / "data").glob("*.parquet")):
            return _hf_motion_entries(HfMotionDataset(local_dir=path))
        raise FileNotFoundError(
            f"No motion files with extensions {supported} found in {path}"
        )

    if is_hf_dataset_id(dataset):
        return _hf_motion_entries(HfMotionDataset(dataset, revision=revision))

    raise FileNotFoundError(
        f"Motion dataset path does not exist: {path}. "
        "Hugging Face dataset IDs must use 'namespace/repo' form."
    )


def resolve_initial_entry(
    entries: list[MotionEntry], initial: str | None
) -> MotionEntry:
    """Resolve the initial clip from an optional stem or filename."""
    if initial is None:
        return entries[0]

    for key in (lambda e: e.name, lambda e: e.stem):
        matches = [entry for entry in entries if key(entry) == initial]
        if matches:
            return matches[0]

    supported = ", ".join(SUPPORTED_MOTION_FILE_EXTENSIONS)
    print(
        f"Warning: could not find initial motion '{initial}' "
        f"(expected matching clip name or stem with {supported}) – using first clip"
    )
    return entries[0]


def create_play_env(task: str, device: str) -> PlayEnvContext:
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = 1
    scene = Scene(env_cfg.scene, device=device)
    sim = Simulation(
        num_envs=scene.num_envs,
        cfg=env_cfg.sim,
        model=scene.compile(),
        device=device,
    )
    scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    return PlayEnvContext(env_cfg=env_cfg, scene=scene, sim=sim)


def resolve_body_id(model: mujoco.MjModel, body_name: str) -> int | None:
    """Return the body id, allowing callers to omit prefixes like ``robot/``."""

    if not body_name:
        return None

    candidates = [body_name]
    if "/" in body_name:
        candidates.append(body_name.split("/")[-1])
    else:
        candidates.append(f"robot/{body_name}")

    for candidate in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if body_id >= 0:
            return body_id

    normalized = body_name.split("/")[-1]
    for body_id in range(model.nbody):
        raw = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if raw.split("/")[-1] == normalized:
            return body_id
    return None


def joint_layout_transform(mj_model) -> MotionTransform | None:
    """Retarget clips onto the task's layout when it isn't the serial K1 one."""
    if mj_model.nq - 7 != len(K1_PARALLEL_JOINT_ORDER):
        return None

    from booster_mjlab.robots.booster_k1.parallel_retarget import to_parallel

    return to_parallel


def load_motion_clip(
    motion_file: MotionFile,
    simulation_dt: float,
    device: str = "cpu",
    transform: MotionTransform | None = None,
) -> MotionClip:
    """Resample a motion file onto the simulation timestep for playback.

    Args:
        motion_file: Motion clip loaded from disk or a Hugging Face dataset
        simulation_dt: Simulation timestep for resampling
        device: Device to place tensors on (cpu or cuda)
        transform: Optional remap of the clip onto the task's joint layout

    Returns:
        MotionClip with resampled motion data
    """
    if transform is not None:
        source_dim = motion_file.dof_pos.shape[1]
        motion_file = transform(motion_file)
        if motion_file.dof_pos.shape[1] != source_dim:
            print(
                f"[motion] Retargeted clip from {source_dim} to "
                f"{motion_file.dof_pos.shape[1]} joints for this task's model"
            )

    # Prepare for AMP (handles resampling, velocity computation)
    torch_device = torch.device(device)
    motion_data = motion_file.prepare(
        simulation_dt=simulation_dt,
        speed_factor=1,
        device=torch_device,
    )

    # Manually resample root positions using same logic as prepare()
    fps_value = motion_file.fps if motion_file.fps > 0 else 30.0
    dt = 1.0 / fps_value
    num_frames = motion_file.num_frames
    original_duration = num_frames * dt

    original_keyframes = torch.linspace(
        0, original_duration, steps=num_frames, device=torch_device
    )
    resampled_keyframes = torch.linspace(
        0,
        original_duration,
        steps=int(original_duration / simulation_dt),
        device=torch_device,
    )

    root_pos_torch = torch.as_tensor(
        motion_file.root_pos, dtype=torch.float32, device=torch_device
    )
    resampled_root_pos = motion_file._resample_Rn(
        root_pos_torch, original_keyframes, resampled_keyframes
    )

    return MotionClip(
        motion_data=motion_data,
        root_positions=resampled_root_pos,
        source_fps=motion_file.fps,
        timestep=simulation_dt,
    )


def update_mujoco_state(
    sim: Simulation,
    clip: MotionClip,
    frame_id: int,
) -> None:
    """Copy one dataset frame into MuJoCo's qpos and run forward kinematics."""
    qpos = np.zeros(sim.mj_model.nq, dtype=np.float64)

    root_pos, root_quat_wxyz, joint_positions = clip.get_frame_data(frame_id)

    # Set floating base (position + quaternion)
    qpos[:3] = root_pos
    qpos[3:7] = root_quat_wxyz

    # Set joint positions (directly, since order matches)
    qpos[7:] = joint_positions

    sim.mj_data.qpos[:] = qpos
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


# -----------------------------------------------------------------------------
# Motion editing functions
# -----------------------------------------------------------------------------


def apply_trim(motion_file: MotionFile, start: int, end: int | None) -> MotionFile:
    """Trim motion file to specified frame range.

    Args:
        motion_file: Input motion file
        start: Start frame (inclusive)
        end: End frame (exclusive), None = end of clip

    Returns:
        Trimmed motion file
    """
    if end is None:
        end = motion_file.num_frames

    # Clamp to valid range
    start = max(0, min(start, motion_file.num_frames - 1))
    end = max(start + 1, min(end, motion_file.num_frames))

    return MotionFile(
        fps=motion_file.fps,
        root_pos=motion_file.root_pos[start:end].copy(),
        root_rot=[quat.copy() for quat in motion_file.root_rot[start:end]],
        dof_pos=motion_file.dof_pos[start:end].copy(),
        local_body_pos=(
            motion_file.local_body_pos[start:end].copy()
            if motion_file.local_body_pos is not None
            else None
        ),
        link_body_list=motion_file.link_body_list,
    )


def apply_speed_modification(motion_file: MotionFile, factor: float) -> MotionFile:
    """Apply speed modification by resampling the motion.

    Args:
        motion_file: Input motion file
        factor: Speed factor (>1 = faster, <1 = slower)

    Returns:
        Speed-modified motion file
    """
    if abs(factor - 1.0) < 1e-6:
        # No change needed
        return motion_file

    return SpeedModifierAugmentation(factor=factor).apply(motion_file)


def apply_mirror(motion_file: MotionFile) -> MotionFile:
    """Mirror motion file left-to-right for K1 robot.

    Args:
        motion_file: Input motion file

    Returns:
        Mirrored motion file
    """
    return MirrorAugmentation().apply(motion_file)


def apply_root_offset(
    motion_file: MotionFile,
    pos_offset: np.ndarray,
    rot_offset_deg: np.ndarray,
) -> MotionFile:
    """Apply root position and rotation offset.

    Args:
        motion_file: Input motion file
        pos_offset: Position offset in XYZ (meters)
        rot_offset_deg: Rotation offset in Euler angles XYZ (degrees)

    Returns:
        Motion file with offset applied
    """
    # Apply position offset
    new_root_pos = motion_file.root_pos + pos_offset

    # Convert rotation offset to quaternion
    rot_offset_rad = np.radians(rot_offset_deg)
    offset_rot = Rotation.from_euler("xyz", rot_offset_rad)

    # Apply rotation offset to all root rotations
    new_root_rot = []
    for quat in motion_file.root_rot:
        # Compose rotations: offset * original
        original_rot = Rotation.from_quat(quat)
        new_rot = offset_rot * original_rot
        new_root_rot.append(new_rot.as_quat())

    return MotionFile(
        fps=motion_file.fps,
        root_pos=new_root_pos,
        root_rot=new_root_rot,
        dof_pos=motion_file.dof_pos.copy(),
        local_body_pos=motion_file.local_body_pos,
        link_body_list=motion_file.link_body_list,
    )


# -----------------------------------------------------------------------------
# Foot contact geometry
# -----------------------------------------------------------------------------


FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")
FOOT_LABELS = ("Left", "Right")

# A foot within this distance of the reference ground counts as in contact.
CONTACT_TOLERANCE = 0.005

# Contact points are thinned onto a grid this coarse, so a flat sole shows a
# readable handful of markers instead of thousands of mesh vertices.
CONTACT_POINT_SPACING = 0.02

CONTACT_COLORS: dict[str, tuple[float, float, float, float]] = {
    "contact": (0.25, 0.85, 0.40, 1.0),
    "air": (1.00, 0.72, 0.15, 1.0),
    "penetrating": (0.95, 0.28, 0.28, 1.0),
}


def classify_contact(clearance: float) -> str:
    """Label a foot clearance as contact, air, or penetrating."""
    if clearance > CONTACT_TOLERANCE:
        return "air"
    if clearance < -CONTACT_TOLERANCE:
        return "penetrating"
    return "contact"


@dataclass(slots=True)
class FootState:
    """How both feet meet the ground in one frame."""

    lowest: np.ndarray  # (2, 3) lowest collision point per foot
    clearances: np.ndarray  # (2,) height of that point above the reference ground
    contacts: list[np.ndarray]  # per foot, the points at or below the ground


def _thin_points(points: np.ndarray, spacing: float) -> np.ndarray:
    """Keep the lowest point in each horizontal grid cell."""
    if len(points) == 0:
        return points
    order = np.argsort(points[:, 2])
    cells = np.round(points[order, :2] / spacing).astype(np.int64)
    _, first = np.unique(cells, axis=0, return_index=True)
    return points[order][first]


def _geom_local_points(mj_model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Geom-local points whose lowest world Z bounds the geom from below."""
    if int(mj_model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(mj_model.geom_dataid[geom_id])
        start = int(mj_model.mesh_vertadr[mesh_id])
        count = int(mj_model.mesh_vertnum[mesh_id])
        return mj_model.mesh_vert[start : start + count].astype(np.float64)

    # Fall back to the geom's local bounding box for primitives.
    center = mj_model.geom_aabb[geom_id, :3]
    half = mj_model.geom_aabb[geom_id, 3:]
    signs = np.array(
        [(sx, sy, sz) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    return center + signs * half


class FootGeometry:
    """Foot collision geometry, used for exact ground-clearance queries.

    The home keyframe defines "standing on the ground": the collision meshes
    may dip slightly below z=0 there, so that level is the reference ground
    rather than z=0 itself.
    """

    def __init__(
        self,
        body_ids: list[int],
        geom_ids: list[list[int]],
        local_points: list[list[np.ndarray]],
        reference_z: float,
    ) -> None:
        self.body_ids = body_ids
        self.geom_ids = geom_ids
        self.local_points = local_points
        self.reference_z = reference_z

    @classmethod
    def create(cls, mj_model: mujoco.MjModel) -> FootGeometry | None:
        """Collect both feet's collision geoms, or None if the model lacks them."""
        body_ids: list[int] = []
        geom_ids: list[list[int]] = []
        local_points: list[list[np.ndarray]] = []

        for name in FOOT_BODY_NAMES:
            body_id = resolve_body_id(mj_model, name)
            if body_id is None:
                print(f"Warning: could not find body '{name}'")
                return None
            ids = [
                geom_id
                for geom_id in range(mj_model.ngeom)
                if int(mj_model.geom_bodyid[geom_id]) == body_id
                and (
                    int(mj_model.geom_contype[geom_id])
                    | int(mj_model.geom_conaffinity[geom_id])
                )
                and int(mj_model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
            ]
            if not ids:
                print(f"Warning: body '{name}' has no collision geoms")
                return None
            body_ids.append(body_id)
            geom_ids.append(ids)
            local_points.append([_geom_local_points(mj_model, g) for g in ids])

        geometry = cls(body_ids, geom_ids, local_points, reference_z=0.0)
        geometry.reference_z = geometry._home_keyframe_z(mj_model)
        return geometry

    def _home_keyframe_z(self, mj_model: mujoco.MjModel) -> float:
        mj_data = mujoco.MjData(mj_model)
        if mj_model.nkey > 0:
            mj_data.qpos[:] = mj_model.key_qpos[0]
        else:
            mj_data.qpos[:] = mj_model.qpos0
        mujoco.mj_forward(mj_model, mj_data)
        return float(self.lowest_points(mj_data)[:, 2].min())

    def _world_points(self, mj_data: mujoco.MjData) -> list[np.ndarray]:
        """Every collision point of each foot, in the world frame."""
        feet: list[np.ndarray] = []
        for ids, locals_ in zip(self.geom_ids, self.local_points):
            parts = [
                mj_data.geom_xpos[geom_id]
                + local @ mj_data.geom_xmat[geom_id].reshape(3, 3).T
                for geom_id, local in zip(ids, locals_)
            ]
            feet.append(parts[0] if len(parts) == 1 else np.concatenate(parts))
        return feet

    def lowest_points(self, mj_data: mujoco.MjData) -> np.ndarray:
        """World-frame lowest collision point of each foot, shape (2, 3)."""
        return np.array(
            [
                points[int(points[:, 2].argmin())]
                for points in self._world_points(mj_data)
            ]
        )

    def clearances(self, mj_data: mujoco.MjData) -> np.ndarray:
        """Height of each foot's lowest point above the reference ground."""
        return self.lowest_points(mj_data)[:, 2] - self.reference_z

    def evaluate(self, mj_data: mujoco.MjData) -> FootState:
        """Lowest point, clearance and contact patch of both feet."""
        lowest = []
        contacts = []
        for points in self._world_points(mj_data):
            lowest.append(points[int(points[:, 2].argmin())])
            touching = points[points[:, 2] - self.reference_z <= CONTACT_TOLERANCE]
            contacts.append(_thin_points(touching, CONTACT_POINT_SPACING))
        lowest = np.array(lowest)
        return FootState(
            lowest=lowest,
            clearances=lowest[:, 2] - self.reference_z,
            contacts=contacts,
        )


def apply_fix_ground_contacts(
    motion_file: MotionFile,
    mj_model: mujoco.MjModel,
    smooth_sigma: float = 0.0,
    transform: MotionTransform | None = None,
) -> MotionFile:
    """Adjust root_pos_z so the stance foot rests on the reference ground.

    Uses MuJoCo FK plus the compiled foot collision geometry to compute exact
    foot-bottom positions per frame; see :class:`FootGeometry` for how the
    reference ground level is derived.

    Args:
        motion_file: Input motion file
        mj_model: MuJoCo model for FK computation
        smooth_sigma: Gaussian smoothing sigma in frames (0 = no smoothing)
        transform: Optional remap of the clip onto the model's joint layout

    Returns:
        Motion file with corrected root Z positions (in the source layout)
    """
    geometry = FootGeometry.create(mj_model)
    if geometry is None:
        print("Warning: no foot collision geometry found, skipping ground fix")
        return motion_file

    # FK needs the model's own joint layout; the result only shifts root height,
    # so the returned clip keeps the layout it came in with.
    dof_pos = (transform(motion_file) if transform is not None else motion_file).dof_pos

    mj_data = mujoco.MjData(mj_model)
    num_frames = motion_file.num_frames
    corrections = np.zeros(num_frames)

    for i in range(num_frames):
        # Set qpos from motion data
        mj_data.qpos[:] = 0.0
        mj_data.qpos[:3] = motion_file.root_pos[i]
        # Convert scipy quaternion (xyzw) to MuJoCo (wxyz)
        quat_xyzw = motion_file.root_rot[i]
        mj_data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        mj_data.qpos[7:] = dof_pos[i]

        mujoco.mj_forward(mj_model, mj_data)

        # Use the highest foot (stance foot) to determine the correction.
        # Using the global minimum would overcorrect when the swing foot dips
        # below the stance foot, causing the stance foot to float.
        corrections[i] = -float(geometry.clearances(mj_data).max())

    # Optional smoothing
    if smooth_sigma > 0 and num_frames > 1:
        corrections = gaussian_filter1d(corrections, sigma=smooth_sigma)

    # Apply corrections to root_pos_z
    new_root_pos = motion_file.root_pos.copy()
    new_root_pos[:, 2] += corrections

    return MotionFile(
        fps=motion_file.fps,
        root_pos=new_root_pos,
        root_rot=motion_file.root_rot,
        dof_pos=motion_file.dof_pos,
        local_body_pos=motion_file.local_body_pos,
        link_body_list=motion_file.link_body_list,
    )


def compile_edits(
    original: MotionFile,
    edit_state: EditState,
    mj_model: mujoco.MjModel | None = None,
    transform: MotionTransform | None = None,
) -> MotionFile:
    """Apply all edits from EditState to create final MotionFile.

    Applies edits in order: trim -> speed -> mirror -> root offset

    Args:
        original: Original motion file
        edit_state: Edit parameters

    Returns:
        Edited motion file with all transformations applied
    """
    result = original

    # 1. Apply trim
    if edit_state.trim_start > 0 or edit_state.trim_end is not None:
        result = apply_trim(result, edit_state.trim_start, edit_state.trim_end)

    # 2. Apply speed modification
    if abs(edit_state.speed_factor - 1.0) > 1e-6:
        result = apply_speed_modification(result, edit_state.speed_factor)

    # 3. Apply mirror
    if edit_state.mirrored:
        result = apply_mirror(result)

    # 4. Apply root offset
    if np.any(np.abs(edit_state.root_position_offset) > 1e-6) or np.any(
        np.abs(edit_state.root_rotation_offset) > 1e-6
    ):
        result = apply_root_offset(
            result,
            edit_state.root_position_offset,
            edit_state.root_rotation_offset,
        )

    # 5. Fix ground contacts (last, depends on final FK state)
    if edit_state.fix_ground and mj_model is not None:
        result = apply_fix_ground_contacts(result, mj_model, transform=transform)

    return result


def save_motion_file(motion_file: MotionFile, path: Path) -> None:
    """Save motion file in the same format as original .pkl files.

    Args:
        motion_file: Motion file to save
        path: Output path
    """
    data = {
        "fps": motion_file.fps,
        "root_pos": motion_file.root_pos,
        "root_rot": motion_file.root_rot,
        "dof_pos": motion_file.dof_pos,
        "local_body_pos": motion_file.local_body_pos,
        "link_body_list": motion_file.link_body_list,
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"✓ Saved edited motion to: {path}")


# -----------------------------------------------------------------------------
# Visualization / UI helpers
# -----------------------------------------------------------------------------


# Camera azimuth and elevation per view, relative to the robot's heading.
CONTACT_VIEWS: dict[str, tuple[float, float]] = {
    "Side": (90.0, 0.0),
    "Front": (180.0, 0.0),
    "Top": (90.0, -89.9),
}


def _free_joint_body_id(mj_model: mujoco.MjModel) -> int | None:
    """Body carrying the floating base, whose yaw is the robot's heading."""
    for joint_id in range(mj_model.njnt):
        if int(mj_model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(mj_model.jnt_bodyid[joint_id])
    return None


GROUND_LINE_RGBA = (0.45, 0.75, 1.0, 1.0)

# Offscreen rendering costs ~15 ms, so cap it below the playback frame rate.
CONTACT_VIEW_MIN_INTERVAL = 0.05

# Refresh status and sparklines at 20 Hz.
STATUS_MIN_INTERVAL = 0.05

# Restart the playback clock after a long stall.
PLAYBACK_RESYNC_THRESHOLD = 0.25


def schedule_next_frame(
    deadline: float, now: float, step_dt: float
) -> tuple[float, int, float]:
    """Return the next deadline, frame advance, and wait after rendering."""
    deadline += step_dt
    remaining = deadline - now
    if remaining > 0.0:
        return deadline, 1, remaining
    if -remaining > PLAYBACK_RESYNC_THRESHOLD:
        return now, 1, 0.0
    # Skip overdue frames to keep rendering overhead from slowing playback.
    skipped = int(-remaining / step_dt)
    return deadline + skipped * step_dt, 1 + skipped, 0.0


class OrthographicFootView:
    """Offscreen orthographic render of the feet against the reference ground.

    Orthographic projection keeps the ground a straight line at every depth,
    which is what makes float and penetration readable by eye.
    """

    WIDTH = 480
    HEIGHT = 300
    # Overlay sizes in pixels: contact marker radius and ground line radius.
    MARKER_PIXELS = 3.0
    GROUND_LINE_PIXELS = 1.5
    # Overlays are drawn this far in front of the camera, clear of the near plane.
    OVERLAY_MARGIN = 0.05

    def __init__(self, mj_model: mujoco.MjModel, geometry: FootGeometry) -> None:
        self._mj_model = mj_model
        self._geometry = geometry
        self._renderer: mujoco.Renderer | None = None
        self._render_thread: int | None = None
        self._unavailable = False
        # Edge-on, the terrain plane smears into a band that hides the ground
        # line and any foot resting on it, so those views drop world geometry.
        self._no_static = mujoco.MjvOption()
        self._no_static.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
        # Pull the camera back far enough that nothing near-clips.
        self._camera_distance = 2.0 * float(mj_model.stat.extent)
        self._root_body_id = _free_joint_body_id(mj_model)

    def _get_renderer(self) -> mujoco.Renderer | None:
        if self._unavailable:
            return None
        if self._renderer is None:
            try:
                # The offscreen buffer must be at least as large as our frame.
                visual = self._mj_model.vis.global_
                visual.offwidth = max(int(visual.offwidth), self.WIDTH)
                visual.offheight = max(int(visual.offheight), self.HEIGHT)
                self._renderer = mujoco.Renderer(
                    self._mj_model, self.HEIGHT, self.WIDTH
                )
                self._render_thread = threading.get_ident()
            except Exception as exc:
                print(f"[contact] Offscreen rendering unavailable: {exc}")
                self._unavailable = True
                return None
        if threading.get_ident() != self._render_thread:
            # The GL context belongs to the thread that created it; on macOS,
            # rendering from any other thread deadlocks that thread for good.
            return None
        return self._renderer

    def render(
        self,
        mj_data: mujoco.MjData,
        state: FootState,
        view: str,
        view_height: float,
    ) -> np.ndarray | None:
        """Render the feet, or None when offscreen rendering is unavailable."""
        renderer = self._get_renderer()
        if renderer is None:
            return None

        azimuth, elevation = CONTACT_VIEWS[view]
        top_down = elevation < -45.0
        center = state.lowest.mean(axis=0)
        ground_z = self._geometry.reference_z

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        # Views are body-relative, so they stay sagittal/coronal as the robot turns.
        cam.azimuth = azimuth + self._heading_deg(mj_data)
        cam.elevation = elevation
        # Looking down, sit just above the feet so the torso falls behind the
        # camera and gets clipped away, leaving a clean footprint view.
        cam.distance = (
            max(0.3, self._feet_top_z(mj_data) - ground_z + 0.1)
            if top_down
            else self._camera_distance
        )
        cam.lookat[:] = (
            center[0],
            center[1],
            ground_z if top_down else ground_z + 0.25 * view_height,
        )

        # For a free camera MuJoCo reads the projection from the model, where
        # an orthographic fovy is the frame height in meters rather than an angle.
        visual = self._mj_model.vis.global_
        previous = int(visual.orthographic), float(visual.fovy)
        visual.orthographic = 1
        visual.fovy = view_height
        try:
            renderer.update_scene(mj_data, cam, None if top_down else self._no_static)
            self._draw_overlays(renderer.scene, state, cam, view_height, top_down)
            return renderer.render()
        finally:
            visual.orthographic, visual.fovy = previous

    def _heading_deg(self, mj_data: mujoco.MjData) -> float:
        """Yaw of the robot's forward axis, in degrees."""
        if self._root_body_id is None:
            return 0.0
        rot = mj_data.xmat[self._root_body_id].reshape(3, 3)
        forward = rot[:, 0]
        if np.hypot(forward[0], forward[1]) < 1e-6:
            # Pitched bolt upright: take the yaw from the lateral axis instead.
            return float(np.degrees(np.arctan2(-rot[0, 1], rot[1, 1])))
        return float(np.degrees(np.arctan2(forward[1], forward[0])))

    def _feet_top_z(self, mj_data: mujoco.MjData) -> float:
        return max(
            float(mj_data.xpos[body_id][2]) for body_id in self._geometry.body_ids
        )

    @staticmethod
    def _add_geom(
        scene: mujoco.MjvScene,
        geom_type: mujoco.mjtGeom,
        rgba: tuple[float, float, float, float],
    ):
        """Claim and initialize the next free geom slot in a rendered scene."""
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            geom,
            int(geom_type),
            np.zeros(3),
            np.zeros(3),
            np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32),
        )
        # Matte, so the overlay colors stay readable.
        geom.specular = 0.0
        geom.shininess = 0.0
        return geom

    def _draw_overlays(
        self,
        scene: mujoco.MjvScene,
        state: FootState,
        cam: mujoco.MjvCamera,
        view_height: float,
        top_down: bool,
    ) -> None:
        ground_z = self._geometry.reference_z
        az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
        forward = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        camera_pos = np.array(cam.lookat) - forward * cam.distance

        def in_front_of_camera(point: np.ndarray) -> np.ndarray:
            """Slide a point along the view axis to just in front of the camera.

            Under orthographic projection this leaves it where it is on screen,
            but nothing in the scene can occlude it any more.
            """
            depth = float(np.dot(point - camera_pos, forward))
            return point - forward * (depth - self.OVERLAY_MARGIN)

        # Overlays are sized in pixels, so they stay legible at any zoom.
        meters_per_pixel = view_height / self.HEIGHT

        if not top_down:
            # Span the frame along the camera's screen-horizontal axis.
            right = np.array([np.sin(az), -np.cos(az), 0.0])
            half_width = 0.5 * view_height * self.WIDTH / self.HEIGHT
            origin = in_front_of_camera(
                np.array([cam.lookat[0], cam.lookat[1], ground_z])
            )
            geom = self._add_geom(
                scene, mujoco.mjtGeom.mjGEOM_CAPSULE, GROUND_LINE_RGBA
            )
            if geom is not None:
                mujoco.mjv_connector(
                    geom,
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                    self.GROUND_LINE_PIXELS * meters_per_pixel,
                    origin - right * half_width,
                    origin + right * half_width,
                )

        for foot, contacts in enumerate(state.contacts):
            # An airborne foot has no contact patch, so mark its lowest point.
            points = contacts if len(contacts) else state.lowest[foot][None]
            for point in points:
                rgba = CONTACT_COLORS[classify_contact(float(point[2]) - ground_z)]
                geom = self._add_geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, rgba)
                if geom is None:
                    return
                geom.size[:] = self.MARKER_PIXELS * meters_per_pixel
                geom.pos[:] = in_front_of_camera(point)


def contact_readout_html(state: FootState | None, message: str = "") -> str:
    """Format per-foot ground clearance as a compact HTML readout."""
    if state is None:
        return f'<div style="font-size:0.85em; padding:0.5em; opacity:0.7;">{message}</div>'

    rows = []
    for label, clearance, contacts in zip(
        FOOT_LABELS, state.clearances, state.contacts
    ):
        status = classify_contact(float(clearance))
        rgba = CONTACT_COLORS[status]
        color = "rgb({}, {}, {})".format(*(int(255 * c) for c in rgba[:3]))
        patch = f" · {len(contacts)} pts" if len(contacts) else ""
        rows.append(
            f'<div style="display:flex; justify-content:space-between; gap:1em;">'
            f"<span>{label}</span>"
            f'<span style="color:{color};">'
            f"{clearance * 1000:+.1f} mm · {status}{patch}</span>"
            f"</div>"
        )
    note = (
        f'<div style="margin-top:0.35em; opacity:0.6;">{message}</div>'
        if message
        else ""
    )
    return f"""
        <div style="font-variant-numeric:tabular-nums; font-size:0.85em; line-height:1.6;">
          {"".join(rows)}
          <div style="margin-top:0.35em; opacity:0.6;">
            Height of each foot's lowest collision point above the reference
            ground (within ±{CONTACT_TOLERANCE * 1000:.0f} mm counts as contact);
            "pts" is the size of its contact patch.
          </div>
          {note}
        </div>
    """


def format_timecode(seconds: float) -> str:
    """Format seconds as a compact, fixed-width player timecode."""
    minutes, seconds = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def setup_viewer(
    ctx: PlayEnvContext,
    state: PlaybackState,
    state_lock: threading.Lock,
    edit_state: EditState,
    edit_lock: threading.Lock,
    contact_state: ContactViewState,
    contact_lock: threading.Lock,
    motion_holder: list[MotionFile],
    clip_holder: list[MotionClip],
    motion_entries: list[MotionEntry],
    current_entry: MotionEntry,
) -> tuple[ViewerContext, EditUI]:
    server = viser.ViserServer()
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(
        dark_mode=False,
        show_logo=False,
        show_share_button=False,
        brand_color=(64, 135, 245),
    )

    mj_scene = ViserMujocoScene(
        server=server,
        mj_model=ctx.sim.mj_model,
        num_envs=ctx.sim.num_envs,
    )
    slider_lock = threading.Lock()

    tabs = server.gui.add_tab_group(order=0)
    with tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
        motion_select = server.gui.add_dropdown(
            "Clip",
            options=[entry.name for entry in motion_entries],
            initial_value=current_entry.name,
        )
        status_html = server.gui.add_html("")
        frame_slider = server.gui.add_slider(
            "Timeline",
            min=0,
            max=max(1, state.total_frames - 1),
            step=1,
            initial_value=state.frame,
            disabled=state.total_frames <= 1,
            hint="Seek to a frame; seeking pauses playback",
        )
        speed_html = server.gui.add_html("")
        transport_buttons = server.gui.add_button_group(
            "Transport",
            (
                "⏮",
                "◀",
                "▶ Play" if state.paused else "Pause ❚❚",
                "▶",
                "⏭",
            ),
            hint="First · Previous · Play/Pause · Next · Last",
        )
        speed_input = server.gui.add_number(
            "Playback speed",
            initial_value=state.speed,
            min=0.001,
            step=0.05,
            hint="Playback multiplier (>1 = faster)",
        )
        loop_checkbox = server.gui.add_checkbox(
            "Loop playback",
            initial_value=state.loop,
            hint="Restart when the clip ends",
        )
    with tabs.add_tab("Edit", icon=viser.Icon.PENCIL):
        edit_info = server.gui.add_html(
            """<div style="font-size:0.85em; padding:0 0.5em 0.5em;">
            Configure edits, then click <strong>Apply Edits</strong> to preview.
            </div>"""
        )

        # Trim controls
        trim_folder = server.gui.add_folder("Trim", expand_by_default=True)
        with trim_folder:
            trim_start_input = server.gui.add_number(
                "Start Frame",
                initial_value=edit_state.trim_start,
                min=0,
                max=motion_holder[0].num_frames - 1,
                step=1,
                hint="First frame to include (inclusive)",
            )
            trim_end_input = server.gui.add_number(
                "End Frame",
                initial_value=(
                    edit_state.trim_end
                    if edit_state.trim_end is not None
                    else motion_holder[0].num_frames
                ),
                min=1,
                max=motion_holder[0].num_frames,
                step=1,
                hint="Last frame to include (exclusive), 0 = end of clip",
            )
            trim_reset_button = server.gui.add_button("Reset Trim")

        # Speed controls
        speed_folder = server.gui.add_folder("Speed", expand_by_default=False)
        with speed_folder:
            speed_factor_input = server.gui.add_number(
                "Speed Factor",
                initial_value=edit_state.speed_factor,
                min=0.1,
                max=10.0,
                step=0.1,
                hint=">1 = faster, <1 = slower",
            )
            speed_info = server.gui.add_html(
                f"""<div style="font-size:0.8em; padding:0 0.5em;">
                Current: {edit_state.speed_factor:.2f}x<br/>
                Original frames: {motion_holder[0].num_frames}
                </div>"""
            )

        # Mirror controls
        mirror_folder = server.gui.add_folder("Mirror", expand_by_default=False)
        with mirror_folder:
            mirror_checkbox = server.gui.add_checkbox(
                "Enable Mirror",
                initial_value=edit_state.mirrored,
                hint="Swap left/right limbs (K1 robot)",
            )
            mirror_info = server.gui.add_html(
                """<div style="font-size:0.8em; padding:0 0.5em;">
                Mirrors left/right limbs for K1 robot.<br/>
                Root Y position will be negated.
                </div>"""
            )

        # Root offset controls
        offset_folder = server.gui.add_folder("Root Offset", expand_by_default=False)
        with offset_folder:
            server.gui.add_html(
                "<div style='font-size:0.8em; margin-bottom:0.5em;'>Position (meters):</div>"
            )
            pos_x = server.gui.add_number(
                "X",
                initial_value=float(edit_state.root_position_offset[0]),
                step=0.01,
            )
            pos_y = server.gui.add_number(
                "Y",
                initial_value=float(edit_state.root_position_offset[1]),
                step=0.01,
            )
            pos_z = server.gui.add_number(
                "Z",
                initial_value=float(edit_state.root_position_offset[2]),
                step=0.01,
            )
            server.gui.add_html(
                "<div style='font-size:0.8em; margin:0.5em 0;'>Rotation (degrees):</div>"
            )
            rot_x = server.gui.add_number(
                "Roll (X)",
                initial_value=float(edit_state.root_rotation_offset[0]),
                step=1.0,
            )
            rot_y = server.gui.add_number(
                "Pitch (Y)",
                initial_value=float(edit_state.root_rotation_offset[1]),
                step=1.0,
            )
            rot_z = server.gui.add_number(
                "Yaw (Z)",
                initial_value=float(edit_state.root_rotation_offset[2]),
                step=1.0,
            )
            offset_reset_button = server.gui.add_button("Reset Offsets")

        # Ground contact fix
        ground_folder = server.gui.add_folder(
            "Ground Contacts", expand_by_default=False
        )
        with ground_folder:
            fix_ground_checkbox = server.gui.add_checkbox(
                "Fix Ground Contacts",
                initial_value=edit_state.fix_ground,
                hint="Adjust root Z so feet sit on the ground (z=0)",
            )

        # Apply/Save controls
        server.gui.add_html("<hr style='margin:1em 0;'/>")
        apply_button = server.gui.add_button(
            "Apply Edits",
            color="blue",
            icon=viser.Icon.REFRESH,
            hint="Recompile motion with current edit settings",
        )
        server.gui.add_html("<div style='height:0.5em;'></div>")

        # Generate default save path
        default_save_path = current_entry.default_save_path
        save_path_input = server.gui.add_text(
            "Save Path",
            initial_value=str(default_save_path),
            hint="Output path for edited motion",
        )
        save_button = server.gui.add_button(
            "Save Edited Clip",
            color="green",
            icon=viser.Icon.DEVICE_FLOPPY,
        )

    with tabs.add_tab("Contacts", icon=viser.Icon.SHOE):
        contact_enabled_checkbox = server.gui.add_checkbox(
            "Show contact view",
            initial_value=contact_state.enabled,
            hint="Render an orthographic view of the feet on every frame",
        )
        contact_view_select = server.gui.add_dropdown(
            "Projection",
            options=tuple(CONTACT_VIEWS),
            initial_value=contact_state.view,
            hint="Camera direction, relative to the robot's heading",
        )
        contact_zoom_slider = server.gui.add_slider(
            "Frame height (m)",
            min=0.15,
            max=1.5,
            step=0.05,
            initial_value=contact_state.view_height,
            hint="Vertical extent of the orthographic frame",
        )
        contact_image = server.gui.add_image(
            np.zeros(
                (OrthographicFootView.HEIGHT, OrthographicFootView.WIDTH, 3),
                dtype=np.uint8,
            ),
            format="jpeg",
        )
        contact_readout = server.gui.add_html("")

    with tabs.add_tab("Scene", icon=viser.Icon.LAYERS_INTERSECT):
        mj_scene.create_scene_gui(
            camera_distance=ctx.env_cfg.viewer.distance,
            camera_azimuth=ctx.env_cfg.viewer.azimuth,
            camera_elevation=ctx.env_cfg.viewer.elevation,
        )
        mj_scene.create_groups_gui()
        step_physics_button = server.gui.add_button(
            "Step Physics",
            icon=viser.Icon.PLAYER_SKIP_FORWARD,
            hint=(
                "Run one policy step from the current pose. "
                "Only available while playback is paused."
            ),
        )
        with server.gui.add_folder("Motion Diagnostics", expand_by_default=False):
            diagnostics_html = server.gui.add_html("")

    server.gui.main_panel.dock_right()
    server.gui.main_panel.set_width(400)

    edit_ui = EditUI(
        trim_start_input=trim_start_input,
        trim_end_input=trim_end_input,
        trim_reset_button=trim_reset_button,
        speed_factor_input=speed_factor_input,
        speed_info=speed_info,
        mirror_checkbox=mirror_checkbox,
        mirror_info=mirror_info,
        pos_x=pos_x,
        pos_y=pos_y,
        pos_z=pos_z,
        rot_x=rot_x,
        rot_y=rot_y,
        rot_z=rot_z,
        offset_reset_button=offset_reset_button,
        fix_ground_checkbox=fix_ground_checkbox,
        apply_button=apply_button,
        save_button=save_button,
        save_path_input=save_path_input,
        edit_info=edit_info,
    )

    ui = PlaybackUI(
        status_html=status_html,
        speed_html=speed_html,
        diagnostics_html=diagnostics_html,
        transport_buttons=transport_buttons,
        frame_slider=frame_slider,
        speed_input=speed_input,
        loop_checkbox=loop_checkbox,
        slider_lock=slider_lock,
        motion_select=motion_select,
    )

    last_status_update = 0.0

    def update_status(force: bool = False) -> None:
        nonlocal last_status_update
        now = time.monotonic()
        if not force and now - last_status_update < STATUS_MIN_INTERVAL:
            return
        last_status_update = now
        with state_lock:
            frame = state.frame
            paused = state.paused
            total_frames = state.total_frames
        source_fps = float(motion_holder[0].fps)
        current_clip = clip_holder[0]
        frame_idx = int(np.clip(frame, 0, current_clip.num_frames - 1))
        vx = float(
            current_clip.motion_data.base_lin_velocities_local[frame_idx, 0].item()
        )
        vy = float(
            current_clip.motion_data.base_lin_velocities_local[frame_idx, 1].item()
        )
        ang_vel_z = float(
            current_clip.motion_data.base_ang_velocities_local[frame_idx, 2].item()
        )
        speed = float(current_clip.planar_speeds[frame_idx])
        max_speed = current_clip.max_speed
        max_speed_frame = current_clip.max_speed_frame
        max_speed_time = format_timecode(max_speed_frame * current_clip.dt)
        speed_baseline = SPARKLINE_HEIGHT
        speed_y = speed_baseline - current_clip.speed_to_y(speed)
        max_speed_y = speed_baseline - current_clip.speed_to_y(max_speed)
        playhead_x = current_clip.frame_to_x(frame_idx)
        max_speed_x = current_clip.frame_to_x(max_speed_frame)
        ui.status_html.content = f"""
            <div style="padding:0 0.25em 0.25em; font-variant-numeric:tabular-nums;">
              <div style="display:flex; justify-content:space-between; gap:1em; font-size:0.95em;">
                <strong>{"Paused" if paused else "Playing"}</strong>
                <span>{format_timecode(frame_idx * current_clip.dt)} / {format_timecode(current_clip.duration)}</span>
              </div>
              <div style="margin-top:0.3em; opacity:0.65; font-size:0.78em;">
                Frame {frame_idx + 1} of {total_frames} · {source_fps:.2f} FPS
              </div>
            </div>
        """
        ui.speed_html.content = f"""
            <div style="padding:0 0.25em 0.5em; font-variant-numeric:tabular-nums;">
              <div style="display:flex; justify-content:space-between; gap:1em;
                          font-size:0.8em;">
                <span><strong>{speed:.3f}</strong> m/s</span>
                <span style="opacity:0.7;">
                  peak {max_speed:.3f} m/s @ {max_speed_time}
                </span>
              </div>
              <svg viewBox="0 0 100 {speed_baseline:.0f}" preserveAspectRatio="none"
                   style="width:100%; height:44px; display:block; margin-top:0.2em;">
                <polyline points="{current_clip.speed_sparkline_points}" fill="none"
                          stroke="currentColor" stroke-opacity="0.45" stroke-width="1"
                          vector-effect="non-scaling-stroke"/>
                <line x1="0" y1="{max_speed_y:.2f}" x2="100" y2="{max_speed_y:.2f}"
                      stroke="#e5484d" stroke-opacity="0.7" stroke-width="1"
                      stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>
                <line x1="{max_speed_x:.2f}" y1="{max_speed_y:.2f}"
                      x2="{max_speed_x:.2f}" y2="{speed_baseline:.2f}"
                      stroke="#e5484d" stroke-opacity="0.7" stroke-width="1"
                      vector-effect="non-scaling-stroke"/>
                <line x1="{playhead_x:.2f}" y1="0" x2="{playhead_x:.2f}"
                      y2="{speed_baseline:.2f}" stroke="#4087f5" stroke-width="1"
                      vector-effect="non-scaling-stroke"/>
                <line x1="{max(0.0, playhead_x - 1.5):.2f}" y1="{speed_y:.2f}"
                      x2="{min(100.0, playhead_x + 1.5):.2f}" y2="{speed_y:.2f}"
                      stroke="#4087f5" stroke-width="2"
                      vector-effect="non-scaling-stroke"/>
              </svg>
            </div>
        """
        ui.diagnostics_html.content = f"""
            <div style="font-variant-numeric:tabular-nums; font-size:0.85em; line-height:1.6;">
              <div><strong>Linear velocity</strong> {vx:+.3f}, {vy:+.3f} m/s</div>
              <div><strong>Yaw velocity</strong> {ang_vel_z:+.3f} rad/s</div>
              <div><strong>Speed</strong> {speed:.3f} m/s (max {max_speed:.3f})</div>
              <div style="margin-top:0.35em; opacity:0.6;">
                Space play/pause · J/L step · Home/End jump
              </div>
            </div>
        """

    def update_transport_buttons() -> None:
        with state_lock:
            paused = state.paused
        ui.transport_buttons.options = (
            "⏮",
            "◀",
            "▶ Play" if paused else "Pause ❚❚",
            "▶",
            "⏭",
        )

    def set_slider_value(value: int) -> None:
        with ui.slider_lock:
            ui.slider_internal = True
        try:
            ui.frame_slider.value = int(value)
        finally:
            with ui.slider_lock:
                ui.slider_internal = False

    def toggle_playback() -> None:
        with state_lock:
            state.paused = not state.paused
            paused = state.paused
            if not paused:
                state.pending_step = False
        print(f"[ui] Pause toggled -> {'paused' if paused else 'playing'}")
        update_transport_buttons()
        update_status(force=True)

    def seek_frame(target: int) -> None:
        with state_lock:
            state.seek_to = int(np.clip(target, 0, state.total_frames - 1))
            state.paused = True
        update_transport_buttons()
        update_status(force=True)

    @transport_buttons.on_click
    def _(_) -> None:
        if transport_buttons.value in ("▶ Play", "Pause ❚❚"):
            toggle_playback()
            return
        with state_lock:
            current_frame = state.frame
            last_frame = state.total_frames - 1
        target = {
            "⏮": 0,
            "◀": current_frame - 1,
            "▶": current_frame + 1,
            "⏭": last_frame,
        }[transport_buttons.value]
        seek_frame(target)

    @step_physics_button.on_click
    def _(_) -> None:
        with state_lock:
            if not state.paused:
                print("[ui] Step Physics ignored – not paused")
                return
            state.pending_step = True
        print("[ui] Step Physics requested")

    @loop_checkbox.on_update
    def _(_) -> None:
        with state_lock:
            state.loop = bool(loop_checkbox.value)
            loop_enabled = state.loop
        print(f"[ui] Loop set -> {loop_enabled}")
        update_status(force=True)

    name_to_entry = {entry.name: entry for entry in motion_entries}

    @motion_select.on_update
    def _(_) -> None:
        selected = motion_select.value
        with state_lock:
            state.pending_motion = name_to_entry[selected]
        print(f"[ui] Motion selected -> {selected}")

    @speed_input.on_update
    def _(_) -> None:
        new_speed = max(float(speed_input.value), 1e-3)
        with state_lock:
            state.speed = new_speed
        print(f"[ui] Speed set -> {new_speed:.2f}")
        update_status(force=True)

    @frame_slider.on_update
    def _(_) -> None:
        with ui.slider_lock:
            if ui.slider_internal:
                return
        target = int(frame_slider.value)
        seek_frame(target)
        print(f"[ui] Timeline seek -> frame {target}")

    toggle_command = server.gui.add_command(
        "Play / pause",
        description="Toggle motion playback",
        hotkey="space",
        icon=viser.Icon.PLAYER_PLAY,
    )
    # J/L rather than the arrows: viser binds WASDQE and all four arrows to
    # camera movement, keyed on event.code, so a modifier can't dodge it.
    previous_command = server.gui.add_command(
        "Previous frame",
        hotkey="J",
        icon=viser.Icon.PLAYER_SKIP_BACK,
    )
    next_command = server.gui.add_command(
        "Next frame",
        hotkey="L",
        icon=viser.Icon.PLAYER_SKIP_FORWARD,
    )
    first_command = server.gui.add_command("First frame", hotkey="home")
    last_command = server.gui.add_command("Last frame", hotkey="end")

    @toggle_command.on_trigger
    def _(_) -> None:
        toggle_playback()

    @previous_command.on_trigger
    def _(_) -> None:
        with state_lock:
            target = state.frame - 1
        seek_frame(target)

    @next_command.on_trigger
    def _(_) -> None:
        with state_lock:
            target = state.frame + 1
        seek_frame(target)

    @first_command.on_trigger
    def _(_) -> None:
        seek_frame(0)

    @last_command.on_trigger
    def _(_) -> None:
        with state_lock:
            target = state.total_frames - 1
        seek_frame(target)

    # Edit tab callbacks
    @trim_start_input.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.trim_start = int(trim_start_input.value)
        print(f"[edit] Trim start -> {edit_state.trim_start}")

    @trim_end_input.on_update
    def _(_) -> None:
        with edit_lock:
            val = int(trim_end_input.value)
            edit_state.trim_end = val if val < motion_holder[0].num_frames else None
        print(f"[edit] Trim end -> {edit_state.trim_end}")

    @trim_reset_button.on_click
    def _(_) -> None:
        with edit_lock:
            edit_state.trim_start = 0
            edit_state.trim_end = None
        trim_start_input.value = 0
        trim_end_input.value = motion_holder[0].num_frames
        print("[edit] Trim reset")

    @speed_factor_input.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.speed_factor = max(0.1, float(speed_factor_input.value))
        print(f"[edit] Speed factor -> {edit_state.speed_factor:.2f}")

    @mirror_checkbox.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.mirrored = bool(mirror_checkbox.value)
        print(f"[edit] Mirror -> {edit_state.mirrored}")

    @fix_ground_checkbox.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.fix_ground = bool(fix_ground_checkbox.value)
        print(f"[edit] Fix ground -> {edit_state.fix_ground}")

    @pos_x.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.root_position_offset[0] = float(pos_x.value)

    @pos_y.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.root_position_offset[1] = float(pos_y.value)

    @pos_z.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.root_position_offset[2] = float(pos_z.value)

    @rot_x.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.root_rotation_offset[0] = float(rot_x.value)

    @rot_y.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.root_rotation_offset[1] = float(rot_y.value)

    @rot_z.on_update
    def _(_) -> None:
        with edit_lock:
            edit_state.root_rotation_offset[2] = float(rot_z.value)

    @offset_reset_button.on_click
    def _(_) -> None:
        with edit_lock:
            edit_state.root_position_offset = np.zeros(3)
            edit_state.root_rotation_offset = np.zeros(3)
        pos_x.value = 0.0
        pos_y.value = 0.0
        pos_z.value = 0.0
        rot_x.value = 0.0
        rot_y.value = 0.0
        rot_z.value = 0.0
        print("[edit] Offsets reset")

    @contact_enabled_checkbox.on_update
    def _(_) -> None:
        with contact_lock:
            contact_state.enabled = bool(contact_enabled_checkbox.value)
            enabled = contact_state.enabled
            contact_state.dirty = True
        print(f"[contact] View enabled -> {enabled}")

    @contact_view_select.on_update
    def _(_) -> None:
        with contact_lock:
            contact_state.view = str(contact_view_select.value)
            view = contact_state.view
            contact_state.dirty = True
        print(f"[contact] Projection -> {view}")

    @contact_zoom_slider.on_update
    def _(_) -> None:
        with contact_lock:
            contact_state.view_height = float(contact_zoom_slider.value)
            contact_state.dirty = True

    @apply_button.on_click
    def _(_) -> None:
        with edit_lock:
            edit_state.needs_recompile = True
        print("[edit] Apply edits requested")

    @save_button.on_click
    def _(_) -> None:
        try:
            # Compile edits and save
            with edit_lock:
                edited_motion = compile_edits(
                    motion_holder[0],
                    edit_state,
                    ctx.sim.mj_model,
                    transform=joint_layout_transform(ctx.sim.mj_model),
                )

            save_path = Path(save_path_input.value)
            save_motion_file(edited_motion, save_path)

            edit_info.content = f"""<div style="font-size:0.85em; padding:0.5em; background:#2d5; border-radius:4px;">
            ✓ Saved to: {save_path.name}
            </div>"""
        except Exception as e:
            print(f"[edit] Save failed: {e}")
            edit_info.content = f"""<div style="font-size:0.85em; padding:0.5em; background:#d25; border-radius:4px;">
            ✗ Save failed: {e}
            </div>"""

    @server.on_client_connect
    def _(_) -> None:
        # Make sure late-joining clients see the current pose and UI state.
        mj_scene.refresh_visualization()
        with contact_lock:
            contact_state.dirty = True
        with state_lock:
            current_frame = state.frame
        set_slider_value(current_frame)
        update_transport_buttons()
        update_status(force=True)

    viewer_ctx = ViewerContext(
        server=server,
        scene=mj_scene,
        ui=ui,
        update_status=update_status,
        update_transport_buttons=update_transport_buttons,
        set_slider_value=set_slider_value,
        contact_image=contact_image,
        contact_readout=contact_readout,
    )

    return viewer_ctx, edit_ui


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


DEFAULT_TASK = "Mjlab-Velocity-Flat-Amp-Booster-K1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="assets/amp",
        help=(
            "Directory containing motion files, a single .pkl/.csv motion file, "
            "or a Hugging Face dataset ID in 'namespace/repo' form"
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Git revision (branch, tag, or commit) of the Hugging Face dataset",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Mjlab registry task to use (loads its play_env_cfg)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Simulation device or 'auto' to prefer CUDA when available",
    )
    parser.add_argument(
        "--initial",
        type=str,
        default=None,
        help="Name or stem of the motion clip to start with (default: first alphabetically)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to start playback from",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback multiplier (>1 = faster, <1 = slower)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Get simulation dt from environment
    device = resolve_device(args.device)
    env_ctx = create_play_env(args.task, device)
    base_dt = env_ctx.env_cfg.sim.mujoco.timestep
    decimation = env_ctx.env_cfg.decimation
    simulation_dt = base_dt * decimation

    # Discover clips in the given directory or Hugging Face dataset
    motion_entries = scan_motion_entries(args.dataset, args.revision)
    print(f"Found {len(motion_entries)} motion clip(s) in {args.dataset}")

    # Pick the initial clip
    initial_entry = resolve_initial_entry(motion_entries, args.initial)

    # Load original motion file (mutable holder – swapped on dropdown change)
    motion_holder: list[MotionFile] = [initial_entry.load()]

    # Serial clips are retargeted when the task uses the parallel-ankle model.
    layout_transform = joint_layout_transform(env_ctx.sim.mj_model)

    # Load using modern utilities with resampling
    clip = load_motion_clip(
        motion_holder[0], simulation_dt, device, transform=layout_transform
    )
    print(
        f"Loaded '{initial_entry.stem}' — {clip.num_frames} frames "
        f"at {clip.source_fps:.1f} source FPS"
    )
    clip_holder: list[MotionClip] = [clip]

    start = int(np.clip(args.start_frame, 0, clip.num_frames - 1))
    update_mujoco_state(env_ctx.sim, clip, start)

    playback_speed = max(args.playback_speed, 1e-3)
    state = PlaybackState(
        frame=start,
        paused=False,
        speed=playback_speed,
        loop=True,
        total_frames=clip.num_frames,
    )
    state_lock = threading.Lock()

    # Create edit state
    edit_state = EditState()
    edit_lock = threading.Lock()

    # Foot contact view state
    foot_geometry = FootGeometry.create(env_ctx.sim.mj_model)
    ortho_view = (
        OrthographicFootView(env_ctx.sim.mj_model, foot_geometry)
        if foot_geometry is not None
        else None
    )
    contact_state = ContactViewState()
    contact_lock = threading.Lock()

    print("Creating viewer")
    viewer, edit_ui = setup_viewer(
        env_ctx,
        state,
        state_lock,
        edit_state,
        edit_lock,
        contact_state,
        contact_lock,
        motion_holder,
        clip_holder,
        motion_entries,
        initial_entry,
    )

    viewer.set_slider_value(state.frame)
    viewer.update_transport_buttons()

    try:
        viewer.scene.update_from_mjdata(env_ctx.sim.mj_data)
    except Exception:
        import traceback

        traceback.print_exc()
        raise

    viewer.update_status()
    stop_event = threading.Event()

    last_contact_update = [0.0]

    def update_contact_view(force: bool = False) -> None:
        """Refresh the orthographic foot view from the current MuJoCo state."""
        now = time.monotonic()
        if not force and now - last_contact_update[0] < CONTACT_VIEW_MIN_INTERVAL:
            return
        last_contact_update[0] = now

        if foot_geometry is None or ortho_view is None:
            viewer.contact_image.visible = False
            viewer.contact_readout.content = contact_readout_html(
                None, "Foot bodies not found in this model."
            )
            return

        with contact_lock:
            enabled = contact_state.enabled
            view = contact_state.view
            view_height = contact_state.view_height
            contact_state.dirty = False

        foot_state = foot_geometry.evaluate(env_ctx.sim.mj_data)
        image = (
            ortho_view.render(env_ctx.sim.mj_data, foot_state, view, view_height)
            if enabled
            else None
        )
        viewer.contact_image.visible = image is not None
        if image is not None:
            viewer.contact_image.image = image

        if not enabled:
            message = "View disabled."
        elif image is None:
            message = "Offscreen rendering unavailable — clearances only."
        else:
            message = ""
        viewer.contact_readout.content = contact_readout_html(foot_state, message)

    def render_frame(frame_idx: int, force_contact: bool = False) -> None:
        update_mujoco_state(env_ctx.sim, clip, frame_idx)
        viewer.scene.update_from_mjdata(env_ctx.sim.mj_data)
        viewer.set_slider_value(frame_idx)
        update_contact_view(force=force_contact)

    def playback_loop() -> None:
        nonlocal clip
        frame = state.frame

        # Reset the deadline after pauses, seeks, or clip changes.
        next_frame_time: float | None = None

        # The offscreen renderer binds to whichever thread first uses it, so
        # every contact render has to happen here, never on the main thread.
        update_contact_view(force=True)

        while not stop_event.is_set():
            # --- Motion switch (dropdown selection changed) ---
            with state_lock:
                pending = state.pending_motion

            if pending is not None:
                with state_lock:
                    state.pending_motion = None
                print(f"[motion] Loading: {pending.stem}")
                try:
                    new_motion_file = pending.load()
                    motion_holder[0] = new_motion_file

                    # Reset edit state
                    with edit_lock:
                        edit_state.trim_start = 0
                        edit_state.trim_end = None
                        edit_state.speed_factor = 1.0
                        edit_state.mirrored = False
                        edit_state.root_position_offset = np.zeros(3)
                        edit_state.root_rotation_offset = np.zeros(3)
                        edit_state.fix_ground = False
                        edit_state.needs_recompile = False

                    # Load new clip
                    clip = load_motion_clip(
                        new_motion_file,
                        simulation_dt,
                        device,
                        transform=layout_transform,
                    )
                    clip_holder[0] = clip

                    frame = 0
                    with state_lock:
                        state.frame = 0
                        state.total_frames = clip.num_frames

                    # Update playback UI
                    viewer.ui.frame_slider.max = max(1, clip.num_frames - 1)
                    viewer.ui.frame_slider.disabled = clip.num_frames <= 1
                    viewer.set_slider_value(0)

                    # Update edit UI
                    edit_ui.trim_start_input.max = new_motion_file.num_frames - 1
                    edit_ui.trim_start_input.value = 0
                    edit_ui.trim_end_input.max = new_motion_file.num_frames
                    edit_ui.trim_end_input.value = new_motion_file.num_frames
                    edit_ui.speed_factor_input.value = 1.0
                    edit_ui.mirror_checkbox.value = False
                    edit_ui.fix_ground_checkbox.value = False
                    edit_ui.pos_x.value = 0.0
                    edit_ui.pos_y.value = 0.0
                    edit_ui.pos_z.value = 0.0
                    edit_ui.rot_x.value = 0.0
                    edit_ui.rot_y.value = 0.0
                    edit_ui.rot_z.value = 0.0
                    edit_ui.save_path_input.value = str(pending.default_save_path)
                    edit_ui.speed_info.content = (
                        f'<div style="font-size:0.8em; padding:0 0.5em;">'
                        f"Current: 1.00x<br/>"
                        f"Original frames: {new_motion_file.num_frames}"
                        f"</div>"
                    )

                    print(
                        f"[motion] Loaded '{pending.stem}' — {clip.num_frames} frames"
                    )
                except Exception as e:
                    print(f"[motion] Failed to load {pending.name}: {e}")
                    import traceback

                    traceback.print_exc()
                next_frame_time = None
                viewer.update_status(force=True)
                continue

            # --- Edit recompile ---
            with edit_lock:
                needs_recompile = edit_state.needs_recompile

            if needs_recompile:
                print("[edit] Recompiling motion with edits...")
                try:
                    with edit_lock:
                        # Compile edits
                        edited_motion = compile_edits(
                            motion_holder[0],
                            edit_state,
                            env_ctx.sim.mj_model,
                            transform=layout_transform,
                        )
                        edit_state.needs_recompile = False

                    clip = load_motion_clip(
                        edited_motion,
                        simulation_dt,
                        device,
                        transform=layout_transform,
                    )
                    clip_holder[0] = clip

                    # Update UI
                    viewer.ui.frame_slider.max = max(1, clip.num_frames - 1)
                    viewer.ui.frame_slider.disabled = clip.num_frames <= 1

                    # Clamp current frame to new range
                    frame = min(frame, clip.num_frames - 1)
                    with state_lock:
                        state.frame = frame
                        state.total_frames = clip.num_frames

                    # Update speed info
                    edit_ui.speed_info.content = f"""<div style="font-size:0.8em; padding:0 0.5em;">
                    Current: {edit_state.speed_factor:.2f}x<br/>
                    New frames: {clip.num_frames}<br/>
                    Original frames: {motion_holder[0].num_frames}
                    </div>"""

                    print(f"[edit] Recompiled! New clip has {clip.num_frames} frames")
                    next_frame_time = None
                    viewer.update_status(force=True)
                except Exception as e:
                    print(f"[edit] Recompilation failed: {e}")
                    import traceback

                    traceback.print_exc()
                    with edit_lock:
                        edit_state.needs_recompile = False

            with state_lock:
                seek_to = state.seek_to
                paused = state.paused
                speed = state.speed
                loop_enabled = state.loop

            if seek_to is not None:
                frame = int(np.clip(seek_to, 0, clip.num_frames - 1))
                with state_lock:
                    state.frame = frame
                    state.seek_to = None
                render_frame(frame, force_contact=True)
                next_frame_time = None
                viewer.update_status(force=True)
                continue

            if paused:
                next_frame_time = None
                with state_lock:
                    pending_step = state.pending_step
                    if pending_step:
                        state.pending_step = False
                if pending_step:
                    # Run one policy-level step (decimation sub-steps) without
                    # overwriting qpos/qvel from the motion clip.  This lets you
                    # inspect contacts, constraint forces, etc. from the current
                    # pose.  Pressing Play afterwards resumes normal kinematic
                    # playback, which re-sets qpos on the next frame.
                    for _ in range(decimation):
                        mujoco.mj_step(env_ctx.sim.mj_model, env_ctx.sim.mj_data)
                    viewer.scene.update_from_mjdata(env_ctx.sim.mj_data)
                    update_contact_view(force=True)
                    print("[sim] Physics step done")
                else:
                    with contact_lock:
                        contact_dirty = contact_state.dirty
                    if contact_dirty:
                        update_contact_view(force=True)
                    time.sleep(0.05)
                continue

            if next_frame_time is None:
                next_frame_time = time.monotonic()

            render_frame(frame)
            viewer.update_status()

            step_dt = clip.dt / max(speed, 1e-3)
            next_frame_time, advance, delay = schedule_next_frame(
                next_frame_time, time.monotonic(), step_dt
            )
            if stop_event.wait(delay):
                break

            frame += advance
            if frame >= clip.num_frames:
                if loop_enabled:
                    frame %= clip.num_frames
                    with state_lock:
                        state.frame = frame
                else:
                    frame = clip.num_frames - 1
                    with state_lock:
                        state.frame = frame
                        state.paused = True
                    render_frame(frame, force_contact=True)
                    viewer.update_transport_buttons()
                    viewer.update_status(force=True)
                    next_frame_time = None
                    continue
            else:
                with state_lock:
                    state.frame = frame

    playback_thread = threading.Thread(target=playback_loop, daemon=True)
    playback_thread.start()

    try:
        viewer.server.sleep_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        playback_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
