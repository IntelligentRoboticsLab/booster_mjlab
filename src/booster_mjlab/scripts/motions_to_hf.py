"""Export each AMP motion dataset under ``assets/amp`` to its own Hugging Face dataset.

Each subfolder of the input directory is staged as a standalone dataset repo
(``<namespace>/<repo-prefix><subset>``) containing ``data/motions.parquet``
with one row per motion clip, mirroring the ``MotionFile`` schema (root_pos,
root_rot, dof_pos, ...), plus a generated dataset card. Exports can be browsed
in the Hub dataset viewer and loaded with ``datasets.load_dataset`` or
:class:`booster_mjlab.motion.HfMotionDataset`.

Examples:

    # Convert only (staged under assets/hf/<repo-name>/):
    uv run motions-to-hf

    # Convert a single dataset and push (requires `hf auth login` or HF_TOKEN):
    uv run motions-to-hf --subsets lafan_curated --namespace whirlwind-team --push
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

from booster_mjlab.motion import SUPPORTED_MOTION_FILE_EXTENSIONS, MotionFile
from booster_mjlab.motion.motion_data import BOOSTER_K1_CSV_HEADER

# Canonical K1 joint order (matches the XML joint order used by `dof_pos`).
K1_JOINT_NAMES = [column.removesuffix("_dof") for column in BOOSTER_K1_CSV_HEADER[7:]]

MOTION_SCHEMA = pa.schema(
    [
        pa.field("name", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("fps", pa.float64()),
        pa.field("num_frames", pa.int64()),
        pa.field("duration_s", pa.float64()),
        pa.field("joint_names", pa.list_(pa.string())),
        pa.field("root_pos", pa.list_(pa.list_(pa.float32()))),
        pa.field("root_rot_xyzw", pa.list_(pa.list_(pa.float32()))),
        pa.field("dof_pos", pa.list_(pa.list_(pa.float32()))),
        pa.field("local_body_pos", pa.list_(pa.list_(pa.list_(pa.float32())))),
        pa.field("link_body_list", pa.list_(pa.string())),
    ]
)


@dataclass(frozen=True)
class Config:
    input_dir: Path = Path("assets/amp")
    output_dir: Path = Path("assets/hf")
    namespace: str | None = None
    """Hub org or user to push the dataset repos to, e.g. `whirlwind-team`."""
    repo_prefix: str = "booster-k1-amp-"
    """Repo name prefix; subset `kicks` becomes `<namespace>/<prefix>kicks`."""
    push: bool = False
    """Upload each staged dataset to the Hub (requires --namespace)."""
    private: bool = True
    subsets: tuple[str, ...] | None = None
    """Subset folder names to export; defaults to every subfolder."""
    commit_message: str = "Upload Booster K1 AMP motion dataset"


def _discover_subsets(
    input_dir: Path, requested: tuple[str, ...] | None
) -> dict[str, list[Path]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    subsets: dict[str, list[Path]] = {}
    for subdir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        if requested is not None and subdir.name not in requested:
            continue
        files = sorted(
            path
            for suffix in SUPPORTED_MOTION_FILE_EXTENSIONS
            for path in subdir.glob(f"*{suffix}")
        )
        if files:
            subsets[subdir.name] = files

    if requested is not None:
        missing = set(requested) - set(subsets)
        if missing:
            raise FileNotFoundError(
                f"Requested subsets not found under {input_dir}: {sorted(missing)}"
            )
    if not subsets:
        raise FileNotFoundError(f"No motion files found under {input_dir}")
    return subsets


def _motion_to_row(motion_path: Path, input_dir: Path) -> dict:
    motion = MotionFile.load(motion_path)

    root_pos = np.asarray(motion.root_pos, dtype=np.float32)
    root_rot = np.asarray(motion.root_rot, dtype=np.float32)
    dof_pos = np.asarray(motion.dof_pos, dtype=np.float32)

    num_frames = root_pos.shape[0]
    if root_rot.shape != (num_frames, 4):
        raise ValueError(
            f"{motion_path}: expected root_rot of shape ({num_frames}, 4), "
            f"got {root_rot.shape}"
        )
    if dof_pos.shape != (num_frames, len(K1_JOINT_NAMES)):
        raise ValueError(
            f"{motion_path}: expected dof_pos of shape "
            f"({num_frames}, {len(K1_JOINT_NAMES)}), got {dof_pos.shape}"
        )

    local_body_pos = None
    if motion.local_body_pos is not None:
        local_body_pos = np.asarray(motion.local_body_pos, dtype=np.float32).tolist()
    link_body_list = (
        [str(name) for name in motion.link_body_list]
        if motion.link_body_list is not None
        else None
    )

    fps = float(motion.fps) if motion.fps > 0 else 30.0
    return {
        "name": motion_path.stem,
        "source_file": str(motion_path.relative_to(input_dir)),
        "fps": fps,
        "num_frames": num_frames,
        "duration_s": num_frames / fps,
        "joint_names": K1_JOINT_NAMES,
        "root_pos": root_pos.tolist(),
        "root_rot_xyzw": root_rot.tolist(),
        "dof_pos": dof_pos.tolist(),
        "local_body_pos": local_body_pos,
        "link_body_list": link_body_list,
    }


def _build_readme(subset: str, repo_name: str, rows: list[dict]) -> str:
    clip_rows = "\n".join(
        f"| `{row['name']}` | {row['num_frames']} | {row['duration_s']:.2f} |"
        for row in rows
    )
    joint_names = ", ".join(f"`{name}`" for name in K1_JOINT_NAMES)
    total_s = sum(row["duration_s"] for row in rows)

    return f"""---
pretty_name: Booster K1 AMP Motions — {subset}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
---

# Booster K1 AMP Motions — `{subset}`

Retargeted motion-capture clips for the Booster Robotics K1 humanoid
({len(K1_JOINT_NAMES)} motion DOF), used as reference motions for Adversarial
Motion Prior (AMP) training in
[booster_mjlab](https://github.com/IntelligentRoboticsLab/booster-mjlab).
{len(rows)} clips, {total_s:.1f} s of motion total; one row per clip.

## Clips

| Clip | Frames | Duration (s) |
|---|---|---|
{clip_rows}

## Schema

| Column | Type | Description |
|---|---|---|
| `name` | `string` | Clip name (source file stem). |
| `source_file` | `string` | Original file path relative to the source dataset root. |
| `fps` | `float64` | Frame rate of the clip. |
| `num_frames` | `int64` | Number of frames `T`. |
| `duration_s` | `float64` | Clip duration in seconds (`num_frames / fps`). |
| `joint_names` | `list<string>` | Joint order for the `dof_pos` columns ({len(K1_JOINT_NAMES)} joints). |
| `root_pos` | `list<list<float32>>` (`T x 3`) | Root (trunk) position in world frame, meters. |
| `root_rot_xyzw` | `list<list<float32>>` (`T x 4`) | Root orientation quaternion, scalar-last `(x, y, z, w)`. |
| `dof_pos` | `list<list<float32>>` (`T x {len(K1_JOINT_NAMES)}`) | Joint positions in radians, ordered as `joint_names`. |
| `local_body_pos` | `list<list<list<float32>>>` (`T x B x 3`), nullable | Per-body positions in the trunk-local frame. |
| `link_body_list` | `list<string>`, nullable | Body names for `local_body_pos`. |

Joint order: {joint_names}.

## Usage

With `datasets`:

```python
from datasets import load_dataset

clips = load_dataset("<namespace>/{repo_name}", split="train")
```

With booster_mjlab:

```python
from booster_mjlab.motion import HfMotionDataset, MotionLoader

dataset = HfMotionDataset("<namespace>/{repo_name}")
motion = dataset[0]  # a booster_mjlab MotionFile

# MotionLoader accepts the Hub ID directly:
loader = MotionLoader(
    dataset_root="<namespace>/{repo_name}",
    simulation_dt=0.02,
    speed_factor=1.0,
)
```
"""


def run(cfg: Config) -> list[Path]:
    if cfg.push and cfg.namespace is None:
        raise ValueError("--push requires --namespace (Hub org or user)")

    subsets = _discover_subsets(cfg.input_dir, cfg.subsets)

    staged: list[Path] = []
    for subset, files in subsets.items():
        repo_name = f"{cfg.repo_prefix}{subset}"
        dataset_dir = cfg.output_dir / repo_name
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        (dataset_dir / "data").mkdir(parents=True)

        rows = [_motion_to_row(path, cfg.input_dir) for path in files]
        table = pa.Table.from_pylist(rows, schema=MOTION_SCHEMA)
        pq.write_table(
            table, dataset_dir / "data" / "motions.parquet", compression="zstd"
        )
        (dataset_dir / "README.md").write_text(_build_readme(subset, repo_name, rows))
        staged.append(dataset_dir)
        print(f"[info] {repo_name}: staged {len(rows)} clips at {dataset_dir}")

        if cfg.push:
            from huggingface_hub import HfApi

            repo_id = f"{cfg.namespace}/{repo_name}"
            api = HfApi()
            api.create_repo(
                repo_id, repo_type="dataset", private=cfg.private, exist_ok=True
            )
            api.upload_folder(
                folder_path=str(dataset_dir),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=cfg.commit_message,
            )
            print(f"[info] Uploaded to https://huggingface.co/datasets/{repo_id}")

    return staged


def main() -> None:
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
