"""Load motion clips from a Hugging Face dataset produced by `motions-to-hf`.

Each Hub repo holds one motion dataset: parquet shards under ``data/`` with one
row per clip (see ``booster_mjlab.scripts.motions_to_hf``). Rows are parsed back
into :class:`MotionFile` objects. :class:`MotionLoader` accepts the Hub repo ID
directly, while ``materialize()`` can optionally write clips back to ``.pkl``.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .motion_data import MotionFile


@dataclass(frozen=True)
class HfMotionClip:
    name: str
    motion: MotionFile


class HfMotionDataset:
    """Motion clips backed by a Hub dataset repo (or a local parquet export).

    Args:
        repo_id: Hub dataset repo, e.g. ``whirlwind-team/booster-k1-amp-kicks``.
            Shards are fetched through the shared huggingface_hub cache, so
            repeated loads don't re-download. Private repos use the cached
            ``hf auth login`` token or ``HF_TOKEN``.
        local_dir: Load from a local export directory (a staged dataset under
            the ``output_dir`` of ``motions-to-hf``) instead of the Hub.
            Mutually exclusive with ``repo_id``.
        revision: Optional git revision (branch, tag, or commit) of the repo.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        *,
        local_dir: PathLike[str] | None = None,
        revision: str | None = None,
    ) -> None:
        if (repo_id is None) == (local_dir is None):
            raise ValueError("Provide exactly one of `repo_id` or `local_dir`.")

        self.repo_id = repo_id
        if repo_id is not None:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=revision,
                    allow_patterns=["data/*.parquet"],
                )
            )
        else:
            assert local_dir is not None
            root = Path(local_dir)

        shards = sorted((root / "data").glob("*.parquet"))
        if not shards:
            shards = sorted(root.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No parquet shards found under {root}")

        self.clips: list[HfMotionClip] = []
        for shard in shards:
            for row in pq.read_table(shard).to_pylist():
                self.clips.append(
                    HfMotionClip(name=row["name"], motion=_row_to_motion_file(row))
                )

    @property
    def clip_names(self) -> list[str]:
        return [clip.name for clip in self.clips]

    def motion_files(self) -> list[MotionFile]:
        return [clip.motion for clip in self.clips]

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> MotionFile:
        return self.clips[index].motion

    def __iter__(self) -> Iterator[MotionFile]:
        return iter(self.motion_files())

    def materialize(self, target_dir: PathLike[str]) -> Path:
        """Write all clips as ``<target_dir>/<name>.pkl``.

        Restores the flat on-disk layout consumed by :class:`MotionLoader`, so
        the returned directory can be used directly as a ``dataset_root``.
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        for clip in self.clips:
            payload = {
                "fps": clip.motion.fps,
                "root_pos": clip.motion.root_pos,
                "root_rot": clip.motion.root_rot,
                "dof_pos": clip.motion.dof_pos,
                "local_body_pos": clip.motion.local_body_pos,
                "link_body_list": clip.motion.link_body_list,
            }
            with (target / f"{clip.name}.pkl").open("wb") as f:
                pickle.dump(payload, f)
        return target

    def __repr__(self) -> str:
        source = self.repo_id if self.repo_id is not None else "local"
        return f"HfMotionDataset({source!r}, num_clips={len(self)})"


def _row_to_motion_file(row: dict) -> MotionFile:
    local_body_pos = (
        np.asarray(row["local_body_pos"], dtype=np.float32)
        if row.get("local_body_pos") is not None
        else None
    )
    return MotionFile(
        fps=row["fps"],
        root_pos=np.asarray(row["root_pos"], dtype=np.float32),
        root_rot=np.asarray(row["root_rot_xyzw"], dtype=np.float32),
        dof_pos=np.asarray(row["dof_pos"], dtype=np.float32),
        local_body_pos=local_body_pos,
        link_body_list=row.get("link_body_list"),
    )
