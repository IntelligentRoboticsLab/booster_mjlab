"""Left/right mirror symmetry of a robot, by joint and body name."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class JointMirror(NamedTuple):
    """Left/right mirror of a joint vector: ``joints[:, perm]`` with ``inverted`` negated."""

    perm: tuple[int, ...]
    """For each joint slot, the slot of its mirror counterpart."""

    inverted: tuple[int, ...]
    """Joint slots whose axes change sign under a left/right mirror (roll and yaw)."""


@dataclass(frozen=True)
class Mirror:
    """A robot's left/right symmetry.

    Names pair up by swapping ``Left``/``Right`` (or ``left``/``right``); names
    without either are central and map to themselves.
    """

    joint_names: tuple[str, ...]
    """Joint order of the robot's motion files (MJCF order of the serial model)."""

    inverted_joints: frozenset[str]
    """Joints whose axes change sign under a left/right mirror (roll and yaw axes)."""

    name_pairs: tuple[tuple[str, str], ...] = ()
    """Left/right body or joint names that do not follow the naming convention."""

    def counterpart(self, name: str) -> str:
        """Return the left/right counterpart of a joint or body name."""
        for a, b in self.name_pairs:
            if name == a:
                return b
            if name == b:
                return a
        for left, right in (("Left", "Right"), ("left", "right")):
            if left in name:
                return name.replace(left, right)
            if right in name:
                return name.replace(right, left)
        return name

    def joint_mirror(self, joint_names: Sequence[str] | None = None) -> JointMirror:
        """Mirror of a joint vector ordered as ``joint_names`` (default: :attr:`joint_names`)."""
        names = tuple(self.joint_names if joint_names is None else joint_names)
        index = {name: i for i, name in enumerate(names)}
        return JointMirror(
            perm=tuple(index[self.counterpart(name)] for name in names),
            inverted=tuple(
                i for i, name in enumerate(names) if name in self.inverted_joints
            ),
        )

    def body_index_map(self, body_names: Sequence[str]) -> np.ndarray:
        """Permutation over ``body_names`` that swaps each left body with its right one.

        Bodies whose counterpart is not in the list map to themselves.
        """
        names = [str(n) for n in body_names]
        index = {name: i for i, name in enumerate(names)}
        return np.array(
            [index.get(self.counterpart(name), i) for i, name in enumerate(names)]
        )
