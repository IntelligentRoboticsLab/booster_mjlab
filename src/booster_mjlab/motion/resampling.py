"""Frame-rate resampling and finite-difference velocity primitives.

All functions are standalone (torch + scipy only) and operate on raw motion
arrays:

  - ``resample_linear``         linear interpolation of an [B, N] signal onto a
                                new time grid.
  - ``resample_rotations``      SO(3) slerp of a quaternion sequence onto a new
                                time grid.
  - ``finite_diff_linear_velocity``   forward difference / dt, last frame held.
  - ``finite_diff_angular_velocity``  angular velocity from a rotation sequence.
"""

from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def resample_linear(
    data: torch.Tensor,
    t_src: torch.Tensor,
    t_dst: torch.Tensor,
) -> torch.Tensor:
    """Linearly resample ``data`` [B, N] from time grid ``t_src`` [B] to ``t_dst`` [R].

    Returns a tensor of shape [R, N] on ``data``'s device.
    """
    f = interp1d(t_src.cpu().numpy(), data.cpu().numpy(), axis=0, kind="linear")
    return torch.as_tensor(f(t_dst.cpu().numpy()), device=data.device)


def resample_rotations(
    quats_xyzw: List[npt.NDArray] | npt.NDArray,
    t_src: torch.Tensor,
    t_dst: torch.Tensor,
) -> Rotation:
    """Slerp a sequence of (xyzw) quaternions from ``t_src`` [B] to ``t_dst`` [R].

    Returns the resampled rotations as a scipy ``Rotation``.
    """
    rotations = Rotation.from_quat(quats_xyzw)
    slerp = Slerp(t_src.cpu().numpy(), rotations)
    return slerp(t_dst.cpu().numpy())


def finite_diff_linear_velocity(data: torch.Tensor, dt: float) -> torch.Tensor:
    """Forward finite-difference velocity of ``data`` [N, D] at step ``dt``.

    The last frame's velocity is held equal to the previous one so the output
    keeps the same length [N, D].
    """
    d = (data[1:] - data[:-1]) / dt
    return torch.vstack([d, d[-1:]])


def finite_diff_angular_velocity(
    rotations: List[Rotation] | Rotation,
    dt: float,
    local: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Angular velocity [N, 3] (rad/s) from a rotation sequence at step ``dt``.

    With ``local=False`` the velocity is expressed in the world frame
    (``R_next @ R_prev^-1``); with ``local=True`` in the body frame
    (``R_prev^-1 @ R_next``). The last frame's velocity is held equal to the
    previous one to preserve length.
    """
    r_prev = rotations[:-1]
    r_next = rotations[1:]
    rel = r_prev.inv() * r_next if local else r_next * r_prev.inv()
    rotation_vector = rel.as_rotvec() / dt
    return torch.as_tensor(
        np.vstack([rotation_vector, rotation_vector[-1:]]),
        dtype=torch.float32,
        device=device,
    )
