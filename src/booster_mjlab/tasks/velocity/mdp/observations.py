from __future__ import annotations

from functools import partial
from typing import NamedTuple

from rsl_rl.env import VecEnv
from tensordict import TensorDict

import torch


class JointMirror(NamedTuple):
    """Left/right mirror of a joint vector: ``joints[:, perm]`` with ``inverted`` negated."""

    perm: tuple[int, ...]
    """For each joint slot, the slot of its mirror counterpart."""

    inverted: tuple[int, ...]
    """Joint slots whose axes change sign under a left/right mirror (roll and yaw)."""


def _swap_blocks(num_joints: int, *pairs: tuple[int, int, int]) -> tuple[int, ...]:
    """Permutation swapping each ``(start_a, start_b, length)`` block pair."""
    perm = list(range(num_joints))
    for a, b, n in pairs:
        perm[a : a + n] = range(b, b + n)
        perm[b : b + n] = range(a, a + n)
    return tuple(perm)


K1_JOINT_DIM = 22
K1_ACTION_DIM = 22

# Joint slots whose axes change sign under a left/right mirror.
K1_INVERTED_JOINT_INDICES: tuple[int, ...] = (
    0,
    3,
    5,
    7,
    9,
    11,
    12,
    15,
    17,
    18,
    21,
)

# The parallel ankle's A/B crank axes both point along +y. Unlike serial ankle
# roll, neither crank changes sign under a left/right mirror.
K1_PARALLEL_INVERTED_JOINT_INDICES: tuple[int, ...] = tuple(
    index for index in K1_INVERTED_JOINT_INDICES if index not in (15, 21)
)

# Head (0-1) stays, arms (2-5 / 6-9) and legs (10-15 / 16-21) swap sides.
K1_MIRROR_PERM = _swap_blocks(K1_JOINT_DIM, (2, 6, 4), (10, 16, 6))
K1_MIRROR = JointMirror(K1_MIRROR_PERM, K1_INVERTED_JOINT_INDICES)
K1_PARALLEL_MIRROR = JointMirror(K1_MIRROR_PERM, K1_PARALLEL_INVERTED_JOINT_INDICES)

# The T1 adds a yaw-only waist between the arms and legs (slot 10):
# 'AAHead_yaw' (Inverted), 'Head_pitch',
# 'Left_Shoulder_Pitch', 'Left_Shoulder_Roll' (Inverted), 'Left_Elbow_Pitch', 'Left_Elbow_Yaw' (Inverted),
# 'Right_Shoulder_Pitch', 'Right_Shoulder_Roll' (Inverted), 'Right_Elbow_Pitch', 'Right_Elbow_Yaw' (Inverted),
# 'Waist' (Inverted),
# 'Left_Hip_Pitch', 'Left_Hip_Roll' (Inverted), 'Left_Hip_Yaw' (Inverted), 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll' (Inverted),
# 'Right_Hip_Pitch', 'Right_Hip_Roll' (Inverted), 'Right_Hip_Yaw' (Inverted), 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll' (Inverted)
T1_JOINT_DIM = 23
T1_MIRROR = JointMirror(
    perm=_swap_blocks(T1_JOINT_DIM, (2, 6, 4), (11, 17, 6)),
    inverted=(0, 3, 5, 7, 9, 10, 12, 13, 16, 18, 19, 22),
)

# The parallel T1's ankle slots hold the A/B cranks, whose axes both point along +y
# and map onto the same crank of the other leg, so neither changes sign.
T1_PARALLEL_MIRROR = JointMirror(
    perm=T1_MIRROR.perm,
    inverted=tuple(i for i in T1_MIRROR.inverted if i not in (16, 22)),
)

CRITIC_EXTRA_DIM = 15


def _policy_dim(mirror: JointMirror) -> int:
    """Actor observation width for a robot with ``len(mirror.perm)`` joints.

    Layout is:
    [base_ang_vel(3), projected_gravity(3), joint_pos(J), joint_vel(J),
     last_action(J), command(3)]
    """
    return sum(_policy_term_dims(mirror))


def _policy_term_dims(mirror: JointMirror) -> tuple[int, ...]:
    """Widths of the actor observation terms, in order (see ``_policy_dim``)."""
    num_joints = len(mirror.perm)
    return (3, 3, num_joints, num_joints, num_joints, 3)


POLICY_DIM_NO_BASE_LIN_VEL = _policy_dim(K1_MIRROR)


def _augment_symmetries(
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    mirror: JointMirror,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply symmetry augmentations to observations and actions."""

    if obs is not None:
        actor_obs = obs["actor"]
        critic_obs = obs["critic"]
        actor_dim = actor_obs.shape[1]
        critic_dim = critic_obs.shape[1]
        policy_dim = _policy_dim(mirror)
        if actor_dim % policy_dim != 0:
            raise ValueError(
                f"Unsupported actor observation dim: {actor_dim}. "
                f"Expected a multiple of {policy_dim} (one per history frame)."
            )
        # The critic never stacks history.
        expected_critic_dim = policy_dim + CRITIC_EXTRA_DIM
        if critic_dim != expected_critic_dim:
            raise ValueError(
                f"Critic observation dim mismatch: got {critic_dim}, "
                f"expected {expected_critic_dim}."
            )

        n_envs = actor_obs.shape[0]

        # augment the batch size to 2 different symmetries
        actor_aug = torch.zeros(n_envs * 2, actor_obs.shape[1], device=actor_obs.device)
        actor_aug[:n_envs] = actor_obs
        actor_aug[n_envs : 2 * n_envs] = flip_actor_obs_left_right(actor_obs, mirror)

        critic_aug = torch.zeros(
            n_envs * 2, critic_obs.shape[1], device=critic_obs.device
        )
        critic_aug[:n_envs] = critic_obs
        critic_aug[n_envs : 2 * n_envs] = flip_critic_obs_left_right(critic_obs, mirror)

        obs = TensorDict(
            {"actor": actor_aug, "critic": critic_aug}, batch_size=(n_envs * 2,)
        )

    if actions is not None:
        if actions.shape[1] != len(mirror.perm):
            raise ValueError(
                f"Unsupported action dim: {actions.shape[1]}. "
                f"Expected {len(mirror.perm)}."
            )

        n_envs = actions.shape[0]

        # augment the batch size to 2 different symmetries
        actions_aug = torch.zeros(n_envs * 2, actions.shape[1], device=actions.device)
        actions_aug[:n_envs] = actions[:]
        actions_aug[n_envs : 2 * n_envs] = flip_action_left_right(actions, mirror)

        actions = actions_aug

    return obs, actions


def augment_symmetries(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the serial-ankle K1."""
    del env
    return _augment_symmetries(obs, actions, K1_MIRROR)


def augment_symmetries_parallel(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the parallel-ankle K1."""
    del env
    return _augment_symmetries(obs, actions, K1_PARALLEL_MIRROR)


def augment_symmetries_t1(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the T1."""
    del env
    return _augment_symmetries(obs, actions, T1_MIRROR)


def augment_symmetries_t1_parallel(
    env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Apply left/right symmetry for the parallel-ankle T1."""
    del env
    return _augment_symmetries(obs, actions, T1_PARALLEL_MIRROR)


def flip_action_left_right(
    action: torch.Tensor, mirror: JointMirror = K1_MIRROR
) -> torch.Tensor:
    action = action.clone()
    # switch left and right joints
    action = _switch_joints_left_right(action, mirror)
    return action


def flip_policy_obs_left_right(
    obs: torch.Tensor, mirror: JointMirror = K1_MIRROR
) -> torch.Tensor:
    obs = obs.clone()
    policy_dim = obs.shape[1]
    num_linear_obs_blocks = _get_num_linear_obs_blocks(policy_dim, mirror)
    if num_linear_obs_blocks is None:
        raise ValueError(
            f"Unsupported policy observation dim: {policy_dim}. "
            f"Expected {_policy_dim(mirror)}."
        )
    num_joints = len(mirror.perm)

    # No leading base linear terms in the current velocity policy layout.
    # Each 3D vector mirrors with [x, y, z] -> [x, -y, z].
    for i in range(num_linear_obs_blocks):
        start = i * 3
        obs[:, start : start + 3] = obs[:, start : start + 3] * obs.new_tensor(
            [1.0, -1.0, 1.0]
        )

    base_offset = num_linear_obs_blocks * 3

    # base ang vel
    obs[:, base_offset : base_offset + 3] = obs[
        :, base_offset : base_offset + 3
    ] * obs.new_tensor([-1.0, 1.0, -1.0])
    # projected gravity
    obs[:, base_offset + 3 : base_offset + 6] = obs[
        :, base_offset + 3 : base_offset + 6
    ] * obs.new_tensor([1.0, -1.0, 1.0])

    joint_pos_start = base_offset + 6
    joint_vel_start = joint_pos_start + num_joints
    last_actions_start = joint_vel_start + num_joints
    command_start = last_actions_start + num_joints

    # joint pos
    obs[:, joint_pos_start:joint_vel_start] = _switch_joints_left_right(
        obs[:, joint_pos_start:joint_vel_start], mirror
    )
    # joint vel
    obs[:, joint_vel_start:last_actions_start] = _switch_joints_left_right(
        obs[:, joint_vel_start:last_actions_start], mirror
    )
    # last actions
    obs[:, last_actions_start:command_start] = _switch_joints_left_right(
        obs[:, last_actions_start:command_start], mirror
    )
    # velocity command
    obs[:, command_start : command_start + 3] = obs[
        :, command_start : command_start + 3
    ] * obs.new_tensor([1.0, -1.0, -1.0])

    return obs


def flip_actor_obs_left_right(
    obs: torch.Tensor, mirror: JointMirror = K1_MIRROR
) -> torch.Tensor:
    """Mirror an actor observation that may stack several history frames.

    mjlab flattens history per term, so a stacked actor observation is term-major:
    ``[base_ang_vel(H*3), projected_gravity(H*3), joint_pos(H*J), ...]``, each term
    oldest frame first. Regroup it into frames, mirror each frame, and restore the
    term-major layout. With ``H == 1`` this is ``flip_policy_obs_left_right``.
    """
    term_dims = _policy_term_dims(mirror)
    policy_dim = sum(term_dims)
    n_envs, actor_dim = obs.shape
    history_len = actor_dim // policy_dim
    if history_len == 1:
        return flip_policy_obs_left_right(obs, mirror)

    blocks = torch.split(obs, [history_len * dim for dim in term_dims], dim=1)
    frames = torch.cat(
        [
            block.reshape(n_envs, history_len, dim)
            for block, dim in zip(blocks, term_dims)
        ],
        dim=-1,
    )
    frames = flip_policy_obs_left_right(frames.reshape(-1, policy_dim), mirror)
    frames = frames.reshape(n_envs, history_len, policy_dim)
    return torch.cat(
        [
            block.reshape(n_envs, -1)
            for block in torch.split(frames, list(term_dims), dim=-1)
        ],
        dim=1,
    )


def flip_critic_obs_left_right(
    obs: torch.Tensor, mirror: JointMirror = K1_MIRROR
) -> torch.Tensor:
    obs = obs.clone()
    policy_dim = obs.shape[1] - CRITIC_EXTRA_DIM
    if policy_dim != _policy_dim(mirror):
        raise ValueError(
            f"Unsupported critic observation dim: {obs.shape[1]}. "
            f"Expected policy part of {_policy_dim(mirror)} "
            f"plus {CRITIC_EXTRA_DIM} extra dims."
        )

    # policy obs
    obs[:, :policy_dim] = flip_policy_obs_left_right(obs[:, :policy_dim], mirror)

    # base lin vel: [x, y, z] -> [x, -y, z]
    base_lin_vel_start = policy_dim
    obs[:, base_lin_vel_start : base_lin_vel_start + 3] = obs[
        :, base_lin_vel_start : base_lin_vel_start + 3
    ] * obs.new_tensor([1.0, -1.0, 1.0])

    foot_height_start = policy_dim + 3
    foot_air_time_start = foot_height_start + 2
    foot_contact_start = foot_air_time_start + 2
    foot_contact_forces_start = foot_contact_start + 2

    # foot height
    obs[:, [foot_height_start, foot_height_start + 1]] = obs[
        :, [foot_height_start + 1, foot_height_start]
    ]
    # foot air time
    obs[:, [foot_air_time_start, foot_air_time_start + 1]] = obs[
        :, [foot_air_time_start + 1, foot_air_time_start]
    ]
    # foot contact
    obs[:, [foot_contact_start, foot_contact_start + 1]] = obs[
        :, [foot_contact_start + 1, foot_contact_start]
    ]
    # foot contact forces
    obs[
        :,
        [
            foot_contact_forces_start,
            foot_contact_forces_start + 1,
            foot_contact_forces_start + 2,
            foot_contact_forces_start + 3,
            foot_contact_forces_start + 4,
            foot_contact_forces_start + 5,
        ],
    ] = obs[
        :,
        [
            foot_contact_forces_start + 3,
            foot_contact_forces_start + 4,
            foot_contact_forces_start + 5,
            foot_contact_forces_start,
            foot_contact_forces_start + 1,
            foot_contact_forces_start + 2,
        ],
    ] * obs.new_tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])

    return obs


SUPPORTED_POLICY_DIMS = (POLICY_DIM_NO_BASE_LIN_VEL, _policy_dim(T1_MIRROR))


def _get_num_linear_obs_blocks(
    policy_dim: int, mirror: JointMirror = K1_MIRROR
) -> int | None:
    """Infer number of leading 3D base linear vectors from policy dim.

    The current velocity policy layout (see ``_policy_dim``) has none.
    """
    if policy_dim != _policy_dim(mirror):
        return None
    return 0


def _switch_joints_left_right(
    joints: torch.Tensor, mirror: JointMirror = K1_MIRROR
) -> torch.Tensor:
    """Switch left and right joints in a joint tensor of shape (..., J).

    K1 joint order:

    'Head_Yaw' (Inverted), 'Head_Pitch',
    'Left_Shoulder_Pitch', 'Left_Shoulder_Roll' (Inverted), 'Left_Elbow_Pitch', 'Left_Elbow_Yaw' (Inverted),
    'Right_Shoulder_Pitch', 'Right_Shoulder_Roll' (Inverted), 'Right_Elbow_Pitch', 'Right_Elbow_Yaw' (Inverted),
    'Left_Hip_Pitch', 'Left_Hip_Roll' (Inverted), 'Left_Hip_Yaw' (Inverted), 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll' (Inverted),
    'Right_Hip_Pitch', 'Right_Hip_Roll' (Inverted), 'Right_Hip_Yaw' (Inverted), 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll' (Inverted)

    Joints marked as "Inverted" need to have their sign flipped when calculating the left-right symmetry.
    See ``T1_MIRROR`` for the T1 layout.
    """
    if joints.shape[1] != len(mirror.perm):
        raise ValueError(
            f"Unsupported joint tensor dim: {joints.shape[1]}. "
            f"Expected {len(mirror.perm)}."
        )
    joints_flipped = joints[:, list(mirror.perm)]

    joints_flipped[:, list(mirror.inverted)] = -joints_flipped[
        :, list(mirror.inverted)
    ]

    return joints_flipped


flip_k1_action_left_right = flip_action_left_right
flip_k1_policy_obs_left_right = flip_policy_obs_left_right
flip_k1_critic_obs_left_right = flip_critic_obs_left_right

flip_k1_parallel_action_left_right = partial(
    flip_action_left_right, mirror=K1_PARALLEL_MIRROR
)
flip_k1_parallel_policy_obs_left_right = partial(
    flip_policy_obs_left_right, mirror=K1_PARALLEL_MIRROR
)
flip_k1_parallel_critic_obs_left_right = partial(
    flip_critic_obs_left_right, mirror=K1_PARALLEL_MIRROR
)
