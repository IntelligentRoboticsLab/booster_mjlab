from __future__ import annotations

from collections.abc import Callable

from rsl_rl.env import VecEnv
from tensordict import TensorDict

import torch

from booster_mjlab.robots.mirror import JointMirror

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


AugmentSymmetries = Callable[
    [VecEnv, TensorDict | None, torch.Tensor | None],
    tuple[TensorDict | None, torch.Tensor | None],
]


def make_augment_symmetries(mirror: JointMirror) -> AugmentSymmetries:
    """rsl_rl ``data_augmentation_func`` applying left/right symmetry with ``mirror``.

    ``mirror`` must be ordered like the policy's joint observations and actions.
    """

    def augment_symmetries(
        env: VecEnv, obs: TensorDict | None, actions: torch.Tensor | None
    ) -> tuple[TensorDict | None, torch.Tensor | None]:
        del env
        return _augment_symmetries(obs, actions, mirror)

    return augment_symmetries


def flip_action_left_right(action: torch.Tensor, mirror: JointMirror) -> torch.Tensor:
    action = action.clone()
    # switch left and right joints
    action = _switch_joints_left_right(action, mirror)
    return action


def flip_policy_obs_left_right(obs: torch.Tensor, mirror: JointMirror) -> torch.Tensor:
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


def flip_actor_obs_left_right(obs: torch.Tensor, mirror: JointMirror) -> torch.Tensor:
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


def flip_critic_obs_left_right(obs: torch.Tensor, mirror: JointMirror) -> torch.Tensor:
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


def _get_num_linear_obs_blocks(policy_dim: int, mirror: JointMirror) -> int | None:
    """Infer number of leading 3D base linear vectors from policy dim.

    The current velocity policy layout (see ``_policy_dim``) has none.
    """
    if policy_dim != _policy_dim(mirror):
        return None
    return 0


def _switch_joints_left_right(
    joints: torch.Tensor, mirror: JointMirror
) -> torch.Tensor:
    """Switch left and right joints in a joint tensor of shape (..., J).

    Each slot takes its mirror counterpart's value, and slots whose axes change sign
    under a left/right mirror (roll and yaw) are negated; see ``JointMirror``.
    """
    if joints.shape[1] != len(mirror.perm):
        raise ValueError(
            f"Unsupported joint tensor dim: {joints.shape[1]}. "
            f"Expected {len(mirror.perm)}."
        )
    joints_flipped = joints[:, list(mirror.perm)]

    joints_flipped[:, list(mirror.inverted)] = -joints_flipped[:, list(mirror.inverted)]

    return joints_flipped
