from __future__ import annotations

from typing import Generator

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from booster_mjlab.amp.modules.discriminator import Discriminator
from booster_mjlab.motion import MotionLoader
from booster_mjlab.amp.utils.default_joint_positions import (
    _resolve_default_joint_positions,
)


def _feed_forward_generator(
    observations: torch.Tensor,
    num_mini_batch: int,
    mini_batch_size: int,
    *,
    allow_replacement: bool,
) -> Generator[torch.Tensor, None, None]:
    """Sample dense data from consecutive shuffled passes."""
    num_samples = observations.shape[0]
    total = num_mini_batch * mini_batch_size
    if total <= 0:
        return
    if num_samples == 0:
        raise ValueError("AMP observation storage is empty.")
    if total > num_samples and not allow_replacement:
        raise ValueError(f"Not enough samples in buffer! ({total}/{num_samples})")

    permutation = torch.randperm(num_samples, device=observations.device)
    head = 0
    for _ in range(num_mini_batch):
        remaining = mini_batch_size
        index_chunks: list[torch.Tensor] = []
        while remaining > 0:
            take = min(remaining, num_samples - head)
            index_chunks.append(permutation[head : head + take])
            head += take
            remaining -= take
            if head == num_samples and remaining > 0:
                permutation = torch.randperm(num_samples, device=observations.device)
                head = 0
        batch_idx = torch.cat(index_chunks)
        yield observations[batch_idx]
        if head == num_samples:
            permutation = torch.randperm(num_samples, device=observations.device)
            head = 0


class AmpReplayBuffer:
    """Fixed-size circular buffer of flattened AMP observation histories."""

    def __init__(
        self,
        obs_dim: int,
        buffer_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.observations = torch.zeros(
            (buffer_size, obs_dim), dtype=torch.float32, device=self.device
        )
        self.step = 0
        self.num_samples = 0
        self._sample_buf = torch.randperm(
            buffer_size, dtype=torch.long, device=self.device
        )
        self._sample_buf_head = 0

    def insert(self, observations: torch.Tensor) -> None:
        batch_size = observations.shape[0]
        if batch_size > self.buffer_size:
            observations = observations[-self.buffer_size :]
            batch_size = self.buffer_size
        end = self.step + batch_size

        if end <= self.buffer_size:
            self.observations[self.step : end] = observations
        else:
            first_part = self.buffer_size - self.step
            self.observations[self.step :] = observations[:first_part]
            remainder = batch_size - first_part
            self.observations[:remainder] = observations[first_part:]

        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, self.num_samples + batch_size)

    def feed_forward_generator(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        allow_replacement: bool = True,
    ) -> Generator[torch.Tensor, None, None]:
        if self.num_samples == 0:
            raise ValueError("AMP replay buffer is empty.")
        total = num_mini_batch * mini_batch_size
        if total > self.num_samples and not allow_replacement:
            raise ValueError(
                f"Not enough samples in buffer! ({total}/{self.num_samples})"
            )
        if mini_batch_size > self.buffer_size:
            raise ValueError(
                "AMP mini-batch cannot exceed replay capacity: "
                f"{mini_batch_size} > {self.buffer_size}"
            )

        if not allow_replacement:
            yield from _feed_forward_generator(
                self.observations[: self.num_samples],
                num_mini_batch,
                mini_batch_size,
                allow_replacement=False,
            )
            return

        for _ in range(num_mini_batch):
            batch_idx = self._sample_rand_idx(mini_batch_size)
            yield self.observations[batch_idx]

    def _reset_sample_buf(self) -> None:
        self._sample_buf = torch.randperm(
            self.buffer_size, dtype=torch.long, device=self.device
        )
        self._sample_buf_head = 0

    def _sample_rand_idx(self, num_samples: int) -> torch.Tensor:
        """Follow MimicKit's persistent shuffled replay sampling."""
        if self._sample_buf_head + num_samples <= self.buffer_size:
            indices = self._sample_buf[
                self._sample_buf_head : self._sample_buf_head + num_samples
            ]
            self._sample_buf_head += num_samples
        else:
            first = self._sample_buf[self._sample_buf_head :]
            remainder = num_samples - first.shape[0]
            self._reset_sample_buf()
            second = self._sample_buf[:remainder]
            indices = torch.cat((first, second))
            self._sample_buf_head = remainder
        return torch.remainder(indices, self.num_samples)

    def is_full(self) -> bool:
        return self.num_samples == self.buffer_size

    def __len__(self) -> int:
        return self.num_samples


class AMP:
    """Shared AMP state: discriminator, motion loader and policy replay buffer."""

    def __init__(
        self,
        discriminator: Discriminator,
        amp_loader: MotionLoader,
        amp_obs_dim: int,
        num_amp_obs_steps: int,
        amp_replay_buffer_size: int = 200_000,
        amp_replay_insert_size: int = 1_000,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.discriminator = discriminator
        self.amp_loader = amp_loader
        self.amp_obs_dim = amp_obs_dim
        self.num_amp_obs_steps = num_amp_obs_steps
        self.amp_history_dim = amp_obs_dim * num_amp_obs_steps
        self.amp_storage = AmpReplayBuffer(
            self.amp_history_dim, amp_replay_buffer_size, self.device
        )
        if amp_replay_insert_size < 1:
            raise ValueError(
                "amp_replay_insert_size must be at least 1, got "
                f"{amp_replay_insert_size}"
            )
        self.amp_replay_insert_size = amp_replay_insert_size
        self._current_policy_batches: list[torch.Tensor] = []
        self._history: torch.Tensor | None = None
        self._history_head = 0

    def reset_history(self, amp_obs: torch.Tensor) -> None:
        """Backfill every history slot with the current observation."""
        self._validate_amp_obs(amp_obs)
        self._history = (
            amp_obs.unsqueeze(1).expand(-1, self.num_amp_obs_steps, -1).clone()
        )
        self._history_head = 0

    def act(self, amp_obs: torch.Tensor) -> None:
        """Initialize history on the first collection step."""
        if self._history is None:
            self.reset_history(amp_obs)

    def process_step(
        self, next_amp_obs: torch.Tensor, dones: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Append a frame, store its K-frame history, and return that history."""
        self._validate_amp_obs(next_amp_obs)
        if self._history is None:
            raise RuntimeError(
                "call act(...) or reset_history(...) before process_step(...)"
            )
        if next_amp_obs.shape[0] != self._history.shape[0]:
            raise ValueError(
                "AMP observation batch size changed from "
                f"{self._history.shape[0]} to {next_amp_obs.shape[0]}"
            )

        self._history[:, self._history_head] = next_amp_obs
        self._history_head = (self._history_head + 1) % self.num_amp_obs_steps

        # The environment returns post-reset observations for done environments.
        # Do not join two episodes into one discriminator history; backfill the
        # reset rows just like a newly initialized circular observation buffer.
        if dones is not None:
            done_mask = dones.reshape(-1).bool()
            if torch.any(done_mask):
                self._history[done_mask] = next_amp_obs[done_mask].unsqueeze(1)

        history = self._flatten_history()
        # Keep the current rollout separate from long-term replay. The clone is
        # required because a flattened history may alias the circular history.
        self._current_policy_batches.append(history.detach().clone())
        return history

    def _flatten_history(self) -> torch.Tensor:
        if self._history is None:
            raise RuntimeError("AMP history has not been initialized")
        if self._history_head == 0:
            ordered = self._history
        else:
            ordered = torch.cat(
                [
                    self._history[:, self._history_head :],
                    self._history[:, : self._history_head],
                ],
                dim=1,
            )
        return ordered.reshape(ordered.shape[0], -1)

    def _validate_amp_obs(self, amp_obs: torch.Tensor) -> None:
        if amp_obs.ndim != 2 or amp_obs.shape[-1] != self.amp_obs_dim:
            raise ValueError(
                "Expected AMP observations with shape "
                f"(batch, {self.amp_obs_dim}), got {tuple(amp_obs.shape)}"
            )

    def predict_style_reward(self, history: torch.Tensor) -> torch.Tensor:
        return self.discriminator.predict_reward(history)

    def has_current_policy_samples(self) -> bool:
        return bool(self._current_policy_batches)

    def _consume_current_policy_samples(self) -> torch.Tensor:
        if not self._current_policy_batches:
            raise ValueError("No current AMP policy samples were collected.")
        current = torch.cat(self._current_policy_batches, dim=0)
        self._current_policy_batches.clear()
        return current

    def _refresh_replay(self, current: torch.Tensor) -> None:
        """Fill replay eagerly, then retain only a small random current subset."""
        if self.amp_storage.is_full():
            num_insert = min(current.shape[0], self.amp_replay_insert_size)
            indices = torch.randperm(current.shape[0], device=current.device)[
                :num_insert
            ]
            self.amp_storage.insert(current[indices])
        else:
            self.amp_storage.insert(current)

    def generators(
        self,
        num_mini_batch: int,
        mini_batch_size: int,
        *,
        allow_policy_replacement: bool = True,
    ) -> tuple[
        Generator[torch.Tensor, None, None],
        Generator[torch.Tensor, None, None],
    ]:
        current = self._consume_current_policy_samples()
        self._refresh_replay(current)

        # MimicKit uses a full current batch and a full replay batch as policy
        # negatives, against one equally sized expert batch.
        current_generator = _feed_forward_generator(
            current,
            num_mini_batch,
            mini_batch_size,
            allow_replacement=allow_policy_replacement,
        )
        replay_generator = self.amp_storage.feed_forward_generator(
            num_mini_batch=num_mini_batch,
            mini_batch_size=mini_batch_size,
            allow_replacement=allow_policy_replacement,
        )

        def policy_generator() -> Generator[torch.Tensor, None, None]:
            for current_batch, replay_batch in zip(
                current_generator, replay_generator, strict=True
            ):
                yield torch.cat((current_batch, replay_batch), dim=0)

        expert_generator = self.amp_loader.feed_forward_generator(
            num_mini_batch=num_mini_batch,
            mini_batch_size=mini_batch_size,
        )
        return policy_generator(), expert_generator

    def compute_discriminator_batch(
        self,
        sample_amp_policy: torch.Tensor,
        sample_amp_expert: torch.Tensor,
        *,
        grad_penalty_lambda: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # Keep raw copies for the normalizer update after the forward pass.
        policy_raw_copy = sample_amp_policy.detach().clone()
        expert_raw_copy = sample_amp_expert.detach().clone()

        batch_policy = sample_amp_policy.size(0)
        discriminator_input = torch.cat((sample_amp_policy, sample_amp_expert), dim=0)
        discriminator_output = self.discriminator(discriminator_input)
        policy_d = discriminator_output[:batch_policy]
        expert_d = discriminator_output[batch_policy:]

        amp_loss, grad_pen_loss = self.discriminator.compute_loss(
            policy_d,
            expert_d,
            sample_amp_expert,
            sample_amp_policy,
            lambda_=grad_penalty_lambda,
        )

        # Update normalizer AFTER the forward pass and loss computation
        self.discriminator.update_normalization(
            expert_raw_copy,
            policy_raw_copy,
        )

        if hasattr(self.discriminator.loss_fn, "eta"):
            eta = self.discriminator.loss_fn.eta
            policy_prob = torch.tanh(eta * policy_d)
            expert_prob = torch.tanh(eta * expert_d)
            policy_target = -torch.ones_like(policy_prob)
            expert_target = torch.ones_like(expert_prob)
        else:
            policy_prob = torch.sigmoid(policy_d)
            expert_prob = torch.sigmoid(expert_d)
            policy_target = torch.zeros_like(policy_prob)
            expert_target = torch.ones_like(expert_prob)

        metrics = {
            "policy_pred": policy_prob.detach().mean(),
            "expert_pred": expert_prob.detach().mean(),
            "accuracy_policy_num": torch.sum(
                torch.round(policy_prob.detach()) == policy_target
            ).detach(),
            "accuracy_expert_num": torch.sum(
                torch.round(expert_prob.detach()) == expert_target
            ).detach(),
            "accuracy_policy_den": torch.tensor(
                float(policy_prob.numel()), device=policy_prob.device
            ),
            "accuracy_expert_den": torch.tensor(
                float(expert_prob.numel()), device=expert_prob.device
            ),
        }
        return amp_loss, grad_pen_loss, metrics


def construct_amp(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str | torch.device,
) -> tuple[AMP, int]:
    """Construct shared AMP runner from runner configuration."""
    amp_groups = cfg["obs_groups"].get("amp")
    if not amp_groups:
        raise ValueError("AMP configuration requires an 'amp' observation group.")

    for required_key in ("dataset_root", "discriminator", "amp_replay_buffer_size"):
        if required_key not in cfg or cfg[required_key] is None:
            raise ValueError(
                f"AMP configuration requires `{required_key}` in runner config."
            )

    amp_obs_dim = sum(obs[group].shape[-1] for group in amp_groups)
    num_amp_obs_steps = int(cfg.get("num_amp_obs_steps", 10))
    if num_amp_obs_steps < 1:
        raise ValueError(
            f"num_amp_obs_steps must be at least 1, got {num_amp_obs_steps}"
        )
    amp_history_dim = amp_obs_dim * num_amp_obs_steps
    discriminator_cfg = cfg["discriminator"]
    loss_fn_kwargs = discriminator_cfg.get("loss_fn_kwargs", {})

    discriminator = Discriminator(
        input_dim=amp_history_dim,
        hidden_layer_sizes=list(discriminator_cfg["hidden_layer_sizes"]),
        reward_scale=float(discriminator_cfg["reward_scale"]),
        reward_clamp_epsilon=float(discriminator_cfg["reward_clamp_epsilon"]),
        loss_type=discriminator_cfg["loss_type"],
        eta_wgan=loss_fn_kwargs.get("eta", 1.0),
        use_minibatch_std=discriminator_cfg.get("use_minibatch_std", True),
        empirical_normalization=discriminator_cfg["empirical_normalization"],
        device=device,
    )

    base_dt = env.cfg.sim.mujoco.timestep
    decimation = env.cfg.decimation
    simulation_dt = base_dt * decimation
    default_pose, joint_indices = _resolve_default_joint_positions(env)
    amp_term_names = set(env.unwrapped.observation_manager._group_obs_term_names["amp"])
    include_base_lin_vel = "base_lin_vel" in amp_term_names
    include_base_ang_vel = "base_ang_vel" in amp_term_names
    include_projected_gravity = "projected_gravity" in amp_term_names

    amp_loader = MotionLoader(
        dataset_root=cfg["dataset_root"],
        simulation_dt=simulation_dt,
        speed_factor=cfg.get("speed_factor", 1.0),
        num_amp_obs_steps=num_amp_obs_steps,
        dataset_weights=cfg.get("dataset_weights"),
        augmentations=cfg.get("dataset_augmentations"),
        dataset_transform=cfg.get("dataset_transform"),
        default_pose=default_pose,
        joint_indices=joint_indices,
        include_base_lin_vel=include_base_lin_vel,
        include_base_ang_vel=include_base_ang_vel,
        include_projected_gravity=include_projected_gravity,
        device=device,
    )

    amp_runner = AMP(
        discriminator=discriminator,
        amp_loader=amp_loader,
        amp_obs_dim=amp_obs_dim,
        num_amp_obs_steps=num_amp_obs_steps,
        amp_replay_buffer_size=int(cfg["amp_replay_buffer_size"]),
        amp_replay_insert_size=int(cfg.get("amp_replay_insert_size", 1_000)),
        device=device,
    )
    return amp_runner, amp_obs_dim
