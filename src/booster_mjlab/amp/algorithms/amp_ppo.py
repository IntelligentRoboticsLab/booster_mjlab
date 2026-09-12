from __future__ import annotations

from itertools import chain
from typing import Union

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from booster_mjlab.amp.modules.amp import (
    AMP,
    construct_amp,
)
from booster_mjlab.rl.muon import HybridMuonOptimizer


class AmpPPO(PPO):
    """Proximal Policy Optimization algorithm with AMP support.

    Based on "AMP: Adverserial Motion Priors for Stylized Physics-Based Character Control" (https://arxiv.org/abs/2104.02180).
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        amp: AMP,
        use_smooth_ratio_clipping: bool = False,
        use_muon: bool = False,
        muon_weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        super().__init__(
            actor=actor, critic=critic, storage=storage, device=device, **kwargs
        )
        self.amp = amp
        self.use_smooth_ratio_clipping = use_smooth_ratio_clipping

        # Rebuild optimizer to include discriminator parameters alongside actor+critic
        params = [
            {
                "params": chain(self.actor.parameters(), self.critic.parameters()),
                "name": "actor_critic",
                "lr_scale": 1.0,
            },
            {
                "params": self.amp.discriminator.trunk.parameters(),
                "weight_decay": 1.0e-3,
                "name": "amp_trunk",
                "lr_scale": 1.0,
            },
            {
                "params": self.amp.discriminator.linear.parameters(),
                "weight_decay": 1.0e-1,
                "name": "amp_head",
                "lr_scale": 1.0,
            },
        ]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        if use_muon:
            matrix_params: list[nn.Parameter] = []
            actor_critic_other_params: list[nn.Parameter] = []
            for module in (self.actor, self.critic):
                for parameter in module.parameters():
                    if parameter.ndim == 2:
                        matrix_params.append(parameter)
                    else:
                        actor_critic_other_params.append(parameter)

            self.optimizer = HybridMuonOptimizer(
                matrix_params,
                [
                    {
                        "params": actor_critic_other_params,
                        "name": "actor_critic_adam",
                        "lr_scale": 1.0,
                        "weight_decay": 0.0,
                    },
                    {
                        "params": self.amp.discriminator.trunk.parameters(),
                        "name": "amp_trunk",
                        "lr_scale": 1.0,
                        "weight_decay": 1.0e-3,
                    },
                    {
                        "params": self.amp.discriminator.linear.parameters(),
                        "name": "amp_head",
                        "lr_scale": 1.0,
                        "weight_decay": 1.0e-1,
                    },
                ],
                lr=self.learning_rate,
                muon_weight_decay=muon_weight_decay,
                muon_momentum=muon_momentum,
                muon_ns_steps=muon_ns_steps,
            )

    def act_amp(self, amp_obs: torch.Tensor) -> None:
        """Initialize AMP history if collection has not started yet."""
        self.amp.act(amp_obs)

    def reset_amp_history(self, amp_obs: torch.Tensor) -> None:
        """Backfill the policy AMP history from the current observation."""
        self.amp.reset_history(amp_obs)

    def process_amp_step(
        self, next_amp_obs: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Append an AMP frame and insert its history into the replay buffer."""
        return self.amp.process_step(next_amp_obs, dones)

    def predict_style_reward(self, amp_history: torch.Tensor) -> torch.Tensor:
        """Predict style reward for a flattened AMP history."""
        return self.amp.predict_style_reward(amp_history)

    def update(self) -> dict[str, dict[str, float]]:  # noqa: C901
        """Perform the AMP/PPO update step.

        Most of this implementatation is a direct copy from PPO.update.
        This merely introduces the AMP update described in Algorithm 1.


        Raises:
            RuntimeError: raised when the transition storage isn't initialized

        Returns:
            dict[str, float]: A dictionary containing metrics data
        """
        if self.storage is None:
            raise RuntimeError("Storage is not initialized. Call init_storage first.")

        # Accumulate on GPU to avoid synchronization
        mean_value_loss = torch.zeros(1, device=self.device)
        mean_surrogate_loss = torch.zeros(1, device=self.device)
        mean_entropy = torch.zeros(1, device=self.device)
        mean_amp_loss = torch.zeros(1, device=self.device)
        mean_grad_pen_loss = torch.zeros(1, device=self.device)
        mean_policy_pred = torch.zeros(1, device=self.device)
        mean_expert_pred = torch.zeros(1, device=self.device)
        mean_accuracy_policy = torch.zeros(1, device=self.device)
        mean_accuracy_expert = torch.zeros(1, device=self.device)
        mean_accuracy_policy_elem = 0.0
        mean_accuracy_expert_elem = 0.0
        mean_kl_divergence = torch.zeros(1, device=self.device)
        mean_rnd_loss = torch.zeros(1, device=self.device) if self.rnd else None
        mean_symmetry_loss = (
            torch.zeros(1, device=self.device) if self.symmetry else None
        )

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        num_updates = self.num_learning_epochs * self.num_mini_batches

        amp_policy_generator, amp_expert_generator = self.amp.generators(
            num_mini_batch=num_updates,
            mini_batch_size=mini_batch_size,
            allow_policy_replacement=True,
        )

        # line 18: sample batch of transitions from motion data
        # line 19: sample batch of transitions from policy replay buffer
        for batch, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            original_batch_size = batch.observations.batch_size[0]

            # normalize advantages, from original PPO impl
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob_batch = self.actor.get_output_log_prob(batch.actions)
            value_batch = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy_batch = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM
                        )
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        # Preserve per-group learning rate ratios (e.g., AMP head/trunk).
                        lr_scale = float(param_group.get("lr_scale", 1.0))
                        param_group["lr"] = self.learning_rate * lr_scale
                    mean_kl_divergence += kl_mean

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(batch.old_actions_log_prob)
            )
            min_ratio = 1.0 - self.clip_param
            max_ratio = 1.0 + self.clip_param
            if self.use_smooth_ratio_clipping:
                clipped_ratio = (
                    1
                    / (
                        1
                        + torch.exp(
                            (-(ratio - min_ratio) / (max_ratio - min_ratio) + 0.5) * 4
                        )
                    )
                    * (max_ratio - min_ratio)
                    + min_ratio
                )
            else:
                clipped_ratio = torch.clamp(ratio, min_ratio, max_ratio)
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * clipped_ratio
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (value_batch - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            rnd_loss = (
                self.rnd.compute_loss(batch.observations[:original_batch_size])
                if self.rnd
                else None
            )
            symmetry_loss = None
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(
                    self.actor, batch, original_batch_size
                )
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            amp_loss, grad_pen_loss, amp_metrics = self.amp.compute_discriminator_batch(
                sample_amp_policy,
                sample_amp_expert,
                grad_penalty_lambda=10.0,
            )

            total_loss = loss + amp_loss + grad_pen_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            # Detach metrics so we don't hold onto autograd graphs.
            entropy_mean = entropy_batch.mean().detach()
            value_loss_det = value_loss.detach()
            surrogate_loss_det = surrogate_loss.detach()
            amp_loss_det = amp_loss.detach()
            grad_pen_loss_det = grad_pen_loss.detach()

            # Update metrics (accumulate on GPU, no synchronization)
            mean_value_loss += value_loss_det
            mean_surrogate_loss += surrogate_loss_det
            mean_entropy += entropy_mean
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.detach()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.detach()
            mean_amp_loss += amp_loss_det
            mean_grad_pen_loss += grad_pen_loss_det
            mean_policy_pred += amp_metrics["policy_pred"]
            mean_expert_pred += amp_metrics["expert_pred"]
            mean_accuracy_policy += amp_metrics["accuracy_policy_num"]
            mean_accuracy_expert += amp_metrics["accuracy_expert_num"]
            mean_accuracy_policy_elem += float(
                amp_metrics["accuracy_policy_den"].item()
            )
            mean_accuracy_expert_elem += float(
                amp_metrics["accuracy_expert_den"].item()
            )

        # Finalize metrics (single GPU sync point)
        mean_value_loss = (mean_value_loss / num_updates).item()
        mean_surrogate_loss = (mean_surrogate_loss / num_updates).item()
        mean_entropy = (mean_entropy / num_updates).item()
        mean_amp_loss = (mean_amp_loss / num_updates).item()
        mean_grad_pen_loss = (mean_grad_pen_loss / num_updates).item()
        mean_policy_pred = (mean_policy_pred / num_updates).item()
        mean_expert_pred = (mean_expert_pred / num_updates).item()
        mean_accuracy_policy = (
            mean_accuracy_policy / max(1.0, mean_accuracy_policy_elem)
        ).item()
        mean_accuracy_expert = (
            mean_accuracy_expert / max(1.0, mean_accuracy_expert_elem)
        ).item()
        mean_kl_divergence = (mean_kl_divergence / num_updates).item()
        if mean_rnd_loss is not None:
            mean_rnd_loss = (mean_rnd_loss / num_updates).item()
        if mean_symmetry_loss is not None:
            mean_symmetry_loss = (mean_symmetry_loss / num_updates).item()

        # Update the observation normalizers from the rollout, as PPO.update does.
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "Discriminator/loss": mean_amp_loss,
            "Discriminator/grad_penalty": mean_grad_pen_loss,
            "kl_divergence": mean_kl_divergence,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return {
            "loss": loss_dict,
            "extra": {
                "policy_pred": mean_policy_pred,
                "expert_pred": mean_expert_pred,
                "accuracy_policy": mean_accuracy_policy,
                "accuracy_expert": mean_accuracy_expert,
            },
        }

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "discriminator_state_dict": self.amp.discriminator.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if "discriminator_state_dict" in loaded_dict:
            self.amp.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"], strict=strict
            )

        return super().load(loaded_dict, load_cfg, strict)

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> AmpPPO:
        """Construct the AmpPPO algorithm.

        Uses the base PPO.construct_algorithm to create actor, critic, and storage,
        then wraps them in AmpPPO with the discriminator and motion loader.
        """
        # Resolve class callables for actor/critic
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND and symmetry configs
        cfg["algorithm"] = resolve_rnd_config(
            cfg["algorithm"], obs, cfg["obs_groups"], env
        )
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Remove class_name from algorithm config (we're constructing AmpPPO, not the resolved class)
        cfg["algorithm"].pop("class_name", None)

        # Create actor and critic
        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")

        # Create storage
        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        amp, _ = construct_amp(obs, env, cfg, device)

        # Create AmpPPO
        alg = AmpPPO(
            actor=actor,
            critic=critic,
            storage=storage,
            amp=amp,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg
