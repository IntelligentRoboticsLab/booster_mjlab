"""FastSAC runner.

Extends the base OnPolicyRunner to handle FastSAC-specific concerns:
- Overrides learn() to remove PPO-specific rnd_cfg references
- No actor/critic model configs (FastSAC constructs them internally)
- Environment state persistence across checkpoints
"""

from __future__ import annotations

import os
import time
from copy import deepcopy

import torch
import wandb
from rsl_rl.utils import WandbLogWriter
from tensordict import TensorDict

from mjlab.rl import MjlabOnPolicyRunner
from booster_mjlab.amp.runners.amp_support import AmpRunner
from booster_mjlab.rl import RslRlVecEnvWrapper
from booster_mjlab.rl.exporter_utils import (
    attach_exclusion_metadata,
    attach_metadata_to_onnx,
    get_base_metadata,
)


class FastSACRunner(AmpRunner, MjlabOnPolicyRunner):
    """Runner for FastSAC that adapts the on-policy loop for off-policy SAC."""

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        del kwargs
        cfg = deepcopy(train_cfg)
        cfg.setdefault("algorithm", {})
        # RSL-RL's logger expects this key to exist.
        cfg["algorithm"].setdefault("rnd_cfg", None)
        super().__init__(env, cfg, log_dir, device)
        self._setup_amp_support(cfg, require_amp_group=False)
        if self._amp_enabled and not getattr(self.alg, "has_amp", False):
            raise ValueError(
                "FastSAC AMP runner requires Adversarial Motion Priors to be configured in the algorithm."
            )

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        """Training loop adapted for FastSAC (no rnd_cfg, no compute_returns)."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        amp_obs = self._get_amp_obs(obs) if self._amp_enabled else None
        if self._amp_enabled:
            self.alg.reset_amp_history(amp_obs)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()
        self._reset_amp_metrics()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Collect one step (num_steps_per_env=1 for off-policy)
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    obs, amp_obs, rewards, dones, extras = self._amp_collect_step(
                        obs, amp_obs
                    )
                    self.logger.process_env_step(rewards, dones, extras, None)

                collect_time = time.time() - start
                start = time.time()

                # No-op for SAC but keeps the interface consistent
                self.alg.compute_returns(obs)

            # Perform SAC gradient updates from replay buffer
            loss_dict = self.alg.update()

            learn_time = time.time() - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )
            self.logger.stop_logging_writer()

    def save(self, path: str, infos=None) -> None:
        """Save checkpoint with environment state and ONNX export."""
        super().save(path, infos)

        # Export ONNX alongside the checkpoint
        if isinstance(self.logger.writer, WandbLogWriter) and wandb.run:
            policy_dir = os.path.dirname(path)
            filename = os.path.basename(policy_dir) + ".onnx"
            try:
                self.export_policy_to_onnx(policy_dir, filename)
                run_name: str = wandb.run.name  # type: ignore[assignment]
                metadata = get_base_metadata(self.env.unwrapped, run_name)
                onnx_path = os.path.join(policy_dir, filename)
                attach_metadata_to_onnx(onnx_path, metadata)
                attach_exclusion_metadata(self.env.unwrapped, onnx_path)
                if self.cfg.get("upload_model", True):
                    wandb.save(onnx_path, base_path=policy_dir)
            except Exception:
                # ONNX export is best-effort; don't fail the checkpoint
                pass

    def get_inference_policy(self, device: str | None = None):
        """Return a deterministic FastSAC policy callable for play.py."""
        self.alg.eval_mode()
        device = device or self.device

        actor = self.alg.get_policy().to(device)
        obs_normalizer = self.alg.obs_normalizer.to(device)
        obs_normalizer.eval()
        obs_group_names = tuple(getattr(self.alg, "_actor_obs_group_names", ("actor",)))

        def _normalize(obs: torch.Tensor) -> torch.Tensor:
            try:
                return obs_normalizer(obs, update=False)
            except TypeError:
                return obs_normalizer(obs)

        @torch.no_grad()
        def policy(obs: TensorDict) -> torch.Tensor:
            flat_obs = torch.cat([obs[name] for name in obs_group_names], dim=-1).to(
                device
            )
            norm_obs = _normalize(flat_obs)
            return actor.explore(norm_obs, deterministic=True)

        return policy
