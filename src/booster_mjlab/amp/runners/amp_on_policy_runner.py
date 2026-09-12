from __future__ import annotations

from dataclasses import dataclass, field
import os
import time

import torch
from mjlab.rl import MjlabOnPolicyRunner

from booster_mjlab.amp.config import DEFAULT_AMP_DATASET, AmpDiscriminatorCfg
from booster_mjlab.amp.runners.amp_support import AmpRunner
from booster_mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlMuonPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)


@dataclass
class AmpRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for AMP PPO algorithm."""

    class_name: str = "booster_mjlab.amp.algorithms.amp_ppo:AmpPPO"
    """Fully qualified class name for the AMP PPO implementation."""


@dataclass
class AmpRslRlMuonPpoAlgorithmCfg(RslRlMuonPpoAlgorithmCfg, AmpRslRlPpoAlgorithmCfg):
    """AMP PPO with shared Muon settings and Adam discriminator updates."""

    class_name: str = "booster_mjlab.amp.algorithms.amp_ppo:AmpPPO"
    use_muon: bool = True


@dataclass
class AmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the AMP on-policy runner."""

    dataset_root: str = DEFAULT_AMP_DATASET
    """Local motion path or Hugging Face dataset ID (``namespace/repo``)."""
    warm_start: bool = False
    """With ``resume``, load only the actor, critic and discriminator weights from the checkpoint.

    Optimizer state and the iteration counter start fresh, so a checkpoint can seed a run with a
    different optimizer (e.g. Adam -> Muon) or a different training setup. Without this flag,
    ``resume`` restores everything and the run continues where it left off.
    """
    class_name: str = "AmpOnPolicyRunner"
    """The runner class name. Default is AmpOnPolicyRunner."""
    discriminator: AmpDiscriminatorCfg = field(default_factory=AmpDiscriminatorCfg)
    """The configuration for the AMP discriminator."""
    style_reward_weight: float = 0.5
    """The weight of the style reward."""
    amp_replay_buffer_size: int = 200_000
    """The size of the AMP replay buffer."""
    amp_replay_insert_size: int = 1_000
    """Samples inserted per rollout once the AMP replay buffer is full."""
    num_amp_obs_steps: int = 10
    """Number of sequential observation frames in each discriminator sample."""
    algorithm: AmpRslRlPpoAlgorithmCfg | AmpRslRlMuonPpoAlgorithmCfg = field(
        default_factory=AmpRslRlPpoAlgorithmCfg
    )
    """The algorithm configuration. Defaults to the AMP PPO implementation."""
    speed_factor: float = 1.0
    """Playback speed multiplier for motion data (`>1` faster, `<1` slower)."""
    dataset_weights: list[float] | None = None
    """The weights for each dataset in the motion dataset. If None, all datasets are weighted equally."""
    dataset_augmentations: list[dict[str, object]] | None = None
    """Additional AMP motion augmentations applied per clip (for example: mirror, speed modifiers)."""
    dataset_transform: str | None = None
    """Optional ``module.path:function`` remapping each clip onto the robot's joint layout."""
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "actor": ("actor",),
            "critic": ("critic",),
            "amp": ("amp",),
        }
    )
    """The observation groups for the runner. Include AMP obs group so resolve_obs_groups picks it up when present."""


class AmpOnPolicyRunner(AmpRunner, MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        if "amp" not in train_cfg["obs_groups"]:
            raise ValueError("AMP runner requires an 'amp' observation group.")
        train_cfg.setdefault("algorithm", {})
        # RSL-RL's logger expects this key to exist.
        train_cfg["algorithm"].setdefault("rnd_cfg", None)

        super().__init__(
            env, train_cfg=train_cfg, log_dir=log_dir, device=device, **kwargs
        )
        self._setup_amp_support(train_cfg, require_amp_group=True)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        if load_cfg is None and self.cfg.get("warm_start", False):
            print(
                "[INFO] Warm start: loading actor, critic and discriminator weights only; "
                "optimizer state and iteration counter start fresh."
            )
            load_cfg = {"actor": True, "critic": True}
        return super().load(
            path, load_cfg=load_cfg, strict=strict, map_location=map_location
        )

    # override the learn method to include AMP loop, most of the code is same as parent
    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        amp_obs = self._get_amp_obs(obs)
        self.alg.reset_amp_history(amp_obs)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()
        self._reset_amp_metrics()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    obs, amp_obs, rewards, dones, extras = self._amp_collect_step(
                        obs, amp_obs
                    )

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards
                        if self.cfg["algorithm"]["rnd_cfg"]
                        else None
                    )

                    # Book keeping via logger
                    self.logger.process_env_step(
                        rewards, dones, extras, intrinsic_rewards
                    )

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            update_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            loss_dict = dict(update_dict["loss"])

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight
                if self.cfg["algorithm"]["rnd_cfg"]
                else None,
            )
            if self.logger.writer is not None:
                for metric_name, metric_value in update_dict.get("extra", {}).items():
                    self.logger.writer.add_scalar(
                        f"Discriminator/{metric_name}", metric_value, it
                    )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
                ),
            )
            self.logger.stop_logging_writer()
