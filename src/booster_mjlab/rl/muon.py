"""Shared hybrid Muon optimizer and standard PPO integration."""

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


class HybridMuonOptimizer:
    """Muon for matrix weights plus Adam for non-matrix parameters."""

    def __init__(
        self,
        matrix_params: list[nn.Parameter],
        adam_param_groups: list[dict],
        *,
        lr: float,
        muon_weight_decay: float,
        muon_momentum: float,
        muon_ns_steps: int,
    ) -> None:
        if not matrix_params:
            raise ValueError("Muon requires at least one 2D parameter.")
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon requires a PyTorch version with torch.optim.Muon.")

        self.muon = torch.optim.Muon(
            [{"params": matrix_params, "name": "muon_matrix", "lr_scale": 1.0}],
            lr=lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
            # Scale the orthogonalized update to Adam's RMS so both halves can share one
            # learning rate and the adaptive-KL schedule that drives it.
            adjust_lr_fn="match_rms_adamw",
        )
        self.adam = torch.optim.Adam(adam_param_groups, lr=lr)
        self.param_groups = [*self.muon.param_groups, *self.adam.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self) -> None:
        self.muon.step()
        self.adam.step()

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict(),
            "adam": self.adam.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
        self.param_groups = [*self.muon.param_groups, *self.adam.param_groups]


class MuonPPO(PPO):
    """PPO using Muon for actor/critic matrices and Adam for other parameters."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        muon_weight_decay: float = 0.0,
        muon_momentum: float = 0.95,
        muon_ns_steps: int = 5,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = HybridMuonOptimizer(
            [p for p in parameters if p.ndim == 2],
            [
                {
                    "params": [p for p in parameters if p.ndim != 2],
                    "name": "actor_critic_adam",
                    "lr_scale": 1.0,
                    "weight_decay": 0.0,
                }
            ],
            lr=self.learning_rate,
            muon_weight_decay=muon_weight_decay,
            muon_momentum=muon_momentum,
            muon_ns_steps=muon_ns_steps,
        )
