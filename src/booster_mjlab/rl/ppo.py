"""PPO with a configurable actor/critic optimizer (including hybrid Muon/Adam)."""

from __future__ import annotations

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from booster_mjlab.rl.optim import build_optimizer


class BoosterPPO(PPO):
    """rsl_rl PPO whose optimizer is picked by ``optimizer`` / ``optimizer_kwargs``.

    ``optimizer`` accepts rsl_rl's names, any ``torch.optim`` class name, or
    ``"muon"`` for Muon on matrix weights with Adam on the rest.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        # The base class builds a plain Adam; replace it below.
        super().__init__(actor, critic, storage, optimizer="adam", **kwargs)
        self.optimizer = build_optimizer(
            [
                {
                    "params": [*self.actor.parameters(), *self.critic.parameters()],
                    "name": "actor_critic",
                    "lr_scale": 1.0,
                }
            ],
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr=self.learning_rate,
        )
