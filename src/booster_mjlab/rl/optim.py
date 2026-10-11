"""Optimizer construction shared by the Booster PPO variants and the AMP discriminator.

``optimizer`` is a name (rsl_rl's ``adam``/``adamw``/``sgd``/``rmsprop``, any
``torch.optim`` class name, or ``muon``) and ``optimizer_kwargs`` go to its
constructor. ``muon`` builds :class:`HybridMuonOptimizer` (``weight_decay``,
``momentum``, ``ns_steps``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.utils import resolve_optimizer


class HybridMuonOptimizer:
    """Muon for matrix weights plus Adam for non-matrix parameters."""

    def __init__(
        self,
        matrix_params: list[nn.Parameter],
        adam_param_groups: list[dict],
        *,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        if not matrix_params:
            raise ValueError("Muon requires at least one 2D parameter.")
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon requires a PyTorch version with torch.optim.Muon.")

        self.muon = torch.optim.Muon(
            [{"params": matrix_params, "name": "muon_matrix", "lr_scale": 1.0}],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
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


def resolve_optimizer_class(name: str) -> type[torch.optim.Optimizer]:
    """Map an optimizer name to a ``torch.optim`` class (rsl_rl names, then any class)."""
    try:
        return resolve_optimizer(name.lower())
    except ValueError:
        pass
    cls = getattr(torch.optim, name, None)
    if isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer):
        return cls
    raise ValueError(
        f"Unknown optimizer {name!r}: expected 'muon', an rsl_rl optimizer name, or a "
        "torch.optim class name."
    )


def build_optimizer(
    param_groups: list[dict],
    *,
    optimizer: str,
    optimizer_kwargs: dict | None,
    lr: float,
) -> torch.optim.Optimizer | HybridMuonOptimizer:
    """Build an optimizer over parameter groups from a name and constructor kwargs.

    Any ``torch.optim`` optimizer gets the groups as-is. ``"muon"`` moves the matrix
    parameters of every group into Muon and keeps the remaining parameters in Adam
    groups that retain their own settings (weight decay, name).
    """
    groups = [{**g, "params": list(g["params"])} for g in param_groups]
    kwargs = dict(optimizer_kwargs or {})
    kwargs.setdefault("lr", lr)
    if optimizer.lower() == "muon":
        matrix_params = [p for g in groups for p in g["params"] if p.ndim == 2]
        adam_groups = [
            {**g, "params": [p for p in g["params"] if p.ndim != 2]} for g in groups
        ]
        return HybridMuonOptimizer(
            matrix_params, [g for g in adam_groups if g["params"]], **kwargs
        )
    return resolve_optimizer_class(optimizer)(groups, **kwargs)
