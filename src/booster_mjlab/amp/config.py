from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DEFAULT_AMP_DATASET = "whirlwind-ams/lafan_locomotion_k1"
"""Hugging Face motion dataset used by the AMP tasks unless overridden."""


@dataclass
class AmpDiscriminatorCfg:
    """Configuration for the AMP discriminator."""

    class_name: str = "Discriminator"
    """The discriminator class name. Default is Discriminator."""
    hidden_layer_sizes: tuple[int, ...] = (256, 128)
    """The hidden layer sizes of the discriminator network."""
    reward_scale: float = 1.0
    """The scale of the reward output by the discriminator."""
    loss_type: Literal["hinge", "bce", "wasserstein"] = "wasserstein"
    """The type of loss function to use."""
    loss_fn_kwargs: dict = field(default_factory=dict)
    """Additional kwargs for the configured discriminator loss (``drift_coef`` for wasserstein)."""
    optimizer: str = "adam"
    """Discriminator optimizer (rsl_rl name or ``torch.optim`` class name). Its lr is
    fixed; it does not follow PPO's adaptive-KL schedule."""
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1.0e-4})
    """Constructor kwargs for that optimizer, including ``lr``."""
    trunk_weight_decay: float = 1.0e-3
    """L2 weight decay on the discriminator trunk."""
    head_weight_decay: float = 1.0e-1
    """L2 weight decay on the discriminator output layer."""
    empirical_normalization: bool = True
    """Whether to enable empirical normalization of AMP observations."""
