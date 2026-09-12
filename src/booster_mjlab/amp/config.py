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
    reward_clamp_epsilon: float = 1.0e-4
    """The epsilon value for reward clamping."""
    loss_type: Literal["hinge", "bce", "wasserstein"] = "wasserstein"
    """The type of loss function to use."""
    loss_fn_kwargs: dict = field(default_factory=lambda: {"eta": 1.0})
    """Additional kwargs for the configured discriminator loss."""
    use_minibatch_std: bool = True
    """Whether to use minibatch standard deviation in the discriminator."""
    empirical_normalization: bool = True
    """Whether to enable empirical normalization of AMP observations."""
