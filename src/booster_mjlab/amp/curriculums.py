"""AMP-related curriculum terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

AMP_STYLE_WEIGHT_ATTR = "amp_style_reward_weight"


def anneal_style_reward(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    start_weight: float,
    end_weight: float,
    start_step: int,
    end_step: int,
) -> torch.Tensor:
    """Linearly anneal the AMP style reward weight between two values.

    The weight is written to ``env.amp_style_reward_weight`` so the AMP runner
    can pick it up in :meth:`AmpRunner._compute_style_weight`.

    Args:
        env: The environment.
        env_ids: Environment IDs (unused).
        start_weight: Style weight at *start_step* (and before).
        end_weight: Style weight at *end_step* (and after).
        start_step: Step at which annealing begins.
        end_step: Step at which annealing ends.

    Returns:
        Current style weight for logging.
    """
    del env_ids

    step = env.common_step_counter
    if step <= start_step:
        weight = start_weight
    elif step >= end_step:
        weight = end_weight
    else:
        t = (step - start_step) / (end_step - start_step)
        weight = start_weight + t * (end_weight - start_weight)

    setattr(env, AMP_STYLE_WEIGHT_ATTR, weight)
    return torch.tensor(weight)
