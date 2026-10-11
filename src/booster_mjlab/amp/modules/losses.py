import torch
import torch.nn as nn


class DiscriminatorLoss(nn.Module):
    """Base class for discriminator loss functions."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        policy_samples: torch.Tensor,
        expert_samples: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_grad_penalty_data(
        self,
        expert_samples: torch.Tensor,
        policy_samples: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        raise NotImplementedError("Subclasses must implement this method.")


def make_loss_function(loss_type: str, **kwargs) -> DiscriminatorLoss:
    """Factory function to create loss functions based on the specified type.

    Args:
        loss_type (str): The type of loss function to create. Supported types are
                         'bce', 'hinge', and 'wasserstein'.
        **kwargs: Keyword arguments for the loss constructor; unknown kwargs raise.
    Returns:
        nn.Module: An instance of the specified loss function.
    """
    normalized_type = (loss_type or "wasserstein").lower()
    if normalized_type in {"bce", "bcewithlogits"}:
        return BCELoss(**kwargs)
    elif normalized_type == "hinge":
        return HingeLoss(**kwargs)
    elif normalized_type in {"wasserstein", "wgan"}:
        return WGANLoss(**kwargs)
    else:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. Supported types are 'hinge', 'bce', and 'wasserstein'."
        )


class BCELoss(DiscriminatorLoss):
    """Binary Cross-Entropy Loss for Discriminator.

    This implements a binary cross-entropy loss function that computes the BCE loss
    for both policy and expert samples using logits. Assuming that expert samples are labeled
    as 1 and policy samples as 0, the loss is computed as:
        L = 0.5 * (BCE(D(expert), 1) + BCE(D(policy), 0))

    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        policy_samples: torch.Tensor,
        expert_samples: torch.Tensor,
    ) -> torch.Tensor:
        expert_loss = self.loss_fn(expert_samples, torch.ones_like(expert_samples))
        policy_loss = self.loss_fn(policy_samples, torch.zeros_like(policy_samples))

        return 0.5 * (expert_loss + policy_loss)

    def get_grad_penalty_data(
        self,
        expert_samples: torch.Tensor,
        policy_samples: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        data = expert_samples.to(device)
        offset = 0.0
        return data, offset


class HingeLoss(DiscriminatorLoss):
    """Hinge Loss for Discriminator.

    This implements a hinge loss function for the discriminator, defined as:
        L = 0.5 * (ReLU(1 - D(expert)) + ReLU(1 + D(policy)))
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        policy_samples: torch.Tensor,
        expert_samples: torch.Tensor,
    ) -> torch.Tensor:
        expert_loss = torch.relu(1.0 - expert_samples).mean()
        policy_loss = torch.relu(1.0 + policy_samples).mean()

        return 0.5 * (expert_loss + policy_loss)

    def get_grad_penalty_data(
        self,
        expert_samples: torch.Tensor,
        policy_samples: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        data = expert_samples.to(device)
        offset = 0.0
        return data, offset


class WGANLoss(DiscriminatorLoss):
    """Wasserstein critic loss on the raw critic score.

        L = E[D(policy)] - E[D(expert)] + drift_coef * (0.5 * (E[D(policy)] + E[D(expert)]))^2

    The Lipschitz constraint comes from the WGAN-GP penalty on expert/policy
    interpolates (:meth:`Discriminator.compute_grad_penalty`). No ``tanh``
    squashing: a saturated tanh zeroes ``dL/dD`` for confidently scored samples on
    both sides, so the critic could never correct a policy sample it mis-scores as
    expert. The Wasserstein term is invariant to ``D -> D + c``; the drift term pins
    the midpoint of the two batch means at zero. The reward standardizes the score by
    the latest policy-score statistics before softplus
    (:meth:`Discriminator.predict_reward`).

        "Wasserstein GAN" (Arjovsky et al., 2017), https://arxiv.org/abs/1701.07875
        "Improved Training of Wasserstein GANs" (Gulrajani et al., 2017),
        https://arxiv.org/abs/1704.00028
    """

    def __init__(self, drift_coef: float = 1.0e-2) -> None:
        super().__init__()
        self.drift_coef = drift_coef

    def forward(
        self,
        policy_samples: torch.Tensor,
        expert_samples: torch.Tensor,
    ) -> torch.Tensor:
        policy_mean = policy_samples.mean()
        expert_mean = expert_samples.mean()
        wasserstein = policy_mean - expert_mean
        drift = self.drift_coef * (0.5 * (policy_mean + expert_mean)).pow(2)
        return wasserstein + drift

    def get_grad_penalty_data(
        self,
        expert_samples: torch.Tensor,
        policy_samples: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        expert = expert_samples
        policy = policy_samples
        if expert.shape[0] != policy.shape[0]:
            # MimicKit-style discriminator batches contain B expert samples and
            # 2B policy samples (B current + B replay). Repeat the expert batch
            # so every policy sample participates in the interpolation penalty.
            repeats = (policy.shape[0] + expert.shape[0] - 1) // expert.shape[0]
            repeat_dims = (repeats,) + (1,) * (expert.ndim - 1)
            expert = expert.repeat(repeat_dims)[: policy.shape[0]]
        alpha = torch.rand(expert.size(0), 1, device=device)
        data = alpha * expert + (1 - alpha) * policy
        offset = 1.0
        return data, offset
