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
        **kwargs: Additional keyword arguments for specific loss functions.
    Returns:
        nn.Module: An instance of the specified loss function.
    """
    normalized_type = (loss_type or "hinge").lower()
    if normalized_type in {"bce", "bcewithlogits"}:
        return BCELoss()
    elif normalized_type == "hinge":
        return HingeLoss()
    elif normalized_type in {"wasserstein", "wgan"}:
        eta_wgan = kwargs.get("eta", 1.0)
        return WGANLoss(eta=eta_wgan)
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
    """Wasserstein loss with tanh-squashed critic outputs.

    This implements a Wasserstein-style objective of the form

        L = E[D(policy)] - E[D(expert)]

    where the critic outputs can be scaled and squashed with a `tanh`
    nonlinearity as

        D̃(x) = tanh(eta_wgan * D(x)),

    to keep values bounded and improve numerical stability when using the
    critic as a reward signal. Note that this deviates slightly from the
    original unbounded critic in:

        "Wasserstein GAN" (Arjovsky et al., 2017), https://arxiv.org/abs/1701.07875
    """

    def __init__(self, eta: float = 1.0) -> None:
        super().__init__()
        self.eta = eta

    def forward(
        self,
        policy_samples: torch.Tensor,
        expert_samples: torch.Tensor,
    ) -> torch.Tensor:
        policy_samples = torch.tanh(self.eta * policy_samples)
        expert_samples = torch.tanh(self.eta * expert_samples)

        return policy_samples.mean() - expert_samples.mean()

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
