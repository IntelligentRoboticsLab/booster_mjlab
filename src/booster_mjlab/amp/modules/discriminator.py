import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F

from booster_mjlab.amp.modules.losses import make_loss_function
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_optimizer


class Discriminator(nn.Module):
    """Small MLP trained to distinguish expert and policy motion histories.

    Args:
        input_dim: Dimension of a flattened K-frame AMP observation history.
        hidden_layer_sizes: List of hidden layer sizes.
        reward_scale: Scale factor for computed rewards.
        device: Device to run the model on.
        loss_type: Type of loss function ('hinge', 'bce', 'wasserstein').
        loss_fn_kwargs: Extra kwargs for the loss (``drift_coef`` for 'wasserstein').
        empirical_normalization: Whether to normalize AMP observations internally.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        device: torch.device = torch.device("cpu"),
        loss_type: str = "wasserstein",
        loss_fn_kwargs: dict | None = None,
        empirical_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.empirical_normalization = empirical_normalization

        if empirical_normalization:
            self.amp_normalizer = EmpiricalNormalization(input_dim).to(device)
        else:
            self.amp_normalizer = None

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers).to(device)
        self.linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)
        self.loss_type = loss_type.lower()
        self.loss_fn = make_loss_function(self.loss_type, **(loss_fn_kwargs or {}))
        if self.loss_type == "wasserstein":
            # Policy-score statistics that standardize the unscaled critic score before
            # the reward squashing; refreshed by update_reward_normalization().
            self.register_buffer("reward_score_mean", torch.zeros((), device=device))
            self.register_buffer("reward_score_std", torch.ones((), device=device))

    def make_optimizer(
        self,
        optimizer: str = "adam",
        optimizer_kwargs: dict | None = None,
        trunk_weight_decay: float = 1.0e-3,
        head_weight_decay: float = 1.0e-1,
    ) -> torch.optim.Optimizer:
        """Build the discriminator's optimizer with separate trunk/head weight decay."""
        kwargs = dict(optimizer_kwargs or {})
        kwargs.setdefault("lr", 1.0e-4)
        try:
            optimizer_cls = resolve_optimizer(optimizer.lower())
        except ValueError:
            optimizer_cls = getattr(torch.optim, optimizer)
        return optimizer_cls(
            [
                {
                    "params": self.trunk.parameters(),
                    "weight_decay": trunk_weight_decay,
                    "name": "amp_trunk",
                },
                {
                    "params": self.linear.parameters(),
                    "weight_decay": head_weight_decay,
                    "name": "amp_head",
                },
            ],
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        Args:
            x: Flattened history tensor of shape (batch, input_dim).

        Returns:
            Discriminator output logits/scores.
        """
        return self._forward_normalized(self._normalize(x))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a raw history exactly once at the model boundary."""
        if self.amp_normalizer is None:
            return x
        return self.amp_normalizer(x)

    def _forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Run the discriminator network on already-normalized histories."""
        return self.linear(self.trunk(x))

    def predict_reward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict reward based on discriminator output.

        Args:
            history: Flattened K-frame AMP observation history.

        Returns:
            Computed adversarial reward.

        Note:
            Normalization is handled internally by forward() if enabled.
        """
        with torch.no_grad():
            logits = self.forward(history)
            if self.loss_type == "wasserstein":
                # The critic score has no fixed scale (the expert/policy gap reaches
                # ~20), which leaves most policy samples in softplus's flat tail.
                # Standardizing by the policy-score statistics keeps the reward graded
                # across the policy distribution and its magnitude stable over training.
                logits = (logits - self.reward_score_mean) / self.reward_score_std
            # softplus(x) = -log(1 - sigmoid(x)) for BCE logits; for the Wasserstein
            # critic it keeps the reward positive and increasing in the score.
            reward = F.softplus(logits)
            return (self.reward_scale * reward).squeeze(-1)

    def update_reward_normalization(self, policy_history: torch.Tensor) -> None:
        """Set the Wasserstein reward statistics from the current critic's policy scores.

        Args:
            policy_history: Raw policy AMP histories, ideally the latest rollout.
        """
        if self.loss_type != "wasserstein":
            return
        with torch.no_grad():
            scores = self.forward(policy_history).squeeze(-1)
            self.reward_score_mean.copy_(scores.mean())
            self.reward_score_std.copy_(scores.std(unbiased=False).clamp_min(1.0e-2))

    def compute_loss(
        self,
        policy_d: torch.Tensor,
        expert_d: torch.Tensor,
        sample_amp_expert: torch.Tensor,
        sample_amp_policy: torch.Tensor,
        lambda_: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute discriminator loss with gradient penalty.

        Args:
            policy_d: Discriminator output for policy samples.
            expert_d: Discriminator output for expert samples.
            sample_amp_expert: Expert motion histories.
            sample_amp_policy: Policy motion histories.
            lambda_: Gradient penalty coefficient.

        Returns:
            Tuple of (amp_loss, grad_penalty_loss).
        """
        grad_pen_loss = self.compute_grad_penalty(
            sample_amp_expert, sample_amp_policy, lambda_
        )
        amp_loss = self.loss_fn(policy_d, expert_d)
        return amp_loss, grad_pen_loss

    def compute_grad_penalty(
        self,
        expert_samples: torch.Tensor,
        policy_samples: torch.Tensor,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """Compute gradient penalty for discriminator regularization.

        Args:
            expert_samples: Expert motion histories.
            policy_samples: Policy motion histories.
            lambda_: Penalty coefficient.

        Returns:
            Gradient penalty loss value.

        Note:
            This method accepts raw histories and normalizes the selected
            gradient-penalty samples exactly once. For Wasserstein loss, uses
            WGAN-GP on interpolated samples.
            For BCE/Hinge loss, uses R1 regularizer on expert data only.
        """
        data, offset = self.loss_fn.get_grad_penalty_data(
            expert_samples, policy_samples, self.device
        )
        data = self._normalize(data).detach().requires_grad_(True)
        scores = self._forward_normalized(data)

        if self.loss_type == "wasserstein":
            # WGAN-GP on the raw critic output: lambda * (||dD/dx|| - 1)^2.
            grad = autograd.grad(
                outputs=scores,
                inputs=data,
                grad_outputs=torch.ones_like(scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return lambda_ * (grad.norm(2, dim=1) - offset).pow(2).mean()
        else:
            # R1 regularizer: 0.5 * lambda * ||grad||^2
            grad = autograd.grad(
                outputs=scores.sum(),
                inputs=data,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return 0.5 * lambda_ * (grad.pow(2).sum(dim=1)).mean()

    def update_normalization(self, *batches: torch.Tensor) -> None:
        """Update empirical normalization statistics from raw AMP histories.

        Args:
            *batches: Raw, unnormalized expert or policy history batches.
        """
        if self.amp_normalizer is None:
            return
        with torch.no_grad():
            for batch in batches:
                self.amp_normalizer.update(batch)
