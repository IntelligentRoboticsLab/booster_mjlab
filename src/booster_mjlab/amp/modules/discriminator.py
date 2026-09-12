import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F

from booster_mjlab.amp.modules.losses import make_loss_function
from rsl_rl.modules import EmpiricalNormalization


class Discriminator(nn.Module):
    """Small MLP trained to distinguish expert and policy motion histories.

    Args:
        input_dim: Dimension of a flattened K-frame AMP observation history.
        hidden_layer_sizes: List of hidden layer sizes.
        reward_scale: Scale factor for computed rewards.
        reward_clamp_epsilon: Epsilon for reward clamping.
        device: Device to run the model on.
        loss_type: Type of loss function ('hinge', 'bce', 'wasserstein').
        eta_wgan: Scaling factor for Wasserstein loss.
        empirical_normalization: Whether to normalize AMP observations internally.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        reward_scale: float,
        reward_clamp_epsilon: float = 1.0e-4,
        device: torch.device = torch.device("cpu"),
        loss_type: str = "hinge",
        eta_wgan: float = 1.0,
        use_minibatch_std: bool = True,
        empirical_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.reward_scale = reward_scale
        self.reward_clamp_epsilon = reward_clamp_epsilon
        self.empirical_normalization = empirical_normalization
        self.use_minibatch_std = use_minibatch_std

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
        final_in_dim = hidden_layer_sizes[-1] + (1 if use_minibatch_std else 0)
        self.linear = nn.Linear(final_in_dim, 1).to(device)
        self.loss_type = loss_type.lower()
        loss_kwargs = {"eta": eta_wgan} if self.loss_type == "wasserstein" else {}
        self.eta_wgan = eta_wgan
        self.loss_fn = make_loss_function(self.loss_type, **loss_kwargs)

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

    def _forward_normalized(
        self, x: torch.Tensor, *, detach_minibatch_std: bool = False
    ) -> torch.Tensor:
        """Run the discriminator network on already-normalized histories."""
        h = self.trunk(x)
        if self.use_minibatch_std:
            if detach_minibatch_std:
                with torch.no_grad():
                    s = self._minibatch_std_scalar(h)
            else:
                s = self._minibatch_std_scalar(h)
            h = torch.cat([h, s], dim=-1)
        return self.linear(h)

    def _minibatch_std_scalar(self, h: torch.Tensor) -> torch.Tensor:
        """Mean over feature-wise std across the batch; shape (B,1)."""
        if h.shape[0] <= 1:
            return h.new_zeros((h.shape[0], 1))
        s = h.float().std(dim=0, unbiased=False).mean()
        return s.expand(h.shape[0], 1).to(h.dtype)

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
                logits = torch.tanh(self.eta_wgan * logits)
                reward = torch.exp(logits)
            else:
                # softplus(x) = log(1 + exp(x)) = -log(1 - sigmoid(x))
                reward = F.softplus(logits)
            return (self.reward_scale * reward).squeeze(-1)

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
        scores = self._forward_normalized(data, detach_minibatch_std=True)

        if self.loss_type == "wasserstein":
            # WGAN-GP: penalty on tanh-transformed output
            scores = torch.tanh(self.eta_wgan * scores)
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
