from __future__ import annotations

from typing import Tuple, Union

import torch


class RunningMeanStd:
    """Tracks running mean/variance for streaming data."""

    def __init__(
        self,
        epsilon: float = 1.0e-4,
        shape: Tuple[int, ...] = (),
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        batch = arr.to(self.device, dtype=torch.float32)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(
            batch.shape[0], dtype=torch.float32, device=self.device
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(total_count)
