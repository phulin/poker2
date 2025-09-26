"""PopArt normalization for value function learning."""

from __future__ import annotations

import torch

from alphaholdem.utils.ema import EMA


class PopArtNormalizer:
    """
    PopArt normalization following the correct algorithm:
    1. Freeze stats (μf, σf) during training
    2. Normalize targets with frozen stats
    3. Update EMAs in background
    4. Rescale weights after training epoch
    """

    def __init__(
        self,
        eps: float = 1e-8,
    ):
        self.mean_ema = EMA()  # EMA over batch means of targets
        self.var_ema = EMA()  # EMA over batch variances of targets
        self.eps = float(eps)

        # Frozen stats for current training epoch
        self.frozen_mu = 0.0
        self.frozen_sigma = 1.0
        self.stats_frozen = False

    def freeze_stats(self) -> None:
        """Freeze current EMA stats for use during training epoch."""
        if self.mean_ema.initialized:
            self.frozen_mu = self.mean_ema.value
            self.frozen_sigma = (self.var_ema.value + self.eps) ** 0.5
        else:
            # First epoch: use identity normalization
            self.frozen_mu = 0.0
            self.frozen_sigma = 1.0

    @torch.no_grad()
    def update_stats(self, values: torch.Tensor) -> None:
        """
        Update EMAs with batch statistics (only when not frozen).

        Args:
            values: Batch of target values to update statistics with
        """

        # Batch stats (population variance for stability with small batches)
        batch_mean = values.mean().item()
        batch_var = values.var(unbiased=False).item()

        self.mean_ema.update(batch_mean)
        self.var_ema.update(batch_var)

    def get_frozen_stats(self) -> tuple[float, float]:
        """
        Get frozen normalization statistics for current epoch.

        Returns:
            Tuple of (mu_frozen, sigma_frozen) as Python floats
        """
        return self.frozen_mu, self.frozen_sigma

    def get_current_stats(self) -> tuple[float, float]:
        """
        Get current EMA statistics (for final rescaling).

        Returns:
            Tuple of (mu_current, sigma_current) as Python floats
        """
        if not self.mean_ema.initialized:
            return 0.0, 1.0
        mu = self.mean_ema.value
        sigma = (self.var_ema.value + self.eps) ** 0.5
        return mu, sigma

    def compute_rescaling_adjustments(self) -> tuple[float, float]:
        """
        Compute weight and bias adjustments for final rescaling.

        Returns:
            Tuple of (weight_scale, bias_adjustment) for final linear layer rescaling
        """

        # Get current EMA stats (after background updates)
        mu_new, sigma_new = self.get_current_stats()

        # Compute rescaling: W' = (σf/σnew) * W, b' = (σf/σnew) * b + (μf - μnew)/σnew
        weight_scale = self.frozen_sigma / max(sigma_new, 1e-6)
        bias_adjustment = (self.frozen_mu - mu_new) / max(sigma_new, 1e-6)

        return weight_scale, bias_adjustment

    def denormalize_value(self, normalized_value: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized value back to original scale: V = σf * v̂ + μf

        Args:
            normalized_value: Value in normalized space

        Returns:
            Value in original scale
        """

        return self.frozen_sigma * normalized_value + self.frozen_mu
