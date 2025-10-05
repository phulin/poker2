"""Quantile calculator for efficient batch quantile computation."""

import torch


class QuantileCalculator:
    """Helper class to efficiently compute quantiles over batched tensors."""

    def __init__(self, device: torch.device):
        self.device = device
        self.tensors: list[torch.Tensor] = []

    def log(self, tensor: torch.Tensor) -> None:
        """Log a tensor to be included in quantile computation."""
        self.tensors.append(tensor)

    def compute(self, n: int) -> list[float]:
        """
        Compute n quantiles of all logged tensors.

        Args:
            n: Number of quantiles to compute

        Returns:
            Tensor of shape (n,) containing the quantiles
        """
        if not self.tensors:
            return torch.tensor([], device=self.device)

        all_values = torch.cat(self.tensors, dim=0)
        quantiles = torch.quantile(
            all_values, torch.arange(0, n + 1, device=self.device) / n
        )
        return quantiles.tolist()

    def compute_wandb(self, n: int) -> dict[str, float]:
        """
        Compute n quantiles of all logged tensors and return a dictionary with the quantiles as values.

        Args:
            n: Number of quantiles to compute

        Returns:
            Dictionary with the quantiles as values
        """
        return {str(i): q for i, q in enumerate(self.compute(n))}

    def reset(self) -> None:
        """Clear all logged tensors."""
        self.tensors.clear()
