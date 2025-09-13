"""Utility functions for KL divergence computation."""

from __future__ import annotations

from typing import Optional
import torch


def compute_kl_divergence(policy1: torch.Tensor, policy2: torch.Tensor) -> float:
    """
    Compute KL divergence between two policy distributions.

    Args:
        policy1: First policy distribution (logits or probabilities)
        policy2: Second policy distribution (logits or probabilities)

    Returns:
        KL divergence value
    """
    # Ensure both are probabilities
    if policy1.dim() > 1:
        policy1 = torch.softmax(policy1, dim=-1)
    if policy2.dim() > 1:
        policy2 = torch.softmax(policy2, dim=-1)

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    policy1 = torch.clamp(policy1, min=eps)
    policy2 = torch.clamp(policy2, min=eps)

    # Compute KL divergence: KL(p1 || p2) = sum(p1 * log(p1 / p2))
    kl_div = torch.sum(policy1 * torch.log(policy1 / policy2))
    return kl_div.item()


def compute_kl_divergence_batch(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """
    Compute KL divergence between two batches of policy logits.

    Args:
        logits1: First batch of policy logits
        logits2: Second batch of policy logits

    Returns:
        Average KL divergence across the batch
    """
    batch_size = min(logits1.shape[0], logits2.shape[0])

    if batch_size == 0:
        return 0.0

    # Convert to probabilities
    probs1 = torch.softmax(logits1, dim=-1)
    probs2 = torch.softmax(logits2, dim=-1)

    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs1 = torch.clamp(probs1, min=eps)
    probs2 = torch.clamp(probs2, min=eps)

    # Compute KL divergence for each sample
    kl_divs = torch.sum(probs1 * torch.log(probs1 / probs2), dim=-1)

    # Return average KL divergence
    return kl_divs.mean().item()
