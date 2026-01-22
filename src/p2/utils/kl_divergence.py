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


def compute_kl_divergence_batch(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    legal_masks: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute KL divergence between two batches of policy logits, with optional legal action masking.

    Args:
        logits1: First (old) batch of policy logits [B, A]
        logits2: Second (new) batch of policy logits [B, A]
        legal_masks: Optional boolean mask of shape [B, A] indicating legal actions (True=legal, False=illegal)

    Returns:
        Average KL divergence KL(logits1 || logits2)
    """
    assert logits1.shape == logits2.shape

    # If legal_masks is provided, mask out illegal actions by setting logits to a large negative value
    if legal_masks is not None:
        logits1 = torch.where(legal_masks, logits1, -1e9)
        logits2 = torch.where(legal_masks, logits2, -1e9)

    # Convert to probabilities
    log_probs1 = torch.log_softmax(logits1, dim=-1)
    log_probs2 = torch.log_softmax(logits2, dim=-1)
    probs1 = log_probs1.exp()

    # Compute KL divergence for each sample
    kl_divs = torch.sum(probs1 * (log_probs1 - log_probs2), dim=-1)

    # Return average KL divergence
    return kl_divs.mean().item()
