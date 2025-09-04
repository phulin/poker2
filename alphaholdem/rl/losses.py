from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict


def trinal_clip_ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    legal_masks: torch.Tensor,
    epsilon: float,
    delta1: float,
    delta2: torch.Tensor,
    delta3: torch.Tensor,
    value_coef: float,
    entropy_coef: float,
) -> Dict[str, torch.Tensor]:
    """
    Trinal-Clip PPO loss with policy and value clipping.

    According to the paper:
    - Policy loss: clip(ratio, clip(ratio, 1-ε, 1+ε), δ1) * advantages
    - Value loss: clip(returns, -δ2, δ3) - values

    Args:
        logits: Policy logits (B, num_actions)
        values: Value predictions (B,)
        actions: Action indices (B,)
        log_probs_old: Old log probabilities (B,)
        advantages: GAE advantages (B,)
        returns: GAE returns (B,)
        legal_masks: Legal action masks (B, num_actions)
        epsilon: PPO clip parameter (typically 0.2)
        delta1: Policy upper bound when advantage < 0 (typically 3.0)
        delta2, delta3: Value clipping bounds (dynamic based on chips)
        value_coef: Value loss coefficient
        entropy_coef: Entropy regularization coefficient
    """
    # Mask illegal actions
    masked_logits = logits.clone()
    masked_logits[legal_masks == 0] = -1e9

    # Compute new log probabilities
    log_probs = F.log_softmax(masked_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute ratio
    ratio = torch.exp(action_log_probs - log_probs_old)

    # Policy loss with Trinal-Clip
    # First clip: standard PPO clip (1-ε, 1+ε)
    first_clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # Second clip: additional upper bound δ1 when advantage < 0
    # This is the "Trinal-Clip" part from the paper
    final_clipped = torch.clamp(ratio, first_clipped, torch.full_like(ratio, delta1))

    # Policy loss: min(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -final_clipped * advantages

    # Value loss with clipping (as per AlphaHoldem paper)
    clipped_returns = torch.clamp(returns, delta2, delta3)

    value_loss = F.mse_loss(values, clipped_returns)

    # Entropy regularization
    probs = F.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    # Total loss
    total_loss = policy_loss.mean() + value_coef * value_loss - entropy_coef * entropy

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss.mean(),
        "value_loss": value_loss,
        "entropy": entropy,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
        "advantage_mean": advantages.mean(),
        "advantage_std": advantages.std(),
        "clipped_ratio_mean": final_clipped.mean(),
        "clipped_ratio_std": final_clipped.std(),
    }


def standard_ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    legal_masks: torch.Tensor,
    epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """Standard PPO loss for comparison."""
    # Mask illegal actions
    masked_logits = logits.clone()
    masked_logits[legal_masks == 0] = -1e9

    # Compute new log probabilities
    log_probs = F.log_softmax(masked_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute ratio
    ratio = torch.exp(action_log_probs - log_probs_old)

    # Standard PPO policy loss
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

    # Value loss
    value_loss = F.mse_loss(values, returns)

    # Entropy regularization
    probs = F.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    # Total loss
    total_loss = policy_loss.mean() + value_coef * value_loss - entropy_coef * entropy

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss.mean(),
        "value_loss": value_loss,
        "entropy": entropy,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
    }
