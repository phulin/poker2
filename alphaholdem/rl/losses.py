from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict

try:
    from line_profiler import profile
except ImportError:  # pragma: no cover

    def profile(f):
        return f


@profile
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
    *,
    value_loss_type: str = "mse",
    huber_delta: float = 1.0,
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
        legal_masks: Legal action masks (B, num_actions) bool
        epsilon: PPO clip parameter (typically 0.2)
        delta1: Policy upper bound when advantage < 0 (typically 3.0)
        delta2, delta3: Value clipping bounds (dynamic based on chips)
        value_coef: Value loss coefficient
        entropy_coef: Entropy regularization coefficient
    """
    # Mask illegal actions
    masked_logits = torch.where(legal_masks, logits, -1e9)

    # Compute new log probabilities
    log_probs = F.log_softmax(masked_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Importance sampling ratio
    ratio = torch.exp(action_log_probs - log_probs_old)
    ppo_low = 1.0 - epsilon
    ppo_high = 1.0 + epsilon
    ppo_clip = torch.clamp(ratio, ppo_low, ppo_high)

    # Trinal-Clip policy:
    #  - For A >= 0: use standard PPO min surrogate (min(ratio, clip(r)))
    #  - For A < 0: clamp ratio into [1-ε, δ1]
    is_neg_adv = advantages < 0.0
    # A>=0 path
    r_pos = torch.minimum(ratio, ppo_clip)
    # A<0 path: clamp to [1-ε, δ1]
    r_neg = torch.clamp(ratio, min=ppo_low, max=delta1)
    r_tc = torch.where(is_neg_adv, r_neg, r_pos)

    # Policy loss
    policy_loss_vec = -(r_tc * advantages)
    policy_loss = policy_loss_vec.mean()

    # Value loss with clipping (as per AlphaHoldem paper)
    # We store delta2 as a negative lower bound (i.e., -chips_opponent/scale),
    # and delta3 as a positive upper bound (chips_self/scale), so clamp directly.
    clipped_returns = torch.clamp(returns, delta2, delta3)

    if value_loss_type == "huber":
        value_loss = F.smooth_l1_loss(values, clipped_returns, beta=huber_delta)
    else:
        value_loss = F.mse_loss(values, clipped_returns)

    # Entropy regularization
    probs = F.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    # Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    # total_loss = value_coef * value_loss

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
        "advantage_mean": advantages.mean(),
        "advantage_std": advantages.std(),
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
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
    }


def dual_clip_ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    legal_masks: torch.Tensor,
    epsilon: float,
    dual_clip: float,
    value_coef: float,
    entropy_coef: float,
) -> Dict[str, torch.Tensor]:
    """Dual-Clip PPO loss (Ye et al. 2020) with legal action masking.

    Policy:
      - For A>=0: use standard PPO min surrogate
      - For A<0: cap ratio by dual_clip (r <= dual_clip)

    Value: standard MSE to returns (no value clipping)
    """
    # Mask illegal actions
    masked_logits = logits.clone()
    masked_logits[legal_masks == 0] = -1e9

    # Log-probs and action log-probs
    log_probs = F.log_softmax(masked_logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Ratios
    ratio = torch.exp(action_log_probs - log_probs_old)
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    # Dual-clip policy surrogate
    surr1 = ratio * advantages
    surr2 = clipped * advantages
    surr_min = torch.min(surr1, surr2)
    ratio_dc = torch.clamp(ratio, max=dual_clip)
    surr_dc = ratio_dc * advantages
    surr = torch.where(advantages < 0.0, surr_dc, surr_min)

    policy_loss = -surr.mean()

    # Value loss (no clipping here)
    value_loss = F.mse_loss(values, returns)

    # Entropy
    probs = F.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
        "clipped_ratio_mean": clipped.mean(),
        "clipped_ratio_std": clipped.std(),
    }
