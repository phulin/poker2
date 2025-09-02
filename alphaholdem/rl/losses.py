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
    epsilon: float = 0.2,
    delta1: float = 3.0,
    delta2: float = 0.0,
    delta3: float = 0.0,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Trinal-Clip PPO loss with policy and value clipping.
    
    Args:
        logits: Policy logits (B, num_actions)
        values: Value predictions (B,)
        actions: Action indices (B,)
        log_probs_old: Old log probabilities (B,)
        advantages: GAE advantages (B,)
        returns: GAE returns (B,)
        legal_masks: Legal action masks (B, num_actions)
        epsilon: PPO clip parameter
        delta1: Policy upper bound when advantage < 0
        delta2, delta3: Value clipping bounds
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
    # Standard PPO clip
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    # Additional upper bound when advantage < 0
    policy_loss1 = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    # Trinal-Clip: additional upper bound for negative advantages
    negative_adv_mask = advantages < 0
    if negative_adv_mask.any():
        upper_clipped = torch.clamp(ratio, 0, delta1)
        policy_loss2 = -torch.min(ratio * advantages, upper_clipped * advantages)
        policy_loss = torch.where(negative_adv_mask, policy_loss2, policy_loss1)
    else:
        policy_loss = policy_loss1
    
    # Value loss with clipping
    if delta2 != 0.0 or delta3 != 0.0:
        clipped_returns = torch.clamp(returns, delta2, delta3)
    else:
        clipped_returns = returns
    
    value_loss = F.mse_loss(values, clipped_returns)
    
    # Entropy regularization
    probs = F.softmax(masked_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    # Total loss
    total_loss = (
        policy_loss.mean() +
        value_coef * value_loss -
        entropy_coef * entropy
    )
    
    return {
        'total_loss': total_loss,
        'policy_loss': policy_loss.mean(),
        'value_loss': value_loss,
        'entropy': entropy,
        'ratio_mean': ratio.mean(),
        'ratio_std': ratio.std(),
        'advantage_mean': advantages.mean(),
        'advantage_std': advantages.std(),
    }
