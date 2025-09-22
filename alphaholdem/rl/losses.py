from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F


from ..models.cnn_embedding_data import CNNEmbeddingData
from ..models.transformer.structured_embedding_data import StructuredEmbeddingData
from ..utils.ema import EMA
from ..utils.profiling import profile
from .vectorized_replay import BatchSample


@dataclass
class LossResult:
    """Dataclass for loss calculation results."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    ratio_mean: torch.Tensor
    ratio_std: torch.Tensor
    epsilon: float
    clipfrac: torch.Tensor
    # Optional fields for specific loss types
    clipped_ratio_mean: torch.Tensor = None
    clipped_ratio_std: torch.Tensor = None


class LossCalculator(ABC):
    """Abstract base class for loss calculators."""

    def __init__(
        self,
        epsilon: float,
        value_coef: float,
        entropy_coef: float,
        value_loss_type: str = "mse",
        huber_delta: float = 1.0,
    ):
        """
        Initialize the loss calculator with configuration parameters.

        Args:
            epsilon: PPO clip parameter (typically 0.2)
            value_coef: Value loss coefficient
            entropy_coef: Entropy regularization coefficient
            value_loss_type: Type of value loss ("mse" or "huber")
            huber_delta: Delta parameter for Huber loss
        """
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.value_loss_type = value_loss_type
        self.huber_delta = huber_delta

    @abstractmethod
    def compute_loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
    ) -> LossResult:
        """
        Compute the loss for the given inputs.

        Args:
            logits: Policy logits (B, num_actions)
            values: Value predictions (B,)
            batch: Batch sample containing actions, advantages, returns, etc.

        Returns:
            LossResult containing loss components and metrics
        """
        pass


class TrinalClipPPOLoss(LossCalculator):
    """
    Trinal-Clip PPO loss with policy and value clipping.

    According to the paper:
    - Policy loss: clip(ratio, clip(ratio, 1-ε, 1+ε), δ1) * advantages
    - Value loss: clip(returns, -δ2, δ3) - values
    """

    def __init__(
        self,
        epsilon: float,
        delta1: float,
        value_coef: float,
        entropy_coef: float,
        value_loss_type: str,
        huber_delta: float,
        target_kl: float,
        kl_ema: EMA,
        value_mean_ema: EMA,
        value_std_ema: EMA,
    ):
        """
        Initialize Trinal-Clip PPO loss calculator.

        Args:
            epsilon: PPO clip parameter (typically 0.2)
            delta1: Policy upper bound when advantage < 0 (typically 3.0)
            value_coef: Value loss coefficient
            entropy_coef: Entropy regularization coefficient
            value_loss_type: Type of value loss ("mse" or "huber")
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__(
            epsilon, value_coef, entropy_coef, value_loss_type, huber_delta
        )
        self.delta1 = delta1
        self.target_kl = target_kl
        self.kl_ema = kl_ema
        self.value_mean_ema = value_mean_ema
        self.value_std_ema = value_std_ema

    def compute_loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
    ) -> LossResult:
        """
        Compute Trinal-Clip PPO loss.

        Args:
            logits: Policy logits (B, num_actions)
            values: Value predictions (B,)
            batch: Batch sample containing actions, advantages, returns, etc.
            kl_divergence: KL divergence between old and new policy logits

        Returns:
            LossResult containing loss components and metrics
        """

        actions = batch.action_indices
        advantages = batch.advantages
        returns = batch.returns
        delta2 = batch.delta2
        delta3 = batch.delta3
        legal_masks = batch.legal_masks

        # Mask illegal actions
        masked_logits = torch.where(legal_masks, logits, -1e9)

        # Compute new log probabilities
        log_probs = F.log_softmax(masked_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        epsilon = self.epsilon
        if self.kl_ema.initialized:
            epsilon = epsilon * (self.target_kl / (self.kl_ema.value + 1e-8))
            epsilon = min(max(epsilon, self.epsilon / 2), self.epsilon * 2)

        # Importance sampling ratio
        ratio = torch.exp(action_log_probs - batch.selected_log_probs)
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
        r_neg = torch.clamp(ratio, min=ppo_low, max=self.delta1)
        r_tc = torch.where(is_neg_adv, r_neg, r_pos)
        clipfrac = (torch.abs(r_tc - ratio) > 1e-8).float().mean()

        # Policy loss
        policy_loss_vec = -(r_tc * advantages)
        policy_loss = policy_loss_vec.mean()

        # Value loss with clipping (as per AlphaHoldem paper)
        # We store delta2 as a negative lower bound (i.e., -chips_opponent/scale),
        # and delta3 as a positive upper bound (chips_self/scale), so clamp directly.
        clipped_returns = torch.clamp(returns, delta2, delta3)

        if self.value_loss_type == "huber":
            value_loss = F.smooth_l1_loss(
                values, clipped_returns, beta=self.huber_delta
            )
        else:
            value_loss = F.mse_loss(values, clipped_returns)

        # Entropy regularization
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        )

        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            ratio_mean=ratio.mean(),
            ratio_std=ratio.std(),
            epsilon=epsilon,
            clipfrac=clipfrac,
        )


class StandardPPOLoss(LossCalculator):
    """Standard PPO loss for comparison."""

    def compute_loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
    ) -> LossResult:
        """Compute standard PPO loss."""
        actions = batch.action_indices
        advantages = batch.advantages
        returns = batch.returns
        legal_masks = batch.embedding_data.legal_masks

        # Mask illegal actions
        masked_logits = torch.where(legal_masks, logits, -1e9)

        # Compute new log probabilities
        log_probs = F.log_softmax(masked_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute ratio
        ratio = torch.exp(action_log_probs - batch.selected_log_probs)

        # Standard PPO policy loss
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy regularization
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        )

        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            ratio_mean=ratio.mean(),
            ratio_std=ratio.std(),
            clipfrac=torch.tensor(
                0.0, device=total_loss.device
            ),  # No clipping in standard PPO
        )


class DualClipPPOLoss(LossCalculator):
    """Dual-Clip PPO loss (Ye et al. 2020) with legal action masking."""

    def __init__(
        self,
        epsilon: float,
        dual_clip: float,
        value_coef: float,
        entropy_coef: float,
        value_loss_type: str = "mse",
        huber_delta: float = 1.0,
    ):
        """
        Initialize Dual-Clip PPO loss calculator.

        Args:
            epsilon: PPO clip parameter
            dual_clip: Dual clip parameter for negative advantages
            value_coef: Value loss coefficient
            entropy_coef: Entropy regularization coefficient
            value_loss_type: Type of value loss ("mse" or "huber")
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__(
            epsilon, value_coef, entropy_coef, value_loss_type, huber_delta
        )
        self.dual_clip = dual_clip

    def compute_loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
    ) -> LossResult:
        """
        Compute Dual-Clip PPO loss.

        Policy:
          - For A>=0: use standard PPO min surrogate
          - For A<0: cap ratio by dual_clip (r <= dual_clip)

        Value: standard MSE to returns (no value clipping)
        """
        actions = batch.action_indices
        advantages = batch.advantages
        returns = batch.returns
        legal_masks = batch.embedding_data.legal_masks

        # Mask illegal actions
        masked_logits = torch.where(legal_masks, logits, -1e9)

        # Log-probs and action log-probs
        log_probs = F.log_softmax(masked_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Ratios
        ratio = torch.exp(action_log_probs - batch.selected_log_probs)
        clipped = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

        # Dual-clip policy surrogate
        surr1 = ratio * advantages
        surr2 = clipped * advantages
        surr_min = torch.min(surr1, surr2)
        ratio_dc = torch.clamp(ratio, max=self.dual_clip)
        surr_dc = ratio_dc * advantages
        surr = torch.where(advantages < 0.0, surr_dc, surr_min)

        policy_loss = -surr.mean()

        # Value loss (no clipping here)
        value_loss = F.mse_loss(values, returns)

        # Entropy
        probs = F.softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        total_loss = (
            policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        )

        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            ratio_mean=ratio.mean(),
            ratio_std=ratio.std(),
            clipfrac=torch.tensor(
                0.0, device=total_loss.device
            ),  # No value clipping in dual-clip PPO
            clipped_ratio_mean=clipped.mean(),
            clipped_ratio_std=clipped.std(),
        )
