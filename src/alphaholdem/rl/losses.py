from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from alphaholdem.rl.popart_normalizer import PopArtNormalizer
from alphaholdem.rl.vectorized_replay import BatchSample
from alphaholdem.utils.ema import EMA


@dataclass
class LossResult:
    """Dataclass for loss calculation results."""

    total_loss: torch.Tensor
    policy_loss: float
    value_loss: float
    entropy: float
    ratio_mean: float
    ratio_std: float
    epsilon: float
    clipfrac: float
    ppo_clipfrac: float
    return_clipfrac: float
    # Optional fields for specific loss types
    clipped_ratio_mean: Optional[float] = None
    clipped_ratio_std: Optional[float] = None


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


class TrinalClipPPOLoss(LossCalculator):
    """
    Trinal-Clip PPO loss with policy and value clipping.

    According to the paper:
    - Policy loss: clip(ratio, clip(ratio, 1-ε, 1+ε), δ1) * advantages
    - Value loss: clip(returns, -δ2, δ3) - values
    """

    def __init__(
        self,
        popart_normalizer: PopArtNormalizer,
        epsilon: float,
        delta1: float,
        value_coef: float,
        entropy_coef: float,
        value_loss_type: str,
        huber_delta: float,
        target_kl: float,
        kl_ema: EMA,
    ):
        """
        Initialize Trinal-Clip PPO loss calculator.

        Args:
            popart_normalizer: PopArtNormalizer instance for value normalization
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
        self.popart = popart_normalizer

    def compute_loss(
        self,
        log_probs: torch.Tensor,
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

        # Compute new log probabilities
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
        ppo_clipfrac = (torch.abs(ppo_clip - ratio) > 1e-8).float().mean()

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

        # Compute return clipping fraction
        return_clipfrac = (torch.abs(clipped_returns - returns) > 1e-8).float().mean()

        # Use frozen stats for normalization during training
        mu_frozen, sigma_frozen = self.popart.get_frozen_stats()
        targets_n = (clipped_returns - mu_frozen) / (sigma_frozen + 1e-8)
        if self.value_loss_type == "huber":
            value_loss = F.smooth_l1_loss(values, targets_n, beta=self.huber_delta)
        else:
            value_loss = F.mse_loss(values, targets_n)

        # Entropy regularization
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = (
            policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        )

        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            ratio_mean=ratio.mean().item(),
            ratio_std=ratio.std().item(),
            epsilon=epsilon,
            clipfrac=clipfrac.item(),
            ppo_clipfrac=ppo_clipfrac.item(),
            return_clipfrac=return_clipfrac.item(),
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

        # Compute PPO clipping fraction
        ppo_clipfrac = (torch.abs(clipped_ratio - ratio) > 1e-8).float().mean()

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
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            ratio_mean=ratio.mean().item(),
            ratio_std=ratio.std().item(),
            epsilon=self.epsilon,
            clipfrac=0.0,
            ppo_clipfrac=ppo_clipfrac.item(),
            return_clipfrac=0.0,
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

        # Compute PPO clipping fraction
        ppo_clipfrac = (torch.abs(clipped - ratio) > 1e-8).float().mean()

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
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            ratio_mean=ratio.mean().item(),
            ratio_std=ratio.std().item(),
            epsilon=self.epsilon,
            clipfrac=0.0,
            ppo_clipfrac=ppo_clipfrac.item(),
            return_clipfrac=0.0,
            clipped_ratio_mean=clipped.mean().item(),
            clipped_ratio_std=clipped.std().item(),
        )


# --- Add this to losses.py -----------------------------------------------
class KLPolicyPPOLoss(LossCalculator):
    """
    PPO with KL penalty instead of ratio clipping.

    Policy loss:
      L = -E[ log π_theta(a|s) * A ] + beta * KL(π_old || π_theta)
    plus value loss and entropy bonus.

    Requirements:
      - batch.old_logits: masked the same way and computed with the frozen π_old
      - batch.legal_masks: boolean mask for legal actions
      - batch.action_indices, batch.advantages, batch.returns
    """

    def __init__(
        self,
        popart_normalizer: PopArtNormalizer,
        value_coef: float,
        entropy_coef: float,
        value_loss_type: str = "huber",
        huber_delta: float = 1.0,
        target_kl: float = 0.015,
        beta_init: float = 1.0,
        beta_min: float = 1e-4,
        beta_max: float = 1e4,
        kl_type: str = "forward",  # "forward" (KL(p_old||p_new)) or "reverse"
        kl_ema: EMA | None = None,  # optional, for logging/smoothing
    ):
        # epsilon unused here; keep API compatibility by passing a dummy (e.g., 0.2)
        super().__init__(
            epsilon=0.2,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            value_loss_type=value_loss_type,
            huber_delta=huber_delta,
        )
        self.beta = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.target_kl = target_kl
        self.kl_type = kl_type
        self.kl_ema = kl_ema
        self.popart = popart_normalizer

    def _adapt_beta(self, kl_value: float) -> None:
        # Simple proportional controller; keep beta bounded.
        with torch.no_grad():
            target = self.target_kl
            # gentle multiplicative adjustments
            if kl_value > 1.5 * target:
                self.beta = min(self.beta * 2.0, self.beta_max)
            elif kl_value < 0.5 * target:
                self.beta = max(self.beta * 0.5, self.beta_min)
            # optional EMA logging
            if self.kl_ema is not None:
                self.kl_ema.update(float(kl_value))

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
    ) -> LossResult:
        actions = batch.action_indices
        advantages = batch.advantages
        returns = batch.returns
        delta2 = batch.delta2
        delta3 = batch.delta3

        # --- Mask illegal actions on BOTH distributions
        legal_masks = batch.legal_masks.bool()
        masked_log_p_new = torch.where(
            legal_masks, log_probs, torch.full_like(log_probs, -1e9)
        )
        masked_old_logits = torch.where(
            legal_masks,
            batch.computed_logits,
            torch.full_like(batch.computed_logits, -1e9),
        )

        # --- Log-probs & distributions
        log_p_new = masked_log_p_new
        log_p_old = torch.log_softmax(masked_old_logits, dim=-1)
        p_new = log_p_new.exp()
        p_old = log_p_old.exp()
        new_logp_a = log_p_new.gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Policy gradient term (no ratio/clipping)
        pg_loss = -(new_logp_a * advantages).mean()

        # --- KL penalty
        if self.kl_type == "reverse":
            # KL(new || old)
            kl_tensor = (p_new * (log_p_new - log_p_old)).sum(dim=-1).mean()
        else:
            # KL(old || new)
            kl_tensor = (p_old * (log_p_old - log_p_new)).sum(dim=-1).mean()
        policy_loss = pg_loss + self.beta * kl_tensor

        # --- Value loss (Huber or MSE)
        clipped_returns = returns  # torch.clamp(returns, delta2, delta3)
        return_clipfrac = (torch.abs(clipped_returns - returns) > 1e-8).float().mean()
        mu_frozen, sigma_frozen = self.popart.get_frozen_stats()
        targets_n = (clipped_returns - mu_frozen) / (sigma_frozen + 1e-8)
        if self.value_loss_type == "huber":
            value_loss = F.smooth_l1_loss(values, targets_n, beta=self.huber_delta)
        else:
            value_loss = F.mse_loss(values, targets_n)

        # --- Entropy bonus of the *new* policy
        entropy = -(p_new * log_p_new).sum(dim=-1).mean()

        # --- Total
        total_loss = (
            policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        )

        # --- Adapt beta toward target KL (after computing loss)
        kl_value = float(kl_tensor.detach().item())
        self._adapt_beta(kl_value)

        # For metrics, reuse fields even if not strictly applicable
        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            ratio_mean=1.0,
            ratio_std=0.0,
            epsilon=0.0,
            clipfrac=0.0,  # no clipping in KL-PPO
            ppo_clipfrac=0.0,  # no PPO clipping in KL-PPO
            return_clipfrac=return_clipfrac.item(),
        )


# --- end of addition ------------------------------------------------------
