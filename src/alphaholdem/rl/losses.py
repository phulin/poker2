from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphaholdem.core.structured_config import KLType, PPOClipping, ValueLossType
from alphaholdem.rl.exponential_controller import ExponentialController
from alphaholdem.rl.popart_normalizer import PopArtNormalizer
from alphaholdem.rl.rebel_replay import RebelBatch
from alphaholdem.rl.vectorized_replay import BatchSample
from alphaholdem.search.cfr_manager import CFRManager
from alphaholdem.rl.rebel_replay import RebelBatch
from alphaholdem.utils.ema import EMA
from alphaholdem.utils.model_utils import compute_masked_logits


@dataclass
class LossResult:
    """Dataclass for loss calculation results."""

    total_loss: torch.Tensor
    policy_loss: float
    value_loss_tensor: torch.Tensor
    entropy: float
    ratio_mean: float
    ratio_std: float
    epsilon: float
    clipfrac: float
    ppo_clipfrac: float
    return_clipfrac: float
    penalty_kl: Optional[float] = None
    forward_kl: Optional[float] = None
    reverse_kl: Optional[float] = None
    cfr_kl: Optional[float] = None
    # Optional fields for specific loss types
    clipped_ratio_mean: Optional[float] = None
    clipped_ratio_std: Optional[float] = None
    value_loss: float = field(init=False)

    def __post_init__(self) -> None:
        # Provide a scalar view for logging/tests while keeping tensor for backprop.
        self.value_loss = float(self.value_loss_tensor.detach().item())


class LossCalculator(ABC):
    """Abstract base class for loss calculators."""

    def __init__(
        self,
        epsilon: float,
        value_coef: float,
        entropy_coef: float,
        value_loss_type: ValueLossType = ValueLossType.mse,
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
        value_quantiles: Optional[torch.Tensor] = None,
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
        value_loss_type: ValueLossType,
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
        logits: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
        value_quantiles: Optional[torch.Tensor] = None,
    ) -> LossResult:
        """
        Compute Trinal-Clip PPO loss.

        Args:
            logits: Policy logits (B, num_actions)
            values: Value predictions (B,)
            batch: Batch sample containing actions, advantages, returns, etc.

        Returns:
            LossResult containing loss components and metrics
        """

        actions = batch.action_indices
        advantages = batch.advantages
        returns = batch.returns
        delta2 = batch.delta2
        delta3 = batch.delta3

        # Mask illegal actions then compute log probabilities
        masked_logits = compute_masked_logits(logits, batch.legal_masks)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        epsilon = self.epsilon
        if self.kl_ema.initialized:
            epsilon = epsilon * (self.target_kl / (self.kl_ema.value + 1e-8))
            epsilon = min(max(epsilon, self.epsilon / 2), self.epsilon * 2)

        # Importance sampling ratio - selected_log_probs computed with frozen model
        ratio = torch.exp(action_log_probs - batch.frozen_selected_log_probs)
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
        if self.value_loss_type == ValueLossType.huber:
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
            value_loss_tensor=value_loss,
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
        value_quantiles: Optional[torch.Tensor] = None,
    ) -> LossResult:
        """Compute standard PPO loss."""
        actions = batch.action_indices
        advantages = batch.advantages
        returns = batch.returns
        legal_masks = batch.embedding_data.legal_masks

        # Mask illegal actions
        masked_logits = compute_masked_logits(logits, legal_masks)

        # Compute new log probabilities
        log_probs = F.log_softmax(masked_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute ratio
        ratio = torch.exp(action_log_probs - batch.frozen_selected_log_probs)

        # Standard PPO policy loss
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

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
            value_loss_tensor=value_loss,
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
        value_loss_type: ValueLossType = ValueLossType.mse,
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
        masked_logits = compute_masked_logits(logits, legal_masks)

        # Log-probs and action log-probs
        log_probs = F.log_softmax(masked_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Ratios
        ratio = torch.exp(action_log_probs - batch.frozen_selected_log_probs)
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
            value_loss_tensor=value_loss,
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
    """PPO variant that penalizes KL divergence instead of clipping ratios."""

    def __init__(
        self,
        popart_normalizer: Optional[PopArtNormalizer],
        beta_controller: ExponentialController,
        value_coef: float,
        entropy_coef: float,
        value_loss_type: ValueLossType = ValueLossType.huber,
        clipping: PPOClipping = PPOClipping.dual,
        return_clipping: bool = True,
        epsilon: float = 0.2,
        dual_clip: float = 3.0,
        huber_delta: float = 1.0,
        kl_type: KLType = KLType.reverse,
        quantile_kappa: float = 1.0,
        num_quantiles: Optional[int] = None,
    ):
        super().__init__(
            epsilon=0.2,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            value_loss_type=value_loss_type,
            huber_delta=huber_delta,
        )
        self.popart = popart_normalizer
        self.beta_controller = beta_controller
        self.clipping = clipping
        self.dual_clip = dual_clip
        self.return_clipping = return_clipping
        self.kl_type = kl_type
        self.quantile_kappa = quantile_kappa
        self.num_quantiles = num_quantiles

    def compute_loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
        value_quantiles: Optional[torch.Tensor] = None,
    ) -> LossResult:
        actions = batch.action_indices
        advantages = batch.advantages
        returns = batch.returns
        delta2 = batch.delta2
        delta3 = batch.delta3
        # use frozen log probs for importance ratio
        log_p_old_a = batch.frozen_selected_log_probs
        # use step log probs for KL penalty
        log_p_step = batch.step_all_log_probs

        # --- Mask illegal actions
        legal_masks = batch.legal_masks.bool()
        masked_new_logits = compute_masked_logits(logits, legal_masks)

        # --- Log-probs & distributions
        log_p_new = torch.log_softmax(masked_new_logits, dim=-1)
        log_p_new_a = log_p_new.gather(1, actions.unsqueeze(1)).squeeze(1)
        p_new = log_p_new.exp()

        # --- Policy gradient term with importance ratio
        # clamp for numerical stability
        ratio = torch.exp(torch.clamp(log_p_new_a - log_p_old_a, -20.0, 20.0))
        ratio_unclipped = ratio
        if self.clipping == PPOClipping.single or self.clipping == PPOClipping.dual:
            ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)

        if self.clipping == PPOClipping.single:
            product = torch.min(ratio_unclipped * advantages, ratio * advantages)
        elif self.clipping == PPOClipping.dual:
            product = torch.min(
                ratio_unclipped * advantages,
                ratio * advantages,
            )
            product = torch.where(
                advantages < 0.0,
                torch.max(self.dual_clip * advantages, product),
                product,
            )
        else:
            product = ratio * advantages
        policy_loss = -product.mean()

        if policy_loss.detach().abs().item() > 50:
            print("Policy loss is too high", policy_loss.detach().abs().item())
            contrib = (ratio * advantages).abs()
            topk = torch.topk(contrib, k=8).indices
            print("Top offenders:")
            for i in topk:
                print(
                    f"Index: {i.item()}, Ratio: {ratio[i].item():.4f}, Advantage: {advantages[i].item():.4f}, LogProbDiff: {(log_p_new_a[i] - log_p_old_a[i]).item():.4f}"
                )
                print("old log probs", batch.frozen_all_log_probs[i].cpu().tolist())
                print("new log probs", log_p_new[i].cpu().tolist())

        # --- KL penalty
        # KL(old || new)
        forward_kl = (log_p_step.exp() * (log_p_step - log_p_new)).sum(dim=-1).mean()
        # KL(new || old)
        reverse_kl = (p_new * (log_p_new - log_p_step)).sum(dim=-1).mean()
        penalty_kl = (
            torch.zeros_like(forward_kl)
            if self.kl_type == KLType.none
            else forward_kl if self.kl_type == KLType.forward else reverse_kl
        )

        # --- Value loss
        if self.return_clipping:
            clipped_returns = torch.clamp(returns, delta2, delta3)
        else:
            clipped_returns = returns
        return_clipfrac = (torch.abs(clipped_returns - returns) > 1e-8).float().mean()
        if self.value_loss_type == ValueLossType.quantile:
            if value_quantiles is None:
                raise ValueError(
                    "value_quantiles must be provided for quantile value loss"
                )
            if not self.num_quantiles:
                raise ValueError("num_quantiles must be set for quantile value loss")
            targets = clipped_returns.unsqueeze(-1)
            diff = targets - value_quantiles
            abs_diff = diff.abs()
            if self.quantile_kappa > 0:
                kappa = self.quantile_kappa
                huber = (
                    torch.where(
                        abs_diff <= kappa,
                        0.5 * diff.pow(2),
                        kappa * (abs_diff - 0.5 * kappa),
                    )
                    / kappa
                )
            else:
                huber = abs_diff
            taus = (
                torch.arange(
                    self.num_quantiles,
                    device=value_quantiles.device,
                    dtype=value_quantiles.dtype,
                )
                + 0.5
            ) / self.num_quantiles
            taus = taus.view(1, -1)
            indicator = (diff.detach() < 0).float()
            quantile_loss = torch.abs(taus - indicator) * huber
            value_loss = quantile_loss.sum(dim=-1).mean() / self.num_quantiles
        else:
            if self.popart is None:
                raise ValueError("PopArt normalizer is required for non-quantile loss")
            mu_frozen, sigma_frozen = self.popart.get_frozen_stats()
            targets_n = (clipped_returns - mu_frozen) / (sigma_frozen + 1e-8)
            if self.value_loss_type == ValueLossType.huber:
                value_loss = F.smooth_l1_loss(values, targets_n, beta=self.huber_delta)
            else:
                value_loss = F.mse_loss(values, targets_n)

        # --- Entropy bonus of the *new* policy
        entropy = -(p_new * log_p_new).sum(dim=-1).mean()

        # --- Total
        total_loss = (
            policy_loss
            + self.beta_controller.current_value * penalty_kl
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        with torch.no_grad():
            ppo_clipfrac = (torch.abs(ratio_unclipped - ratio) > 1e-8).float().mean()
            clipfrac = (torch.abs(product - ratio * advantages) > 1e-8).float().mean()

        # For metrics, reuse fields even if not strictly applicable
        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss.item(),
            value_loss_tensor=value_loss,
            entropy=entropy.item(),
            penalty_kl=penalty_kl.item(),
            forward_kl=forward_kl.item(),
            reverse_kl=reverse_kl.item(),
            ratio_mean=ratio.mean().item(),
            ratio_std=ratio.std().item(),
            epsilon=self.epsilon,
            clipfrac=clipfrac.item(),
            ppo_clipfrac=ppo_clipfrac.item(),
            return_clipfrac=return_clipfrac.item(),
        )


class CFRDistillationLoss(LossCalculator):
    """
    CFR Distillation Loss that trains policy to match CFR equilibrium targets.

    Uses KL divergence between model policy and CFR target policy for policy loss,
    while keeping standard value loss and entropy regularization.
    """

    def __init__(
        self,
        popart_normalizer: Optional[PopArtNormalizer],
        value_coef: float = 1.0,
        entropy_coef: float = 0.01,
        value_loss_type: ValueLossType = ValueLossType.mse,
        huber_delta: float = 1.0,
    ):
        """
        Initialize CFR Distillation loss calculator.

        Args:
            popart_normalizer: PopArtNormalizer instance for value normalization
            value_coef: Value loss coefficient
            entropy_coef: Entropy regularization coefficient
            value_loss_type: Type of value loss ("mse" or "huber")
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__(
            epsilon=0.2,  # Not used but required by parent
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            value_loss_type=value_loss_type,
            huber_delta=huber_delta,
        )
        self.popart = popart_normalizer

    def compute_loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        batch: BatchSample,
        value_quantiles: Optional[torch.Tensor] = None,
        cfr_target: Optional[torch.Tensor] = None,
    ) -> LossResult:
        """
        Compute CFR Distillation loss.

        Args:
            logits: Policy logits (B, num_actions)
            values: Value predictions (B,)
            batch: Batch sample containing actions, advantages, returns, etc.
            cfr_target: CFR target policy for distillation (B, 4)

        Returns:
            LossResult containing loss components and metrics
        """
        if cfr_target is None:
            raise ValueError("CFRDistillationLoss requires cfr_target to be provided")

        returns = batch.returns
        delta2 = batch.delta2
        delta3 = batch.delta3

        # Mask illegal actions
        legal_masks = batch.legal_masks.bool()
        masked_logits = compute_masked_logits(logits, legal_masks)

        # Get model policy in full action space
        model_probs_full = F.softmax(masked_logits, dim=-1)

        # Collapse model policy to 4 actions for comparison with CFR target
        model_probs_4 = CFRManager.collapse_policy_full_to_4(model_probs_full)

        # Compute KL divergence: KL(cfr_target || model_probs_4)
        # Add small epsilon for numerical stability
        cfr_target_stable = cfr_target + 1e-8
        model_probs_4_stable = model_probs_4 + 1e-8

        # Normalize probabilities
        cfr_target_norm = cfr_target_stable / cfr_target_stable.sum(
            dim=-1, keepdim=True
        )
        model_probs_4_norm = model_probs_4_stable / model_probs_4_stable.sum(
            dim=-1, keepdim=True
        )

        # Compute KL divergence per sample
        kl_div_per_sample = (
            cfr_target_norm * torch.log(cfr_target_norm / model_probs_4_norm)
        ).sum(dim=-1)

        # Policy loss is mean KL divergence
        policy_loss = kl_div_per_sample.mean()

        # Value loss with clipping (as per AlphaHoldem paper)
        clipped_returns = torch.clamp(returns, delta2, delta3)
        return_clipfrac = (torch.abs(clipped_returns - returns) > 1e-8).float().mean()

        # Use frozen stats for normalization during training
        mu_frozen, sigma_frozen = self.popart.get_frozen_stats()
        targets_n = (clipped_returns - mu_frozen) / (sigma_frozen + 1e-8)

        if self.value_loss_type == ValueLossType.huber:
            value_loss = F.smooth_l1_loss(values, targets_n, beta=self.huber_delta)
        else:
            value_loss = F.mse_loss(values, targets_n)

        # Entropy regularization (compute only if enabled)
        if self.entropy_coef != 0.0:
            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        else:
            entropy = torch.tensor(0.0, dtype=values.dtype, device=values.device)

        # Total loss (pass back through total_loss; policy_loss field can be 0)
        total_loss = (
            policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        )

        # Compute CFR vs model KL for logging
        cfr_model_kl = kl_div_per_sample.mean().item()

        return LossResult(
            total_loss=total_loss,
            policy_loss=policy_loss.item(),
            value_loss_tensor=value_loss,
            entropy=entropy.item(),
            ratio_mean=0.0,  # Not applicable for CFR
            ratio_std=0.0,  # Not applicable for CFR
            epsilon=0.0,  # Not applicable for CFR
            clipfrac=0.0,  # Not applicable for CFR
            ppo_clipfrac=0.0,  # Not applicable for CFR
            return_clipfrac=return_clipfrac.item(),
            cfr_kl=cfr_model_kl,
        )


class RebelSupervisedLoss(nn.Module):
    """Supervised loss for ReBeL-style CFR training."""

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        entropy_coef: float | None = None,
    ) -> None:
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_coef = entropy_coef

    def forward(
        self,
        logits: torch.Tensor,
        hand_values: torch.Tensor,
        batch: RebelBatch,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, num_actions] or [B, num_hands, num_actions] raw policy logits.
            hand_values: [B, num_players, num_combos] per-hand value predictions.
            batch: RebelBatch with policy/value targets.
        Returns:
            Dict of scalar tensors for loss components and diagnostics.
        """

        legal_masks = batch.legal_masks[:, None, :]
        masked_logits = compute_masked_logits(logits, legal_masks)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()

        if batch.policy_targets is None:
            policy_loss = torch.zeros(1, device=logits.device)
        else:
            policy_loss = F.huber_loss(probs, batch.policy_targets, delta=1.0)

        if batch.value_targets is None:
            value_loss = torch.zeros(1, device=logits.device)
        else:
            value_loss = F.mse_loss(hand_values, batch.value_targets)

        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss

        entropy = -(probs * log_probs).sum(dim=1).mean()
        if self.entropy_coef is not None and self.entropy_coef != 0.0:
            total_loss -= self.entropy_coef * entropy

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
