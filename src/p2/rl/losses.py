from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2.core.structured_config import (
    KLType,
    PPOClipping,
    PolicyLossType,
    PolicyNodeWeighting,
    ValueLossType,
)
from p2.env.card_utils import (
    NUM_HANDS,
    combo_suit_permutation_tensor,
    hand_combos_tensor,
)
from p2.models.model_output import ModelOutput
from p2.rl.exponential_controller import ExponentialController
from p2.rl.popart_normalizer import PopArtNormalizer
from p2.rl.rebel_batch import RebelBatch
from p2.rl.vectorized_replay import BatchSample
from p2.search.cfr_manager import CFRManager
from p2.utils.ema import EMA
from p2.utils.model_utils import compute_masked_logits


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
            else forward_kl
            if self.kl_type == KLType.forward
            else reverse_kl
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
        permutation_weight: float = 0.01,
        entropy_coef: float | None = None,
        num_players: int = 2,
        policy_node_weighting: PolicyNodeWeighting | str = PolicyNodeWeighting.uniform,
        policy_loss_type: PolicyLossType | str = PolicyLossType.cross_entropy,
    ) -> None:
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_coef = entropy_coef
        self.permutation_weight = permutation_weight
        self.num_players = num_players
        self.policy_node_weighting = (
            policy_node_weighting
            if isinstance(policy_node_weighting, PolicyNodeWeighting)
            else PolicyNodeWeighting(policy_node_weighting)
        )
        self.policy_loss_type = (
            policy_loss_type
            if isinstance(policy_loss_type, PolicyLossType)
            else PolicyLossType(policy_loss_type)
        )
        combos = hand_combos_tensor()
        self.register_buffer("_combo_card_a", combos[:, 0].long(), persistent=False)
        self.register_buffer("_combo_card_b", combos[:, 1].long(), persistent=False)
        self.register_buffer(
            "_combo_suit_permutations",
            combo_suit_permutation_tensor(),
            persistent=False,
        )

    def compile_forward_modes(self, **kwargs):
        """Compile fixed-mode loss forwards without compiling optional dispatch."""
        self._compiled_forward_policy = torch.compile(self.forward_policy, **kwargs)
        self._compiled_forward_value = torch.compile(self.forward_value, **kwargs)
        self._compiled_forward_both = torch.compile(self.forward_both, **kwargs)
        return self

    def _call_forward_policy(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_policy", None)
        if fn is None:
            fn = self.forward_policy
        return fn(*args, **kwargs)

    def _call_forward_value(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_value", None)
        if fn is None:
            fn = self.forward_value
        return fn(*args, **kwargs)

    def _call_forward_both(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_both", None)
        if fn is None:
            fn = self.forward_both
        return fn(*args, **kwargs)

    def _board_allowed_hands(self, board: torch.Tensor) -> torch.Tensor:
        """Return private-hand mask using precomputed combo card buffers."""
        if board.shape[-1] == 0:
            return torch.ones(
                *board.shape[:-1], NUM_HANDS, dtype=torch.bool, device=board.device
            )

        flat_board = board.reshape(-1, board.shape[-1]).long()
        valid = flat_board >= 0
        flat_board_safe = torch.where(
            valid, flat_board, torch.full_like(flat_board, 52)
        )
        board_onehot = torch.zeros(
            flat_board.shape[0], 53, dtype=torch.bool, device=board.device
        )
        board_onehot.scatter_(1, flat_board_safe, valid)
        board_onehot = board_onehot[:, :52]
        allowed = ~(
            board_onehot[:, self._combo_card_a] | board_onehot[:, self._combo_card_b]
        )
        return allowed.reshape(*board.shape[:-1], NUM_HANDS)

    def _calculate_unblocked_mass(self, target: torch.Tensor) -> torch.Tensor:
        """PIE unblocked mass using precomputed combo card buffers."""
        target_batched = target.view(-1, NUM_HANDS).float()
        card_a = self._combo_card_a
        card_b = self._combo_card_b

        total = target_batched.sum(dim=-1, keepdim=True)
        cardsum = torch.zeros(
            target_batched.shape[0],
            52,
            dtype=target_batched.dtype,
            device=target_batched.device,
        )
        card_a_idx = card_a[None, :].expand(target_batched.shape[0], -1)
        card_b_idx = card_b[None, :].expand(target_batched.shape[0], -1)
        cardsum.scatter_add_(1, card_a_idx, target_batched)
        cardsum.scatter_add_(1, card_b_idx, target_batched)

        multiply = total - cardsum[:, card_a] - cardsum[:, card_b] + target_batched
        return multiply.view_as(target).clamp(min=0.0)

    def _base_weights(
        self, batch: RebelBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        player_beliefs = batch.features.beliefs.view(-1, 2, NUM_HANDS)
        allowed_hands = self._board_allowed_hands(batch.features.board)
        allowed_hands_float = allowed_hands.to(dtype=player_beliefs.dtype)
        unblocked_mass = self._calculate_unblocked_mass(player_beliefs)
        return player_beliefs, allowed_hands_float, unblocked_mass

    def _policy_weights(
        self, batch: RebelBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = batch.features.to_act
        opp = 1 - actor
        player_beliefs, allowed_hands_float, unblocked_mass = self._base_weights(batch)
        actor_belief = player_beliefs.gather(
            1, actor[:, None, None].expand(-1, 1, NUM_HANDS)
        ).squeeze(1)
        opp_matchup = (
            unblocked_mass.gather(
                1, opp[:, None, None].expand(-1, 1, NUM_HANDS)
            ).squeeze(1)
            * allowed_hands_float
        )
        return (
            player_beliefs,
            allowed_hands_float,
            unblocked_mass,
            actor_belief,
            opp_matchup,
        )

    def _zero(self, device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)

    def _policy_node_weights(
        self, batch: RebelBatch, dtype: torch.dtype
    ) -> torch.Tensor | None:
        if self.policy_node_weighting == PolicyNodeWeighting.uniform:
            return None
        reach = batch.statistics.get("policy_node_reach")
        if reach is None:
            return None
        reach = reach.to(dtype=dtype).clamp(min=0.0)
        if self.policy_node_weighting == PolicyNodeWeighting.reach:
            return reach
        if self.policy_node_weighting == PolicyNodeWeighting.sqrt_reach:
            return reach.sqrt()
        if self.policy_node_weighting == PolicyNodeWeighting.clipped_reach:
            relative = reach / reach.mean().clamp(min=1e-8)
            return relative.clamp(min=0.1, max=10.0)
        raise ValueError(
            f"Unsupported policy node weighting: {self.policy_node_weighting}"
        )

    def _reduce_policy_node_metric(
        self, per_node: torch.Tensor, node_weights: torch.Tensor | None
    ) -> torch.Tensor:
        if node_weights is None:
            return per_node.mean()
        return (per_node * node_weights).sum() / node_weights.sum().clamp(min=1e-8)

    def _policy_objective_per_hand(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        policy_ce_per_hand: torch.Tensor,
    ) -> torch.Tensor:
        if self.policy_loss_type == PolicyLossType.cross_entropy:
            return policy_ce_per_hand
        if self.policy_loss_type == PolicyLossType.mse:
            return F.mse_loss(probs, targets, reduction="none").mean(dim=-1)
        raise ValueError(f"Unsupported policy loss type: {self.policy_loss_type}")

    def _permutation_loss(
        self,
        output: ModelOutput,
        output_permuted: ModelOutput,
        suit_permutation_idxs: torch.Tensor,
    ) -> torch.Tensor:
        combo_permutations = self._combo_suit_permutations[suit_permutation_idxs]
        hand_values_permuted_reversed = torch.gather(
            output_permuted.hand_values,
            2,
            combo_permutations[:, None, :].expand(-1, self.num_players, -1),
        )
        return F.mse_loss(output.hand_values, hand_values_permuted_reversed)

    def forward_policy(
        self,
        output: ModelOutput,
        batch: RebelBatch,
    ) -> dict[str, torch.Tensor]:
        logits = output.policy_logits
        device = logits.device
        _, _, _, actor_belief, opp_matchup = self._policy_weights(batch)

        legal_masks = batch.legal_masks[:, None, :]
        masked_logits = compute_masked_logits(logits, legal_masks)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()

        policy_weights_unnormalized = actor_belief * opp_matchup
        policy_weight_sum = policy_weights_unnormalized.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        policy_weights = policy_weights_unnormalized / policy_weight_sum
        target_log_probs = batch.policy_targets.clamp_min(1e-8).log()
        target_entropy_per_hand = -(batch.policy_targets * target_log_probs).sum(dim=-1)
        policy_ce_per_hand = -(batch.policy_targets * log_probs).sum(dim=-1)
        policy_objective_per_hand = self._policy_objective_per_hand(
            probs, batch.policy_targets, policy_ce_per_hand
        )
        policy_loss_per_hand = policy_objective_per_hand * policy_weights
        policy_loss_all = policy_loss_per_hand.sum(dim=-1)
        node_weights = self._policy_node_weights(batch, policy_loss_all.dtype)
        policy_loss = self._reduce_policy_node_metric(policy_loss_all, node_weights)
        policy_loss_all = policy_loss_all.detach()
        target_entropy_per_sample = (target_entropy_per_hand * policy_weights).sum(
            dim=-1
        )
        target_entropy = self._reduce_policy_node_metric(
            target_entropy_per_sample, node_weights
        )
        model_entropy_per_hand = -(probs * log_probs).sum(dim=-1)
        model_entropy_per_sample = (model_entropy_per_hand * policy_weights).sum(
            dim=-1
        )
        model_entropy = self._reduce_policy_node_metric(
            model_entropy_per_sample, node_weights
        )
        entropy_gap_all = model_entropy_per_sample - target_entropy_per_sample
        entropy_gap = self._reduce_policy_node_metric(entropy_gap_all, node_weights)
        target_model_kl_all = (
            (policy_ce_per_hand - target_entropy_per_hand) * policy_weights
        ).sum(dim=-1)
        target_model_kl = self._reduce_policy_node_metric(
            target_model_kl_all, node_weights
        )

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        total_loss = self.policy_weight * policy_loss
        if self.entropy_coef is not None and self.entropy_coef != 0.0:
            total_loss -= self.entropy_coef * entropy

        zero = self._zero(device)
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "policy_loss_all": policy_loss_all,
            "target_entropy": target_entropy,
            "target_entropy_all": target_entropy_per_sample.detach(),
            "model_entropy": model_entropy,
            "model_entropy_all": model_entropy_per_sample.detach(),
            "entropy_gap": entropy_gap,
            "entropy_gap_all": entropy_gap_all.detach(),
            "target_model_kl": target_model_kl,
            "target_model_kl_all": target_model_kl_all.detach(),
            "policy_weights": policy_weights,
            "policy_node_weights": node_weights,
            "value_loss": zero,
            "value_loss_all": None,
            "value_weights": None,
            "entropy": entropy,
            "permutation_loss": zero,
        }

    def forward_value(
        self,
        output: ModelOutput,
        batch: RebelBatch,
    ) -> dict[str, torch.Tensor]:
        hand_values = output.hand_values
        device = hand_values.device
        _, allowed_hands_float, unblocked_mass = self._base_weights(batch)

        value_weights = unblocked_mass.flip(dims=[1]) * allowed_hands_float[:, None]
        value_loss = F.mse_loss(hand_values, batch.value_targets, weight=value_weights)
        value_loss_all = F.mse_loss(
            hand_values.detach(),
            batch.value_targets,
            reduction="none",
            weight=value_weights,
        )
        total_loss = self.value_weight * value_loss

        zero = self._zero(device)
        return {
            "total_loss": total_loss,
            "policy_loss": zero,
            "policy_loss_all": None,
            "target_entropy": zero,
            "target_entropy_all": None,
            "model_entropy": zero,
            "model_entropy_all": None,
            "entropy_gap": zero,
            "entropy_gap_all": None,
            "target_model_kl": zero,
            "target_model_kl_all": None,
            "policy_weights": None,
            "policy_node_weights": None,
            "value_loss": value_loss,
            "value_loss_all": value_loss_all,
            "value_weights": value_weights,
            "entropy": zero,
            "permutation_loss": zero,
        }

    def forward_both(
        self,
        output: ModelOutput,
        batch: RebelBatch,
    ) -> dict[str, torch.Tensor]:
        logits = output.policy_logits
        hand_values = output.hand_values
        device = logits.device
        _, allowed_hands_float, unblocked_mass, actor_belief, opp_matchup = (
            self._policy_weights(batch)
        )

        legal_masks = batch.legal_masks[:, None, :]
        masked_logits = compute_masked_logits(logits, legal_masks)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = log_probs.exp()

        policy_weights_unnormalized = actor_belief * opp_matchup
        policy_weight_sum = policy_weights_unnormalized.sum(dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        policy_weights = policy_weights_unnormalized / policy_weight_sum
        target_log_probs = batch.policy_targets.clamp_min(1e-8).log()
        target_entropy_per_hand = -(batch.policy_targets * target_log_probs).sum(dim=-1)
        policy_ce_per_hand = -(batch.policy_targets * log_probs).sum(dim=-1)
        policy_objective_per_hand = self._policy_objective_per_hand(
            probs, batch.policy_targets, policy_ce_per_hand
        )
        policy_loss_per_hand = policy_objective_per_hand * policy_weights
        policy_loss_all = policy_loss_per_hand.sum(dim=-1)
        node_weights = self._policy_node_weights(batch, policy_loss_all.dtype)
        policy_loss = self._reduce_policy_node_metric(policy_loss_all, node_weights)
        policy_loss_all = policy_loss_all.detach()
        target_entropy_per_sample = (target_entropy_per_hand * policy_weights).sum(
            dim=-1
        )
        target_entropy = self._reduce_policy_node_metric(
            target_entropy_per_sample, node_weights
        )
        model_entropy_per_hand = -(probs * log_probs).sum(dim=-1)
        model_entropy_per_sample = (model_entropy_per_hand * policy_weights).sum(
            dim=-1
        )
        model_entropy = self._reduce_policy_node_metric(
            model_entropy_per_sample, node_weights
        )
        entropy_gap_all = model_entropy_per_sample - target_entropy_per_sample
        entropy_gap = self._reduce_policy_node_metric(entropy_gap_all, node_weights)
        target_model_kl_all = (
            (policy_ce_per_hand - target_entropy_per_hand) * policy_weights
        ).sum(dim=-1)
        target_model_kl = self._reduce_policy_node_metric(
            target_model_kl_all, node_weights
        )
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        value_weights = unblocked_mass.flip(dims=[1]) * allowed_hands_float[:, None]
        value_loss = F.mse_loss(hand_values, batch.value_targets, weight=value_weights)
        value_loss_all = F.mse_loss(
            hand_values.detach(),
            batch.value_targets,
            reduction="none",
            weight=value_weights,
        )

        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        if self.entropy_coef is not None and self.entropy_coef != 0.0:
            total_loss -= self.entropy_coef * entropy

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "policy_loss_all": policy_loss_all,
            "target_entropy": target_entropy,
            "target_entropy_all": target_entropy_per_sample.detach(),
            "model_entropy": model_entropy,
            "model_entropy_all": model_entropy_per_sample.detach(),
            "entropy_gap": entropy_gap,
            "entropy_gap_all": entropy_gap_all.detach(),
            "target_model_kl": target_model_kl,
            "target_model_kl_all": target_model_kl_all.detach(),
            "policy_weights": policy_weights,
            "policy_node_weights": node_weights,
            "value_loss": value_loss,
            "value_loss_all": value_loss_all,
            "value_weights": value_weights,
            "entropy": entropy,
            "permutation_loss": self._zero(device),
        }

    def forward(
        self,
        output: ModelOutput,
        batch: RebelBatch,
        output_permuted: ModelOutput | None = None,
        suit_permutation_idxs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            output: Model output with policy logits and hand values.
            batch: RebelBatch with policy/value targets.
            output_permuted: Model output from permuted inputs (optional).
            suit_permutation_idxs: Indices of suit permutations used (B,) (optional).
        Returns:
            Dict of scalar tensors for loss components and diagnostics.
        """

        if batch.policy_targets is not None and batch.value_targets is not None:
            result = self._call_forward_both(output, batch)
        elif batch.policy_targets is not None:
            result = self._call_forward_policy(output, batch)
        elif batch.value_targets is not None:
            result = self._call_forward_value(output, batch)
        else:
            device = output.value.device
            zero = self._zero(device)
            result = {
                "total_loss": zero,
                "policy_loss": zero,
                "policy_loss_all": None,
                "policy_weights": None,
                "value_loss": zero,
                "value_loss_all": None,
                "value_weights": None,
                "entropy": zero,
                "permutation_loss": zero,
            }

        if output_permuted is not None and suit_permutation_idxs is not None:
            permutation_loss = self._permutation_loss(
                output, output_permuted, suit_permutation_idxs
            )
            result["permutation_loss"] = permutation_loss
            result["total_loss"] = (
                result["total_loss"] + self.permutation_weight * permutation_loss
            )

        return result
