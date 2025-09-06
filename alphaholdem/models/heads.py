from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from line_profiler import profile

from ..core.interfaces import Policy
from ..core.registry import register_policy


@profile
@register_policy("categorical_v1")
class CategoricalPolicyV1(Policy):
    def action(
        self, logits: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, float]:
        if legal_mask is not None:
            # mask out illegal actions
            logits = logits.clone()
            logits[legal_mask == 0] = float("-inf")
        # Fast sampling: log_softmax -> exp -> multinomial
        log_probs_vec = F.log_softmax(logits.float(), dim=-1)
        probs_vec = log_probs_vec.exp()
        a = torch.multinomial(probs_vec, num_samples=1)
        action_idx = int(a.item())
        logp = log_probs_vec[action_idx]
        return action_idx, float(logp.item())

    def action_batch(
        self, logits: torch.Tensor, legal_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions for a batch of environments.

        Args:
            logits: [N, B] tensor of action logits
            legal_masks: [N, B] tensor of legal action masks

        Returns:
            Tuple of (action_indices, log_probs) both of shape [N]
        """
        if legal_masks is not None:
            # Mask out illegal actions more efficiently
            # Use torch.where instead of boolean indexing for better performance
            masked_logits = torch.where(legal_masks.bool(), logits, -1e9)

            # Ensure at least one action is legal to avoid numerical issues
            # If all actions are illegal, make the first action legal
            all_illegal = legal_masks.sum(dim=1) == 0
            if all_illegal.any():
                print("WARNING: All actions are illegal")
                masked_logits[all_illegal, 0] = logits[all_illegal, 0]
        else:
            masked_logits = logits

        # Compute log probabilities
        log_probs = F.log_softmax(masked_logits.float(), dim=-1)

        # Sample actions
        probs = log_probs.exp()

        # Debug: Check for invalid probabilities
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            print(f"WARNING: Invalid probabilities detected!")
            print(f"  NaN: {torch.isnan(probs).any()}")
            print(f"  Inf: {torch.isinf(probs).any()}")
            print(f"  Negative: {(probs < 0).any()}")
            print(f"  Probs min/max: {probs.min():.6f}/{probs.max():.6f}")
            print(
                f"  Logits min/max: {masked_logits.min():.6f}/{masked_logits.max():.6f}"
            )
            print(
                f"  Legal masks sum: {legal_masks.sum(dim=1) if legal_masks is not None else 'None'}"
            )

            # Fallback: use uniform probabilities for problematic rows
            invalid_rows = (
                torch.isnan(probs).any(dim=1)
                | torch.isinf(probs).any(dim=1)
                | (probs < 0).any(dim=1)
            )
            if invalid_rows.any():
                print(
                    f"  Fixing {invalid_rows.sum()} invalid rows with uniform probabilities"
                )
                probs[invalid_rows] = (
                    torch.ones_like(probs[invalid_rows]) / probs.shape[1]
                )

        action_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Get log probabilities for selected actions
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        return action_indices, selected_log_probs
