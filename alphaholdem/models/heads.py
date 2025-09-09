from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

try:
    from line_profiler import profile
except ImportError:  # pragma: no cover

    def profile(f):
        return f


from ..core.interfaces import Policy
from ..core.registry import register_policy


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
        else:
            masked_logits = logits

        # Compute log probabilities
        log_probs = F.log_softmax(masked_logits.float(), dim=-1)

        # Sample actions
        probs = log_probs.exp()

        action_indices = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Get log probabilities for selected actions
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

        return action_indices, selected_log_probs
