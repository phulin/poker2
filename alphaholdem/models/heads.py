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
