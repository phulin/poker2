from __future__ import annotations

from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F

from ..core.interfaces import Policy
from ..core.registry import register_policy


@register_policy("categorical_v1")
class CategoricalPolicyV1(Policy):
    def action(
        self, logits: torch.Tensor, legal_mask: Optional[torch.Tensor] = None
    ) -> Tuple[int, float]:
        if legal_mask is not None:
            # mask out illegal by setting to large negative
            logits = logits.clone()
            logits[legal_mask == 0] = -1e9
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        logp = dist.log_prob(a)
        return int(a.item()), float(logp.item())
