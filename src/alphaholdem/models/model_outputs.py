"""Output dataclass for poker models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class ModelOutput:
    """Unified output from poker models (both CNN and transformer)."""

    policy_logits: torch.Tensor
    """Policy logits of shape (batch_size, num_actions)"""

    value: torch.Tensor
    """Value estimates of shape (batch_size,)"""

    value_quantiles: Optional[torch.Tensor] = None
    """Optional quantile value estimates of shape (batch_size, num_quantiles)"""

    kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    """KV cache dictionary keyed by layer ID for incremental generation (transformer only)"""

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            "policy_logits": self.policy_logits,
            "value": self.value,
        }
        if self.value_quantiles is not None:
            result["value_quantiles"] = self.value_quantiles
        if self.kv_cache is not None:
            result["kv_cache"] = self.kv_cache
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> ModelOutput:
        """Create from dictionary format."""
        return cls(
            policy_logits=data["policy_logits"],
            value=data["value"],
            value_quantiles=data.get("value_quantiles"),
            kv_cache=data.get("kv_cache"),
        )
