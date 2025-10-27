"""Output dataclass for poker models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """Unified output from poker models (both CNN and transformer)."""

    policy_logits: torch.Tensor
    """Policy logits of shape (batch_size, num_actions)"""

    value: torch.Tensor
    """Value estimates of shape (batch_size,)"""

    value_quantiles: torch.Tensor | None = None
    """Optional quantile value estimates of shape (batch_size, num_quantiles)"""

    hand_values: torch.Tensor | None = None
    """Optional per-hand value estimates of shape (batch_size, num_players, num_combos)"""

    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    """KV cache dictionary keyed by layer ID for incremental generation (transformer only)"""

    encoded_with_permutation: torch.Tensor | None = None
    """Encoded belief features with permutation applied (PBS-style only)"""

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            "policy_logits": self.policy_logits,
            "value": self.value,
        }
        if self.value_quantiles is not None:
            result["value_quantiles"] = self.value_quantiles
        if self.hand_values is not None:
            result["hand_values"] = self.hand_values
        if self.kv_cache is not None:
            result["kv_cache"] = self.kv_cache
        if self.encoded_with_permutation is not None:
            result["encoded_with_permutation"] = self.encoded_with_permutation
        return result

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor]) -> ModelOutput:
        """Create from dictionary format."""
        return cls(
            policy_logits=data["policy_logits"],
            value=data["value"],
            value_quantiles=data.get("value_quantiles"),
            hand_values=data.get("hand_values"),
            kv_cache=data.get("kv_cache"),
            encoded_with_permutation=data.get("encoded_with_permutation"),
        )
