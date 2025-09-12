"""Data structures for transformer embedding components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import torch


@dataclass
class StructuredEmbeddingData:
    """Data structure containing all components of structured embeddings.

    This class encapsulates all the structured embedding components used by the
    transformer poker model, providing a clean interface and type safety.
    """

    # Card components [batch_size, seq_len]
    token_ids: torch.Tensor  # Token IDs (0-51 for cards, 52 for CLS, -1 for padding)
    card_ranks: torch.Tensor  # Card ranks (0-12)
    card_suits: torch.Tensor  # Card suits (0-3)
    card_stages: torch.Tensor  # Stage indices (0-3: hole, flop, turn, river)

    # Action components [batch_size, seq_len]
    action_actors: torch.Tensor  # Actor indices (0-1: player indices)
    action_streets: torch.Tensor  # Street indices (0-3: hole, flop, turn, river)
    action_legal_masks: torch.Tensor  # Legal action masks [batch_size, seq_len, 8]

    # Context components [batch_size, seq_len, 10]
    context_features: (
        torch.Tensor
    )  # All context info in one tensor [batch_size, seq_len, 10]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for model forward pass."""
        return {
            "card_indices": self.token_ids,
            "card_stages": self.card_stages,
            "action_actors": self.action_actors,
            "action_streets": self.action_streets,
            "action_legal_masks": self.action_legal_masks,
            "context_features": self.context_features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> StructuredEmbeddingData:
        """Create from dictionary format."""
        return cls(
            token_ids=data["card_indices"],
            card_stages=data["card_stages"],
            action_actors=data["action_actors"],
            action_streets=data["action_streets"],
            action_legal_masks=data["action_legal_masks"],
            context_features=data["context_features"],
        )

    def to_device(self, device: torch.device) -> StructuredEmbeddingData:
        """Move all tensors to specified device."""
        return StructuredEmbeddingData(
            token_ids=self.token_ids.to(device),
            card_stages=self.card_stages.to(device),
            action_actors=self.action_actors.to(device),
            action_streets=self.action_streets.to(device),
            action_legal_masks=self.action_legal_masks.to(device),
            context_features=self.context_features.to(device),
        )

    def __len__(self) -> int:
        """Return batch size."""
        return self.token_ids.shape[0]

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.token_ids.shape[0]

    @property
    def seq_len(self) -> int:
        """Get sequence length."""
        return self.token_ids.shape[1]

    @property
    def device(self) -> torch.device:
        """Get device of tensors."""
        return self.token_ids.device
