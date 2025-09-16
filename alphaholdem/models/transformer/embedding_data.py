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

    Dtype conventions:
    - All integer fields (token_ids, card_ranks, card_suits, card_streets, action_actors, action_streets) are long
    - Context fields (action_legal_masks, context_features) should match self.dtype (default: float32)
    """

    # Card components [batch_size, seq_len] - all should be long
    token_ids: torch.Tensor  # Token IDs (0-51 for cards, 52 for CLS, -1 for padding)
    card_ranks: torch.Tensor  # Card ranks (0-12)
    card_suits: torch.Tensor  # Card suits (0-3)
    card_streets: torch.Tensor  # Street indices for cards (0-3)

    # Action components [batch_size, seq_len] - all should be long
    action_actors: torch.Tensor  # Actor indices (0-1: player indices)
    action_streets: torch.Tensor  # Street indices (0-3: hole, flop, turn, river)

    # Context components - should match self.dtype (default: float32)
    action_legal_masks: (
        torch.Tensor
    )  # Legal action masks [batch_size, seq_len, 8] - dtype should match self.dtype
    context_features: (
        torch.Tensor
    )  # Numeric features per token [batch_size, seq_len, 10]

    # Sequence metadata
    lengths: torch.Tensor  # Actual sequence lengths [batch_size]

    # Dtype control
    dtype: torch.dtype = torch.float32  # Target dtype for context fields

    def __post_init__(self):
        """Ensure proper dtypes on creation."""
        # Convert integer fields to long
        self.token_ids = self.token_ids.long()
        self.card_ranks = self.card_ranks.long()
        self.card_suits = self.card_suits.long()
        self.card_streets = self.card_streets.long()
        self.action_actors = self.action_actors.long()
        self.action_streets = self.action_streets.long()
        self.lengths = self.lengths.long()

        # Convert context fields to specified dtype (self.dtype)
        self.action_legal_masks = self.action_legal_masks.to(self.dtype)  # Float mask
        self.context_features = self.context_features.to(self.dtype)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for model forward pass."""
        return {
            "token_ids": self.token_ids,
            "card_ranks": self.card_ranks,
            "card_suits": self.card_suits,
            "card_streets": self.card_streets,
            "action_actors": self.action_actors,
            "action_streets": self.action_streets,
            "action_legal_masks": self.action_legal_masks,
            "context_features": self.context_features,
            "lengths": self.lengths,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> StructuredEmbeddingData:
        """Create from dictionary format."""
        lengths = data.get("lengths")
        if lengths is None:
            lengths = (data["token_ids"] >= 0).long().sum(dim=1)
        return cls(
            token_ids=data["token_ids"],
            card_ranks=data["card_ranks"],
            card_suits=data["card_suits"],
            card_streets=data["card_streets"],
            action_actors=data["action_actors"],
            action_streets=data["action_streets"],
            action_legal_masks=data["action_legal_masks"],
            context_features=data["context_features"],
            lengths=lengths,
        )

    def to_device(self, device: torch.device) -> StructuredEmbeddingData:
        """Move all tensors to specified device."""
        return StructuredEmbeddingData(
            token_ids=self.token_ids.to(device),
            card_ranks=self.card_ranks.to(device),
            card_suits=self.card_suits.to(device),
            card_streets=self.card_streets.to(device),
            action_actors=self.action_actors.to(device),
            action_streets=self.action_streets.to(device),
            action_legal_masks=self.action_legal_masks.to(device),
            context_features=self.context_features.to(device),
            lengths=self.lengths.to(device),
            dtype=self.dtype,  # Preserve the dtype
        )

    def to(self, dtype: torch.dtype) -> StructuredEmbeddingData:
        """Convert context fields to specified dtype and update self.dtype. Integer fields remain long."""
        # Create new instance without calling __post_init__
        result = StructuredEmbeddingData.__new__(StructuredEmbeddingData)
        result.token_ids = self.token_ids  # Keep as long
        result.card_ranks = self.card_ranks  # Keep as long
        result.card_suits = self.card_suits  # Keep as long
        result.card_streets = self.card_streets  # Keep as long
        result.action_actors = self.action_actors  # Keep as long
        result.action_streets = self.action_streets  # Keep as long
        result.action_legal_masks = self.action_legal_masks.to(
            dtype
        )  # Convert to new dtype
        result.context_features = self.context_features.to(
            dtype
        )  # Convert to new dtype
        result.lengths = self.lengths  # Keep as long
        result.dtype = dtype  # Update self.dtype to match
        return result

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

    def __getitem__(self, indices) -> StructuredEmbeddingData:
        """Slice the embedding data by indices."""
        return StructuredEmbeddingData(
            token_ids=self.token_ids[indices],
            card_ranks=self.card_ranks[indices],
            card_suits=self.card_suits[indices],
            card_streets=self.card_streets[indices],
            action_actors=self.action_actors[indices],
            action_streets=self.action_streets[indices],
            action_legal_masks=self.action_legal_masks[indices],
            context_features=self.context_features[indices],
            lengths=self.lengths[indices],
        )

    @property
    def attention_mask(self) -> torch.Tensor:
        """Return boolean mask indicating valid tokens."""
        return self.token_ids >= 0
