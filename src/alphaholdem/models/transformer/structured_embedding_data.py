"""Data structures for transformer embedding components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from alphaholdem.models.transformer.tokens import Context


@dataclass
class StructuredEmbeddingData:
    """Data structure containing all components of structured embeddings.

    This class encapsulates all the structured embedding components used by the
    transformer poker model, providing a clean interface and type safety.

    Dtype conventions:
    - All integer fields (token_ids, token_streets, card_ranks, card_suits, action_actors) are long
    - Context fields (action_legal_masks, context_features) should match self.dtype (default: float32)
    """

    # Card components [batch_size, seq_len] - all should be long
    token_ids: torch.Tensor  # Token IDs (0-51 for cards, 52 for CLS, -1 for padding)
    token_streets: torch.Tensor  # Street indices for all events
    card_ranks: torch.Tensor  # Card ranks (0-12)
    card_suits: torch.Tensor  # Card suits (0-3)

    # Action components [batch_size, seq_len] - all should be long
    action_actors: torch.Tensor  # Actor indices (0-1: player indices)

    # Legal action masks [batch_size, seq_len, 8] - torch.bool
    action_legal_masks: torch.Tensor
    # Context components - stored as int16 for memory efficiency
    # Raw numeric features per token [batch_size, seq_len, NUM_RAW_CONTEXT] - int16
    context_features: torch.Tensor

    # Sequence metadata
    lengths: torch.Tensor  # Actual sequence lengths [batch_size]

    def __post_init__(self):
        """Ensure proper dtypes on creation."""
        # Convert integer fields to storage type
        self.token_ids = self.token_ids.to(torch.int8)
        self.token_streets = self.token_streets.to(torch.uint8)
        self.card_ranks = self.card_ranks.to(torch.uint8)
        self.card_suits = self.card_suits.to(torch.uint8)
        self.action_actors = self.action_actors.to(torch.uint8)
        self.lengths = self.lengths.to(torch.uint8)
        assert self.action_legal_masks.dtype == torch.bool

        # Convert context features to int16 for storage efficiency
        self.context_features = self.context_features.to(torch.int16)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for model forward pass."""
        return {
            "token_ids": self.token_ids,
            "token_streets": self.token_streets,
            "card_ranks": self.card_ranks,
            "card_suits": self.card_suits,
            "action_actors": self.action_actors,
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
            token_streets=data["token_streets"],
            card_ranks=data["card_ranks"],
            card_suits=data["card_suits"],
            action_actors=data["action_actors"],
            action_legal_masks=data["action_legal_masks"],
            context_features=data["context_features"],
            lengths=lengths,
        )

    def to_device(self, device: torch.device) -> StructuredEmbeddingData:
        """Move all tensors to specified device."""
        return StructuredEmbeddingData(
            token_ids=self.token_ids.to(device),
            token_streets=self.token_streets.to(device),
            card_ranks=self.card_ranks.to(device),
            card_suits=self.card_suits.to(device),
            action_actors=self.action_actors.to(device),
            action_legal_masks=self.action_legal_masks.to(device),
            context_features=self.context_features.to(device),
            lengths=self.lengths.to(device),
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

    def __getitem__(self, indices) -> StructuredEmbeddingData:
        """Slice the embedding data by indices."""
        return StructuredEmbeddingData(
            token_ids=self.token_ids[indices],
            token_streets=self.token_streets[indices],
            card_ranks=self.card_ranks[indices],
            card_suits=self.card_suits[indices],
            action_actors=self.action_actors[indices],
            action_legal_masks=self.action_legal_masks[indices],
            context_features=self.context_features[indices],
            lengths=self.lengths[indices],
        )

    @property
    def attention_mask(self) -> torch.Tensor:
        """
        SDPA-style attention mask where True means 'allow' attention, False means 'block'.
        Valid tokens (token_ids >= 0) should allow attention, padding tokens (token_ids < 0) should block.
        """
        return self.token_ids >= 0  # dtype: bool, shape: [B, S]

    def clone(self) -> StructuredEmbeddingData:
        """Return a copy of the embedding data."""
        return StructuredEmbeddingData(
            token_ids=self.token_ids.clone(),
            token_streets=self.token_streets.clone(),
            card_ranks=self.card_ranks.clone(),
            card_suits=self.card_suits.clone(),
            action_actors=self.action_actors.clone(),
            action_legal_masks=self.action_legal_masks.clone(),
            context_features=self.context_features.clone(),
            lengths=self.lengths.clone(),
        )

    @classmethod
    def empty(
        cls,
        batch_size: int,
        seq_len: int,
        num_bet_bins: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> StructuredEmbeddingData:
        return cls(
            token_ids=torch.full(
                (batch_size, seq_len), -1, dtype=torch.int8, device=device
            ),
            token_streets=torch.zeros(
                (batch_size, seq_len), dtype=torch.uint8, device=device
            ),
            card_ranks=torch.zeros(
                (batch_size, seq_len), dtype=torch.uint8, device=device
            ),
            card_suits=torch.zeros(
                (batch_size, seq_len), dtype=torch.uint8, device=device
            ),
            action_actors=torch.zeros(
                (batch_size, seq_len), dtype=torch.uint8, device=device
            ),
            action_legal_masks=torch.zeros(
                (batch_size, seq_len, num_bet_bins), dtype=torch.bool, device=device
            ),
            context_features=torch.zeros(
                (batch_size, seq_len, Context.NUM_RAW_CONTEXT.value),
                dtype=torch.int16,
                device=device,
            ),
            lengths=torch.zeros((batch_size), dtype=torch.long, device=device),
        )
