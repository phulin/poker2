"""Data structures for CNN embedding components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class CNNEmbeddingData:
    """Data structure containing all components of CNN embeddings.

    This class encapsulates all the CNN embedding components used by the
    CNN poker models, providing a clean interface and type safety.

    Dtype conventions:
    - All fields should match self.dtype (default: float32)
    """

    # Card components [batch_size, 6_channels, 4_suits, 13_ranks] - should match self.dtype
    cards: (
        torch.Tensor
    )  # Card planes with 6 channels (hole, flop, turn, river, public, all)

    # Action components [batch_size, 24_channels, 4_players, num_bet_bins] - should match self.dtype
    actions: torch.Tensor  # Action history with 24 channels (4 streets × 6 slots)

    # Dtype control
    dtype: torch.dtype = torch.float32  # Target dtype for all fields

    def __post_init__(self):
        """Ensure proper dtypes on creation."""
        # Convert both fields to specified dtype (self.dtype)
        self.cards = self.cards.to(self.dtype)
        self.actions = self.actions.to(self.dtype)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for model forward pass."""
        return {
            "cards": self.cards,
            "actions": self.actions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> CNNEmbeddingData:
        """Create from dictionary format."""
        return cls(
            cards=data["cards"],
            actions=data["actions"],
        )

    def to_device(self, device: torch.device) -> CNNEmbeddingData:
        """Move all tensors to specified device."""
        return CNNEmbeddingData(
            cards=self.cards.to(device),
            actions=self.actions.to(device),
            dtype=self.dtype,  # Preserve the dtype
        )

    def to(self, dtype: torch.dtype) -> CNNEmbeddingData:
        """Convert all fields to specified dtype and update self.dtype."""
        # Create new instance without calling __post_init__
        result = CNNEmbeddingData.__new__(CNNEmbeddingData)
        result.cards = self.cards.to(dtype)
        result.actions = self.actions.to(dtype)
        result.dtype = dtype  # Update self.dtype to match
        return result

    def __len__(self) -> int:
        """Return batch size."""
        return self.cards.shape[0]

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.cards.shape[0]

    @property
    def device(self) -> torch.device:
        """Get device of tensors."""
        return self.cards.device

    def __getitem__(self, indices) -> CNNEmbeddingData:
        """Slice the embedding data by indices."""
        return CNNEmbeddingData(
            cards=self.cards[indices],
            actions=self.actions[indices],
        )
