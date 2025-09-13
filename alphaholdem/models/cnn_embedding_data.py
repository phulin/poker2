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
    """

    # Card components [batch_size, 6_channels, 4_suits, 13_ranks]
    cards: (
        torch.Tensor
    )  # Card planes with 6 channels (hole, flop, turn, river, public, all)

    # Action components [batch_size, 24_channels, 4_players, num_bet_bins]
    actions: torch.Tensor  # Action history with 24 channels (4 streets × 6 slots)

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
        )

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
