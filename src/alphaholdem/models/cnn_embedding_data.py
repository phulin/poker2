"""Data structures for CNN embedding components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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

    # Optional raw hole card indices [batch_size, 2]; when provided, enables get_hole_cards
    hole_indices: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Ensure proper dtypes on creation."""
        # Convert both fields to specified dtype (self.dtype)
        self.cards = self.cards.to(self.dtype)
        self.actions = self.actions.to(self.dtype)
        if self.hole_indices is not None:
            self.hole_indices = self.hole_indices.to(torch.long)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for model forward pass."""
        result = {
            "cards": self.cards,
            "actions": self.actions,
        }
        if self.hole_indices is not None:
            result["hole_indices"] = self.hole_indices
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> CNNEmbeddingData:
        """Create from dictionary format."""
        return cls(
            cards=data["cards"],
            actions=data["actions"],
            hole_indices=data.get("hole_indices"),
        )

    def to_device(self, device: torch.device) -> CNNEmbeddingData:
        """Move all tensors to specified device."""
        return CNNEmbeddingData(
            cards=self.cards.to(device),
            actions=self.actions.to(device),
            dtype=self.dtype,  # Preserve the dtype
            hole_indices=(
                self.hole_indices.to(device) if self.hole_indices is not None else None
            ),
        )

    def to(self, dtype: torch.dtype) -> CNNEmbeddingData:
        """Convert all fields to specified dtype and update self.dtype."""
        # Create new instance without calling __post_init__
        result = CNNEmbeddingData.__new__(CNNEmbeddingData)
        result.cards = self.cards.to(dtype)
        result.actions = self.actions.to(dtype)
        result.dtype = dtype  # Update self.dtype to match
        result.hole_indices = self.hole_indices  # stays long
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
            hole_indices=(
                self.hole_indices[indices] if self.hole_indices is not None else None
            ),
        )

    def get_hole_cards(self) -> torch.Tensor:
        """
        Return raw hole card indices per batch item.

        Returns:
            Tensor [B, 2] of integer card indices in [0..51].
        """
        if self.hole_indices is None:
            # Derive from cards planes channel 0 if needed
            # cards[:, 0] is [B, 4, 13] onehot of hole cards; decode top-2 positions
            hole_planes = self.cards[:, 0].to(torch.bool)  # [B, 4, 13]
            # Flatten suit-major indexing to match env encoding (suit*13 + rank)
            # Get indices of True entries per batch
            b, s, r = torch.where(hole_planes)
            # Group by batch: expect exactly two per batch
            hole_indices = torch.full(
                (self.batch_size, 2), -1, dtype=torch.long, device=self.device
            )
            for i in range(b.numel()):
                idx = b[i].item()
                pos = 0 if hole_indices[idx, 0] < 0 else 1
                hole_indices[idx, pos] = (s[i] * 13 + r[i]).to(torch.long)
            return hole_indices
        return self.hole_indices
