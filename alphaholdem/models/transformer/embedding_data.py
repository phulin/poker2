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
    card_indices: torch.Tensor  # Token IDs (0-51 for cards, 52 for CLS, -1 for padding)
    card_stages: torch.Tensor  # Stage indices (0-3: hole, flop, turn, river)

    # Action components [batch_size, seq_len]
    action_actors: torch.Tensor  # Actor indices (0-1: player indices)
    action_streets: torch.Tensor  # Street indices (0-3: hole, flop, turn, river)
    action_legal_masks: torch.Tensor  # Legal action masks [batch_size, seq_len, 8]

    # Context components [batch_size, seq_len, ...]
    context_pot_sizes: torch.Tensor  # Pot sizes [batch_size, seq_len, 1]
    context_stack_sizes: torch.Tensor  # Stack sizes [batch_size, seq_len, 2]
    context_committed_sizes: torch.Tensor  # Committed sizes [batch_size, seq_len, 2]
    context_positions: torch.Tensor  # Position indices [batch_size, seq_len]
    context_street: torch.Tensor  # Street context [batch_size, seq_len, 4]
    context_actions_this_round: torch.Tensor  # Actions this round [batch_size, seq_len]
    context_min_raise: torch.Tensor  # Min raise amounts [batch_size, seq_len]
    context_bet_to_call: torch.Tensor  # Bet to call amounts [batch_size, seq_len]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for model forward pass."""
        return {
            "card_indices": self.card_indices,
            "card_stages": self.card_stages,
            "action_actors": self.action_actors,
            "action_streets": self.action_streets,
            "action_legal_masks": self.action_legal_masks,
            "context_pot_sizes": self.context_pot_sizes,
            "context_stack_sizes": self.context_stack_sizes,
            "context_committed_sizes": self.context_committed_sizes,
            "context_positions": self.context_positions,
            "context_street": self.context_street,
            "context_actions_this_round": self.context_actions_this_round,
            "context_min_raise": self.context_min_raise,
            "context_bet_to_call": self.context_bet_to_call,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> StructuredEmbeddingData:
        """Create from dictionary format."""
        return cls(
            card_indices=data["card_indices"],
            card_stages=data["card_stages"],
            action_actors=data["action_actors"],
            action_streets=data["action_streets"],
            action_legal_masks=data["action_legal_masks"],
            context_pot_sizes=data["context_pot_sizes"],
            context_stack_sizes=data["context_stack_sizes"],
            context_committed_sizes=data["context_committed_sizes"],
            context_positions=data["context_positions"],
            context_street=data["context_street"],
            context_actions_this_round=data["context_actions_this_round"],
            context_min_raise=data["context_min_raise"],
            context_bet_to_call=data["context_bet_to_call"],
        )

    def to_device(self, device: torch.device) -> StructuredEmbeddingData:
        """Move all tensors to specified device."""
        return StructuredEmbeddingData(
            card_indices=self.card_indices.to(device),
            card_stages=self.card_stages.to(device),
            action_actors=self.action_actors.to(device),
            action_streets=self.action_streets.to(device),
            action_legal_masks=self.action_legal_masks.to(device),
            context_pot_sizes=self.context_pot_sizes.to(device),
            context_stack_sizes=self.context_stack_sizes.to(device),
            context_committed_sizes=self.context_committed_sizes.to(device),
            context_positions=self.context_positions.to(device),
            context_street=self.context_street.to(device),
            context_actions_this_round=self.context_actions_this_round.to(device),
            context_min_raise=self.context_min_raise.to(device),
            context_bet_to_call=self.context_bet_to_call.to(device),
        )

    def __len__(self) -> int:
        """Return batch size."""
        return self.card_indices.shape[0]

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.card_indices.shape[0]

    @property
    def seq_len(self) -> int:
        """Get sequence length."""
        return self.card_indices.shape[1]

    @property
    def device(self) -> torch.device:
        """Get device of tensors."""
        return self.card_indices.device
