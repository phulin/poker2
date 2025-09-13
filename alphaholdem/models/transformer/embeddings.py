"""Embedding modules for transformer poker model."""

from __future__ import annotations

from typing import Optional
from ...utils.profiling import profile
import torch
import torch.nn as nn
import torch.nn.functional as F


class CardEmbedding(nn.Module):
    """Embedding module for card tokens.

    Composes card embeddings from multiple components:
    - Rank embedding (A, 2, 3, ..., K)
    - Suit embedding (s, h, d, c)
    - Stage embedding (hole, flop, turn, river)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Component embeddings
        self.rank_emb = nn.Embedding(13, d_model)  # A,2,3,...,K
        self.suit_emb = nn.Embedding(4, d_model)  # s,h,d,c
        self.stage_emb = nn.Embedding(4, d_model)  # hole,flop,turn,river

        # Relational attention biases for poker-specific patterns
        self.rank_bias = nn.Parameter(torch.zeros(13, 13))
        self.suit_bias = nn.Parameter(torch.zeros(4, 4))

        # Initialize biases
        self._init_biases()

    def _init_biases(self):
        """Initialize attention biases for poker relationships."""
        # Rank adjacency bias (for straights)
        with torch.no_grad():
            for i in range(13):
                for j in range(13):
                    if abs(i - j) == 1:  # Adjacent ranks
                        self.rank_bias[i, j] = 0.1
                    elif i == j:  # Same rank
                        self.rank_bias[i, j] = 0.2

        # Suit matching bias (for flushes)
        with torch.no_grad():
            for i in range(4):
                for j in range(4):
                    if i == j:  # Same suit
                        self.suit_bias[i, j] = 0.3

    @profile
    def forward(
        self,
        card_indices: torch.Tensor,
        stages: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for card embeddings.

        Args:
            card_indices: Card indices [batch_size, seq_len] (0-51, with 52 for CLS)
            stages: Stage indices [batch_size, seq_len] (0-3)

        Returns:
            Card embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = card_indices.shape

        # Handle CLS token (index 52) and padding (index -1)
        # Create masks for valid card indices
        valid_card_mask = (card_indices >= 0) & (card_indices < 52)
        cls_mask = card_indices == 52

        # Initialize embeddings
        embeddings = torch.zeros(
            batch_size, seq_len, self.d_model, device=card_indices.device
        )

        # Process valid cards (0-51)
        if valid_card_mask.any():
            valid_indices = card_indices[valid_card_mask]
            valid_stages = stages[valid_card_mask]

            # Clamp embedding indices to valid ranges to avoid index errors
            valid_stages = torch.clamp(valid_stages, 0, 3)  # 0-3: hole,flop,turn,river

            # Convert card indices to rank and suit
            ranks = valid_indices % 13
            suits = valid_indices // 13

            # Compose embeddings additively for valid cards
            valid_embeddings = (
                self.rank_emb(ranks)
                + self.suit_emb(suits)
                + self.stage_emb(valid_stages)
            )

            # Place valid embeddings back
            embeddings[valid_card_mask] = valid_embeddings

        # Handle CLS tokens (index 52) - use zero embedding for now
        # In the future, we could add a dedicated CLS embedding
        if cls_mask.any():
            embeddings[cls_mask] = 0.0

        # Padding tokens (index -1) remain zero

        return embeddings

    def get_attention_biases(
        self, ranks: torch.Tensor, suits: torch.Tensor
    ) -> torch.Tensor:
        """Get attention biases for given ranks and suits.

        Args:
            ranks: Rank indices [batch_size, seq_len]
            suits: Suit indices [batch_size, seq_len]

        Returns:
            Attention biases [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = ranks.shape

        # Get rank biases
        rank_biases = self.rank_bias[ranks, ranks.unsqueeze(-1)]  # [batch, seq, seq]

        # Get suit biases
        suit_biases = self.suit_bias[suits, suits.unsqueeze(-1)]  # [batch, seq, seq]

        # Combine biases
        total_biases = rank_biases + suit_biases

        return total_biases


class ActionEmbedding(nn.Module):
    """Embedding module for action tokens.

    Composes action embeddings from:
    - Actor embedding (P1, P2)
    - Street embedding (preflop, flop, turn, river)
    - Legal action masks (8 action types)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Component embeddings
        self.actor_emb = nn.Embedding(2, d_model)  # P1, P2
        self.street_emb = nn.Embedding(4, d_model)  # pre,flop,turn,river

        # Legal action mask processing
        self.legal_mask_mlp = nn.Sequential(
            nn.Linear(8, d_model),  # 8 action types
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Combine embeddings
        self.combine_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # 3 component embeddings
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        actors: torch.Tensor,
        streets: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for action embeddings.

        Args:
            actors: Actor indices [batch_size, seq_len] (0-1)
            streets: Street indices [batch_size, seq_len] (0-3)
            legal_masks: Legal action masks [batch_size, seq_len, 8]

        Returns:
            Action embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = actors.shape

        # Clamp all embedding indices to valid ranges to avoid index errors
        actors = torch.clamp(actors, 0, 1)  # 0-1: player indices
        streets = torch.clamp(streets, 0, 3)  # 0-3: hole,flop,turn,river

        # Get component embeddings
        actor_emb = self.actor_emb(actors)
        street_emb = self.street_emb(streets)

        # Process legal action masks
        legal_mask_emb = self.legal_mask_mlp(legal_masks.float())

        # Combine embeddings
        combined = torch.cat(
            [actor_emb, street_emb, legal_mask_emb], dim=-1
        )  # [batch, seq, d_model * 3]

        # Final combination
        embeddings = self.combine_mlp(combined)

        return embeddings


class ContextEmbedding(nn.Module):
    """Embedding module for context tokens (numeric game state).

    Handles:
    - Pot size embedding
    - Stack size embedding (both players)
    - Committed size embedding (both players)
    - Position embedding (SB, BB)
    - Street-specific context
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Context embeddings
        self.pot_emb = nn.Sequential(
            nn.Linear(1, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Dropout(0.1)
        )

        self.stack_emb = nn.Sequential(
            nn.Linear(2, d_model),  # effective stacks for both players
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.committed_emb = nn.Sequential(
            nn.Linear(2, d_model),  # committed chips for both players
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.position_emb = nn.Embedding(2, d_model)  # SB, BB

        # Street-specific context
        self.street_context_emb = nn.Sequential(
            nn.Linear(4, d_model),  # street, actions_this_round, min_raise, bet_to_call
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for context embeddings.

        Args:
            context_features: Consolidated context features [batch_size, seq_len, 10] (long)
                - 0: pot size
                - 1: our stack
                - 2: opponent stack
                - 3: our committed
                - 4: opponent committed
                - 5: position (0-1)
                - 6: street
                - 7: actions this round
                - 8: min raise
                - 9: bet to call

        Returns:
            Context embeddings [batch_size, seq_len, d_model]
        """
        # Extract individual components from consolidated tensor
        pot_sizes = context_features[:, :, 0:1].float()  # [batch_size, seq_len, 1]
        stack_sizes = context_features[:, :, 1:3].float()  # [batch_size, seq_len, 2]
        committed_sizes = context_features[
            :, :, 3:5
        ].float()  # [batch_size, seq_len, 2]
        positions = context_features[:, :, 5]  # [batch_size, seq_len] - already long
        street_context = context_features[
            :, :, 6:10
        ].float()  # [batch_size, seq_len, 4]

        # Clamp embedding indices to valid ranges
        positions = torch.clamp(positions, 0, 1).long()  # Convert to long for embedding

        # Get component embeddings
        pot_emb = self.pot_emb(pot_sizes)
        stack_emb = self.stack_emb(stack_sizes)
        committed_emb = self.committed_emb(committed_sizes)
        position_emb = self.position_emb(positions)
        street_context_emb = self.street_context_emb(street_context)

        # Combine embeddings additively
        embeddings = (
            pot_emb + stack_emb + committed_emb + position_emb + street_context_emb
        )

        return embeddings
