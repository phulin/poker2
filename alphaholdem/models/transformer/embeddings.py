"""Embedding modules for transformer poker model."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CardEmbedding(nn.Module):
    """Embedding module for card tokens.

    Composes card embeddings from multiple components:
    - Rank embedding (A, 2, 3, ..., K)
    - Suit embedding (s, h, d, c)
    - Stage embedding (hole, flop, turn, river)
    - Visibility embedding (self, opponent, public)
    - Order embedding (position within stage)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Component embeddings
        self.rank_emb = nn.Embedding(13, d_model)  # A,2,3,...,K
        self.suit_emb = nn.Embedding(4, d_model)  # s,h,d,c
        self.stage_emb = nn.Embedding(4, d_model)  # hole,flop,turn,river
        self.visibility_emb = nn.Embedding(3, d_model)  # self,opponent,public
        self.order_emb = nn.Embedding(5, d_model)  # position within stage

        # Relational attention biases for poker-specific patterns
        self.rank_bias = nn.Parameter(torch.zeros(13, 13))
        self.suit_bias = nn.Parameter(torch.zeros(4, 4))

        # Initialize biases
        self._init_biases()

    def _init_biases(self):
        """Initialize attention biases for poker relationships."""
        # Rank adjacency bias (for straights)
        for i in range(13):
            for j in range(13):
                if abs(i - j) == 1:  # Adjacent ranks
                    self.rank_bias[i, j] = 0.1
                elif i == j:  # Same rank
                    self.rank_bias[i, j] = 0.2

        # Suit matching bias (for flushes)
        for i in range(4):
            for j in range(4):
                if i == j:  # Same suit
                    self.suit_bias[i, j] = 0.3

    def forward(
        self,
        card_indices: torch.Tensor,
        stages: torch.Tensor,
        visibility: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for card embeddings.

        Args:
            card_indices: Card indices [batch_size, seq_len] (0-51)
            stages: Stage indices [batch_size, seq_len] (0-3)
            visibility: Visibility indices [batch_size, seq_len] (0-2)
            order: Order indices [batch_size, seq_len] (0-4)

        Returns:
            Card embeddings [batch_size, seq_len, d_model]
        """
        # Convert card indices to rank and suit
        ranks = card_indices % 13  # [batch_size, seq_len]
        suits = card_indices // 13  # [batch_size, seq_len]

        # Compose embeddings additively
        embeddings = (
            self.rank_emb(ranks)
            + self.suit_emb(suits)
            + self.stage_emb(stages)
            + self.visibility_emb(visibility)
            + self.order_emb(order)
        )

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
    - Action type embedding (fold, check, call, bet, raise, allin)
    - Street embedding (preflop, flop, turn, river)
    - Size bin embedding (coarse bet size categories)
    - Size MLP (fine bet size features)
    """

    def __init__(self, d_model: int = 256, num_action_types: int = 6):
        super().__init__()
        self.d_model = d_model
        self.num_action_types = num_action_types

        # Component embeddings
        self.actor_emb = nn.Embedding(2, d_model)  # P1, P2
        self.action_type_emb = nn.Embedding(num_action_types, d_model)
        self.street_emb = nn.Embedding(4, d_model)  # pre,flop,turn,river
        self.size_bin_emb = nn.Embedding(20, d_model)  # coarse size bins

        # Fine size features MLP
        self.size_mlp = nn.Sequential(
            nn.Linear(3, d_model),  # fraction_of_pot, fraction_of_stack, log_chips
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Combine embeddings
        self.combine_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model),  # 4 component embeddings
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        actors: torch.Tensor,
        action_types: torch.Tensor,
        streets: torch.Tensor,
        size_bins: torch.Tensor,
        size_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for action embeddings.

        Args:
            actors: Actor indices [batch_size, seq_len] (0-1)
            action_types: Action type indices [batch_size, seq_len] (0-5)
            streets: Street indices [batch_size, seq_len] (0-3)
            size_bins: Size bin indices [batch_size, seq_len] (0-19)
            size_features: Size features [batch_size, seq_len, 3]

        Returns:
            Action embeddings [batch_size, seq_len, d_model]
        """
        # Get component embeddings
        actor_emb = self.actor_emb(actors)
        action_type_emb = self.action_type_emb(action_types)
        street_emb = self.street_emb(streets)
        size_bin_emb = self.size_bin_emb(size_bins)

        # Process fine size features
        size_mlp_emb = self.size_mlp(size_features)

        # Combine embeddings
        combined = torch.cat(
            [actor_emb, action_type_emb, street_emb, size_bin_emb], dim=-1
        )  # [batch, seq, d_model * 4]

        embeddings = self.combine_mlp(combined)

        return embeddings


class ContextEmbedding(nn.Module):
    """Embedding module for context tokens (numeric game state).

    Handles:
    - Pot size embedding
    - Stack size embedding (both players)
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
        pot_sizes: torch.Tensor,
        stack_sizes: torch.Tensor,
        positions: torch.Tensor,
        street_context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for context embeddings.

        Args:
            pot_sizes: Pot sizes [batch_size, seq_len, 1]
            stack_sizes: Stack sizes [batch_size, seq_len, 2]
            positions: Position indices [batch_size, seq_len] (0-1)
            street_context: Street context [batch_size, seq_len, 4]

        Returns:
            Context embeddings [batch_size, seq_len, d_model]
        """
        # Get component embeddings
        pot_emb = self.pot_emb(pot_sizes)
        stack_emb = self.stack_emb(stack_sizes)
        position_emb = self.position_emb(positions)
        street_context_emb = self.street_context_emb(street_context)

        # Combine embeddings additively
        embeddings = pot_emb + stack_emb + position_emb + street_context_emb

        return embeddings
