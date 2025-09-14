"""Embedding modules for transformer poker model."""

from __future__ import annotations


import torch
import torch.nn as nn

from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder

from ...utils.profiling import profile


class CardEmbedding(nn.Module):
    """Embedding module for card tokens.

    Composes card embeddings from multiple components:
    - Rank embedding (A, 2, 3, ..., K)
    - Suit embedding (s, h, d, c)
    - Stage embedding (hole, flop, turn, river)
    """

    def __init__(self, num_bet_bins: int, d_model: int = 256):
        super().__init__()
        self.num_bet_bins = num_bet_bins
        self.d_model = d_model

        # Component embeddings
        self.rank_emb = nn.Embedding(13, d_model)  # A,2,3,...,K
        self.suit_emb = nn.Embedding(4, d_model)  # s,h,d,c
        self.street_emb = nn.Embedding(4, d_model)  # hole,flop,turn,river

        # Relational attention biases for poker-specific patterns
        self.rank_bias = nn.Parameter(torch.zeros(13, 13))
        self.suit_bias = nn.Parameter(torch.zeros(4, 4))

        # Initialize biases
        self._init_biases()

    def _init_biases(self):
        """Initialize attention biases for poker relationships."""
        with torch.no_grad():
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

    @profile
    def forward(
        self,
        token_ids: torch.Tensor,
        card_streets: torch.Tensor,
        card_ranks: torch.Tensor,
        card_suits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for card embeddings.

        Args:
            token_ids: Token IDs [batch_size, seq_len] (0-51, with 52 for CLS)
            card_streets: Street indices [batch_size, seq_len] (0-3)
            card_ranks: Card ranks [batch_size, seq_len] (0-12, with -1 for invalid)
            card_suits: Card suits [batch_size, seq_len] (0-3, with -1 for invalid)

        Returns:
            Card embeddings [batch_size, seq_len, d_model]
        """
        batch_size, _ = token_ids.shape

        start = TransformerStateEncoder.get_card_index_offset()
        end = TransformerStateEncoder.get_action_index_offset()

        # Get card data for the card range
        card_ranks_slice = card_ranks[:, start:end]
        card_suits_slice = card_suits[:, start:end]
        card_streets_slice = card_streets[:, start:end]

        # Create mask for valid cards (not -1)
        valid_mask = token_ids[:, start:end] >= 0

        # Create full-sized tensor for this slice
        slice_embeddings = torch.zeros(
            batch_size,
            end - start,
            self.d_model,
            device=token_ids.device,
            dtype=self.rank_emb.weight.dtype,
        )

        # Get embeddings for valid positions only
        valid_indices = torch.where(valid_mask)
        rank_emb = self.rank_emb(card_ranks_slice[valid_indices])
        suit_emb = self.suit_emb(card_suits_slice[valid_indices])
        stage_emb = self.street_emb(card_streets_slice[valid_indices])

        # Combine embeddings additively
        combined_emb = rank_emb + suit_emb + stage_emb

        # Place valid embeddings back
        slice_embeddings[valid_indices] = combined_emb

        # Return only the card range embeddings
        return slice_embeddings

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

    def __init__(self, num_bet_bins: int, d_model: int = 256):
        super().__init__()
        self.num_bet_bins = num_bet_bins
        self.d_model = d_model

        # Component embeddings
        self.actor_emb = nn.Embedding(2, d_model)  # P1, P2
        self.street_emb = nn.Embedding(4, d_model)  # pre,flop,turn,river

        start = TransformerStateEncoder.get_action_index_offset()
        end = TransformerStateEncoder.get_context_index_offset()

        self.pos_emb = nn.Embedding(
            end - start, d_model
        )  # Positional embedding for 24 action slots

        # Legal action mask processing
        self.legal_mask_mlp = nn.Sequential(
            nn.Linear(num_bet_bins, d_model),  # num_bet_bins action types
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        action_actors: torch.Tensor,
        action_streets: torch.Tensor,
        action_legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for action embeddings.

        Args:
            action_actors: Actor indices [batch_size, seq_len] (0-1)
            action_streets: Street indices [batch_size, seq_len] (0-3)
            action_legal_masks: Legal action masks [batch_size, seq_len, num_bet_bins]

        Returns:
            Action embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = action_actors.shape

        start = TransformerStateEncoder.get_action_index_offset()
        end = TransformerStateEncoder.get_context_index_offset()

        # Get action data for the action range
        action_actors_slice = action_actors[:, start:end]
        action_streets_slice = action_streets[:, start:end]
        action_legal_masks_slice = action_legal_masks[:, start:end, :]

        # Create mask for valid tokens (non-negative token_ids)
        valid_mask = token_ids[:, start:end] >= 0
        valid_indices = torch.where(valid_mask)

        # Create full-sized tensor for this slice
        slice_embeddings = torch.zeros(
            batch_size,
            end - start,
            self.d_model,
            device=action_actors.device,
            dtype=self.actor_emb.weight.dtype,
        )

        # Get embeddings for valid positions only
        if valid_indices[0].numel() > 0:  # Only compute if there are valid tokens
            actor_emb = self.actor_emb(action_actors_slice[valid_indices])
            street_emb = self.street_emb(action_streets_slice[valid_indices])

            # Create positional embeddings for valid action slots
            valid_positions = valid_indices[
                1
            ]  # Position indices within the action range
            pos_emb = self.pos_emb(valid_positions)

            # Process legal action masks for valid positions
            legal_mask_emb = self.legal_mask_mlp(
                action_legal_masks_slice[valid_indices].float()
            )

            # Combine embeddings additively
            combined_emb = actor_emb + street_emb + pos_emb + legal_mask_emb

            # Place valid embeddings back
            slice_embeddings[valid_indices] = combined_emb

        # Return only the action range embeddings
        return slice_embeddings


class ContextEmbedding(nn.Module):
    """Embedding module for context tokens (numeric game state).

    Handles:
    - Pot size embedding
    - Stack size embedding (both players)
    - Committed size embedding (both players)
    - Position embedding (SB, BB)
    - Street-specific context
    """

    def __init__(self, num_bet_bins: int, d_model: int = 256):
        super().__init__()
        self.num_bet_bins = num_bet_bins
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
        token_ids: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for context embeddings.

        Args:
            token_ids: Token IDs [batch_size, seq_len] (0-51, with 52 for CLS)
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

        start = TransformerStateEncoder.get_context_index_offset()
        end = start + 10  # Context has 10 features

        value_start = TransformerStateEncoder.get_context_token_offset(
            self.num_bet_bins
        )
        value_end = value_start + 10

        if (
            (token_ids[:, start:end] < value_start)
            | (token_ids[:, start:end] > value_end)
        ).any():
            print("WARNING: Invalid tokens found in context embeddings.")

        # Extract individual components from consolidated tensor for context range
        pot_sizes = context_features[
            :, start:end, 0:1
        ].float()  # [batch_size, seq_len, 1]
        stack_sizes = context_features[
            :, start:end, 1:3
        ].float()  # [batch_size, seq_len, 2]
        committed_sizes = context_features[
            :, start:end, 3:5
        ].float()  # [batch_size, seq_len, 2]
        positions = context_features[
            :, start:end, 5
        ]  # [batch_size, seq_len] - already long
        street_context = context_features[
            :, start:end, 6:10
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

        # Return only the context range embeddings
        return embeddings
