"""Custom attention layers with poker-specific biases."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PokerAttention(nn.Module):
    """Multi-head attention with poker-specific biases.

    Incorporates card relationship biases (rank adjacency, suit matching)
    and positional biases for poker-specific patterns.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_poker_biases: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_poker_biases = use_poker_biases

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Poker-specific biases
        if use_poker_biases:
            self.rank_bias = nn.Parameter(torch.zeros(13, 13))
            self.suit_bias = nn.Parameter(torch.zeros(4, 4))
            self.position_bias = nn.Parameter(torch.zeros(50, 50))  # Max seq len
            self._init_poker_biases()

    def _init_poker_biases(self):
        """Initialize poker-specific attention biases."""
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

            # Position bias (temporal relationships)
            for i in range(50):
                for j in range(50):
                    if abs(i - j) <= 2:  # Close positions
                        self.position_bias[i, j] = 0.1

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        card_indices: Optional[torch.Tensor] = None,
        position_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with poker-specific biases.

        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            card_indices: Card indices for poker biases [batch_size, seq_len]
            position_indices: Position indices for poker biases [batch_size, seq_len]

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = query.shape

        # Linear projections
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        # Add poker-specific biases
        if self.use_poker_biases and card_indices is not None:
            poker_bias = self._compute_poker_bias(card_indices, position_indices)
            scores = scores + poker_bias.unsqueeze(1)  # Add to all heads

        # Apply mask
        if mask is not None:
            # mask is key_padding_mask: [batch_size, seq_len] with True for positions to mask
            # We need to expand it to [batch_size, n_heads, seq_len, seq_len]
            key_padding_mask = mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(key_padding_mask, -1e9)

        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Reshape and project output
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        output = self.w_o(context)

        return output, attention_weights

    def _compute_poker_bias(
        self,
        card_indices: torch.Tensor,
        position_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute poker-specific attention biases (vectorized version).

        Args:
            card_indices: Card indices [batch_size, seq_len]
            position_indices: Position indices [batch_size, seq_len]

        Returns:
            Bias tensor [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = card_indices.shape
        device = card_indices.device

        # Initialize bias tensor
        bias = torch.zeros(batch_size, seq_len, seq_len, device=device)

        # Create valid card mask (exclude CLS token and padding)
        valid_mask = (card_indices >= 0) & (card_indices < 52)

        if not valid_mask.any():
            return bias

        # Vectorized computation for valid cards only
        valid_cards = card_indices[valid_mask]

        # Convert to ranks and suits for valid cards
        ranks = valid_cards % 13
        suits = valid_cards // 13

        # Create expanded tensors for pairwise computation
        # ranks_i: [num_valid, 1], ranks_j: [1, num_valid]
        ranks_i = ranks.unsqueeze(1)  # [num_valid, 1]
        ranks_j = ranks.unsqueeze(0)  # [1, num_valid]
        suits_i = suits.unsqueeze(1)  # [num_valid, 1]
        suits_j = suits.unsqueeze(0)  # [1, num_valid]

        # Compute rank biases vectorized
        rank_diff = torch.abs(ranks_i - ranks_j)
        rank_bias = torch.zeros_like(rank_diff, dtype=torch.float)
        rank_bias[rank_diff == 1] = 0.1  # Adjacent ranks
        rank_bias[rank_diff == 0] = 0.2  # Same rank

        # Compute suit biases vectorized
        suit_match = suits_i == suits_j
        suit_bias = torch.zeros_like(suit_match, dtype=torch.float)
        suit_bias[suit_match] = 0.3  # Same suit

        # Combine biases
        combined_bias = rank_bias + suit_bias

        # Add position bias if available
        if position_indices is not None:
            valid_positions = position_indices[valid_mask]
            pos_i = valid_positions.unsqueeze(1)  # [num_valid, 1]
            pos_j = valid_positions.unsqueeze(0)  # [1, num_valid]
            pos_diff = torch.abs(pos_i - pos_j)
            pos_bias = torch.zeros_like(pos_diff, dtype=torch.float)
            pos_bias[pos_diff <= 2] = 0.1  # Close positions
            combined_bias += pos_bias

        # Map back to full bias tensor
        # Create indices for valid positions
        valid_indices = torch.where(valid_mask)
        batch_indices = valid_indices[0]  # [num_valid]
        seq_indices = valid_indices[1]  # [num_valid]

        # Create meshgrid for pairwise indices
        i_indices = seq_indices.unsqueeze(1)  # [num_valid, 1]
        j_indices = seq_indices.unsqueeze(0)  # [1, num_valid]

        # Map biases back to full tensor
        batch_i = batch_indices.unsqueeze(1).expand(-1, len(seq_indices))
        batch_j = batch_indices.unsqueeze(0).expand(len(seq_indices), -1)

        bias[batch_i, i_indices, j_indices] = combined_bias

        return bias


class PokerTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with poker-specific attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_poker_biases: bool = True,
    ):
        super().__init__()
        self.self_attn = PokerAttention(d_model, n_heads, dropout, use_poker_biases)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        card_indices: Optional[torch.Tensor] = None,
        position_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with poker-specific attention."""
        # Convert attention mask to key padding mask format
        # attention_mask: [batch_size, seq_len] with 1.0 for valid positions, 0.0 for padding
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0.0  # True for positions to mask out

        # Self-attention with poker biases
        attn_output, _ = self.self_attn(
            src, src, src, key_padding_mask, card_indices, position_indices
        )
        src = self.norm1(src + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))

        return src
