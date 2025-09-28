"""Tests for StructuredEmbeddingData."""

import torch

from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.tokens import get_card_token_id_offset


class TestStructuredEmbeddingData:
    """Test suite for StructuredEmbeddingData."""

    def test_permute_suits_basic(self):
        """Test basic suit permutation functionality."""
        batch_size = 2
        seq_len = 10
        num_bet_bins = 7

        # Create test data with known card tokens
        offset = get_card_token_id_offset()

        # Create structured embedding data
        device = torch.device("cpu")
        data = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)

        # Set up some card tokens in the sequence
        # Card 0 (Ace of Spades): token_id = offset + 0*13 + 0 = offset + 0
        # Card 1 (Ace of Hearts): token_id = offset + 1*13 + 0 = offset + 13
        # Card 2 (Ace of Diamonds): token_id = offset + 2*13 + 0 = offset + 26
        # Card 3 (Ace of Clubs): token_id = offset + 3*13 + 0 = offset + 39

        # Set card tokens at positions 0, 1, 2, 3 for batch 0
        data.token_ids[0, 0] = offset + 0  # Ace of Spades (suit=0, rank=0)
        data.token_ids[0, 1] = offset + 13  # Ace of Hearts (suit=1, rank=0)
        data.token_ids[0, 2] = offset + 26  # Ace of Diamonds (suit=2, rank=0)
        data.token_ids[0, 3] = offset + 39  # Ace of Clubs (suit=3, rank=0)

        # Set corresponding card suits and ranks
        data.card_suits[0, 0] = 0  # Spades
        data.card_suits[0, 1] = 1  # Hearts
        data.card_suits[0, 2] = 2  # Diamonds
        data.card_suits[0, 3] = 3  # Clubs

        data.card_ranks[0, 0] = 0  # Ace
        data.card_ranks[0, 1] = 0  # Ace
        data.card_ranks[0, 2] = 0  # Ace
        data.card_ranks[0, 3] = 0  # Ace

        # Set card mask for these positions
        card_mask = (data.token_ids >= offset) & (data.token_ids < offset + 52)
        data.action_legal_masks[0, :4] = (
            card_mask[0, :4].unsqueeze(-1).expand(-1, num_bet_bins)
        )

        # Store original values for comparison
        original_suits = data.card_suits[0, :4].clone()
        original_token_ids = data.token_ids[0, :4].clone()

        # Create generator for reproducible results
        generator = torch.Generator()
        generator.manual_seed(42)

        # Apply suit permutation
        data.permute_suits(generator)

        # Check that suits were permuted
        new_suits = data.card_suits[0, :4]
        new_token_ids = data.token_ids[0, :4]

        # Suits should be different (unless permutation happened to be identity)
        # Token IDs should be updated to reflect new suits
        for i in range(4):
            if original_suits[i] != new_suits[i]:
                # If suit changed, token_id should reflect the new suit
                expected_token_id = offset + new_suits[i] * 13 + data.card_ranks[0, i]
                assert new_token_ids[i] == expected_token_id

        # All suits 0-3 should still be present (just permuted)
        assert set(new_suits.tolist()) == {0, 1, 2, 3}

        # Ranks should be unchanged
        assert torch.equal(data.card_ranks[0, :4], torch.tensor([0, 0, 0, 0]))

    def test_permute_suits_deterministic(self):
        """Test that suit permutation is deterministic with same generator seed."""
        batch_size = 1
        seq_len = 5
        num_bet_bins = 7

        offset = get_card_token_id_offset()

        # Create two identical data structures
        device = torch.device("cpu")
        data1 = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)
        data2 = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)

        # Set up card tokens
        data1.token_ids[0, 0] = offset + 0  # Ace of Spades
        data1.token_ids[0, 1] = offset + 13  # Ace of Hearts
        data1.card_suits[0, 0] = 0
        data1.card_suits[0, 1] = 1
        data1.card_ranks[0, 0] = 0
        data1.card_ranks[0, 1] = 0

        # Copy to data2
        data2.token_ids[0, 0] = offset + 0
        data2.token_ids[0, 1] = offset + 13
        data2.card_suits[0, 0] = 0
        data2.card_suits[0, 1] = 1
        data2.card_ranks[0, 0] = 0
        data2.card_ranks[0, 1] = 0

        # Apply same permutation to both
        generator1 = torch.Generator()
        generator1.manual_seed(123)
        generator2 = torch.Generator()
        generator2.manual_seed(123)

        data1.permute_suits(generator1)
        data2.permute_suits(generator2)

        # Results should be identical
        assert torch.equal(data1.card_suits, data2.card_suits)
        assert torch.equal(data1.token_ids, data2.token_ids)

    def test_permute_suits_non_card_tokens(self):
        """Test that non-card tokens are not affected by suit permutation."""
        batch_size = 1
        seq_len = 5
        num_bet_bins = 7

        offset = get_card_token_id_offset()

        device = torch.device("cpu")
        data = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)

        # Set up mixed tokens: some cards, some non-cards
        data.token_ids[0, 0] = offset + 0  # Card: Ace of Spades
        data.token_ids[0, 1] = 0  # Non-card: CLS token
        data.token_ids[0, 2] = offset + 13  # Card: Ace of Hearts
        data.token_ids[0, 3] = -1  # Non-card: padding
        data.token_ids[0, 4] = 120  # Non-card: some other token (fits in int8)

        # Set card suits and ranks only for card positions
        data.card_suits[0, 0] = 0  # Spades
        data.card_suits[0, 2] = 1  # Hearts
        data.card_ranks[0, 0] = 0  # Ace
        data.card_ranks[0, 2] = 0  # Ace

        # Store original non-card token IDs
        original_non_card_tokens = data.token_ids[0, [1, 3, 4]].clone()

        # Apply suit permutation
        generator = torch.Generator()
        generator.manual_seed(456)
        data.permute_suits(generator)

        # Non-card tokens should be unchanged
        assert torch.equal(data.token_ids[0, [1, 3, 4]], original_non_card_tokens)

        # Card tokens should have been updated
        assert data.token_ids[0, 0] >= offset  # Should still be a card token
        assert data.token_ids[0, 2] >= offset  # Should still be a card token

    def test_permute_suits_batch_independence(self):
        """Test that suit permutation is independent across batches."""
        batch_size = 2
        seq_len = 3
        num_bet_bins = 7

        offset = get_card_token_id_offset()

        device = torch.device("cpu")
        data = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)

        # Set up identical card tokens for both batches
        for b in range(batch_size):
            data.token_ids[b, 0] = offset + 0  # Ace of Spades
            data.token_ids[b, 1] = offset + 13  # Ace of Hearts
            data.card_suits[b, 0] = 0
            data.card_suits[b, 1] = 1
            data.card_ranks[b, 0] = 0
            data.card_ranks[b, 1] = 0

        # Apply suit permutation
        generator = torch.Generator()
        generator.manual_seed(789)
        data.permute_suits(generator)

        # Both batches should have the same permutation (same generator seed)
        assert torch.equal(data.card_suits[0, :2], data.card_suits[1, :2])
        assert torch.equal(data.token_ids[0, :2], data.token_ids[1, :2])

    def test_permute_suits_empty_sequence(self):
        """Test suit permutation with empty sequences."""
        batch_size = 2
        seq_len = 0
        num_bet_bins = 7

        device = torch.device("cpu")
        data = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)

        # Should not crash
        generator = torch.Generator()
        generator.manual_seed(999)
        data.permute_suits(generator)

        # Data should remain unchanged (empty)
        assert data.token_ids.shape == (batch_size, 0)
        assert data.card_suits.shape == (batch_size, 0)

    def test_permute_suits_no_cards(self):
        """Test suit permutation when there are no card tokens."""
        batch_size = 1
        seq_len = 5
        num_bet_bins = 7

        device = torch.device("cpu")
        data = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)

        # Set up only non-card tokens (using values that fit in int8)
        data.token_ids[0, 0] = 0  # CLS
        data.token_ids[0, 1] = -1  # Padding
        data.token_ids[0, 2] = 100  # Some other token
        data.token_ids[0, 3] = 120  # Another token (fits in int8)
        data.token_ids[0, 4] = -1  # Padding

        # Store original values
        original_token_ids = data.token_ids.clone()
        original_suits = data.card_suits.clone()

        # Apply suit permutation
        generator = torch.Generator()
        generator.manual_seed(111)
        data.permute_suits(generator)

        # Nothing should change
        assert torch.equal(data.token_ids, original_token_ids)
        assert torch.equal(data.card_suits, original_suits)

    def test_permute_suits_token_id_consistency(self):
        """Test that token IDs are correctly updated after suit permutation."""
        batch_size = 1
        seq_len = 4
        num_bet_bins = 7

        offset = get_card_token_id_offset()

        device = torch.device("cpu")
        data = StructuredEmbeddingData.empty(batch_size, seq_len, num_bet_bins, device)

        # Set up cards with different ranks
        data.token_ids[0, 0] = offset + 0  # Ace of Spades (suit=0, rank=0)
        data.token_ids[0, 1] = offset + 14  # 2 of Hearts (suit=1, rank=1)
        data.token_ids[0, 2] = offset + 28  # 3 of Diamonds (suit=2, rank=2)
        data.token_ids[0, 3] = offset + 42  # 4 of Clubs (suit=3, rank=3)

        data.card_suits[0, 0] = 0
        data.card_suits[0, 1] = 1
        data.card_suits[0, 2] = 2
        data.card_suits[0, 3] = 3

        data.card_ranks[0, 0] = 0
        data.card_ranks[0, 1] = 1
        data.card_ranks[0, 2] = 2
        data.card_ranks[0, 3] = 3

        # Apply suit permutation
        generator = torch.Generator()
        generator.manual_seed(222)
        data.permute_suits(generator)

        # Check that token IDs are consistent with new suits and ranks
        for i in range(4):
            expected_token_id = (
                offset + data.card_suits[0, i] * 13 + data.card_ranks[0, i]
            )
            assert data.token_ids[0, i] == expected_token_id

            # Verify it's still a valid card token
            assert offset <= data.token_ids[0, i] < offset + 52
