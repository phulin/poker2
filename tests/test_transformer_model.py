"""Tests for transformer model and embeddings."""

import pytest
import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.factory import ModelFactory
from alphaholdem.models.transformer.embeddings import (
    ActionEmbedding,
    CardEmbedding,
    ContextEmbedding,
)
from alphaholdem.models.transformer.embedding_data import StructuredEmbeddingData
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder


class TestCardEmbedding:
    """Test CardEmbedding functionality."""

    def test_card_embedding_creation(self):
        """Test CardEmbedding can be created with proper parameters."""
        embedding = CardEmbedding(num_bet_bins=8, d_model=128)
        assert embedding.d_model == 128
        assert embedding.num_bet_bins == 8

    def test_card_embedding_forward(self):
        """Test CardEmbedding forward pass."""
        batch_size = 2
        max_seq_len = 42  # Full sequence length
        d_model = 128
        num_bet_bins = 8

        embedding = CardEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

        # Create dummy input data with full sequence length
        token_ids = torch.randint(0, 52, (batch_size, max_seq_len))
        card_ranks = torch.randint(0, 13, (batch_size, max_seq_len))  # 0-12 for ranks
        card_suits = torch.randint(0, 4, (batch_size, max_seq_len))  # 0-3 for suits
        card_streets = torch.randint(0, 4, (batch_size, max_seq_len))  # 0-3 for streets

        # Test forward pass
        output = embedding(token_ids, card_ranks, card_suits, card_streets)
        assert output.shape == (
            batch_size,
            7,
            d_model,
        )  # CardEmbedding returns 7 positions

    def test_card_embedding_with_invalid_tokens(self):
        """Test CardEmbedding handles invalid tokens correctly."""
        batch_size = 2
        max_seq_len = 42
        d_model = 128
        num_bet_bins = 8

        embedding = CardEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

        # Create input with some invalid tokens (-1) in the full sequence
        token_ids = torch.randint(0, 52, (batch_size, max_seq_len))
        token_ids[0, 3] = -1  # Invalid token at position 3
        token_ids[0, 5] = -1  # Invalid token at position 5
        token_ids[1, 6] = -1  # Invalid token at position 6

        card_ranks = torch.randint(0, 13, (batch_size, max_seq_len))
        card_suits = torch.randint(0, 4, (batch_size, max_seq_len))
        card_streets = torch.randint(0, 4, (batch_size, max_seq_len))

        # Test forward pass
        output = embedding(token_ids, card_ranks, card_suits, card_streets)
        assert output.shape == (batch_size, 7, d_model)
        # Note: The invalid tokens are in the full sequence, but CardEmbedding only processes positions 1-8
        # So we need to check the corresponding positions in the card range


class TestActionEmbedding:
    """Test ActionEmbedding functionality."""

    def test_action_embedding_creation(self):
        """Test ActionEmbedding can be created with proper parameters."""
        embedding = ActionEmbedding(num_bet_bins=8, d_model=128)
        assert embedding.d_model == 128
        assert embedding.num_bet_bins == 8

    def test_action_embedding_forward(self):
        """Test ActionEmbedding forward pass."""
        batch_size = 2
        max_seq_len = 42  # Full sequence length
        d_model = 128
        num_bet_bins = 8

        embedding = ActionEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

        # Create dummy input data with full sequence length
        token_ids = torch.randint(0, 60, (batch_size, max_seq_len))
        action_actors = torch.randint(0, 2, (batch_size, max_seq_len))
        action_streets = torch.randint(0, 4, (batch_size, max_seq_len))
        action_legal_masks = torch.randint(
            0, 2, (batch_size, max_seq_len, num_bet_bins)
        ).float()

        # Test forward pass
        output = embedding(token_ids, action_actors, action_streets, action_legal_masks)
        assert output.shape == (
            batch_size,
            24,
            d_model,
        )  # ActionEmbedding returns 24 positions

    def test_action_embedding_with_invalid_tokens(self):
        """Test ActionEmbedding handles invalid tokens correctly."""
        batch_size = 2
        max_seq_len = 42
        d_model = 128
        num_bet_bins = 8

        embedding = ActionEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

        # Create input with some invalid tokens (-1) in the full sequence
        token_ids = torch.randint(-1, 60, (batch_size, max_seq_len))
        action_actors = torch.randint(0, 2, (batch_size, max_seq_len))
        action_streets = torch.randint(0, 4, (batch_size, max_seq_len))
        action_legal_masks = torch.randint(
            0, 2, (batch_size, max_seq_len, num_bet_bins)
        ).float()

        # Test forward pass
        output = embedding(token_ids, action_actors, action_streets, action_legal_masks)
        assert output.shape == (batch_size, 24, d_model)


class TestContextEmbedding:
    """Test ContextEmbedding functionality."""

    def test_context_embedding_creation(self):
        """Test ContextEmbedding can be created with proper parameters."""
        embedding = ContextEmbedding(num_bet_bins=8, d_model=128)
        assert embedding.d_model == 128
        assert embedding.num_bet_bins == 8

    def test_context_embedding_forward(self):
        """Test ContextEmbedding forward pass."""
        batch_size = 2
        max_seq_len = 42  # Full sequence length
        d_model = 128
        num_bet_bins = 8

        embedding = ContextEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

        # Create dummy input data with full sequence length
        token_ids = torch.randint(0, 60, (batch_size, max_seq_len))
        context_features = torch.randint(0, 1000, (batch_size, max_seq_len, 10)).float()

        # Test forward pass
        output = embedding(token_ids, context_features)
        assert output.shape == (
            batch_size,
            10,
            d_model,
        )  # ContextEmbedding returns 10 positions


class TestPokerTransformerV1:
    """Test PokerTransformerV1 model functionality."""

    def test_transformer_creation(self):
        """Test PokerTransformerV1 can be created with proper parameters."""
        model = PokerTransformerV1(
            d_model=128,
            n_layers=4,
            n_heads=4,
            num_bet_bins=8,
            dropout=0.1,
        )
        assert model.d_model == 128
        assert model.n_layers == 4
        assert model.n_heads == 4
        # Note: num_bet_bins is not stored as an attribute in the model

    def test_transformer_forward_with_dummy_data(self):
        """Test PokerTransformerV1 forward pass with dummy structured data."""
        batch_size = 2
        d_model = 128
        num_bet_bins = 8

        model = PokerTransformerV1(
            d_model=d_model,
            n_layers=2,  # Smaller for testing
            n_heads=4,
            num_bet_bins=num_bet_bins,
            dropout=0.1,
        )

        # Create dummy structured data with proper sequence length
        max_seq_len = 42  # CLS(1) + cards(7) + actions(24) + context(10) = 42
        structured_data = StructuredEmbeddingData(
            token_ids=torch.randint(0, 60, (batch_size, max_seq_len)),
            card_ranks=torch.randint(0, 13, (batch_size, max_seq_len)),
            card_suits=torch.randint(0, 4, (batch_size, max_seq_len)),
            card_streets=torch.randint(0, 4, (batch_size, max_seq_len)),
            action_actors=torch.randint(0, 2, (batch_size, max_seq_len)),
            action_streets=torch.randint(0, 4, (batch_size, max_seq_len)),
            action_legal_masks=torch.randint(
                0, 2, (batch_size, max_seq_len, num_bet_bins)
            ).float(),
            context_features=torch.randint(
                0, 1000, (batch_size, max_seq_len, 10)
            ).float(),
        )

        # Test forward pass
        with torch.no_grad():
            outputs = model(structured_data)
            assert "policy_logits" in outputs
            assert "value" in outputs
            assert outputs["policy_logits"].shape == (batch_size, num_bet_bins)
            assert outputs["value"].shape == (batch_size,)

    def test_transformer_with_real_environment(self):
        """Test PokerTransformerV1 with real environment data."""
        device = torch.device("cpu")

        # Create environment
        env = HUNLTensorEnv(
            num_envs=2,
            starting_stack=20000,
            sb=50,
            bb=100,
            bet_bins=[0.5, 1.0, 1.5, 2.0],
            device=device,  # Pass device to environment
        )

        # Create state encoder
        state_encoder = ModelFactory.create_state_encoder(
            "transformer", device, tensor_env=env
        )

        # Create model
        model_config = {
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "num_bet_bins": 8,
            "dropout": 0.1,
            "use_auxiliary_loss": False,
            "use_gradient_checkpointing": False,
        }
        model = ModelFactory.create_model("transformer", model_config, device)

        # Get environment observation
        obs = env.reset()

        # Encode observation
        structured_data = state_encoder.encode_tensor_states(0, torch.arange(2))

        # Test forward pass
        with torch.no_grad():
            outputs = model(structured_data)
            assert "policy_logits" in outputs
            assert "value" in outputs
            assert outputs["policy_logits"].shape == (2, 8)
            assert outputs["value"].shape == (2,)

    def test_transformer_gradient_checkpointing(self):
        """Test PokerTransformerV1 with gradient checkpointing enabled."""
        batch_size = 2
        d_model = 128
        num_bet_bins = 8

        model = PokerTransformerV1(
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            num_bet_bins=num_bet_bins,
            dropout=0.1,
            use_gradient_checkpointing=True,
        )

        # Create dummy structured data
        max_seq_len = 42
        structured_data = StructuredEmbeddingData(
            token_ids=torch.randint(0, 60, (batch_size, max_seq_len)),
            card_ranks=torch.randint(0, 13, (batch_size, max_seq_len)),
            card_suits=torch.randint(0, 4, (batch_size, max_seq_len)),
            card_streets=torch.randint(0, 4, (batch_size, max_seq_len)),
            action_actors=torch.randint(0, 2, (batch_size, max_seq_len)),
            action_streets=torch.randint(0, 4, (batch_size, max_seq_len)),
            action_legal_masks=torch.randint(
                0, 2, (batch_size, max_seq_len, num_bet_bins)
            ).float(),
            context_features=torch.randint(
                0, 1000, (batch_size, max_seq_len, 10)
            ).float(),
        )

        # Test forward pass
        outputs = model(structured_data)
        assert "policy_logits" in outputs
        assert "value" in outputs
        assert outputs["policy_logits"].shape == (batch_size, num_bet_bins)
        assert outputs["value"].shape == (batch_size,)

    def test_transformer_different_batch_sizes(self):
        """Test PokerTransformerV1 with different batch sizes."""
        d_model = 128
        num_bet_bins = 8

        model = PokerTransformerV1(
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            num_bet_bins=num_bet_bins,
            dropout=0.1,
        )

        # Test with different batch sizes
        for batch_size in [1, 4, 8]:
            max_seq_len = 42
            structured_data = StructuredEmbeddingData(
                token_ids=torch.randint(0, 60, (batch_size, max_seq_len)),
                card_ranks=torch.randint(0, 13, (batch_size, max_seq_len)),
                card_suits=torch.randint(0, 4, (batch_size, max_seq_len)),
                card_streets=torch.randint(0, 4, (batch_size, max_seq_len)),
                action_actors=torch.randint(0, 2, (batch_size, max_seq_len)),
                action_streets=torch.randint(0, 4, (batch_size, max_seq_len)),
                action_legal_masks=torch.randint(
                    0, 2, (batch_size, max_seq_len, num_bet_bins)
                ).float(),
                context_features=torch.randint(
                    0, 1000, (batch_size, max_seq_len, 10)
                ).float(),
            )

            with torch.no_grad():
                outputs = model(structured_data)
                assert outputs["policy_logits"].shape == (batch_size, num_bet_bins)
                assert outputs["value"].shape == (batch_size,)


class TestEmbeddingIntegration:
    """Test integration between different embedding types."""

    def test_embedding_dimensions_match(self):
        """Test that all embeddings produce consistent dimensions."""
        d_model = 128
        num_bet_bins = 8
        batch_size = 2

        # Create embeddings
        card_embedding = CardEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)
        action_embedding = ActionEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)
        context_embedding = ContextEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

        # Create dummy data with full sequence length
        max_seq_len = 42  # Full sequence length

        token_ids = torch.randint(0, 60, (batch_size, max_seq_len))

        # Test card embedding
        card_ranks = torch.randint(0, 13, (batch_size, max_seq_len))
        card_suits = torch.randint(0, 4, (batch_size, max_seq_len))
        card_streets = torch.randint(0, 4, (batch_size, max_seq_len))
        card_output = card_embedding(token_ids, card_ranks, card_suits, card_streets)
        assert card_output.shape == (
            batch_size,
            7,
            d_model,
        )  # CardEmbedding returns 7 positions

        # Test action embedding
        action_actors = torch.randint(0, 2, (batch_size, max_seq_len))
        action_streets = torch.randint(0, 4, (batch_size, max_seq_len))
        action_legal_masks = torch.randint(
            0, 2, (batch_size, max_seq_len, num_bet_bins)
        ).float()
        action_output = action_embedding(
            token_ids, action_actors, action_streets, action_legal_masks
        )
        assert action_output.shape == (
            batch_size,
            24,
            d_model,
        )  # ActionEmbedding returns 24 positions

        # Test context embedding
        context_features = torch.randint(0, 1000, (batch_size, max_seq_len, 10)).float()
        context_output = context_embedding(token_ids, context_features)
        assert context_output.shape == (
            batch_size,
            10,
            d_model,
        )  # ContextEmbedding returns 10 positions

    def test_embedding_device_consistency(self):
        """Test that embeddings work on different devices."""
        d_model = 128
        num_bet_bins = 8
        batch_size = 2
        seq_len = 7

        # Test on CPU
        device = torch.device("cpu")
        embedding = CardEmbedding(num_bet_bins=num_bet_bins, d_model=d_model).to(device)

        max_seq_len = 42
        token_ids = torch.randint(0, 60, (batch_size, max_seq_len), device=device)
        card_ranks = torch.randint(0, 13, (batch_size, max_seq_len), device=device)
        card_suits = torch.randint(0, 4, (batch_size, max_seq_len), device=device)
        card_streets = torch.randint(0, 4, (batch_size, max_seq_len), device=device)

        output = embedding(token_ids, card_ranks, card_suits, card_streets)
        assert output.device == device
        assert output.shape == (
            batch_size,
            7,
            d_model,
        )  # CardEmbedding returns 7 positions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
