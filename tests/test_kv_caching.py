"""Tests for KV caching functionality in poker transformer models."""

import pytest
import torch
import torch.nn as nn

from alphaholdem.models.transformer.poker_transformer import (
    PokerTransformerV1,
    TransformerLayer,
)
from alphaholdem.models.transformer.rotary_attention import RotarySelfAttention
from alphaholdem.models.transformer.kv_cache_manager import (
    KVCacheManager,
    SelfPlayKVCacheManager,
)
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.model_outputs import ModelOutput


class TestRotaryAttentionKVCaching:
    """Test KV caching in RotarySelfAttention."""

    def test_rotary_attention_kv_cache_basic(self):
        """Test basic KV caching functionality."""
        d_model = 128
        n_heads = 4
        batch_size = 2
        seq_len = 10

        attention = RotarySelfAttention(d_model, n_heads)

        # Create test data
        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        cos = torch.randn(seq_len, d_model // n_heads)
        sin = torch.randn(seq_len, d_model // n_heads)

        # First forward pass without cache
        output1, cache = attention(x, mask, cos, sin)
        assert cache is not None
        assert len(cache) == 2  # k, v
        assert cache[0].shape == (batch_size, n_heads, seq_len, d_model // n_heads)
        assert cache[1].shape == (batch_size, n_heads, seq_len, d_model // n_heads)

        # Second forward pass with cache (single token)
        x2 = torch.randn(batch_size, 1, d_model)
        mask2 = torch.ones(batch_size, 1, dtype=torch.bool)
        cos2 = torch.randn(1, d_model // n_heads)
        sin2 = torch.randn(1, d_model // n_heads)

        output2, updated_cache = attention(x2, mask2, cos2, sin2, kv_cache=cache)
        assert updated_cache is not None
        assert updated_cache[0].shape == (
            batch_size,
            n_heads,
            seq_len + 1,
            d_model // n_heads,
        )
        assert updated_cache[1].shape == (
            batch_size,
            n_heads,
            seq_len + 1,
            d_model // n_heads,
        )

        # Verify cache grows correctly
        assert updated_cache[0].shape[2] == cache[0].shape[2] + 1

    def test_rotary_attention_always_returns_cache(self):
        """Test that cache is always returned."""
        d_model = 128
        n_heads = 4
        batch_size = 2
        seq_len = 10

        attention = RotarySelfAttention(d_model, n_heads)

        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        cos = torch.randn(seq_len, d_model // n_heads)
        sin = torch.randn(seq_len, d_model // n_heads)

        output, cache = attention(x, mask, cos, sin)
        assert cache is not None


class TestTransformerLayerKVCaching:
    """Test KV caching in TransformerLayer."""

    def test_transformer_layer_kv_cache(self):
        """Test KV caching through transformer layer."""
        d_model = 128
        n_heads = 4
        batch_size = 2
        seq_len = 10

        layer = TransformerLayer(d_model, n_heads)

        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        cos = torch.randn(seq_len, d_model // n_heads)
        sin = torch.randn(seq_len, d_model // n_heads)

        # First forward pass
        output1, cache = layer(x, mask, cos, sin)
        assert cache is not None

        # Second forward pass with cache
        x2 = torch.randn(batch_size, 1, d_model)
        mask2 = torch.ones(batch_size, 1, dtype=torch.bool)
        cos2 = torch.randn(1, d_model // n_heads)
        sin2 = torch.randn(1, d_model // n_heads)

        output2, updated_cache = layer(x2, mask2, cos2, sin2, kv_cache=cache)
        assert updated_cache is not None


class TestPokerTransformerKVCaching:
    """Test KV caching in PokerTransformerV1."""

    def test_poker_transformer_kv_cache(self):
        """Test KV caching in the full poker transformer."""
        d_model = 128
        n_layers = 2
        n_heads = 4
        num_bet_bins = 8

        model = PokerTransformerV1(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            num_bet_bins=num_bet_bins,
        )

        # Create mock structured data
        batch_size = 2
        seq_len = 10

        # Mock the structured data (simplified for testing)
        structured_data = StructuredEmbeddingData(
            token_ids=torch.randint(0, 100, (batch_size, seq_len)),
            attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
            position_ids=torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        )

        # First forward pass
        output1 = model(structured_data)
        assert isinstance(output1, ModelOutput)
        assert output1.kv_cache is not None
        assert len(output1.kv_cache) == n_layers  # One cache per layer

        # Second forward pass with cache
        structured_data2 = StructuredEmbeddingData(
            token_ids=torch.randint(0, 100, (batch_size, 1)),
            attention_mask=torch.ones(batch_size, 1, dtype=torch.bool),
            position_ids=torch.zeros(batch_size, 1, dtype=torch.long),
        )

        output2 = model(structured_data2, kv_cache=output1.kv_cache)
        assert isinstance(output2, ModelOutput)
        assert output2.kv_cache is not None

        # Verify cache grows
        for layer_id in output1.kv_cache:
            old_cache = output1.kv_cache[layer_id]
            new_cache = output2.kv_cache[layer_id]
            assert (
                new_cache[0].shape[2] > old_cache[0].shape[2]
            )  # Sequence length grows


class TestKVCacheManager:
    """Test KV cache manager functionality."""

    def test_kv_cache_manager_basic(self):
        """Test basic KV cache manager functionality."""
        d_model = 128
        n_heads = 4
        batch_size = 2

        # Create a simple model with create_empty_cache method
        class MockModel(nn.Module):
            def create_empty_cache(self, batch_size, device):
                return {
                    id(self): (
                        torch.zeros(
                            batch_size, n_heads, 0, d_model // n_heads, device=device
                        ),
                        torch.zeros(
                            batch_size, n_heads, 0, d_model // n_heads, device=device
                        ),
                    )
                }

        model = MockModel()
        device = torch.device("cpu")
        manager = KVCacheManager(model, device)

        # Test cache creation
        manager.create_player_cache(0, batch_size)
        assert manager.has_player_cache(0)

        cache = manager.get_player_cache(0)
        assert cache is not None

        # Test cache update
        new_cache = {
            id(model): (
                torch.randn(batch_size, n_heads, 5, d_model // n_heads),
                torch.randn(batch_size, n_heads, 5, d_model // n_heads),
            )
        }
        manager.update_player_cache(0, new_cache)

        updated_cache = manager.get_player_cache(0)
        assert updated_cache[list(updated_cache.keys())[0]][0].shape[2] == 5

        # Test cache clearing
        manager.clear_player_cache(0)
        assert not manager.has_player_cache(0)


class TestSelfPlayKVCacheManager:
    """Test self-play KV cache manager functionality."""

    def test_self_play_kv_cache_manager(self):
        """Test self-play KV cache manager functionality."""
        d_model = 128
        n_heads = 4
        batch_size = 2

        # Create a simple model with create_empty_cache method
        class MockModel(nn.Module):
            def create_empty_cache(self, batch_size, device):
                return {
                    id(self): (
                        torch.zeros(
                            batch_size, n_heads, 0, d_model // n_heads, device=device
                        ),
                        torch.zeros(
                            batch_size, n_heads, 0, d_model // n_heads, device=device
                        ),
                    )
                }

        model = MockModel()
        device = torch.device("cpu")
        manager = SelfPlayKVCacheManager(model, device)

        # Test initialization
        manager.initialize_self_cache(batch_size)
        manager.initialize_opponent_cache(1, batch_size)

        assert manager.get_self_cache() is not None
        assert manager.get_opponent_cache(1) is not None

        # Test cache updates
        self_cache = manager.get_self_cache()
        new_self_cache = {
            id(model): (
                torch.randn(batch_size, n_heads, 3, d_model // n_heads),
                torch.randn(batch_size, n_heads, 3, d_model // n_heads),
            )
        }
        manager.update_self_cache(new_self_cache)

        opp_cache = manager.get_opponent_cache(1)
        new_opp_cache = {
            id(model): (
                torch.randn(batch_size, n_heads, 2, d_model // n_heads),
                torch.randn(batch_size, n_heads, 2, d_model // n_heads),
            )
        }
        manager.update_opponent_cache(1, new_opp_cache)

        # Test reset
        manager.reset_for_new_game()
        assert manager.get_self_cache() is None
        assert manager.get_opponent_cache(1) is None

        # Test cache info
        manager.initialize_self_cache(batch_size)
        info = manager.get_cache_info()
        assert info["has_self_cache"] is True
        assert info["num_opponents"] == 0


class TestKVCachingIntegration:
    """Integration tests for KV caching in poker scenarios."""

    def test_incremental_generation_simulation(self):
        """Test simulating incremental generation (like during inference)."""
        d_model = 128
        n_layers = 2
        n_heads = 4
        num_bet_bins = 8

        model = PokerTransformerV1(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            num_bet_bins=num_bet_bins,
        )

        batch_size = 1
        device = torch.device("cpu")
        manager = SelfPlayKVCacheManager(model, device)

        # Simulate a poker hand with multiple actions
        actions = [
            torch.randint(0, 100, (batch_size, 5)),  # Initial cards + context
            torch.randint(0, 100, (batch_size, 1)),  # First action
            torch.randint(0, 100, (batch_size, 1)),  # Second action
            torch.randint(0, 100, (batch_size, 1)),  # Third action
        ]

        # Initialize cache
        manager.initialize_self_cache(batch_size)

        # Process each action incrementally
        for i, action_tokens in enumerate(actions):
            seq_len = action_tokens.shape[1]

            structured_data = StructuredEmbeddingData(
                token_ids=action_tokens,
                attention_mask=torch.ones(batch_size, seq_len, dtype=torch.bool),
                position_ids=torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            )

            # Get current cache
            current_cache = manager.get_self_cache()

            # Forward pass with caching
            output = model(structured_data, kv_cache=current_cache)

            # Update cache
            manager.update_self_cache(output.kv_cache)

            # Verify cache grows
            if i > 0:  # After first action
                cache_info = manager.get_cache_info()
                assert cache_info["has_self_cache"] is True

        # Verify final cache has accumulated all tokens
        final_cache = manager.get_self_cache()
        assert final_cache is not None
        # The cache should have grown through all the actions
        total_tokens = sum(action.shape[1] for action in actions)
        first_layer_cache = next(iter(final_cache.values()))
        assert first_layer_cache[0].shape[2] == total_tokens


if __name__ == "__main__":
    pytest.main([__file__])
