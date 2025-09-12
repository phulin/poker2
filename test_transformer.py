#!/usr/bin/env python3
"""Unit tests for transformer poker model."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from alphaholdem.models.transformer import PokerTransformerV1
from alphaholdem.models.transformer.embedding_data import StructuredEmbeddingData
from alphaholdem.models.factory import ModelFactory


def test_transformer_model_creation():
    """Test creating transformer model."""
    model = PokerTransformerV1(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=80,
        num_actions=8,
        use_auxiliary_loss=True,
    )

    assert model.d_model == 256
    assert model.n_layers == 4
    assert model.n_heads == 4
    assert model.vocab_size == 80
    assert model.num_actions == 8
    assert model.use_auxiliary_loss == True


def test_transformer_forward_pass():
    """Test transformer forward pass."""
    model = PokerTransformerV1(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=80,
        num_actions=8,
        use_auxiliary_loss=True,
    )

    # Test forward pass with structured embeddings
    batch_size = 2
    seq_len = 20

    # Create structured embedding inputs using the new StructuredEmbeddingData structure
    structured_data = StructuredEmbeddingData(
        token_ids=torch.randint(0, 100, (batch_size, seq_len)),
        card_ranks=torch.randint(0, 13, (batch_size, seq_len)),
        card_suits=torch.randint(0, 4, (batch_size, seq_len)),
        card_streets=torch.randint(0, 4, (batch_size, seq_len)),
        action_actors=torch.randint(0, 2, (batch_size, seq_len)),
        action_streets=torch.randint(0, 4, (batch_size, seq_len)),
        action_legal_masks=torch.ones(batch_size, seq_len, 8).bool(),
        context_features=torch.randint(0, 10, (batch_size, seq_len, 10)),
    )

    outputs = model(structured_data)

    # Check output shapes
    assert outputs["policy_logits"].shape == (batch_size, 8)
    assert outputs["size_params"].shape == (batch_size, 2)
    assert outputs["value"].shape == (batch_size,)
    # Note: hand_range_logits is not included since use_auxiliary_loss is True but hand_range_head is commented out


def test_transformer_via_factory():
    """Test creating transformer model via ModelFactory."""
    device = torch.device("cpu")
    config = {
        "d_model": 256,
        "n_layers": 4,
        "n_heads": 4,
        "vocab_size": 80,
        "num_actions": 8,
        "use_auxiliary_loss": True,
    }

    model = ModelFactory.create_model("transformer", config, device)

    assert isinstance(model, PokerTransformerV1)
    assert model.d_model == 256


def test_transformer_model_info():
    """Test model info method."""
    model = PokerTransformerV1(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=80,
        num_actions=8,
        use_auxiliary_loss=True,
    )

    info = model.get_model_info()

    assert info["model_type"] == "poker_transformer_v1"
    assert info["d_model"] == 256
    assert info["n_layers"] == 4
    assert info["n_heads"] == 4
    assert info["vocab_size"] == 80
    assert info["num_actions"] == 8
    assert info["use_auxiliary_loss"] == True
    assert info["total_parameters"] > 0
    assert info["trainable_parameters"] > 0


def test_transformer_gradient_checkpointing():
    """Test gradient checkpointing functionality."""
    model = PokerTransformerV1(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=80,
        num_actions=8,
        use_auxiliary_loss=True,
        use_gradient_checkpointing=True,
    )

    # Set to training mode
    model.train()

    # Test forward pass with gradient checkpointing
    batch_size = 2
    seq_len = 20

    # Create structured embedding inputs using the new StructuredEmbeddingData structure
    structured_data = StructuredEmbeddingData(
        token_ids=torch.randint(0, 100, (batch_size, seq_len)),
        card_ranks=torch.randint(0, 13, (batch_size, seq_len)),
        card_suits=torch.randint(0, 4, (batch_size, seq_len)),
        card_streets=torch.randint(0, 4, (batch_size, seq_len)),
        action_actors=torch.randint(0, 2, (batch_size, seq_len)),
        action_streets=torch.randint(0, 4, (batch_size, seq_len)),
        action_legal_masks=torch.ones(batch_size, seq_len, 8).bool(),
        context_features=torch.randint(0, 10, (batch_size, seq_len, 10)),
    )

    outputs = model(structured_data)

    # Check that outputs are still correct
    assert outputs["policy_logits"].shape == (batch_size, 8)
    assert outputs["size_params"].shape == (batch_size, 2)
    assert outputs["value"].shape == (batch_size,)
    # Note: hand_range_logits is not included since use_auxiliary_loss is True but hand_range_head is commented out


if __name__ == "__main__":
    # Run tests
    test_transformer_model_creation()
    print("✅ Model creation test passed")

    test_transformer_forward_pass()
    print("✅ Forward pass test passed")

    test_transformer_via_factory()
    print("✅ Factory test passed")

    test_transformer_model_info()
    print("✅ Model info test passed")

    test_transformer_gradient_checkpointing()
    print("✅ Gradient checkpointing test passed")

    print("\n🎉 All transformer tests passed!")


def test_structured_embeddings():
    """Test structured embeddings functionality."""
    model = PokerTransformerV1(
        d_model=256,
        n_layers=4,
        n_heads=4,
        vocab_size=80,
        num_actions=8,
        use_auxiliary_loss=True,
    )

    # Test structured embeddings forward pass
    batch_size = 2
    seq_len = 20

    # Create structured embedding inputs using the new StructuredEmbeddingData structure
    structured_data = StructuredEmbeddingData(
        token_ids=torch.randint(0, 100, (batch_size, seq_len)),
        card_ranks=torch.randint(0, 13, (batch_size, seq_len)),
        card_suits=torch.randint(0, 4, (batch_size, seq_len)),
        card_streets=torch.randint(0, 4, (batch_size, seq_len)),
        action_actors=torch.randint(0, 2, (batch_size, seq_len)),
        action_streets=torch.randint(0, 4, (batch_size, seq_len)),
        action_legal_masks=torch.ones(batch_size, seq_len, 8).bool(),
        context_features=torch.randint(0, 10, (batch_size, seq_len, 10)),
    )

    # Test forward pass with structured embeddings
    outputs = model(structured_data)

    # Check output shapes
    assert outputs["policy_logits"].shape == (batch_size, 8)
    assert outputs["size_params"].shape == (batch_size, 2)
    assert outputs["value"].shape == (batch_size,)
    # Note: hand_range_logits is not included since use_auxiliary_loss is True but hand_range_head is commented out

    print("✅ Structured embeddings test passed")


if __name__ == "__main__":
    # Run tests
    test_transformer_model_creation()
    print("✅ Model creation test passed")

    test_transformer_forward_pass()
    print("✅ Forward pass test passed")

    test_transformer_via_factory()
    print("✅ Factory test passed")

    test_transformer_model_info()
    print("✅ Model info test passed")

    test_transformer_gradient_checkpointing()
    print("✅ Gradient checkpointing test passed")

    test_tokenizer()
    print("✅ Tokenizer test passed")

    test_structured_embeddings()
    print("✅ Structured embeddings test passed")

    print("\n🎉 All transformer tests passed!")
