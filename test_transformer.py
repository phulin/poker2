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
from alphaholdem.models.factory import ModelFactory
from alphaholdem.models.transformer.tokenizer import PokerTokenizer


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

    # Test forward pass
    batch_size = 2
    seq_len = 20
    token_ids = torch.randint(0, 80, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    outputs = model(token_ids, attention_mask)

    # Check output shapes
    assert outputs["policy_logits"].shape == (batch_size, 8)
    assert outputs["size_params"].shape == (batch_size, 2)
    assert outputs["value"].shape == (batch_size,)
    assert outputs["hand_range_logits"].shape == (batch_size, 1326)


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
    token_ids = torch.randint(0, 80, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    outputs = model(token_ids, attention_mask)

    # Check that outputs are still correct
    assert outputs["policy_logits"].shape == (batch_size, 8)
    assert outputs["size_params"].shape == (batch_size, 2)
    assert outputs["value"].shape == (batch_size,)
    assert outputs["hand_range_logits"].shape == (batch_size, 1326)


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


def test_tokenizer():
    """Test tokenizer functionality."""
    tokenizer = PokerTokenizer(max_sequence_length=50)

    # Test special tokens
    assert tokenizer.special_tokens["CLS"] == 0
    assert tokenizer.special_tokens["SEP"] == 1
    assert tokenizer.special_tokens["MASK"] == 2
    assert tokenizer.special_tokens["PAD"] == 3

    # Test vocab size
    assert tokenizer.get_vocab_size() == 80

    # Test token decoding
    tokens = torch.tensor([0, 1, 4, 5, 3, 3])  # CLS, SEP, card 0, card 1, PAD, PAD
    decoded = tokenizer.decode_tokens(tokens)
    expected = ["[CLS]", "[SEP]", "[CARD_0]", "[CARD_1]", "[PAD]", "[PAD]"]
    assert decoded == expected

    print("✅ Tokenizer test passed")


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

    print("\n🎉 All transformer tests passed!")
