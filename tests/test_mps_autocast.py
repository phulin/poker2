#!/usr/bin/env python3
"""
Tests for MPS autocast support in SelfPlayTrainer.
"""

import os
import sys

import pytest
import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.rl.self_play import SelfPlayTrainer

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath("."))


@pytest.fixture
def mps_config():
    """Create a minimal config for MPS testing."""
    return Config(
        train=TrainingConfig(
            batch_size=4,
            use_mixed_precision=True,  # Enable mixed precision
            loss_scale=128.0,
        ),
        model=ModelConfig(),
        env=EnvConfig(),
        use_tensor_env=True,
        num_envs=8,
        device="mps",
        use_wandb=False,  # Disable wandb for testing
        wandb_project="test",
        wandb_name="mps-autocast-test",
        wandb_tags=["test"],
        wandb_run_id=None,
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_autocast_initialization(mps_config):
    """Test that MPS autocast is properly initialized."""
    device = torch.device("mps")

    # Create trainer with MPS autocast enabled
    trainer = SelfPlayTrainer(mps_config, device)

    # Check that scaler is initialized for MPS
    assert trainer.scaler is not None, "GradScaler should be initialized for MPS"
    assert trainer.scaler._device == "mps", "GradScaler should be configured for MPS"
    assert trainer.use_mixed_precision is True, "Mixed precision should be enabled"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_autocast_forward_pass(mps_config):
    """Test that MPS autocast works in forward pass."""
    device = torch.device("mps")
    trainer = SelfPlayTrainer(mps_config, device)

    # Create test tensors
    batch_size = 4
    cards_features = torch.zeros(batch_size, 6, 4, 13, dtype=torch.bool, device=device)
    actions_features = torch.zeros(
        batch_size, 24, 4, 8, dtype=torch.bool, device=device
    )

    # Test autocast forward pass (now uses bfloat16 with mixed precision)
    cards_float = cards_features.to(torch.bfloat16)
    actions_float = actions_features.to(torch.bfloat16)

    # Create CNNEmbeddingData
    embedding_data = CNNEmbeddingData(cards=cards_float, actions=actions_float)

    with torch.amp.autocast("mps", dtype=torch.bfloat16):
        outputs = trainer.model(embedding_data)
        logits = outputs.policy_logits
        values = outputs.value

    # Verify outputs
    assert (
        logits.dtype == torch.bfloat16
    ), "Logits should be bfloat16 with mixed precision"
    assert (
        values.dtype == torch.bfloat16
    ), "Values should be bfloat16 with mixed precision"
    assert logits.shape == (batch_size, 8), "Logits shape should be correct"
    assert values.shape == (batch_size,), "Values shape should be correct (1D tensor)"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_autocast_backward_pass(mps_config):
    """Test that MPS autocast works in backward pass with scaler."""
    device = torch.device("mps")
    trainer = SelfPlayTrainer(mps_config, device)

    # Create test tensors
    batch_size = 4
    cards_features = torch.zeros(batch_size, 6, 4, 13, dtype=torch.bool, device=device)
    actions_features = torch.zeros(
        batch_size, 24, 4, 8, dtype=torch.bool, device=device
    )

    # Forward pass with autocast (now uses bfloat16)
    cards_float = cards_features.to(torch.bfloat16)
    actions_float = actions_features.to(torch.bfloat16)

    # Create CNNEmbeddingData
    embedding_data = CNNEmbeddingData(cards=cards_float, actions=actions_float)

    with torch.amp.autocast("mps", dtype=torch.bfloat16):
        outputs = trainer.model(embedding_data)
        logits = outputs.policy_logits
        values = outputs.value

    # Create a dummy loss
    loss = torch.mean(logits) + torch.mean(values)

    # Test scaler operations
    trainer.optimizer.zero_grad()

    # Scale the loss
    scaled_loss = trainer.scaler.scale(loss)
    assert scaled_loss.dtype == torch.float32, "Scaled loss should be float32"

    # Backward pass
    scaled_loss.backward()

    # Unscale gradients
    trainer.scaler.unscale_(trainer.optimizer)

    # Step optimizer
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()

    # Verify scaler state
    assert trainer.scaler.get_scale() > 0, "Scaler scale should be positive"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_autocast_device_consistency(mps_config):
    """Test that MPS autocast maintains device consistency."""
    device = torch.device("mps")
    trainer = SelfPlayTrainer(mps_config, device)

    # Create test tensors
    batch_size = 4
    cards_features = torch.zeros(batch_size, 6, 4, 13, dtype=torch.bool, device=device)
    actions_features = torch.zeros(
        batch_size, 24, 4, 8, dtype=torch.bool, device=device
    )

    # Test autocast forward pass (now uses bfloat16)
    cards_float = cards_features.to(torch.bfloat16)
    actions_float = actions_features.to(torch.bfloat16)

    # Create CNNEmbeddingData
    embedding_data = CNNEmbeddingData(cards=cards_float, actions=actions_float)

    with torch.amp.autocast("mps", dtype=torch.bfloat16):
        outputs = trainer.model(embedding_data)
        logits = outputs.policy_logits
        values = outputs.value

    # Verify device consistency (account for device index)
    assert logits.device.type == device.type, "Logits should be on MPS device"
    assert values.device.type == device.type, "Values should be on MPS device"
    assert cards_float.device.type == device.type, "Cards float should be on MPS device"
    assert (
        actions_float.device.type == device.type
    ), "Actions float should be on MPS device"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_autocast_mixed_precision_disabled():
    """Test that MPS autocast is disabled when mixed precision is off."""
    config = Config(
        train=TrainingConfig(
            batch_size=4,
            use_mixed_precision=False,  # Disable mixed precision
        ),
        model=ModelConfig(),
        env=EnvConfig(),
        use_tensor_env=True,
        num_envs=8,
        device="mps",
        use_wandb=False,
        wandb_project="test",
        wandb_name="mps-test",
        wandb_tags=["test"],
        wandb_run_id=None,
    )

    device = torch.device("mps")
    trainer = SelfPlayTrainer(config, device)

    # GradScaler exists but is disabled when mixed precision is off
    assert trainer.scaler is not None
    assert trainer.use_mixed_precision is False, "Mixed precision should be disabled"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_autocast_vs_cuda_consistency():
    """Test that MPS autocast behaves consistently with CUDA autocast."""
    # This test verifies that the autocast logic works the same way for both devices
    device = torch.device("mps")

    # Test autocast context manager
    with torch.amp.autocast("mps"):
        x = torch.randn(4, 4, device=device, dtype=torch.float32)
        y = torch.randn(4, 4, device=device, dtype=torch.float32)
        result = torch.matmul(x, y)

    # Verify autocast behavior (default autocast uses float16)
    assert result.dtype == torch.float16, "MPS autocast should produce float16 results"
    assert result.device.type == device.type, "Result should be on MPS device"

    # Test GradScaler
    scaler = torch.amp.GradScaler("mps", init_scale=128.0)
    assert scaler._device == "mps", "GradScaler should be configured for MPS"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_autocast_integration_with_selfplay_trainer(mps_config):
    """Test that MPS autocast is properly integrated into SelfPlayTrainer's update_model method."""
    device = torch.device("mps")
    trainer = SelfPlayTrainer(mps_config, device)

    # Create a mock batch that would come from the replay buffer
    batch_size = 4
    mock_batch = {
        "cards_features": torch.zeros(
            batch_size, 6, 4, 13, dtype=torch.bool, device=device
        ),
        "actions_features": torch.zeros(
            batch_size, 24, 4, 8, dtype=torch.bool, device=device
        ),
        "action_indices": torch.randint(0, 8, (batch_size,), device=device),
        "log_probs_old": torch.randn(batch_size, device=device),
        "advantages": torch.randn(batch_size, device=device),
        "returns": torch.randn(batch_size, device=device),
        "legal_masks": torch.ones(batch_size, 8, dtype=torch.bool, device=device),
        "delta2": torch.randn(batch_size, device=device),
        "delta3": torch.randn(batch_size, device=device),
    }

    # Mock the update_model method to test autocast integration
    # We'll test the forward pass part that uses autocast
    cards_features = mock_batch["cards_features"]
    actions_features = mock_batch["actions_features"]

    # Convert bool tensors to appropriate dtype for model forward pass (as done in update_model)
    if trainer.use_mixed_precision and trainer.device.type in ["cuda", "mps"]:
        cards_float = cards_features.to(torch.bfloat16)
        actions_float = actions_features.to(torch.bfloat16)

        with torch.amp.autocast(trainer.device.type, dtype=torch.bfloat16):
            embedding_data = CNNEmbeddingData(cards=cards_float, actions=actions_float)
            outputs = trainer.model(embedding_data)
            logits = outputs.policy_logits
            values = outputs.value
    else:
        cards_float = cards_features.float()
        actions_float = actions_features.float()
        embedding_data = CNNEmbeddingData(cards=cards_float, actions=actions_float)
        outputs = trainer.model(embedding_data)
        logits = outputs.policy_logits
        values = outputs.value

    # Verify autocast is working
    assert (
        logits.dtype == torch.bfloat16
    ), "Logits should be bfloat16 with MPS mixed precision"
    assert (
        values.dtype == torch.bfloat16
    ), "Values should be bfloat16 with MPS mixed precision"
    assert logits.device.type == device.type, "Logits should be on MPS device"
    assert values.device.type == device.type, "Values should be on MPS device"

    # Verify scaler is available for backward pass
    assert trainer.scaler is not None, "GradScaler should be available for MPS"
    assert trainer.scaler._device == "mps", "GradScaler should be configured for MPS"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
