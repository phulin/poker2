#!/usr/bin/env python3
"""
Test script to verify checkpoint compression with bfloat16 works correctly.
"""

import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alphaholdem.core.structured_config import Config, SearchConfig, TrainingConfig
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer


def test_checkpoint_compression():
    """Test that bfloat16 checkpoints save and load correctly."""

    print("Creating test configuration...")
    config = Config()
    config.device = "cpu"
    config.seed = 42
    config.use_wandb = False
    config.checkpoint_dir = "checkpoints-test"

    config.train = TrainingConfig()
    config.search = SearchConfig()
    config.search.enabled = False  # Skip search to speed up test

    print("Creating trainer...")
    device = torch.device(config.device)
    trainer = RebelCFRTrainer(cfg=config, device=device)

    # Get initial model parameters
    initial_params = {k: v.clone() for k, v in trainer.model.named_parameters()}

    # Test 1: Save checkpoint with bfloat16
    print("\nTest 1: Saving checkpoint with bfloat16...")
    test_path = "checkpoints-test/test_bfloat16.pt"
    trainer.save_checkpoint(
        test_path,
        step=100,
        save_optimizer=False,
        save_dtype=torch.bfloat16,
    )

    # Check file size
    file_size = os.path.getsize(test_path)
    print(f"  File size: {file_size / (1024**2):.2f} MB")

    # Test 2: Load checkpoint and verify model loads correctly
    print("\nTest 2: Loading bfloat16 checkpoint...")
    trainer2 = RebelCFRTrainer(cfg=config, device=device)
    loaded_step = trainer2.load_checkpoint(test_path)

    print(f"  Loaded step: {loaded_step}")

    # Verify parameters match
    for name, param in trainer2.model.named_parameters():
        initial_param = initial_params[name]
        if not torch.allclose(param, initial_param, rtol=1e-2):
            print(f"  ERROR: Parameter {name} doesn't match after loading!")
            return False
        else:
            # Check that parameters are back in float32
            assert param.dtype == torch.float32, f"Parameter {name} should be float32"

    print("  ✓ All parameters match and are in float32")

    # Test 3: Save checkpoint with full state (float32)
    print("\nTest 3: Saving full checkpoint with optimizer state...")
    test_path2 = "checkpoints-test/test_full.pt"
    trainer.save_checkpoint(
        test_path2,
        step=200,
        save_optimizer=True,
        save_dtype=None,
    )

    file_size2 = os.path.getsize(test_path2)
    print(f"  File size: {file_size2 / (1024**2):.2f} MB")
    print(f"  Size reduction: {(1 - file_size / file_size2) * 100:.1f}%")

    # Test 4: Load full checkpoint
    print("\nTest 4: Loading full checkpoint...")
    trainer3 = RebelCFRTrainer(cfg=config, device=device)
    loaded_step3 = trainer3.load_checkpoint(test_path2)
    print(f"  Loaded step: {loaded_step3}")
    print("  ✓ Full checkpoint loaded successfully")

    # Cleanup
    print("\nCleaning up test files...")
    if os.path.exists(test_path):
        os.remove(test_path)
    if os.path.exists(test_path2):
        os.remove(test_path2)

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_checkpoint_compression()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
