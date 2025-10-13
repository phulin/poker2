#!/usr/bin/env python3
"""
Test script to verify RebelCFREvaluator integration works correctly.
"""

import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from src.alphaholdem.core.structured_config import Config, SearchConfig
from src.alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from src.alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator


def test_rebel_cfr_evaluator():
    """Test basic functionality of RebelCFREvaluator."""
    print("Testing RebelCFREvaluator...")

    # Create a simple configuration
    device = torch.device("cpu")
    float_dtype = torch.float32

    # Create a simple environment
    env = HUNLTensorEnv(
        num_envs=4,
        starting_stack=1000,
        sb=50,
        bb=100,
        default_bet_bins=[0.5, 1.0, 2.0],
        device=device,
        float_dtype=float_dtype,
    )

    # Create CFR evaluator
    evaluator = RebelCFREvaluator(
        search_batch_size=2,
        env_proto=env,
        bet_bins=[0.5, 1.0, 2.0],
        max_depth=2,
        device=device,
        float_dtype=float_dtype,
    )

    print(
        f"Created evaluator with batch_size={evaluator.search_batch_size}, max_depth={evaluator.max_depth}"
    )

    # Initialize search
    env_indices = torch.arange(2, device=device)
    evaluator.initialize_search(env, env_indices)
    print("Initialized search successfully")

    # Create a simple mock value network
    def mock_value_network(features):
        batch_size = features.shape[0]
        # Return dummy hand values
        return type(
            "MockOutput",
            (),
            {
                "hand_values": torch.zeros(
                    batch_size, 2, 1326, device=device, dtype=float_dtype
                )
            },
        )()

    # Run a few CFR iterations
    result = evaluator.run_cfr_iterations(3, mock_value_network)
    print(f"CFR iterations completed. Root policy shape: {result.root_policy.shape}")
    print(f"Average policy shape: {result.root_policy_avg.shape}")
    print(f"Sampled policy shape: {result.root_policy_sampled.shape}")
    print(f"Root values shape: {result.root_values.shape}")

    # Test training data extraction
    training_data = result.training_targets
    print(f"Training data keys: {list(training_data.keys())}")

    # Test next PBS (for continued search)
    print(
        f"Next PBS shape: {evaluator.next_pbs.shape if evaluator.next_pbs is not None else 'None'}"
    )

    print("All tests passed successfully!")
    return True


if __name__ == "__main__":
    try:
        test_rebel_cfr_evaluator()
        print("✓ RebelCFREvaluator integration test successful!")
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
