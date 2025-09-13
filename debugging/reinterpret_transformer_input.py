#!/usr/bin/env python3
"""Comprehensive example of transformer state debugging utilities."""

import torch
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder
from alphaholdem.models.transformer.debug_utils import (
    debug_transformer_state,
    TransformerStateDebugger,
    create_debugger_with_encoder,
)


def test_basic_debugging():
    """Test basic debugging functionality."""
    print("=" * 60)
    print("BASIC DEBUGGING TEST")
    print("=" * 60)

    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )

    env.reset()

    # Use the convenience function to create both encoder and debugger
    encoder, debugger = create_debugger_with_encoder(env, device)

    # Play some actions
    env.step_bins(torch.tensor([1], device=device))  # Call
    env.step_bins(torch.tensor([1], device=device))  # Call

    # Debug
    embedding_data = encoder.encode_tensor_states(
        player=0, idxs=torch.tensor([0], device=device)
    )
    comparison = debug_transformer_state(embedding_data, env, env_idx=0)

    return comparison


def test_multiple_environments():
    """Test debugging with multiple environments."""
    print("\n" + "=" * 60)
    print("MULTIPLE ENVIRONMENTS TEST")
    print("=" * 60)

    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=3,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )

    env.reset()
    encoder = TransformerStateEncoder(env, device)

    # Play different actions in each environment
    env.step_bins(torch.tensor([1, 0, 2], device=device))  # Call, Fold, Raise
    env.step_bins(torch.tensor([1, 1, 1], device=device))  # Call, Call, Call

    # Debug all environments
    embedding_data = encoder.encode_tensor_states(
        player=0, idxs=torch.tensor([0, 1, 2], device=device)
    )

    debugger = TransformerStateDebugger(env)
    results = []

    for env_idx in range(3):
        print(f"\n--- Environment {env_idx} ---")
        comparison = debugger.compare_reconstructed_vs_env(embedding_data, env_idx)
        results.append(comparison)

        # Print summary
        matches = sum(comparison.values())
        total = len(comparison)
        print(f"Matches: {matches}/{total}")

    return results


def test_different_game_states():
    """Test debugging at different stages of the game."""
    print("\n" + "=" * 60)
    print("DIFFERENT GAME STATES TEST")
    print("=" * 60)

    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )

    env.reset()
    encoder = TransformerStateEncoder(env, device)

    stages = [
        ("Preflop", []),
        ("After preflop betting", [1, 1]),  # Call, Call
        ("After flop betting", [1, 1, 1, 1]),  # Call, Call, Call, Call
    ]

    results = []

    for stage_name, actions in stages:
        print(f"\n--- {stage_name} ---")

        # Play actions for this stage
        for action in actions:
            env.step_bins(torch.tensor([action], device=device))

        # Debug current state
        embedding_data = encoder.encode_tensor_states(
            player=0, idxs=torch.tensor([0], device=device)
        )
        comparison = debug_transformer_state(
            embedding_data, env, env_idx=0, analyze_context=False
        )

        matches = sum(comparison.values())
        total = len(comparison)
        print(f"Stage: {stage_name}, Matches: {matches}/{total}")
        results.append((stage_name, comparison))

    return results


def test_perspective_swapping():
    """Test debugging with different player perspectives."""
    print("\n" + "=" * 60)
    print("PERSPECTIVE SWAPPING TEST")
    print("=" * 60)

    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )

    env.reset()
    encoder = TransformerStateEncoder(env, device)

    # Play some actions
    env.step_bins(torch.tensor([1], device=device))  # Call
    env.step_bins(torch.tensor([1], device=device))  # Call

    # Debug from both perspectives
    print("\n--- Player 0 Perspective ---")
    embedding_data_p0 = encoder.encode_tensor_states(
        player=0, idxs=torch.tensor([0], device=device)
    )
    comparison_p0 = debug_transformer_state(
        embedding_data_p0, env, env_idx=0, analyze_context=False
    )

    print("\n--- Player 1 Perspective ---")
    embedding_data_p1 = encoder.encode_tensor_states(
        player=1, idxs=torch.tensor([0], device=device)
    )
    comparison_p1 = debug_transformer_state(
        embedding_data_p1, env, env_idx=0, analyze_context=False
    )

    # Compare the two perspectives
    print("\n--- Perspective Comparison ---")
    print("Player 0 matches:", sum(comparison_p0.values()))
    print("Player 1 matches:", sum(comparison_p1.values()))

    return comparison_p0, comparison_p1


def main():
    """Run all debugging tests."""
    print("🧪 TRANSFORMER STATE DEBUGGING TESTS")
    print("=" * 60)

    # Run all tests
    basic_results = test_basic_debugging()
    multi_env_results = test_multiple_environments()
    game_state_results = test_different_game_states()
    perspective_results = test_perspective_swapping()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Basic debugging: {sum(basic_results.values())}/{len(basic_results)} matches"
    )
    print(
        f"Multiple environments: {sum(sum(r.values()) for r in multi_env_results)}/{sum(len(r) for r in multi_env_results)} total matches"
    )
    print(
        f"Game states: {sum(sum(r[1].values()) for r in game_state_results)}/{sum(len(r[1]) for r in game_state_results)} total matches"
    )
    print(
        f"Perspectives: P0={sum(perspective_results[0].values())}, P1={sum(perspective_results[1].values())}"
    )

    print("\n🎉 All debugging tests completed!")


if __name__ == "__main__":
    main()
