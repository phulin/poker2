#!/usr/bin/env python3
"""
Policy Analysis Script

Analyzes how the model's policy changes across different game states
to verify it's learning to differentiate between situations.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.models.cnn import CardsPlanesV1, ActionsHUEncoderV1
from alphaholdem.env import rules


def analyze_policy_diversity(trainer):
    """Analyze how the model's policy varies across different game states."""
    print("=== Policy Diversity Analysis ===")

    # Create encoders
    cards_encoder = CardsPlanesV1()
    actions_encoder = ActionsHUEncoderV1()

    # Test different game states
    test_cases = [
        ("Preflop, small pot", HUNLEnv(starting_stack=400, sb=1, bb=2, seed=1)),
        ("Preflop, larger pot", HUNLEnv(starting_stack=400, sb=1, bb=2, seed=2)),
        ("Flop, small pot", HUNLEnv(starting_stack=400, sb=1, bb=2, seed=3)),
        ("Turn, medium pot", HUNLEnv(starting_stack=400, sb=1, bb=2, seed=4)),
    ]

    all_policies = []

    for name, env in test_cases:
        print(f"\n--- {name} ---")

        # Reset and play a few steps to get to interesting state
        state = env.reset()

        # Play a few actions to get to different game states
        for _ in range(3):
            if state.terminal:
                break

            # Get legal actions
            legal_actions = env.legal_actions()
            legal_mask = torch.zeros(9)
            for action in legal_actions:
                bin_idx = actions_encoder._action_to_bin(action, state, 9)
                if bin_idx is not None:
                    legal_mask[bin_idx] = 1.0

            # Encode state
            cards_tensor = cards_encoder.encode_cards(state, seat=state.to_act)
            actions_tensor = actions_encoder.encode_actions(state, seat=state.to_act)

            # Model forward pass
            with torch.no_grad():
                logits, value = trainer.model(
                    cards_tensor.unsqueeze(0), actions_tensor.unsqueeze(0)
                )
                logits = logits.squeeze(0)
                value = value.squeeze(0)

            # Apply legal mask and compute policy
            masked_logits = logits.clone()
            masked_logits[legal_mask == 0] = -1e9
            probs = F.softmax(masked_logits, dim=-1)

            print(f"  State: {state.street}, Pot: {state.pot}, To act: {state.to_act}")
            print(f"  Value: {value.item():.6f}")
            print(f"  Policy: {probs.tolist()}")

            # Store policy for comparison
            all_policies.append(probs.tolist())

            # Take a random action to progress
            if legal_actions:
                state, _, done, _ = env.step(legal_actions[0])
                if done:
                    break

    # Analyze policy diversity
    print(f"\n=== Policy Diversity Analysis ===")
    print(f"Number of policies analyzed: {len(all_policies)}")

    if len(all_policies) > 1:
        # Calculate variance in each action probability
        policies_tensor = torch.tensor(all_policies)
        variances = policies_tensor.var(dim=0)

        action_names = [
            "fold",
            "check/call",
            "bet 1/2",
            "bet 3/4",
            "bet pot",
            "bet 1.5x",
            "bet 2x",
            "bet 3x",
            "all-in",
        ]

        print(f"\nVariance in action probabilities across states:")
        for i, (name, var) in enumerate(zip(action_names, variances)):
            print(f"  {name:12}: {var.item():.6f}")

        # Calculate overall diversity metric
        total_variance = variances.sum().item()
        print(f"\nTotal policy variance: {total_variance:.6f}")

        if total_variance < 0.001:
            print("⚠️  WARNING: Very low policy diversity! Model may not be learning.")
        elif total_variance < 0.01:
            print("⚠️  WARNING: Low policy diversity. Model may need more training.")
        else:
            print("✅ Good policy diversity detected.")
    else:
        print("⚠️  Not enough policies to analyze diversity.")


def analyze_model_sensitivity(trainer):
    """Test if the model responds to different inputs."""
    print("\n=== Model Sensitivity Analysis ===")

    cards_encoder = CardsPlanesV1()
    actions_encoder = ActionsHUEncoderV1()

    # Create two different game states
    env1 = HUNLEnv(starting_stack=400, sb=1, bb=2, seed=10)
    env2 = HUNLEnv(starting_stack=400, sb=1, bb=2, seed=20)

    state1 = env1.reset()
    state2 = env2.reset()

    # Get policies for both states
    def get_policy(state, env):
        legal_actions = env.legal_actions()
        legal_mask = torch.zeros(9)
        for action in legal_actions:
            bin_idx = actions_encoder._action_to_bin(action, state, 9)
            if bin_idx is not None:
                legal_mask[bin_idx] = 1.0

        cards_tensor = cards_encoder.encode_cards(state, seat=state.to_act)
        actions_tensor = actions_encoder.encode_actions(state, seat=state.to_act)

        with torch.no_grad():
            logits, value = trainer.model(
                cards_tensor.unsqueeze(0), actions_tensor.unsqueeze(0)
            )
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = -1e9
        probs = F.softmax(masked_logits, dim=-1)

        return probs, value

    policy1, value1 = get_policy(state1, env1)
    policy2, value2 = get_policy(state2, env2)

    print(
        f"State 1 - Hole cards: {state1.players[0].hole_cards}, Value: {value1.item():.6f}"
    )
    print(
        f"State 2 - Hole cards: {state2.players[0].hole_cards}, Value: {value2.item():.6f}"
    )

    # Calculate differences
    policy_diff = torch.abs(policy1 - policy2)
    value_diff = abs(value1.item() - value2.item())

    print(f"\nPolicy differences:")
    action_names = [
        "fold",
        "check/call",
        "bet 1/2",
        "bet 3/4",
        "bet pot",
        "bet 1.5x",
        "bet 2x",
        "bet 3x",
        "all-in",
    ]
    for i, (name, diff) in enumerate(zip(action_names, policy_diff)):
        print(f"  {name:12}: {diff.item():.6f}")

    print(f"\nValue difference: {value_diff:.6f}")

    total_policy_diff = policy_diff.sum().item()
    print(f"Total policy difference: {total_policy_diff:.6f}")

    if total_policy_diff < 0.01:
        print("⚠️  WARNING: Model shows very little sensitivity to different inputs!")
    elif total_policy_diff < 0.1:
        print("⚠️  WARNING: Model shows low sensitivity to different inputs.")
    else:
        print("✅ Model shows good sensitivity to different inputs.")


def main():
    """Main function."""
    print("=== Policy Analysis Script ===")
    print(
        "This script analyzes how the model's policy varies across different game states."
    )
    print()

    # Initialize trainer
    trainer = SelfPlayTrainer()

    # Train the model briefly
    print("Training model for 20 steps...")
    for step in range(20):
        stats = trainer.train_step(num_trajectories=4)
        if step % 5 == 0:
            print(f"Step {step:2d}: Reward: {stats['avg_reward']:6.2f}")

    print("Training completed!\n")

    # Analyze policy diversity
    analyze_policy_diversity(trainer)

    # Analyze model sensitivity
    analyze_model_sensitivity(trainer)

    print("\n=== Analysis Complete ===")
    print("This analysis helps determine if the model is learning to differentiate")
    print("between different game states and make context-appropriate decisions.")


if __name__ == "__main__":
    main()
