#!/usr/bin/env python3
"""
Hand Analysis Script

Trains the model for 50 steps, then runs through a complete poker hand
showing value and policy calculations at each decision point.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.encoding.cards_encoder import CardsPlanesV1
from alphaholdem.encoding.actions_encoder import ActionsHUEncoderV1
from alphaholdem.env import rules


def card_number_to_name(card_num):
    """Convert card number to human-readable name."""
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["h", "d", "c", "s"]  # hearts, diamonds, clubs, spades

    rank_idx = card_num % 13
    suit_idx = card_num // 13

    return f"{ranks[rank_idx]}{suits[suit_idx]}"


def cards_to_names(cards):
    """Convert list of card numbers to human-readable names."""
    return [card_number_to_name(card) for card in cards]


def train_model(trainer, num_steps=50):
    """Train the model for specified number of steps."""
    print(f"Training model for {num_steps} steps...")

    for step in range(num_steps):
        stats = trainer.train_step(num_trajectories=4)
        if step % 10 == 0:
            print(
                f"Step {step:2d}: Reward: {stats['avg_reward']:6.2f}, "
                f"Episodes: {stats['episode_count']}"
            )

    print("Training completed!\n")


def load_checkpoint(trainer, checkpoint_path="checkpoints/final_checkpoint.pt"):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded successfully!")
        print(f"  Training step: {checkpoint.get('step', 'Unknown')}")
        print(f"  Episode count: {checkpoint.get('episode_count', 'Unknown')}")
    else:
        print(f"Warning: Checkpoint file {checkpoint_path} not found!")
        print("Training model from scratch...")
        train_model(trainer, num_steps=50)

    print()


def analyze_hand(trainer):
    """Analyze a complete poker hand step by step."""
    print("=== Hand Analysis ===")

    # Create a fresh environment for analysis
    env = HUNLEnv(
        starting_stack=400, sb=1, bb=2, seed=123
    )  # Different seed for variety
    cards_encoder = CardsPlanesV1()
    actions_encoder = ActionsHUEncoderV1()

    # Reset environment
    state = env.reset()
    print(f"Initial state: {state}")
    print(
        f"Player 0 stack: {state.players[0].stack}, Player 1 stack: {state.players[1].stack}"
    )
    print(f"Pot: {state.pot}, To act: {state.to_act}")
    print(f"Player 0 hole cards: {cards_to_names(state.players[0].hole_cards)}")
    print(f"Player 1 hole cards: {cards_to_names(state.players[1].hole_cards)}")
    print()

    step_count = 0
    max_steps = 20  # Safety limit

    while not state.terminal and step_count < max_steps:
        step_count += 1
        print(f"--- Step {step_count} ---")

        # Encode current state
        cards_tensor = cards_encoder.encode_cards(state, seat=state.to_act)
        actions_tensor = actions_encoder.encode_actions(
            state, seat=state.to_act, num_bet_bins=9
        )

        # Get legal actions and convert to indices
        legal_actions = env.legal_actions()
        legal_mask = torch.zeros(9)
        for action in legal_actions:
            bin_idx = actions_encoder._action_to_bin(action, state, 9)
            if bin_idx is not None:
                legal_mask[bin_idx] = 1.0

        print(f"Player {state.to_act} to act")
        print(f"Legal actions: {legal_actions}")
        print(f"Current pot: {state.pot}")
        print(f"Player {state.to_act} stack: {state.players[state.to_act].stack}")

        # Show board cards if any
        if state.board:
            print(f"Board: {cards_to_names(state.board)}")

        # Model forward pass
        with torch.no_grad():
            logits, value = trainer.model(
                cards_tensor.unsqueeze(0), actions_tensor.unsqueeze(0)
            )
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        # Apply legal mask
        masked_logits = logits.clone()
        masked_logits[legal_mask == 0] = -1e9

        # Compute policy
        log_probs = F.log_softmax(masked_logits, dim=-1)
        probs = F.softmax(masked_logits, dim=-1)

        print(f"\nModel Analysis:")
        print(f"  Raw value: {value.item():.6f}")
        print(f"  Value interpretation: {value.item():.2f} chips expected")

        print(f"\nPolicy (action probabilities):")
        action_names = [
            "fold",
            "check/call",
            "bet 1/2",
            "bet 3/4",
            "bet pot",
            "bet 1.5x",
            "bet 2x",
            "all-in",
        ]

        for i, (name, prob) in enumerate(zip(action_names, probs)):
            if legal_mask[i] == 1:
                print(f"    {name:12}: {prob.item():.3f} ({log_probs[i].item():.3f})")
            else:
                print(f"    {name:12}: ILLEGAL")

        # Find best action
        best_action = torch.argmax(masked_logits).item()
        best_action_name = action_names[best_action]
        best_prob = probs[best_action].item()

        print(f"\nBest action: {best_action_name} (prob: {best_prob:.3f})")
        print(
            f"Action confidence: {'High' if best_prob > 0.7 else 'Medium' if best_prob > 0.4 else 'Low'}"
        )

        # Take the best action
        print(f"Taking action: {best_action}")
        # Convert index back to Action object
        best_action_obj = legal_actions[0]  # Default to first legal action
        for action in legal_actions:
            bin_idx = actions_encoder._action_to_bin(action, state, 9)
            if bin_idx == best_action:
                best_action_obj = action
                break

        state, reward, done, _ = env.step(best_action_obj)

        if done:
            print(f"\nHand finished!")
            print(f"Final reward: {reward}")
            print(f"Winner: {state.winner if hasattr(state, 'winner') else 'Unknown'}")
            print(f"Final pot: {state.pot}")
            print(f"Player 0 final stack: {state.players[0].stack}")
            print(f"Player 1 final stack: {state.players[1].stack}")
            break

        print(f"Action result: reward={reward}, done={done}")
        print()

    if step_count >= max_steps:
        print("Reached maximum steps, forcing termination")


def main():
    """Main function."""
    print("=== Hand Analysis Script ===")
    print("This script will:")
    print("1. Load a trained model from checkpoint")
    print("2. Analyze a complete poker hand step by step")
    print("3. Show value and policy calculations at each decision point")
    print()

    # Initialize trainer
    trainer = SelfPlayTrainer(
        num_bet_bins=9,
        learning_rate=1e-4,
        batch_size=256,
        grad_clip=0.5,
    )

    # Load trained model from checkpoint
    load_checkpoint(trainer)

    # Analyze a hand
    analyze_hand(trainer)

    print("\n=== Analysis Complete ===")
    print("This shows how the model evaluates poker positions and makes decisions.")
    print("The value function estimates expected chip value, while the policy")
    print("determines action probabilities based on the current game state.")


if __name__ == "__main__":
    main()
