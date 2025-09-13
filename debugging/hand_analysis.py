#!/usr/bin/env python3
"""
Hand Analysis Script

Trains the model for 50 steps, then runs through a complete poker hand
showing value and policy calculations at each decision point.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from alphaholdem.core.structured_config import Config
from alphaholdem.encoding.action_mapping import bin_to_action, get_legal_mask
from alphaholdem.env import rules
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.models.cnn import ActionsHUEncoderV1, CardsPlanesV1
from alphaholdem.rl.self_play import SelfPlayTrainer


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
        stats = trainer.train_step()
        if step % 10 == 0:
            print(
                f"Step {step:2d}: Reward: {stats['avg_reward']:6.2f}, "
                f"Trajectories: {stats['trajectories_collected']}"
            )

    print("Training completed!\n")


def load_checkpoint(trainer, checkpoint_path="checkpoints/final_checkpoint.pt"):
    """Load a trained model from checkpoint (CPU/MPS compatible)."""
    print(f"Loading model from {checkpoint_path}...")

    if os.path.exists(checkpoint_path):
        step = trainer.load_checkpoint(checkpoint_path)
        print(f"Model loaded successfully!")
        print(f"  Training step: {step}")
        print(f"  Episode count: {trainer.episode_count}")
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
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        seed=123,
    )
    # Use trainer's encoders to respect config (e.g., bet_bins)
    cards_encoder = trainer.cards_encoder
    actions_encoder = trainer.actions_encoder
    # Ensure model is on the trainer's device
    device = trainer.device if hasattr(trainer, "device") else torch.device("cpu")
    trainer.model.to(device)

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
        num_bet_bins = trainer.num_bet_bins
        cards_tensor = cards_encoder.encode_cards(state, seat=state.to_act)
        actions_tensor = actions_encoder.encode_actions(state, seat=state.to_act)
        # Move inputs to model device
        cards_tensor = cards_tensor.to(device)
        actions_tensor = actions_tensor.to(device)

        # Legal mask aligned with training/inference
        legal_mask = get_legal_mask(state, num_bet_bins).cpu()

        print(f"Player {state.to_act} to act")
        # Build names from config bet_bins (needed now to print legal actions)
        bin_labels = [f"bet/raise {m:g}x total" for m in trainer.cfg.bet_bins]
        action_names = ["fold", "check/call", *bin_labels, "all-in"]
        legal_idxs = [i for i in range(num_bet_bins) if legal_mask[i] == 1]
        print(
            f"Legal actions: {[action_names[i] if i < len(action_names) else i for i in legal_idxs]}"
        )
        print(f"Current pot: {state.pot}")
        print(f"Player {state.to_act} stack: {state.players[state.to_act].stack}")

        # Show board cards if any
        if state.board:
            print(f"Board: {cards_to_names(state.board)}")

        # Model forward pass
        with torch.no_grad():
            was_training = trainer.model.training
            trainer.model.eval()
            logits, value = trainer.model(
                cards_tensor.unsqueeze(0), actions_tensor.unsqueeze(0)
            )
            if was_training:
                trainer.model.train()
            # Bring outputs to CPU for analysis and masking with CPU tensors
            logits = logits.squeeze(0).detach().cpu()
            value = value.squeeze(0).detach().cpu()

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

        for i, (name, prob) in enumerate(zip(action_names, probs)):
            if legal_mask[i] == 1:
                print(f"    {name:12}: {prob.item():.3f} ({log_probs[i].item():.3f})")
            else:
                print(f"    {name:12}: ILLEGAL")

        # Find best action
        best_action = torch.argmax(masked_logits).item()
        best_action_name = (
            action_names[best_action]
            if best_action < len(action_names)
            else str(best_action)
        )
        best_prob = probs[best_action].item()

        print(f"\nBest action: {best_action_name} (prob: {best_prob:.3f})")
        print(
            f"Action confidence: {'High' if best_prob > 0.7 else 'Medium' if best_prob > 0.4 else 'Low'}"
        )

        # Take the best action
        print(f"Taking action index: {best_action}")
        # Convert index back to Action object using same mapping as training
        best_action_obj = bin_to_action(best_action, state, num_bet_bins)
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    """
    Analyze a poker hand using a trained model checkpoint.

    Args:
        cfg: Hydra configuration object
    """
    print("=== Hand Analysis Script ===")
    print("This script will:")
    print("1. Load a trained model from checkpoint")
    print("2. Analyze a complete poker hand step by step")
    print("3. Show value and policy calculations at each decision point")
    print()

    # Initialize trainer with config
    trainer = SelfPlayTrainer(cfg=cfg, device=torch.device("cpu"))

    # Load trained model from checkpoint
    load_checkpoint(trainer, cfg.resume_from)

    # Analyze a hand
    analyze_hand(trainer)

    print("\n=== Analysis Complete ===")
    print("This shows how the model evaluates poker positions and makes decisions.")
    print("The value function estimates expected chip value, while the policy")
    print("determines action probabilities based on the current game state.")


if __name__ == "__main__":
    main()
