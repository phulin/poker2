#!/usr/bin/env python3
"""
Simplified script to analyze preflop action probabilities for a given checkpoint and hand.
Shows what the model would do when acting first (SB) and as BB when called to act.
"""

import argparse
from pathlib import Path

import torch

from alphaholdem.encoding.action_mapping import get_legal_mask
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.models.cnn import ActionsHUEncoderV1, CardsPlanesV1, SiameseConvNetV1


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with correct parameters for high-perf checkpoints
    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=192,
        actions_hidden=192,
        fusion_hidden=[2048, 2048],  # High-perf uses 2048 instead of 1024
        num_actions=8,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully on {device}")
    return model


def card_to_index(card: str) -> int:
    """Convert card string to index."""
    suits = {"s": 0, "h": 1, "d": 2, "c": 3}
    ranks = {
        "2": 0,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 4,
        "7": 5,
        "8": 6,
        "9": 7,
        "T": 8,
        "J": 9,
        "Q": 10,
        "K": 11,
        "A": 12,
    }

    if len(card) != 2:
        raise ValueError(f"Invalid card format: {card}")

    rank, suit = card[0], card[1]
    if rank not in ranks or suit not in suits:
        raise ValueError(f"Invalid card: {card}")

    # Use suit-major indexing consistent with HUNLTensorEnv (card // 13 = suit, card % 13 = rank)
    return suits[suit] * 13 + ranks[rank]


def create_preflop_state_with_hand(hand_cards: list, device: str = "cpu"):
    """Create a preflop game state with specific hole cards."""
    # Convert hand cards to indices
    card_indices = [card_to_index(card) for card in hand_cards]

    print(f"Hand cards: {hand_cards} -> indices: {card_indices}")

    # Create environment
    env = HUNLEnv(starting_stack=1000, sb=5, bb=10)

    # Reset to get initial state
    state = env.reset()

    # Manually setting exact hole cards is not wired; using default random cards
    print("Note: Using default preflop state (cards will be random)")

    return state, env


def get_action_probabilities(model, state, seat: int, device: str = "cpu"):
    """Get action probabilities and value for the given seat (0=SB,1=BB)."""
    with torch.no_grad():
        # Create encoders
        cards_encoder = CardsPlanesV1()
        actions_encoder = ActionsHUEncoderV1()

        # Encode the state for the specified seat
        cards_features = (
            cards_encoder.encode_cards(state, seat=seat).unsqueeze(0).to(device)
        )
        actions_features = (
            actions_encoder.encode_actions(state, seat=seat).unsqueeze(0).to(device)
        )

        # Convert to float
        cards_float = cards_features.to(torch.float32)
        actions_float = actions_features.to(torch.float32)

        # Get model predictions
        logits, values = model(cards_float, actions_float)

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        return probs.squeeze(0), values.squeeze(0)


def get_action_name(action_idx: int) -> str:
    """Convert action index to human-readable name."""
    if action_idx == 0:
        return "Fold"
    elif action_idx == 1:
        return "Call"
    elif action_idx == 2:
        return "Raise (min)"
    elif action_idx < 10:
        return f"Raise ({action_idx-1}x)"
    else:
        return f"Action {action_idx}"


def print_top3_for_position(model, state, seat: int, title: str, device: str):
    """Compute and print top-3 actions for a given seat using current game_state."""
    # Temporarily set to_act to this seat so legality matches encodings
    prev_to_act = state.to_act
    state.to_act = seat
    try:
        probs, value = get_action_probabilities(model, state, seat, device)
        legal_mask = get_legal_mask(
            state, num_bet_bins=8, dtype=torch.bool, device=device
        )

        # Extract legal probs and sort
        legal_indices = torch.where(legal_mask)[0]
        legal_probs = probs[legal_mask]
        action_prob_pairs = list(zip(legal_indices.tolist(), legal_probs.tolist()))
        action_prob_pairs.sort(key=lambda x: x[1], reverse=True)

        # Print header and top-3
        print(f"\n{title}")
        print(f"Value estimate: {value.item():.4f}")
        print("Top 3 Actions:")
        print("-" * 30)
        for i, (action_idx, prob) in enumerate(action_prob_pairs[:3]):
            action_name = get_action_name(action_idx)
            print(f"{i+1}. {action_name}: {prob:.4f} ({prob*100:.1f}%)")
    finally:
        state.to_act = prev_to_act


def analyze_preflop_actions(
    checkpoint_path: str, hand_cards: list, device: str = "cpu"
):
    """Analyze preflop action probabilities for SB and BB."""
    print("=" * 80)
    print("PREFLOP ACTION ANALYSIS")
    print("=" * 80)

    # Load model
    model = load_model(checkpoint_path, device)

    # Create preflop state with specific hand
    state, env = create_preflop_state_with_hand(hand_cards, device)

    # Small Blind acts first
    print(f"\nHand: {' '.join(hand_cards)}")
    print_top3_for_position(
        model, state, seat=0, title="Small Blind (acts first)", device=device
    )

    # Big Blind when called to act (preflop second to act)
    print_top3_for_position(
        model, state, seat=1, title="Big Blind (called to act)", device=device
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze preflop action probabilities")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("hand", nargs=2, help="Hole cards (e.g., As Kh)")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/mps/cuda)")

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    # Validate hand cards
    hand_cards = args.hand
    if len(hand_cards) != 2:
        print("Error: Must provide exactly 2 hole cards")
        return

    # Validate device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    try:
        analyze_preflop_actions(str(checkpoint_path), hand_cards, args.device)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
