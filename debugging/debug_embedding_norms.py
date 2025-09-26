#!/usr/bin/env python3
"""
Debugging script to analyze embedding norms in a trained transformer model.

This script:
1. Creates a SelfPlayTrainer
2. Loads a checkpoint using load_checkpoint
3. Extracts card ID, rank, and suit embeddings
4. Calculates and prints average norms for each embedding type
"""

import argparse
import itertools
import os
import sys

import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.tokens import (
    Special,
    get_card_token_id_offset,
    get_special_token_id_offset,
)
from alphaholdem.rl.self_play import SelfPlayTrainer


def analyze_embedding_norms(trainer: SelfPlayTrainer) -> None:
    """Analyze and print embedding norms for card ID, rank, and suit embeddings."""

    # Get the transformer model
    model = trainer.model

    # Access the embedding module
    embedding_module = model.embedding

    # Get token offsets
    special_offset = get_special_token_id_offset()
    card_offset = get_card_token_id_offset()

    print(f"Token offsets:")
    print(f"  Special tokens: {special_offset}")
    print(f"  Card tokens: {card_offset}")
    print(f"  Number of special tokens: {Special.NUM_SPECIAL.value}")
    print()

    # Extract base embedding table
    base_embedding = embedding_module.base_embedding.weight.data

    # Extract card ID embeddings (token IDs 7-58, which are cards 0-51)
    card_id_embeddings = base_embedding[card_offset : card_offset + 52]

    # Extract rank and suit embeddings
    rank_embeddings = embedding_module.card_rank_emb.weight.data  # 13 ranks
    suit_embeddings = embedding_module.card_suit_emb.weight.data  # 4 suits

    # Calculate norms
    card_id_norm = torch.norm(card_id_embeddings, dim=1).mean().item()
    rank_norm = torch.norm(rank_embeddings, dim=1).mean().item()
    suit_norm = torch.norm(suit_embeddings, dim=1).mean().item()

    # Print results
    print("Embedding Norms:")
    print(f"  Card ID embeddings (avg norm): {card_id_norm:.6f}")
    print(f"  Rank embeddings (avg norm):    {rank_norm:.6f}")
    print(f"  Suit embeddings (avg norm):    {suit_norm:.6f}")
    print()

    # Additional statistics
    print("Additional Statistics:")
    print(f"  Card ID embedding shape: {card_id_embeddings.shape}")
    print(f"  Rank embedding shape:    {rank_embeddings.shape}")
    print(f"  Suit embedding shape:    {suit_embeddings.shape}")
    print()

    # Show individual card norms
    card_norms = torch.norm(card_id_embeddings, dim=1)
    print("Individual Card ID Norms:")
    for i, norm in enumerate(card_norms):
        print(f"  Card {i:2d}: {norm:.6f}")
    print()

    # Show rank norms
    rank_norms = torch.norm(rank_embeddings, dim=1)
    rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    print("Individual Rank Norms:")
    for i, norm in enumerate(rank_norms):
        print(f"  {rank_names[i]:2s}: {norm:.6f}")
    print()

    # Show suit norms
    suit_norms = torch.norm(suit_embeddings, dim=1)
    suit_names = ["♠", "♥", "♦", "♣"]
    print("Individual Suit Norms:")
    for i, norm in enumerate(suit_norms):
        print(f"  {suit_names[i]:2s}: {norm:.6f}")


def create_river_state_with_cards(
    trainer: SelfPlayTrainer,
    hole_cards: list[int],
    board_cards: list[int],
    device: torch.device,
) -> HUNLTensorEnv:
    """Create a river game state with specific cards."""

    # Create tensor environment
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=50,
        bb=100,
        bet_bins=trainer.tensor_env.bet_bins,
        device=device,
    )

    # Reset environment
    env.reset()

    # Force specific cards by setting the deck
    # Create a deck with our desired cards first (9 cards total: 2 hole + 5 board + 2 extra)
    forced_deck = torch.zeros(9, dtype=torch.long, device=device)
    forced_deck[:7] = torch.tensor(hole_cards + board_cards, device=device)

    # Fill remaining deck with unused cards
    used_cards = set(hole_cards + board_cards)
    unused_cards = [i for i in range(52) if i not in used_cards]
    forced_deck[7:] = torch.tensor(unused_cards[:2], device=device)

    # Reset with forced deck
    env.reset(force_deck=forced_deck.unsqueeze(0))

    # Advance to river by dealing flop, turn, river
    # The cards are already in the deck in the right order
    env.step_bins(torch.tensor([1], device=device))  # Call preflop
    env.step_bins(torch.tensor([1], device=device))  # Call preflop

    # Deal flop (cards 4, 5, 6)
    env.step_bins(torch.tensor([1], device=device))  # Call flop
    env.step_bins(torch.tensor([1], device=device))  # Call flop

    # Deal turn (card 7)
    env.step_bins(torch.tensor([1], device=device))  # Call turn
    env.step_bins(torch.tensor([1], device=device))  # Call turn

    # Deal river (card 8)
    env.step_bins(torch.tensor([1], device=device))  # Call river
    # Now it's player 0's turn to act on the river

    return env


def get_action_predictions(
    trainer: SelfPlayTrainer, env: HUNLTensorEnv, device: torch.device
):
    """Get action predictions from the model for the current state."""

    with torch.no_grad():
        # Encode the current state
        embedding_data = trainer.state_encoder.encode_tensor_states(
            player=0, idxs=torch.tensor([0], device=device)
        )

        # Get model predictions
        outputs = trainer.model(embedding_data)
        logits = outputs.policy_logits.squeeze(0)  # [num_bet_bins]
        value = outputs.value.squeeze(0).item()

        # Get legal actions
        legal_mask = env.legal_bins_mask()[0]  # [num_bet_bins]

        # Apply legal mask
        masked_logits = torch.where(legal_mask == 0, -1e9, logits)

        # Convert to probabilities
        probs = torch.softmax(masked_logits, dim=-1)

        return probs, value, legal_mask


def test_suit_permutations(trainer: SelfPlayTrainer, device: torch.device) -> None:
    """Test how suit permutations affect river action predictions."""

    print("=" * 80)
    print("RIVER ACTION PREDICTION TEST WITH SUIT PERMUTATIONS")
    print("=" * 80)

    # Define a sample hand: pocket aces vs a board that makes a straight
    # Hole cards: A♠ A♥ (0, 13)
    # Board: K♠ Q♠ J♠ T♠ 9♠ (12, 11, 10, 9, 8) - royal flush board
    hole_cards = [0, 13]  # A♠ A♥
    base_board = [12, 11, 10, 9, 8]  # K♠ Q♠ J♠ T♠ 9♠

    print(f"Hole cards: A♠ A♥ (indices: {hole_cards})")
    print(f"Base board: K♠ Q♠ J♠ T♠ 9♠ (indices: {base_board})")
    print()

    # Get all 6 permutations of suits (0,1,2,3)
    suit_permutations = list(itertools.permutations([0, 1, 2, 3]))
    suit_names = ["♠", "♥", "♦", "♣"]

    results = []

    for i, suit_perm in enumerate(suit_permutations):
        print(f"Permutation {i+1}: {[suit_names[s] for s in suit_perm]}")

        # Create board with permuted suits
        # Keep ranks the same, just change suits
        permuted_board = []
        for j, card in enumerate(base_board):
            rank = card % 13
            new_suit = suit_perm[j % len(suit_perm)]  # Cycle through permutation
            new_card = new_suit * 13 + rank
            permuted_board.append(new_card)

        print(
            f"  Board cards: {[f'{chr(65+12-rank)}{suit_names[card//13]}' for card in permuted_board]}"
        )
        print(f"  Board indices: {permuted_board}")

        # Create river state
        env = create_river_state_with_cards(trainer, hole_cards, permuted_board, device)

        # Get predictions
        probs, value, legal_mask = get_action_predictions(trainer, env, device)

        # Store results
        results.append(
            {
                "permutation": suit_perm,
                "board": permuted_board,
                "probs": probs,
                "value": value,
                "legal_mask": legal_mask,
            }
        )

        # Print action probabilities
        action_names = [
            "Fold",
            "Call",
            "Bet 0.5x",
            "Bet 1.0x",
            "Bet 1.5x",
            "Bet 2.0x",
            "Bet 2.5x",
            "Bet 3.0x",
        ]
        print(f"  Action probabilities:")
        for j, (prob, legal) in enumerate(zip(probs, legal_mask)):
            if legal:
                print(f"    {action_names[j]:12s}: {prob:.4f}")
        print(f"  Value estimate: {value:.4f}")
        print()

    # Compare results
    print("=" * 80)
    print("COMPARISON OF PREDICTIONS")
    print("=" * 80)

    # Find the most different predictions
    max_diff = 0
    max_diff_pair = None

    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            prob_diff = (
                torch.abs(results[i]["probs"] - results[j]["probs"]).max().item()
            )
            value_diff = abs(results[i]["value"] - results[j]["value"])

            if prob_diff > max_diff:
                max_diff = prob_diff
                max_diff_pair = (i, j)

            print(f"Perm {i+1} vs Perm {j+1}:")
            print(f"  Max prob difference: {prob_diff:.4f}")
            print(f"  Value difference: {value_diff:.4f}")
            print()

    if max_diff_pair:
        i, j = max_diff_pair
        print(f"Largest difference: Permutation {i+1} vs {j+1}")
        print(f"Max probability difference: {max_diff:.4f}")

        # Show the specific differences
        prob_diff = torch.abs(results[i]["probs"] - results[j]["probs"])
        print("Action probability differences:")
        for k, diff in enumerate(prob_diff):
            if results[i]["legal_mask"][k] or results[j]["legal_mask"][k]:
                print(f"  {action_names[k]:12s}: {diff:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze embedding norms in a trained transformer model"
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the checkpoint file to load"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict model loading (default: False)",
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint_path}")

    # Load checkpoint to get config info
    device_obj = torch.device(args.device)

    # Create a minimal config for debugging
    config = Config(
        train=TrainingConfig(batch_size=32),
        model=ModelConfig(
            name="poker_transformer_v1",
            kwargs={
                "max_sequence_length": 50,
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 2,
                "num_bet_bins": 8,
                "dropout": 0.1,
                "use_gradient_checkpointing": False,
            },
        ),
        env=EnvConfig(bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0]),
        use_tensor_env=True,
        num_envs=1,
        device=args.device,
        use_wandb=False,
    )

    print(f"Creating SelfPlayTrainer...")

    # Create trainer
    trainer = SelfPlayTrainer(config, device_obj)

    print(f"Loading checkpoint into trainer...")

    # Load the checkpoint
    if args.strict:
        # Load checkpoint manually with strict=True
        checkpoint = torch.load(
            args.checkpoint_path, weights_only=False, map_location=device_obj
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        step = checkpoint.get("step", 0)
        wandb_run_id = checkpoint.get("wandb_run_id")
        print(f"Checkpoint loaded with strict=True")
    else:
        # Use the trainer's load_checkpoint method (strict=False by default)
        step, wandb_run_id = trainer.load_checkpoint(args.checkpoint_path)

    print(f"✅ Checkpoint loaded successfully")
    print(f"   Step: {step}")
    print(f"   ELO: {trainer.opponent_pool.current_elo:.1f}")
    print()

    # Analyze embedding norms
    analyze_embedding_norms(trainer)

    # Test suit permutations on river
    test_suit_permutations(trainer, device_obj)


if __name__ == "__main__":
    main()
