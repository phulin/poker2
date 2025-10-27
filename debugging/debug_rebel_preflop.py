#!/usr/bin/env python3
"""
Debugging script to load a ReBeL checkpoint and analyze poker strategies.

This script loads a trained ReBeL model checkpoint and can perform two types of analysis:

1. PREFLOP ANALYSIS (default):
   - SB preflop action probabilities and value estimates
   - BB response probabilities when facing SB all-in
   - Suited vs offsuit breakdowns

2. RIVER ANALYSIS (--river flag):
   - Analyze specific river situations with custom board and hole cards
   - Show action probabilities and value estimates for both players
   - Display recommended actions based on model predictions

Usage:
    # Preflop analysis
    python debug_rebel_preflop.py --checkpoint checkpoints-rebel/rebel_final.pt
    python debug_rebel_preflop.py --checkpoint checkpoints-rebel/rebel_latest.pt --device cuda

    # River analysis
    python debug_rebel_preflop.py --checkpoint checkpoints-rebel/rebel_final.pt --river
    python debug_rebel_preflop.py --checkpoint checkpoints-rebel/rebel_final.pt --river --board "As Kh Qd Jc Ts" --sb-cards "Ac Kc" --bb-cards "Ad Kd"
"""

import argparse
import os
import sys
from typing import Optional, List, Tuple

import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from alphaholdem.core.structured_config import (
    CFRType,
    Config,
    EnvConfig,
    ModelConfig,
    SearchConfig,
    TrainingConfig,
)
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.utils.training_utils import print_preflop_range_grid
from alphaholdem.env.types import GameState, PlayerState
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.utils.model_utils import get_probs_and_values
from alphaholdem.encoding.action_mapping import get_legal_mask


def create_default_config() -> Config:
    """Create a default configuration for ReBeL debugging."""
    config = Config()

    # Basic settings
    config.num_steps = 1000  # Not used for debugging
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.seed = 42
    config.use_wandb = False
    config.checkpoint_dir = "checkpoints-rebel"
    config.checkpoint_interval = 50
    config.economize_checkpoints = False

    # Training config
    config.train = TrainingConfig()
    config.train.learning_rate = 3e-4
    config.train.batch_size = 1024
    config.train.replay_buffer_batches = 8
    config.train.value_coef = 1.0
    config.train.entropy_coef = 0.0
    config.train.grad_clip = 1.0
    config.train.max_sequence_length = 32

    # Model config
    config.model = ModelConfig()
    config.model.name = "rebel_ffn"
    config.model.input_dim = 2661  # Will be overridden by RebelCFRTrainer
    config.model.num_actions = -1  # Will be overridden by RebelCFRTrainer
    config.model.hidden_dim = 1536
    config.model.num_hidden_layers = 6
    config.model.value_head_type = "scalar"
    config.model.value_head_num_quantiles = 1
    config.model.detach_value_head = True

    # Environment config
    config.env = EnvConfig()
    config.env.stack = 1000
    config.env.sb = 5
    config.env.bb = 10
    config.env.bet_bins = [0.5, 1.5]
    config.env.flop_showdown = False

    # Search config (reduced for debugging to save memory)
    config.search = SearchConfig()
    config.search.enabled = True
    config.search.depth = 3  # Reduced depth
    config.search.iterations = 50  # Reduced iterations
    config.search.warm_start_iterations = 5  # Reduced warm start
    config.search.branching = 4
    config.search.belief_samples = 4  # Reduced belief samples
    config.search.dcfr_alpha = 1.5
    config.search.dcfr_beta = 0.0
    config.search.dcfr_gamma = 2.0
    config.search.include_average_policy = True
    config.search.cfr_type = CFRType.linear
    config.search.cfr_avg = True

    return config


def load_rebel_checkpoint_and_trainer(
    checkpoint_path: str, device: str = "auto"
) -> RebelCFRTrainer:
    """Load a ReBeL checkpoint and create a trainer instance."""

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading ReBeL checkpoint: {checkpoint_path}")

    # Determine device
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    print(f"Using device: {device_obj}")

    # Create config
    config = create_default_config()
    config.device = device_obj.type

    # Load checkpoint to inspect its contents
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location=device_obj
    )

    # Extract step information
    step = checkpoint.get("step", "unknown")
    wandb_run_id = checkpoint.get("wandb_run_id", None)

    print(f"Checkpoint step: {step}")
    if wandb_run_id:
        print(f"Wandb run ID: {wandb_run_id}")

    # Create trainer
    trainer = RebelCFRTrainer(cfg=config, device=device_obj)

    # Load the checkpoint
    loaded_step = trainer.load_checkpoint(checkpoint_path)

    print(f"✅ ReBeL checkpoint loaded successfully")
    print(f"   Loaded step: {loaded_step}")
    print(
        f"   Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}"
    )
    print(f"   Bet bins: {trainer.bet_bins}")
    print(f"   Number of actions: {trainer.num_actions}")

    return trainer


def create_river_state(
    board_cards: List[int],
    hole_cards_sb: List[int],
    hole_cards_bb: List[int],
    pot_size: int = 100,
    sb_amount: int = 5,
    bb_amount: int = 10,
    button: int = 0,
) -> GameState:
    """Create a river GameState with specific board cards and all-call betting history.

    Args:
        board_cards: List of 5 card indices (0-51) for the board
        hole_cards_sb: List of 2 card indices for SB hole cards
        hole_cards_bb: List of 2 card indices for BB hole cards
        pot_size: Total pot size (default 100)
        sb_amount: Small blind amount (default 5)
        bb_amount: Big blind amount (default 10)
        button: Button position (0 or 1, default 0)

    Returns:
        GameState on the river with all-call betting history
    """
    # Create a dummy deck (not used for river analysis)
    deck = list(range(52))

    # Create player states
    sb_player = PlayerState(stack=1000 - sb_amount)
    sb_player.hole_cards = hole_cards_sb
    sb_player.committed = sb_amount
    sb_player.stack_after_posting = sb_player.stack

    bb_player = PlayerState(stack=1000 - bb_amount)
    bb_player.hole_cards = hole_cards_bb
    bb_player.committed = bb_amount
    bb_player.stack_after_posting = bb_player.stack

    # Create all-call betting history
    # Preflop: SB calls BB
    # Flop: Both check
    # Turn: Both check
    # River: Both check (current state)
    action_history = [
        (
            "preflop",
            0,
            "call",
            bb_amount - sb_amount,
            bb_amount - sb_amount,
            sb_amount,
        ),  # SB calls BB
        ("flop", 0, "check", 0, 0, bb_amount),  # SB checks
        ("flop", 1, "check", 0, 0, bb_amount),  # BB checks
        ("turn", 0, "check", 0, 0, bb_amount),  # SB checks
        ("turn", 1, "check", 0, 0, bb_amount),  # BB checks
        ("river", 0, "check", 0, 0, bb_amount),  # SB checks
    ]

    # Create game state
    game_state = GameState(
        button=button,
        street="river",
        deck=deck,
        board=board_cards,
        pot=pot_size,
        to_act=1,  # BB to act after SB checks
        small_blind=sb_amount,
        big_blind=bb_amount,
        min_raise=bb_amount,
        last_aggressive_amount=bb_amount,
        players=(sb_player, bb_player),
        terminal=False,
        winner=None,
        action_history=action_history,
    )

    return game_state


def analyze_river_state(
    trainer: RebelCFRTrainer,
    board_cards: List[int],
    hole_cards_sb: List[int],
    hole_cards_bb: List[int],
    pot_size: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analyze a river state and get model predictions for both players.

    Args:
        trainer: ReBeL trainer with loaded model
        board_cards: List of 5 card indices (0-51) for the board
        hole_cards_sb: List of 2 card indices for SB hole cards
        hole_cards_bb: List of 2 card indices for BB hole cards
        pot_size: Total pot size

    Returns:
        Tuple of (sb_probs, sb_values, bb_probs, bb_values)
    """
    # Create a temporary tensor environment for encoding
    temp_env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=trainer.bet_bins,
        device=trainer.device,
        float_dtype=torch.float32,
    )

    # Create feature encoder
    feature_encoder = RebelFeatureEncoder(
        env=temp_env,
        device=trainer.device,
        dtype=torch.float32,
    )

    # Set up the environment state
    temp_env.reset()

    # Set board cards
    temp_env.board_indices[0, :5] = torch.tensor(board_cards, device=trainer.device)

    # Set hole cards
    temp_env.hole_indices[0, 0, :2] = torch.tensor(hole_cards_sb, device=trainer.device)
    temp_env.hole_indices[0, 1, :2] = torch.tensor(hole_cards_bb, device=trainer.device)

    # Set pot size
    temp_env.pot[0] = pot_size

    # Set street to river
    temp_env.street[0] = 4  # river

    # Set to_act to BB (player 1)
    temp_env.to_act[0] = 1

    # Create beliefs tensor (uniform distribution for simplicity)
    beliefs = torch.ones(1, 2, 1326, device=trainer.device) / 1326

    # Get predictions for both players
    results = {}

    for player in [0, 1]:  # SB and BB
        # Set to_act to current player
        temp_env.to_act[0] = player

        # Encode features for this player
        features = feature_encoder.encode(
            agents=torch.tensor([player], device=trainer.device), beliefs=beliefs
        )

        # Get legal mask - create a simple legal mask for river analysis
        legal_mask = torch.zeros(
            1, trainer.num_actions, dtype=torch.float32, device=trainer.device
        )
        legal_mask[0, 0] = 1.0  # Fold (if facing bet)
        legal_mask[0, 1] = 1.0  # Check/Call
        legal_mask[0, 2] = 1.0  # Bet 0.5x
        legal_mask[0, 3] = 1.0  # Bet 1.5x
        legal_mask[0, 4] = 1.0  # All-in

        # Get model predictions
        with torch.no_grad():
            outputs = trainer.model(features)

            # For ReBeL models, we need to extract the right hand combination
            # For simplicity, we'll use the first hand combination (index 0)
            # In a real implementation, you'd need to map the actual hole cards to combo index
            logits = outputs.policy_logits[0, 0, :]  # [num_actions]
            values = outputs.hand_values[0, player, 0]  # scalar value

            # Apply legal mask
            masked_logits = torch.where(legal_mask[0] == 0, -1e9, logits)

            # Get probabilities
            probs = torch.softmax(masked_logits, dim=-1)

            results[player] = (probs, values)

    return results[0][0], results[0][1], results[1][0], results[1][1]


def card_index_to_string(card_idx: int) -> str:
    """Convert card index (0-51) to string representation like 'As' or 'Kh'."""
    if card_idx < 0 or card_idx > 51:
        return "??"

    rank = card_idx % 13
    suit = card_idx // 13

    rank_str = "23456789TJQKA"[rank]
    suit_str = "shdc"[suit]  # spades, hearts, diamonds, clubs

    return f"{rank_str}{suit_str}"


def display_river_analysis(
    board_cards: List[int],
    hole_cards_sb: List[int],
    hole_cards_bb: List[int],
    sb_probs: torch.Tensor,
    sb_values: torch.Tensor,
    bb_probs: torch.Tensor,
    bb_values: torch.Tensor,
    bet_bins: List[float],
) -> None:
    """Display river analysis results in a readable format."""

    print(f"\n{'='*80}")
    print("RIVER ANALYSIS")
    print(f"{'='*80}")

    # Display board and hole cards
    board_str = " ".join(card_index_to_string(card) for card in board_cards)
    sb_cards_str = " ".join(card_index_to_string(card) for card in hole_cards_sb)
    bb_cards_str = " ".join(card_index_to_string(card) for card in hole_cards_bb)

    print(f"Board: {board_str}")
    print(f"SB Cards: {sb_cards_str}")
    print(f"BB Cards: {bb_cards_str}")
    print()

    # Action names
    action_names = ["Fold", "Check/Call"]
    for i, bet_size in enumerate(bet_bins):
        action_names.append(f"Bet {bet_size:.1f}x")
    action_names.append("All-in")

    # Display SB analysis
    print("SMALL BLIND (Player 0):")
    print(f"  Value Estimate: {sb_values.item():.4f}")
    print("  Action Probabilities:")
    for i, (prob, action_name) in enumerate(zip(sb_probs, action_names)):
        if prob > 0.001:  # Only show actions with >0.1% probability
            print(f"    {action_name}: {prob.item():.4f} ({prob.item()*100:.1f}%)")
    print()

    # Display BB analysis
    print("BIG BLIND (Player 1):")
    print(f"  Value Estimate: {bb_values.item():.4f}")
    print("  Action Probabilities:")
    for i, (prob, action_name) in enumerate(zip(bb_probs, action_names)):
        if prob > 0.001:  # Only show actions with >0.1% probability
            print(f"    {action_name}: {prob.item():.4f} ({prob.item()*100:.1f}%)")
    print()

    # Display best actions
    sb_best_action = torch.argmax(sb_probs).item()
    bb_best_action = torch.argmax(bb_probs).item()

    print("RECOMMENDED ACTIONS:")
    print(
        f"  SB: {action_names[sb_best_action]} (prob: {sb_probs[sb_best_action].item():.4f})"
    )
    print(
        f"  BB: {action_names[bb_best_action]} (prob: {bb_probs[bb_best_action].item():.4f})"
    )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Debug ReBeL checkpoint preflop analyzer probabilities and river analysis"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ReBeL checkpoint file (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for computation",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step number to display (defaults to checkpoint step)",
    )
    parser.add_argument(
        "--river",
        action="store_true",
        help="Run river analysis instead of preflop analysis",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="As Kh Qd Jc Ts",
        help="Board cards (e.g., 'As Kh Qd Jc Ts')",
    )
    parser.add_argument(
        "--sb-cards",
        type=str,
        default="Ac Kc",
        help="SB hole cards (e.g., 'Ac Kc')",
    )
    parser.add_argument(
        "--bb-cards",
        type=str,
        default="Ad Kd",
        help="BB hole cards (e.g., 'Ad Kd')",
    )
    parser.add_argument(
        "--pot-size",
        type=int,
        default=100,
        help="Pot size for river analysis",
    )

    args = parser.parse_args()

    try:
        # Load trainer from checkpoint
        trainer = load_rebel_checkpoint_and_trainer(args.checkpoint, args.device)

        # Determine step to display
        display_step = args.step if args.step is not None else trainer.cfg.num_steps

        if args.river:
            # Parse card strings
            def parse_cards(card_str: str) -> List[int]:
                """Parse card string like 'As Kh' to list of card indices."""
                cards = []
                for card in card_str.split():
                    if len(card) != 2:
                        raise ValueError(f"Invalid card format: {card}")

                    rank_char = card[0].upper()
                    suit_char = card[1].lower()

                    # Convert rank
                    if rank_char == "T":
                        rank = 8  # 10
                    elif rank_char == "J":
                        rank = 9
                    elif rank_char == "Q":
                        rank = 10
                    elif rank_char == "K":
                        rank = 11
                    elif rank_char == "A":
                        rank = 12
                    else:
                        rank = int(rank_char) - 2

                    # Convert suit
                    suit_map = {"s": 0, "h": 1, "d": 2, "c": 3}
                    if suit_char not in suit_map:
                        raise ValueError(f"Invalid suit: {suit_char}")
                    suit = suit_map[suit_char]

                    # Card index = suit * 13 + rank
                    card_idx = suit * 13 + rank
                    cards.append(card_idx)

                return cards

            # Parse cards
            board_cards = parse_cards(args.board)
            sb_cards = parse_cards(args.sb_cards)
            bb_cards = parse_cards(args.bb_cards)

            if len(board_cards) != 5:
                raise ValueError(
                    f"Board must have exactly 5 cards, got {len(board_cards)}"
                )
            if len(sb_cards) != 2:
                raise ValueError(
                    f"SB cards must have exactly 2 cards, got {len(sb_cards)}"
                )
            if len(bb_cards) != 2:
                raise ValueError(
                    f"BB cards must have exactly 2 cards, got {len(bb_cards)}"
                )

            # Run river analysis
            print(f"\n{'='*80}")
            print(f"ReBeL River Analysis (Step {display_step})")
            print(f"{'='*80}")

            sb_probs, sb_values, bb_probs, bb_values = analyze_river_state(
                trainer=trainer,
                board_cards=board_cards,
                hole_cards_sb=sb_cards,
                hole_cards_bb=bb_cards,
                pot_size=args.pot_size,
            )

            display_river_analysis(
                board_cards=board_cards,
                hole_cards_sb=sb_cards,
                hole_cards_bb=bb_cards,
                sb_probs=sb_probs,
                sb_values=sb_values,
                bb_probs=bb_probs,
                bb_values=bb_values,
                bet_bins=trainer.bet_bins,
            )

        else:
            # Run preflop analysis
            print(f"\n{'='*80}")
            print(f"ReBeL Preflop Analyzer Probabilities (Step {display_step})")
            print(f"{'='*80}")

            # Print preflop range grids using the same function as training
            print_preflop_range_grid(
                trainer=trainer,
                step=display_step,
                title=f"ReBeL Preflop Analysis (Step {display_step})",
                rebel=True,
            )

        print(f"\n{'='*80}")
        print("Analysis complete!")
        print(f"{'='*80}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
