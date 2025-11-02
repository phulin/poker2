#!/usr/bin/env python3
"""
Debugging script to show a depth-1 CFR tree with regrets and policy probabilities.

This script creates a depth-1 CFR tree and displays:
- Each child node on one line
- The action taken to reach it
- Policy probability to take that action
- Regret for taking that action
- Current average policy probability
- Iterations 16-25 (post-warm-start)

Usage:
    # Preflop mode (default)
    python debug_cfr_depth1.py --cfr-type linear
    python debug_cfr_depth1.py --cfr-type discounted --checkpoint checkpoints-rebel/rebel_final.pt

    # River mode
    python debug_cfr_depth1.py --river
    python debug_cfr_depth1.py --river --checkpoint rebel_step_1250.pt
"""

import argparse
import os
import random

import torch

from alphaholdem.core.structured_config import (
    CFRType,
    Config,
    EnvConfig,
    ModelConfig,
    ModelType,
    SearchConfig,
    TrainingConfig,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.types import GameState, PlayerState
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator


def action_to_string(action_idx: int, bet_bins: list[float]) -> str:
    """Convert action index to string representation."""
    if action_idx == 0:
        return "FOLD"
    elif action_idx == 1:
        return "CALL"
    elif action_idx == len(bet_bins) + 2:
        return "ALLIN"
    elif action_idx >= 2 and action_idx < len(bet_bins) + 2:
        bet_size = bet_bins[action_idx - 2]
        return f"BET{bet_size:.1f}x"
    else:
        return f"ACTION_{action_idx}"


def load_model_from_checkpoint(
    checkpoint_path: str, config: Config, device: torch.device
) -> RebelFFN | BetterFFN:
    """Load a model from checkpoint using RebelCFRTrainer."""
    print(f"Loading model from {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Falling back to random model initialization.")
        return create_random_model(config, device)

    # Inspect checkpoint to determine model type and architecture
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]

    # Detect model type from checkpoint keys
    is_better_ffn = "street_embedding.weight" in model_state
    if is_better_ffn:
        print("Detected BetterFFN model in checkpoint")
        config.model.name = ModelType.better_ffn

        # Extract hidden dimension from checkpoint
        if "post_norm.weight" in model_state:
            hidden_dim = model_state["post_norm.weight"].shape[0]
            print(f"Detected hidden_dim: {hidden_dim}")
            config.model.hidden_dim = hidden_dim
            config.model.range_hidden_dim = 128  # Default for BetterFFN
            config.model.ffn_dim = hidden_dim * 2  # Common pattern

        # Extract number of hidden layers from checkpoint
        trunk_keys = [k for k in model_state.keys() if k.startswith("trunk.")]
        if trunk_keys:
            layer_indices = set([int(k.split(".")[1]) for k in trunk_keys])
            num_hidden_layers = len(layer_indices)
            print(f"Detected num_hidden_layers: {num_hidden_layers}")
            config.model.num_hidden_layers = num_hidden_layers
    else:
        print("Detected RebelFFN model in checkpoint")
        config.model.name = ModelType.rebel_ffn

    # Create trainer with the adjusted config
    trainer = RebelCFRTrainer(cfg=config, device=device)

    # Load checkpoint using trainer's method
    loaded_step = trainer.load_checkpoint(checkpoint_path)

    print(f"Model loaded successfully (step: {loaded_step})")

    # Extract and return the model
    model = trainer.model
    model.eval()
    return model


def create_random_model(config: Config, device: torch.device) -> RebelFFN:
    """Create a random model for testing."""
    print("Creating random model for testing")

    model = RebelFFN(
        input_dim=config.model.input_dim,
        num_actions=len(config.env.bet_bins) + 3,
        hidden_dim=config.model.hidden_dim,
        num_hidden_layers=config.model.num_hidden_layers,
        detach_value_head=config.model.detach_value_head,
        num_players=2,
    )

    # Initialize with random weights
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(42)
    model.init_weights(cpu_rng)
    model.to(device)
    model.eval()

    print("Random model created successfully")
    return model


def create_river_state(
    sb_amount: int = 5,
    bb_amount: int = 10,
    starting_stack: int = 1000,
    button: int = 0,
    seed: int | None = None,
) -> GameState:
    """Create a river GameState with reasonable betting history.

    Betting history:
    - Preflop: SB calls BB (pot=20)
    - Flop: Both check (pot=20)
    - Turn: SB checks, BB bets 2x pot (40), SB calls (pot=100)
    - River: SB checks (current state, pot=100)

    Args:
        sb_amount: Small blind amount
        bb_amount: Big blind amount
        starting_stack: Starting stack for each player
        button: Button position (0 or 1)
        seed: Random seed for shuffling deck

    Returns:
        GameState at the beginning of the river
    """
    # Create a deck
    deck = list(range(52))

    if seed is not None:
        random.seed(seed)
    random.shuffle(deck)

    # Deal board cards (5 cards for river)
    board = [deck.pop() for _ in range(5)]

    # Create player states with hole cards
    sb_player = PlayerState(
        stack=starting_stack - sb_amount - 40
    )  # Lost blinds + call turn bet
    sb_player.hole_cards = [deck.pop(), deck.pop()]
    sb_player.committed = sb_amount + 40  # Posted blind + called turn bet
    sb_player.stack_after_posting = sb_player.stack

    bb_player = PlayerState(
        stack=starting_stack - bb_amount - 40
    )  # Lost blind + bet on turn
    bb_player.hole_cards = [deck.pop(), deck.pop()]
    bb_player.committed = bb_amount + 40  # Posted blind + bet on turn
    bb_player.stack_after_posting = bb_player.stack

    # Create betting history
    action_history = [
        # Preflop
        ("preflop", 0, "call", bb_amount - sb_amount, bb_amount - sb_amount, sb_amount),
        # Flop
        ("flop", 0, "check", 0, 0, bb_amount),
        ("flop", 1, "check", 0, 0, bb_amount),
        # Turn
        ("turn", 0, "check", 0, 0, bb_amount),
        ("turn", 1, "bet", 40, 40, bb_amount),  # BB bets 2x pot
        ("turn", 0, "call", 40, 40, bb_amount + 40),
        # River
        ("river", 0, "check", 0, 0, bb_amount + 40),  # SB checks
    ]

    # Create game state
    game_state = GameState(
        button=button,
        street="river",
        deck=deck,
        board=board,
        pot=bb_amount
        + sb_amount
        + 40
        + 40,  # Initial blinds (15) + turn betting (80) = 95
        to_act=1,  # BB to act after SB checks
        small_blind=sb_amount,
        big_blind=bb_amount,
        min_raise=bb_amount,
        last_aggressive_amount=40,  # Last bet was 40 on turn
        players=(sb_player, bb_player),
        terminal=False,
        winner=None,
        action_history=action_history,
    )

    return game_state


def create_default_config() -> Config:
    """Create a default configuration for debugging."""
    config = Config()
    config.num_envs = 1
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.seed = 42

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
    config.model.name = ModelType.rebel_ffn
    config.model.input_dim = 2661
    config.model.hidden_dim = 1536
    config.model.num_hidden_layers = 6
    config.model.value_head_type = "scalar"
    config.model.value_head_num_quantiles = 1
    config.model.detach_value_head = True
    config.model.compile = False

    # Environment config
    config.env = EnvConfig()
    config.env.stack = 1000
    config.env.sb = 5
    config.env.bb = 10
    config.env.bet_bins = [0.5, 1.5]
    config.env.flop_showdown = False

    # Search config
    config.search = SearchConfig()
    config.search.enabled = True
    config.search.depth = 1  # Depth-1 tree for debugging
    config.search.iterations = 201  # Show up to iteration 25 (range goes to 26)
    config.search.warm_start_iterations = 15  # Warm start 15 iterations
    config.search.branching = 4
    config.search.belief_samples = 1
    config.search.dcfr_alpha = 1.5
    config.search.dcfr_beta = 0.0
    config.search.dcfr_gamma = 2.0
    config.search.dcfr_plus_delay = 50
    config.search.include_average_policy = True
    config.search.cfr_type = CFRType.linear
    config.search.cfr_avg = True

    return config


def print_single_iteration_data(
    evaluator: RebelCFREvaluator,
    iteration: int,
    bet_bins: list[float],
    num_actions: int,
) -> None:
    """Print data for a single CFR iteration."""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}")
    print(f"{'='*80}")

    # Get root node (depth 0)
    root_idx = evaluator.depth_offsets[0]

    # Get depth 1 nodes
    depth1_offset = evaluator.depth_offsets[1]
    depth1_end = evaluator.depth_offsets[2]

    print(f"\nChild Nodes (depth 1):")
    print(
        f"{'Node':>6} | {'Action':>10} | {'Policy':>7} | {'PolicyAvg':>7} | {'Regret [min, max]':>24} | {'Value':>7} | {'BeliefSum':>9}"
    )
    print("-" * 80)

    for child_idx in range(depth1_offset, depth1_end):
        if not evaluator.valid_mask[child_idx]:
            continue

        # Determine which action was taken to reach this node
        # For depth 1: child_idx = depth1_offset + action
        action_idx = child_idx - depth1_offset

        # Policy probs and regrets are stored on the child node itself
        # So we look at policy_probs[child_idx] and cumulative_regrets[child_idx]
        allowed_hands = evaluator.allowed_hands[child_idx]
        policy_probs = evaluator.policy_probs[child_idx].clone()
        policy_probs.masked_fill_(~allowed_hands, 0.0)
        policy_probs_avg = evaluator.policy_probs_avg[child_idx].clone()
        policy_probs_avg.masked_fill_(~allowed_hands, 0.0)
        policy_prob = policy_probs.sum().item() / allowed_hands.sum().item()
        policy_avg_prob = policy_probs_avg.sum().item() / allowed_hands.sum().item()
        # Note: cumulative_regrets can be negative; policy uses clamped version (regret matching)
        regret = evaluator.cumulative_regrets[child_idx].mean().item()
        regret_max = evaluator.cumulative_regrets[child_idx].max().item()
        regret_min = evaluator.cumulative_regrets[child_idx].min().item()

        # Get value at this node - average over both players and all hands
        # values_avg has shape [M, 2, NUM_HANDS]
        value = evaluator.values_avg[child_idx, :, :].mean().item()

        # Get belief sum at this node - sum over both players and all hands
        belief_sum = evaluator.beliefs_avg[child_idx, :, :].sum().item()

        action_name = action_to_string(action_idx, bet_bins)

        print(
            f"{child_idx:>6} | {action_name:>10} | {policy_prob:7.4f} | {policy_avg_prob:7.4f} | "
            f"{regret:7.2f} [{regret_min:7.2f}, {regret_max:7.2f}] | {value:7.4f} | {belief_sum:9.4f}"
        )


def debug_cfr_depth1(
    checkpoint_path: str | None = None,
    cfr_type_str: str = "linear",
    river_mode: bool = False,
    river_seed: int | None = None,
    iterations: int | None = None,
    no_cfr_avg: bool = False,
    dcfr_delay: int | None = None,
) -> None:
    """Main debugging function."""
    print("=== ReBeL CFR Depth-1 Debugger ===")
    if river_mode:
        print("Mode: RIVER")
    else:
        print("Mode: PREFLOP")

    # Determine CFR type
    if cfr_type_str == "linear":
        cfr_type = CFRType.linear
        dcfr_delay = 0
    elif cfr_type_str == "discounted":
        cfr_type = CFRType.discounted
        dcfr_delay = 0
    elif cfr_type_str == "discounted_plus":
        cfr_type = CFRType.discounted_plus
        if dcfr_delay is None:
            dcfr_delay = 70
    else:
        raise ValueError(f"Unknown CFR type: {cfr_type_str}")

    print(f"CFR Type: {cfr_type_str}")
    if cfr_type == CFRType.discounted:
        print(f"DCFR Delay: {dcfr_delay}")

    # Create configuration
    config = create_default_config()
    config.search.cfr_type = cfr_type
    config.search.cfr_avg = not no_cfr_avg
    config.seed = random.randint(0, 1000000) if river_seed is None else river_seed
    if iterations is not None:
        # Ensure warm_start < iterations
        config.search.iterations = iterations
        if config.search.warm_start_iterations >= config.search.iterations:
            config.search.warm_start_iterations = max(0, config.search.iterations - 1)
    device = torch.device(config.device)

    print(f"Using device: {device}")
    print(f"Tree depth: {config.search.depth}")
    print(f"Warm start iterations: {config.search.warm_start_iterations}")
    print(f"Total iterations: {config.search.iterations}")

    # Load or create model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_model_from_checkpoint(checkpoint_path, config, device)
    else:
        model = create_random_model(config, device)

    # Create environment
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=config.env.stack,
        sb=config.env.sb,
        bb=config.env.bb,
        default_bet_bins=config.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=config.env.flop_showdown,
    )

    env.reset()

    if river_mode:
        # Set up river state manually
        river_state = create_river_state(
            sb_amount=config.env.sb,
            bb_amount=config.env.bb,
            starting_stack=config.env.stack,
            button=0,
            seed=river_seed,
        )

        # Convert GameState to tensor format
        # Deck order: board cards first, then hole cards
        deck_cards = torch.tensor(
            river_state.board
            + river_state.players[0].hole_cards
            + river_state.players[1].hole_cards,
            device=device,
            dtype=torch.long,
        ).unsqueeze(
            0
        )  # Shape: [1, 9]

        # Set up the environment tensors manually for river state
        env_idx = 0
        env.deck[env_idx, :9] = deck_cards[0]
        env.deck_pos[env_idx] = 9  # All cards dealt

        # Set hole cards
        env.hole_indices[env_idx, 0, 0] = river_state.players[0].hole_cards[0]
        env.hole_indices[env_idx, 0, 1] = river_state.players[0].hole_cards[1]
        env.hole_indices[env_idx, 1, 0] = river_state.players[1].hole_cards[0]
        env.hole_indices[env_idx, 1, 1] = river_state.players[1].hole_cards[1]

        # Set board cards
        for i, card in enumerate(river_state.board):
            env.board_indices[env_idx, i] = card
            # Set one-hot encoding for board
            rank = card // 4
            suit = card % 4
            env.board_onehot[env_idx, i, suit, rank] = True

        # Set one-hot encoding for hole cards
        for player in range(2):
            for card_idx, card in enumerate(river_state.players[player].hole_cards):
                rank = card // 4
                suit = card % 4
                env.hole_onehot[env_idx, player, card_idx, suit, rank] = True

        # Set street to river (street 3)
        env.street[env_idx] = 3

        # Set stacks and committed amounts
        env.stacks[env_idx, 0] = river_state.players[0].stack
        env.stacks[env_idx, 1] = river_state.players[1].stack
        env.committed[env_idx, 0] = river_state.players[0].committed
        env.committed[env_idx, 1] = river_state.players[1].committed

        # Set other state
        env.button[env_idx] = river_state.button
        env.pot[env_idx] = river_state.pot
        env.to_act[env_idx] = river_state.to_act
        env.min_raise[env_idx] = river_state.min_raise
        env.actions_this_round[env_idx] = 0

        print(f"\nStarting River CFR Tree Construction...")
        print(f"Pot size: {river_state.pot}")
        print(f"Board: {river_state.board}")
        print(f"SB hole cards: {river_state.players[0].hole_cards}")
        print(f"BB hole cards: {river_state.players[1].hole_cards}")
        print(f"To act: {river_state.to_act}")
    else:
        print("\nStarting Preflop CFR Tree Construction...")

    # Create evaluator
    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=config.env.bet_bins,
        max_depth=config.search.depth,
        cfr_iterations=config.search.iterations,
        device=device,
        float_dtype=torch.float32,
        warm_start_iterations=config.search.warm_start_iterations,
        cfr_type=cfr_type,
        cfr_avg=config.search.cfr_avg,
        dcfr_delay=dcfr_delay,
    )

    # Initialize search
    roots = torch.tensor([0], device=device)
    evaluator.initialize_search(env, roots)

    # We need to manually run CFR iterations and capture data at specific iterations
    # The evaluator's self_play_iteration() runs all iterations internally
    # So we need to hook into it to extract data at specific iterations

    # We'll modify the evaluator to capture intermediate state
    # Or run iterations manually by calling the internal methods

    # Initialize subgame once
    evaluator.construct_subgame()
    evaluator.initialize_policy_and_beliefs()

    # Run warm start
    print("\nRunning warm start iterations...")
    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.warm_start()

    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values

    # Now run CFR iterations 16-25 and capture state
    print(
        f"\nRunning CFR iterations {config.search.warm_start_iterations}-{config.search.iterations}..."
    )

    evaluator.sample_count = 0

    # Run iterations 15-25, but only print 16-25
    for t in range(config.search.warm_start_iterations, config.search.iterations):
        evaluator.cfr_iteration(t, training_mode=False)

        print_single_iteration_data(
            evaluator=evaluator,
            iteration=t,
            bet_bins=config.env.bet_bins,
            num_actions=evaluator.num_actions,
        )

        # Compute and display exploitability after updating averages
        exploit_stats = evaluator._compute_exploitability()
        if exploit_stats.local_exploitability.numel() > 0:
            total_expl = exploit_stats.local_exploitability.mean().item()
            imp_p0 = exploit_stats.local_br_improvement[:, 0].mean().item()
            imp_p1 = exploit_stats.local_br_improvement[:, 1].mean().item()
            print(
                f"Exploitability (avg best-response improv): total={total_expl:.6f} | P0={imp_p0:.6f}, P1={imp_p1:.6f}"
            )

    print(f"\n{'='*80}")
    print("Debug Complete!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug ReBeL CFR depth-1 tree with regrets and policies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint file (optional, uses random model if not provided)",
    )
    parser.add_argument(
        "--cfr-type",
        type=str,
        default="discounted_plus",
        choices=["linear", "discounted", "discounted_plus"],
        help="CFR type to use (default: discounted_plus)",
    )
    parser.add_argument(
        "--river",
        action="store_true",
        help="Start at river with betting history (default: preflop)",
    )
    parser.add_argument(
        "--river-seed",
        type=int,
        default=None,
        help="Random seed for river state deck shuffling (default: 42)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of CFR iterations to run (overrides default)",
    )
    parser.add_argument(
        "--no-cfr-avg",
        action="store_true",
        help="Disable CFR-AVG (use current policy instead of average policy)",
    )
    parser.add_argument(
        "--dcfr-delay",
        type=int,
        default=None,
        help="DCFR delay parameter (default: 0 for linear, 70 for discounted)",
    )

    args = parser.parse_args()

    debug_cfr_depth1(
        checkpoint_path=args.checkpoint,
        cfr_type_str=args.cfr_type,
        river_mode=args.river,
        river_seed=args.river_seed,
        iterations=args.iterations,
        no_cfr_avg=args.no_cfr_avg,
        dcfr_delay=args.dcfr_delay,
    )


if __name__ == "__main__":
    main()
