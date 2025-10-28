#!/usr/bin/env python3
"""
Debugging script to show a depth-1 CFR tree preflop with regrets and policy probabilities.

This script creates a depth-1 CFR tree at preflop and displays:
- Each child node on one line
- The action taken to reach it
- Policy probability to take that action
- Regret for taking that action
- Current average policy probability
- Iterations 16-25 (post-warm-start)

Usage:
    python debug_cfr_depth1.py --cfr-type linear
    python debug_cfr_depth1.py --cfr-type discounted --checkpoint checkpoints-rebel/rebel_final.pt
"""

import argparse
import os

import torch

from alphaholdem.core.structured_config import CFRType, Config
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
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
) -> RebelFFN:
    """Load a model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Create model
    model = RebelFFN(
        input_dim=config.model.input_dim,
        num_actions=len(config.env.bet_bins) + 3,  # fold, call, bets, allin
        hidden_dim=config.model.hidden_dim,
        num_hidden_layers=config.model.num_hidden_layers,
        detach_value_head=config.model.detach_value_head,
        num_players=2,
    )

    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Falling back to random model initialization.")
        return create_random_model(config, device)

    # Handle different checkpoint formats
    try:
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Assume the checkpoint is just the model state dict
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print(f"Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("Falling back to random model initialization.")
        return create_random_model(config, device)


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


def create_default_config() -> Config:
    """Create a default configuration for debugging."""
    config = Config()
    config.num_envs = 1
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.seed = 42

    # Model config
    config.model = Config()
    config.model.name = "rebel_ffn"
    config.model.input_dim = 2661
    config.model.num_actions = -1
    config.model.hidden_dim = 1536
    config.model.num_hidden_layers = 6
    config.model.value_head_type = "scalar"
    config.model.value_head_num_quantiles = 1
    config.model.detach_value_head = True

    # Environment config
    config.env = Config()
    config.env.stack = 1000
    config.env.sb = 5
    config.env.bb = 10
    config.env.bet_bins = [0.5, 1.5]
    config.env.flop_showdown = False

    # Search config
    config.search = Config()
    config.search.enabled = True
    config.search.depth = 1  # Depth-1 tree for debugging
    config.search.iterations = 26  # Show up to iteration 25 (range goes to 26)
    config.search.warm_start_iterations = 15  # Warm start 15 iterations
    config.search.branching = 4
    config.search.belief_samples = 1
    config.search.dcfr_alpha = 1.5
    config.search.dcfr_beta = 0.0
    config.search.dcfr_gamma = 2.0
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
        f"{'Node':>6} | {'Action':>10} | {'Policy':>7} | {'PolicyAvg':>7} | {'Regret':>7} | {'Value':>7} | {'BeliefSum':>9}"
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
        policy_prob = evaluator.policy_probs[child_idx].mean().item()
        policy_avg_prob = evaluator.policy_probs_avg[child_idx].mean().item()
        regret = evaluator.cumulative_regrets[child_idx].mean().item()

        # Get value at this node - average over both players and all hands
        # values_avg has shape [M, 2, NUM_HANDS]
        value = evaluator.values_avg[child_idx, :, :].mean().item()

        # Get belief sum at this node - sum over both players and all hands
        belief_sum = evaluator.beliefs_avg[child_idx, :, :].sum().item()

        action_name = action_to_string(action_idx, bet_bins)

        print(
            f"{child_idx:>6} | {action_name:>10} | {policy_prob:7.4f} | {policy_avg_prob:7.4f} | "
            f"{regret:7.2f} | {value:7.4f} | {belief_sum:9.4f}"
        )


def debug_cfr_depth1(
    checkpoint_path: str | None = None,
    cfr_type_str: str = "linear",
) -> None:
    """Main debugging function."""
    print("=== ReBeL CFR Depth-1 Debugger ===")

    # Determine CFR type
    if cfr_type_str == "linear":
        cfr_type = CFRType.linear
        dcfr_delay = 0
    elif cfr_type_str == "discounted":
        cfr_type = CFRType.discounted
        dcfr_delay = 20
    else:
        raise ValueError(f"Unknown CFR type: {cfr_type_str}")

    print(f"CFR Type: {cfr_type_str}")
    if cfr_type == CFRType.discounted:
        print(f"DCFR Delay: {dcfr_delay}")

    # Create configuration
    config = create_default_config()
    config.search.cfr_type = cfr_type
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
        cfr_avg=True,
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
    evaluator.set_leaf_values()
    evaluator.compute_expected_values()
    evaluator.warm_start()

    # Now run CFR iterations 16-25 and capture state
    print(f"\nRunning CFR iterations 16-{config.search.iterations}...")

    # Run iterations 15-25, but only print 16-25
    for t in range(config.search.warm_start_iterations, config.search.iterations):
        # Skip iterations before 16 silently
        if t < 16:
            evaluator.set_leaf_values()
            evaluator.compute_expected_values()
            evaluator.values_avg[:] = evaluator.values

            regrets = evaluator.compute_instantaneous_regrets(evaluator.values)
            if evaluator.cfr_type == CFRType.linear:
                # Alternate updates
                regrets.masked_fill_(
                    evaluator.env.to_act[:, None] == t % evaluator.num_players, 0.0
                )
            elif evaluator.cfr_type == CFRType.discounted:
                factor = torch.where(
                    regrets > 0,
                    (t - 1) ** evaluator.dcfr_alpha,
                    (t - 1) ** evaluator.dcfr_beta,
                )
                evaluator.cumulative_regrets *= factor / (factor + 1)

            evaluator.cumulative_regrets += regrets
            evaluator.update_policy(t)

            evaluator.set_leaf_values()
            evaluator.compute_expected_values()

            if evaluator.cfr_type == CFRType.discounted:
                if t > evaluator.dcfr_delay:
                    evaluator.values_avg *= max(0, t - 1 - evaluator.dcfr_delay)
                    evaluator.values_avg += evaluator.values
                    evaluator.values_avg /= t + 1
                else:
                    evaluator.values_avg[:] = evaluator.values
            else:
                evaluator.values_avg *= t
                evaluator.values_avg += evaluator.values
                evaluator.values_avg /= t + 1
            continue

        # For iterations 16-25, print the data
        # Run the iteration and capture state
        evaluator.set_leaf_values()
        evaluator.compute_expected_values()
        evaluator.values_avg[:] = evaluator.values

        regrets = evaluator.compute_instantaneous_regrets(evaluator.values)
        if evaluator.cfr_type == CFRType.linear:
            # Alternate updates
            regrets.masked_fill_(
                evaluator.env.to_act[:, None] == t % evaluator.num_players, 0.0
            )
        elif evaluator.cfr_type == CFRType.discounted:
            factor = torch.where(
                regrets > 0,
                (t - 1) ** evaluator.dcfr_alpha,
                (t - 1) ** evaluator.dcfr_beta,
            )
            evaluator.cumulative_regrets *= factor / (factor + 1)

        evaluator.cumulative_regrets += regrets

        # Print AFTER regret accumulation but BEFORE policy update
        print_single_iteration_data(
            evaluator=evaluator,
            iteration=t,
            bet_bins=config.env.bet_bins,
            num_actions=evaluator.num_actions,
        )

        evaluator.update_policy(t)

        evaluator.set_leaf_values()
        evaluator.compute_expected_values()

        if evaluator.cfr_type == CFRType.discounted:
            if t > evaluator.dcfr_delay:
                evaluator.values_avg *= max(0, t - 1 - evaluator.dcfr_delay)
                evaluator.values_avg += evaluator.values
                evaluator.values_avg /= t + 1
            else:
                evaluator.values_avg[:] = evaluator.values
        else:
            evaluator.values_avg *= t
            evaluator.values_avg += evaluator.values
            evaluator.values_avg /= t + 1

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
        default="linear",
        choices=["linear", "discounted"],
        help="CFR type to use (default: linear)",
    )

    args = parser.parse_args()

    debug_cfr_depth1(checkpoint_path=args.checkpoint, cfr_type_str=args.cfr_type)


if __name__ == "__main__":
    main()
