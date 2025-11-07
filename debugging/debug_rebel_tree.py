#!/usr/bin/env python3
"""
Debug script for visualizing ReBeL CFR tree structure.

This script loads a trained model (optional) and creates a CFR tree using RebelCFREvaluator,
showing all valid nodes with their computed values, policies, and other relevant information.
"""

import argparse
import os
from typing import Dict, List, Optional

import torch

from alphaholdem.core.structured_config import CFRType, Config
from alphaholdem.env.card_utils import hand_combos_tensor
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp import RebelFFN
from alphaholdem.search.rebel_cfr_evaluator import NUM_HANDS, RebelCFREvaluator


def card_to_string(card_idx: int) -> str:
    """Convert card index to string representation."""
    rank = card_idx // 4
    suit = card_idx % 4
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["♠", "♥", "♦", "♣"]
    return f"{ranks[rank]}{suits[suit]}"


def hand_to_string(hand_idx: int) -> str:
    """Convert hand index to string representation."""
    combos = hand_combos_tensor(device=torch.device("cpu"))
    c1, c2 = combos[hand_idx]
    return f"{card_to_string(c1)}{card_to_string(c2)}"


def action_to_string(action_idx: int, bet_bins: List[float]) -> str:
    """Convert action index to string representation."""
    if action_idx == 0:
        return "FOLD"
    elif action_idx == 1:
        return "CALL"
    elif action_idx == len(bet_bins) + 2:
        return "ALLIN"
    elif action_idx >= 2 and action_idx < len(bet_bins) + 2:
        bet_size = bet_bins[action_idx - 2]
        return f"B{bet_size:.1f}x"
    else:
        return f"ACTION_{action_idx}"


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
    config.search.depth = 3  # Shallow for debugging
    config.search.iterations = 150
    config.search.warm_start_iterations = 15
    config.search.branching = 4
    config.search.belief_samples = 1
    config.search.dcfr_alpha = 1.5
    config.search.dcfr_beta = 0.0
    config.search.dcfr_gamma = 2.0
    config.search.include_average_policy = True
    config.search.cfr_type = CFRType.linear
    config.search.cfr_avg = True

    return config


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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
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
    cpu_rng.manual_seed(config.seed)
    model.init_weights(cpu_rng)
    model.to(device)
    model.eval()

    print("Random model created successfully")
    return model


def analyze_tree_node(
    evaluator: RebelCFREvaluator,
    node_idx: int,
    depth: int,
    action_path: List[str],
    bet_bins: List[float],
    parent_action: Optional[int] = None,
) -> Dict:
    """Analyze a single tree node and return relevant information."""
    env = evaluator.env

    # Basic node information
    node_info = {
        "node_idx": node_idx,
        "depth": depth,
        "action_path": " -> ".join(action_path) if action_path else "ROOT",
        "to_act": env.to_act[node_idx].item(),
        "street": env.street[node_idx].item(),
        "pot": env.pot[node_idx].item(),
        "stacks": env.stacks[node_idx].tolist(),
        "committed": env.committed[node_idx].tolist(),
        "done": env.done[node_idx].item(),
        "valid": evaluator.valid_mask[node_idx].item(),
        "leaf": evaluator.leaf_mask[node_idx].item(),
    }

    # Action taken to reach this node (inferred from node structure)
    if depth == 0:
        node_info["action_taken"] = "ROOT"
        node_info["action_prob"] = None
    else:
        # For non-root nodes, we can infer the action from the node index
        # The tree structure is: offset_next + (current_idx - offset) * B + action
        # So action = (node_idx - offset_next) % B
        offset_next = evaluator.depth_offsets[depth]
        B = len(bet_bins) + 3  # num_actions

        if node_idx >= offset_next:
            action_idx = (node_idx - offset_next) % B
            node_info["action_taken"] = action_to_string(action_idx, bet_bins)

            # Get the probability of taking this action from the current node's policy
            # policy_probs[node_idx] has shape [NUM_HANDS] - probability of taking action to reach this node for each hand
            if (
                node_idx < len(evaluator.policy_probs_avg)
                and evaluator.valid_mask[node_idx]
            ):
                current_policy = evaluator.policy_probs_avg[node_idx]
                # Sum across all hands to get overall probability of reaching this node
                node_info["action_prob"] = current_policy.mean().item()
        else:
            node_info["action_taken"] = "UNKNOWN"
            node_info["action_prob"] = None

    # Board cards
    board = env.board_indices[node_idx]
    board_cards = [card_to_string(card.item()) for card in board if card.item() >= 0]
    node_info["board"] = " ".join(board_cards) if board_cards else "PREFLOP"

    # Legal actions
    if evaluator.legal_mask is not None:
        legal_actions = evaluator.legal_mask[node_idx]
        legal_action_names = [
            action_to_string(i, bet_bins)
            for i in range(len(legal_actions))
            if legal_actions[i]
        ]
        node_info["legal_actions"] = legal_action_names

        # Show action probabilities for legal actions only
        if evaluator.valid_mask[node_idx] and not evaluator.leaf_mask[node_idx]:
            policy_probs = evaluator.policy_probs[node_idx]
            policy_avg = evaluator.policy_probs_avg[node_idx]

            # Get probabilities for legal actions
            legal_probs = [
                (action_to_string(i, bet_bins), policy_probs[i].item())
                for i in range(len(legal_actions))
                if legal_actions[i]
            ]
            legal_probs_avg = [
                (action_to_string(i, bet_bins), policy_avg[i].item())
                for i in range(len(legal_actions))
                if legal_actions[i]
            ]

            # Sort by probability
            legal_probs.sort(key=lambda x: x[1], reverse=True)
            legal_probs_avg.sort(key=lambda x: x[1], reverse=True)

            node_info["legal_action_probs"] = legal_probs[:3]  # Top 3
            node_info["legal_action_probs_avg"] = legal_probs_avg[:3]  # Top 3

    # Policy information
    if evaluator.valid_mask[node_idx] and not evaluator.leaf_mask[node_idx]:
        policy_probs = evaluator.policy_probs[node_idx]
        policy_avg = evaluator.policy_probs_avg[node_idx]

        # Get top 3 actions by probability
        top_actions = torch.topk(policy_probs, min(3, len(policy_probs)))
        top_actions_avg = torch.topk(policy_avg, min(3, len(policy_avg)))

        node_info["top_policy"] = [
            (action_to_string(idx.item(), bet_bins), prob.item())
            for idx, prob in zip(top_actions.indices, top_actions.values)
        ]
        node_info["top_policy_avg"] = [
            (action_to_string(idx.item(), bet_bins), prob.item())
            for idx, prob in zip(top_actions_avg.indices, top_actions_avg.values)
        ]

    # Value information
    if evaluator.valid_mask[node_idx]:
        values = evaluator.values_avg[node_idx]

        # Get mean value over beliefs
        beliefs = evaluator.beliefs_avg[node_idx]
        mean_value = (values * beliefs).sum().item()

        node_info["mean_value"] = mean_value

        # Get value range
        valid_hands = beliefs.sum(dim=-1) > 1e-6
        if valid_hands.any():
            valid_values = values[valid_hands]
            node_info["value_range"] = [
                valid_values.min().item(),
                valid_values.max().item(),
            ]

        # Belief information
        belief_sum = beliefs.sum().item()
        node_info["belief_sum"] = belief_sum

        # Top hands by belief
        top_hands = torch.topk(beliefs.sum(dim=0), min(5, NUM_HANDS))
        node_info["top_hands"] = [
            (hand_to_string(idx.item()), prob.item())
            for idx, prob in zip(top_hands.indices, top_hands.values)
        ]

    # Regret information
    if evaluator.valid_mask[node_idx] and not evaluator.leaf_mask[node_idx]:
        regrets = evaluator.cumulative_regrets[node_idx]
        max_regret = regrets.max().item()
        node_info["max_regret"] = max_regret

        # Top regrets
        top_regrets = torch.topk(regrets, min(3, len(regrets)))
        node_info["top_regrets"] = [
            (action_to_string(idx.item(), bet_bins), regret.item())
            for idx, regret in zip(top_regrets.indices, top_regrets.values)
        ]

    return node_info


def print_node_info(node_info: Dict, indent: int = 0) -> None:
    """Print formatted node information concisely."""
    prefix = "  " * indent
    postfix = ""  # "  " * max(0, 3 - indent)

    # Basic node info
    action_taken = node_info.get("action_taken", "ROOT")
    player = node_info["to_act"]
    street = node_info["street"]
    board = node_info["board"]
    pot = node_info["pot"]
    stacks = node_info["stacks"]
    valid = node_info["valid"]
    leaf = node_info["leaf"]

    # Build the main line with action probability
    action_prob = node_info.get("action_prob")
    if action_prob is not None:
        prob_str = f"{action_prob*100:3.0f}%"
    else:
        prob_str = "   "

    main_info = (
        f"{prefix}{node_info['node_idx']:<4}{postfix} | {prob_str} | {action_taken:>5}"
    )

    # Add value right after action if available
    if "mean_value" in node_info and valid:
        main_info += f" {1000 * node_info['mean_value']:5.0f} "
        if "value_range" in node_info:
            main_info += f"[{1000 * node_info['value_range'][0]:5.0f},{1000 * node_info['value_range'][1]:5.0f}]"

    main_info += f" | P{player} | {board} | Pot:{pot:.0f} | Stacks:{stacks[0]:.0f},{stacks[1]:.0f}"

    # Add status flags
    status_flags = []
    if not valid:
        status_flags.append("INVALID")
    if leaf:
        status_flags.append("LEAF")
    if node_info["done"]:
        status_flags.append("DONE")

    if status_flags:
        main_info += f" | {'/'.join(status_flags)}"

    print(main_info)


def debug_rebel_tree(
    checkpoint_path: Optional[str] = None,
    max_depth: int = 3,
    delta_beliefs: bool = False,
    filter_zero_reach: bool = True,
) -> None:
    """Main debugging function."""
    print("=== ReBeL CFR Tree Debugger ===")

    # Create configuration
    config = create_default_config()
    config.search.depth = max_depth
    device = torch.device(config.device)

    print(f"Using device: {device}")
    print(f"Tree depth: {max_depth}")

    # Load or create model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_model_from_checkpoint(checkpoint_path, config, device)
    else:
        model = create_random_model(config, device)

    # Create environment
    env = HUNLTensorEnv(
        num_envs=config.num_envs,
        starting_stack=config.env.stack,
        sb=config.env.sb,
        bb=config.env.bb,
        default_bet_bins=config.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=config.env.flop_showdown,
    )
    env.reset()

    # Create three specific game states
    print("Setting up three specific game states...")

    # State 1: Preflop (root)
    env1 = HUNLTensorEnv(
        num_envs=1,
        starting_stack=config.env.stack,
        sb=config.env.sb,
        bb=config.env.bb,
        default_bet_bins=config.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=config.env.flop_showdown,
    )
    env1.reset()

    # State 2: Flop after bet0.5-call
    env2 = HUNLTensorEnv(
        num_envs=1,
        starting_stack=config.env.stack,
        sb=config.env.sb,
        bb=config.env.bb,
        default_bet_bins=config.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=config.env.flop_showdown,
    )
    env2.reset()
    # Simulate bet0.5-call sequence
    # Player 0 bets 0.5x pot
    bet_action = 2  # BET0.5x is action index 2
    call_action = 1  # CALL is action index 1
    env2.step_bins(torch.tensor([bet_action], device=device))
    env2.step_bins(torch.tensor([call_action], device=device))

    # State 3: Turn after bet0.5-call-check-bet0.5-call
    env3 = HUNLTensorEnv(
        num_envs=1,
        starting_stack=config.env.stack,
        sb=config.env.sb,
        bb=config.env.bb,
        default_bet_bins=config.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=config.env.flop_showdown,
    )
    env3.reset()
    # Simulate bet0.5-call-check-bet0.5-call sequence
    env3.step_bins(torch.tensor([bet_action], device=device))  # bet0.5
    env3.step_bins(torch.tensor([call_action], device=device))  # call
    env3.step_bins(
        torch.tensor([call_action], device=device)
    )  # check (call with 0 to call)
    env3.step_bins(torch.tensor([bet_action], device=device))  # bet0.5
    env3.step_bins(torch.tensor([call_action], device=device))  # call

    # State 4: River after bet0.5-call-check-bet0.5-call-check-bet0.5-call
    env4 = HUNLTensorEnv(
        num_envs=1,
        starting_stack=config.env.stack,
        sb=config.env.sb,
        bb=config.env.bb,
        default_bet_bins=config.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=config.env.flop_showdown,
    )
    env4.reset()
    # Simulate bet0.5-call-check-bet0.5-call-check-bet0.5-call sequence
    env4.step_bins(torch.tensor([bet_action], device=device))  # bet0.5
    env4.step_bins(torch.tensor([call_action], device=device))  # call
    env4.step_bins(torch.tensor([call_action], device=device))  # check
    env4.step_bins(torch.tensor([bet_action], device=device))  # bet0.5
    env4.step_bins(torch.tensor([call_action], device=device))  # call
    env4.step_bins(torch.tensor([call_action], device=device))  # check
    env4.step_bins(torch.tensor([bet_action], device=device))  # bet0.5
    env4.step_bins(torch.tensor([call_action], device=device))  # call

    # environments = [env1, env2, env3, env4]
    environments = [env4]
    state_names = [
        # "Preflop",
        # "Flop (bet0.5-call)",
        # "Turn (bet0.5-call-check-bet0.5-call)",
        "River (bet0.5-call-check-bet0.5-call-check-bet0.5-call)",
    ]

    # Create delta beliefs if requested
    initial_beliefs = None
    if delta_beliefs:
        print("Using delta beliefs: AA vs KK")
        from alphaholdem.env.card_utils import combo_index

        # Define card function locally
        def card(rank: int, suit: int) -> int:
            return rank * 4 + suit

        # Create delta beliefs: Player 0 has AA, Player 1 has KK
        aa_hand = combo_index(card(12, 0), card(12, 1))  # A♠A♥
        kk_hand = combo_index(card(11, 2), card(11, 3))  # K♦K♣

        initial_beliefs = torch.zeros(
            1, 2, NUM_HANDS, device=device, dtype=torch.float32
        )
        initial_beliefs[0, 0, aa_hand] = 1.0  # Player 0 has AA
        initial_beliefs[0, 1, kk_hand] = 1.0  # Player 1 has KK

        print(f"Player 0 hand: {hand_to_string(aa_hand)}")
        print(f"Player 1 hand: {hand_to_string(kk_hand)}")

    # Process each of the four game states
    for i, (env_state, state_name) in enumerate(zip(environments, state_names)):
        print(f"\n=== {state_name} ===")

        # Create evaluator for this state
        evaluator = RebelCFREvaluator(
            search_batch_size=1,
            env_proto=env_state,
            model=model,
            bet_bins=config.env.bet_bins,
            max_depth=config.search.depth,
            cfr_iterations=config.search.iterations,
            device=device,
            float_dtype=torch.float32,
            warm_start_iterations=config.search.warm_start_iterations,
            cfr_type=config.search.cfr_type,
            cfr_avg=config.search.cfr_avg,
        )

        # Initialize search
        roots = torch.tensor([0], device=device)
        evaluator.initialize_search(env_state, roots, initial_beliefs=initial_beliefs)

        # Run CFR iterations
        print(f"Running {config.search.iterations} CFR iterations...")
        evaluator.evaluate_cfr()
        print("CFR iterations completed.")

        # Get all valid nodes (optionally filter by reach)
        valid_nodes = []
        for depth in range(config.search.depth + 1):
            offset = evaluator.depth_offsets[depth]
            offset_next = evaluator.depth_offsets[depth + 1]

            for node_idx in range(offset, offset_next):
                if evaluator.valid_mask[node_idx]:
                    # Check if we should filter by reach
                    if filter_zero_reach:
                        # Check if this node has non-zero reach probability
                        # reach_weights has shape [M, 2, NUM_HANDS], sum across players and hands
                        reach_weight = evaluator.reach_weights[node_idx].sum().item()
                        if reach_weight < 1e-6:  # Essentially zero reach
                            continue
                    valid_nodes.append((node_idx, depth))

        print(f"Total valid nodes: {len(valid_nodes)}")
        if filter_zero_reach:
            print(f"Filtered out nodes with zero reach (threshold: 1e-6)")

        # Analyze each node in depth-first order
        print("\nNode Details:")

        def get_children(node_idx: int, depth: int) -> List[int]:
            """Get children of a node in depth-first order."""
            if depth >= config.search.depth:
                return []

            children = []
            offset_next = evaluator.depth_offsets[depth + 1]
            offset_next_next = evaluator.depth_offsets[depth + 2]
            B = len(config.env.bet_bins) + 3  # num_actions

            # Calculate the range of child nodes
            start_child = offset_next + (node_idx - evaluator.depth_offsets[depth]) * B
            end_child = min(start_child + B, offset_next_next)

            for child_idx in range(start_child, end_child):
                if (
                    child_idx < len(evaluator.valid_mask)
                    and evaluator.valid_mask[child_idx]
                ):
                    children.append(child_idx)

            return children

        def print_node_recursive(node_idx: int, depth: int) -> None:
            """Print node and its children recursively."""
            # Determine action path (simplified)
            action_path = []
            if depth > 0:
                action_path = [f"Action_{depth}"]

            node_info = analyze_tree_node(
                evaluator, node_idx, depth, action_path, config.env.bet_bins
            )
            print_node_info(node_info, indent=depth)

            # Print children recursively
            children = get_children(node_idx, depth)
            for child_idx in children:
                print_node_recursive(child_idx, depth + 1)

        # Start from root node
        root_nodes = [idx for idx, depth in valid_nodes if depth == 0]
        for root_idx in root_nodes:
            print_node_recursive(root_idx, 0)

        # Summary statistics for this state
        valid_mask = evaluator.valid_mask
        leaf_mask = evaluator.leaf_mask

        print(
            f"\nSummary: {valid_mask.sum().item()} valid nodes, {leaf_mask.sum().item()} leaf nodes"
        )

        # Value statistics
        if valid_mask.any():
            valid_values = evaluator.values_avg[valid_mask]
            print(
                f"Value range: [{valid_values.min().item():.4f}, {valid_values.max().item():.4f}]"
            )

    print("\n=== Debug Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Debug ReBeL CFR tree structure")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint file (optional)"
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Maximum tree depth (default: 3)"
    )
    parser.add_argument(
        "--delta-beliefs",
        action="store_true",
        help="Use delta beliefs (AA vs KK) instead of uniform beliefs",
    )
    parser.add_argument(
        "--show-zero-reach",
        action="store_true",
        help="Show nodes with zero reach (default: filter them out)",
    )

    args = parser.parse_args()

    debug_rebel_tree(
        args.checkpoint, args.depth, args.delta_beliefs, not args.show_zero_reach
    )


if __name__ == "__main__":
    main()
