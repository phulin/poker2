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
    python debug_cfr_iterations.py --cfr-type linear
    python debug_cfr_iterations.py --cfr-type discounted --checkpoint checkpoints-rebel/rebel_final.pt

    # Different streets
    python debug_cfr_iterations.py --street=flop
    python debug_cfr_iterations.py --street=turn
    python debug_cfr_iterations.py --street=river
    python debug_cfr_iterations.py --street=river --checkpoint rebel_step_1250.pt
    python debug_cfr_iterations.py --street=river_plus

    # Force a particular board (3 cards for flop, 4 for turn, 5 for river/river_plus)
    python debug_cfr_iterations.py --street=flop --board=AsKhQd
    python debug_cfr_iterations.py --street=turn --board=AsKhQd2c
    python debug_cfr_iterations.py --street=river --board=AsKhQd2c3h
    python debug_cfr_iterations.py --street=river_plus --board=AsKhQd2c3h

    # Use sparse CFR evaluator instead of ReBeL CFR evaluator
    python debug_cfr_iterations.py --sparse=true
    python debug_cfr_iterations.py --sparse=true --street=flop --checkpoint checkpoints-rebel/rebel_final.pt
"""

import os
import random
from dataclasses import dataclass
from typing import Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from alphaholdem.core.structured_config import Config, ModelType
from alphaholdem.env import card_utils
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
from alphaholdem.search.rebel_cfr_evaluator_old import (
    RebelCFREvaluator as RebelCFREvaluatorOld,
)
from alphaholdem.search.sparse_cfr_evaluator import SparseCFREvaluator


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
        return f"B{bet_size:.1f}x"
    else:
        return f"ACTION_{action_idx}"


# Fixed width for the Node column when printing depth-first (includes indentation)
NODE_COL_WIDTH = 20


def _resolve_sparse_action_idx(
    evaluator: RebelCFREvaluator | RebelCFREvaluatorOld | SparseCFREvaluator,
    node_idx: int,
    fallback: int,
) -> int:
    if isinstance(evaluator, SparseCFREvaluator):
        action_tensor = evaluator.action_from_parent
        if node_idx < action_tensor.shape[0]:
            action_idx = int(action_tensor[node_idx].item())
            if action_idx >= 0:
                return action_idx
    return fallback


def _card_index_to_str(card_idx: int) -> str:
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["♠", "♥", "♦", "♣"]
    suit = card_idx // 13
    rank = card_idx % 13
    return f"{ranks[rank]}{suits[suit]}"


def _parse_hole_cards_str(hole: str) -> tuple[int, int]:
    """Parse hole string like 'AsTc' into two 0-51 card indices."""
    s = hole.strip()
    if len(s) != 4:
        raise ValueError("Hole cards must be 4 chars like AsTc")
    ranks = {
        c: i
        for i, c in enumerate(
            ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        )
    }
    suits = {"s": 0, "h": 1, "d": 2, "c": 3}
    r1, su1, r2, su2 = s[0], s[1].lower(), s[2], s[3].lower()
    if r1 not in ranks or r2 not in ranks or su1 not in suits or su2 not in suits:
        raise ValueError("Invalid hole card string; use format AsTc with suits shdc")
    c1 = suits[su1] * 13 + ranks[r1]
    c2 = suits[su2] * 13 + ranks[r2]
    return c1, c2


def _parse_board_str(board: str) -> list[int]:
    """Parse board string like 'AsKhQd' (flop), 'AsKhQd2c' (turn), or 'AsKhQd2c3h' (river) into card indices."""
    s = board.strip()
    if len(s) % 2 != 0:
        raise ValueError("Board string must have even length (each card is 2 chars)")
    if len(s) < 6 or len(s) > 10:
        raise ValueError("Board must have 3-5 cards (6-10 chars)")

    ranks = {
        c: i
        for i, c in enumerate(
            ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        )
    }
    suits = {"s": 0, "h": 1, "d": 2, "c": 3}

    card_indices = []
    for i in range(0, len(s), 2):
        rank_char = s[i]
        suit_char = s[i + 1].lower()
        if rank_char not in ranks or suit_char not in suits:
            raise ValueError(f"Invalid card at position {i}: {rank_char}{suit_char}")
        card_idx = suits[suit_char] * 13 + ranks[rank_char]
        card_indices.append(card_idx)

    return card_indices


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


def advance_to_street_with_history(
    env: HUNLTensorEnv,
    street: str,
    button: int = 1,
    forced_board: Optional[list[int]] = None,
) -> None:
    """Reset env and step through a fixed action history to the specified street.

    History pattern:
    - Preflop: SB raise 1.5x pot, BB call
    - Flop: SB checks, BB checks
    - Turn: SB checks, BB bets 1.5x pot, SB calls
    - River: SB checks → BB to act
    - River_plus: River + call + bet 0.5x + bet 0.5x → next player to act

    Args:
        env: Environment to mutate in-place (expects N=1)
        street: Target street ("preflop", "flop", "turn", "river", "river_plus")
        button: 0=SB is button, 1=BB is button. Use 1 so SB acts first postflop.
        forced_board: Optional list of card indices to force as board cards.
                      Must match street: 3 cards for flop, 4 for turn, 5 for river/river_plus.
    """
    assert env.N == 1, "This helper expects a single-env instance (N=1)"

    if street == "preflop":
        env.reset(force_button=torch.tensor([button], device=env.device))
        if forced_board is not None:
            raise ValueError("Cannot force board in preflop")
        return

    # Reset with forced button
    env.reset(force_button=torch.tensor([button], device=env.device))

    bet_bin_index = min(3, len(env.default_bet_bins) + 2)
    call_action = torch.ones(env.N, dtype=torch.long, device=env.device)

    # Preflop: SB raise 1.5x pot, BB call
    env.step_bins(
        torch.full((env.N,), bet_bin_index, dtype=torch.long, device=env.device)
    )
    env.step_bins(call_action)

    if street == "flop":
        if forced_board is not None:
            if len(forced_board) != 3:
                raise ValueError(
                    f"Flop requires 3 board cards, got {len(forced_board)}"
                )
            _force_board_cards(env, forced_board, street)
        return

    # Flop: check, check
    env.step_bins(call_action)
    env.step_bins(call_action)

    if street == "turn":
        if forced_board is not None:
            if len(forced_board) != 4:
                raise ValueError(
                    f"Turn requires 4 board cards, got {len(forced_board)}"
                )
            _force_board_cards(env, forced_board, street)
        return

    # Turn: SB check
    env.step_bins(call_action)

    # Turn: BB bet 1.5x pot
    env.step_bins(
        torch.full((env.N,), bet_bin_index, dtype=torch.long, device=env.device)
    )

    # Turn: SB call
    env.step_bins(call_action)

    if street == "river":
        if forced_board is not None:
            if len(forced_board) != 5:
                raise ValueError(
                    f"River requires 5 board cards, got {len(forced_board)}"
                )
            _force_board_cards(env, forced_board, street)
        return

    # River_plus: advance to river first, then execute call-bet0.5-bet0.5
    if street == "river_plus":
        if forced_board is not None:
            if len(forced_board) != 5:
                raise ValueError(
                    f"River_plus requires 5 board cards, got {len(forced_board)}"
                )
            _force_board_cards(env, forced_board, "river")
        # River: SB check (call action when no bet to call)
        env.step_bins(call_action)
        # Find bet bin index for 0.5x pot
        bet_bins = env.default_bet_bins
        if 0.5 not in bet_bins:
            raise ValueError(
                f"Bet bin 0.5x not found in bet_bins: {bet_bins}. Required for river_plus."
            )
        bet_05_index = bet_bins.index(0.5) + 2  # Bet bins start at index 2
        # River: BB bet 0.5x pot
        env.step_bins(
            torch.full((env.N,), bet_05_index, dtype=torch.long, device=env.device)
        )
        # River: SB bet 0.5x pot (raise)
        env.step_bins(
            torch.full((env.N,), bet_05_index, dtype=torch.long, device=env.device)
        )
        return

    raise ValueError(
        f"Unknown street: {street}. Must be one of: preflop, flop, turn, river, river_plus"
    )


def _force_board_cards(
    env: HUNLTensorEnv, board_card_indices: list[int], street: str
) -> None:
    """Force board cards in the environment after advancing to a street.

    Args:
        env: Environment to mutate (expects N=1)
        board_card_indices: List of card indices (0-51) to set as board cards
        street: Current street ("flop", "turn", "river")
    """
    assert env.N == 1, "This helper expects a single-env instance (N=1)"

    env_idx = 0
    num_cards = len(board_card_indices)

    # Set board_indices
    for i in range(num_cards):
        env.board_indices[env_idx, i] = board_card_indices[i]

    # Set board_onehot using the cached one-hot encodings
    for i in range(num_cards):
        card_idx = board_card_indices[i]
        env.board_onehot[env_idx, i] = env.card_onehot_cache[card_idx]


def print_single_iteration_data(
    evaluator: RebelCFREvaluator | RebelCFREvaluatorOld | SparseCFREvaluator,
    iteration: int,
    bet_bins: list[float],
    selected_hand_idx: Optional[int] = None,
) -> None:
    """Print data for a single CFR iteration."""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration + 1}")
    print(f"{'='*80}")

    # Get depth 1 nodes
    root_idx = 0
    depth1_offset = evaluator.depth_offsets[1]
    depth1_end = evaluator.depth_offsets[2]
    actor = evaluator.env.to_act[root_idx].item()

    print(f"\nChild Nodes (depth 1):")
    print(
        f"{'Node':>6} | {'Action':>10} | {'Policy':>7} | {'PolicyAvg':>7} | {'Regret [min, max]':>24} | {'Value':>7} | {'ValLatest':>9} | {'ValAvgPol':>9}"
    )
    print("-" * 100)

    for child_idx in range(depth1_offset, depth1_end):
        if hasattr(evaluator, "valid_mask") and not evaluator.valid_mask[child_idx]:
            continue

        # Determine which action was taken to reach this node
        # For depth 1: child_idx = depth1_offset + action
        action_idx = _resolve_sparse_action_idx(
            evaluator, child_idx, child_idx - depth1_offset
        )

        # Policy probs and regrets are stored on the child node itself
        # So we look at policy_probs[child_idx] and cumulative_regrets[child_idx]
        policy_probs = evaluator.policy_probs[child_idx].clone()
        policy_probs_avg = evaluator.policy_probs_avg[child_idx].clone()

        if selected_hand_idx is not None:
            # Specific hand stats only
            policy_prob = float(policy_probs[selected_hand_idx].item())
            policy_avg_prob = float(policy_probs_avg[selected_hand_idx].item())
            regret_val = float(
                evaluator.cumulative_regrets[child_idx][selected_hand_idx].item()
            )
            regret = regret_val
            regret_min = regret_val
            regret_max = regret_val
            value = (
                evaluator.values_avg[child_idx, actor, selected_hand_idx].mean().item()
            )
            value_latest = (
                evaluator.latest_values[child_idx, actor, selected_hand_idx]
                .mean()
                .item()
            )
            value_avg_policy = value
        else:
            # Weighting: use current beliefs for current policy, avg beliefs for avg policy
            # Use beliefs from parent (root) so all actions share the same weighting
            beliefs_current = evaluator.beliefs[root_idx, :, :]  # [2, NUM_HANDS]
            beliefs_avg = evaluator.beliefs_avg[root_idx, :, :]  # [2, NUM_HANDS]
            weights_current = beliefs_current.sum(dim=0)  # [NUM_HANDS]
            weights_avg = beliefs_avg.sum(dim=0)  # [NUM_HANDS]

            parent_allowed = evaluator.allowed_hands[root_idx]
            child_allowed = evaluator.allowed_hands[child_idx]
            allowed_hands = parent_allowed & child_allowed

            policy_probs = policy_probs.masked_fill(~allowed_hands, 0.0)
            policy_probs_avg = policy_probs_avg.masked_fill(~allowed_hands, 0.0)
            weights_current = weights_current.masked_fill(~allowed_hands, 0.0)
            weights_avg = weights_avg.masked_fill(~allowed_hands, 0.0)

            weights_current_sum = float(weights_current.sum().item())
            weights_avg_sum = float(weights_avg.sum().item())

            policy_prob = (
                (policy_probs * weights_current).sum().item() / weights_current_sum
                if weights_current_sum > 0.0
                else 0.0
            )
            policy_avg_prob = (
                (policy_probs_avg * weights_avg).sum().item() / weights_avg_sum
                if weights_avg_sum > 0.0
                else 0.0
            )
            # Note: cumulative_regrets can be negative; policy uses clamped version (regret matching)
            regret = evaluator.cumulative_regrets[child_idx].mean().item()
            regret_max = evaluator.cumulative_regrets[child_idx].max().item()
            regret_min = evaluator.cumulative_regrets[child_idx].min().item()

            # Get value at this node - average over both players and all hands
            # values_avg has shape [M, 2, NUM_HANDS]
            value = evaluator.values_avg[child_idx, actor, :].mean().item()
            value_latest = evaluator.latest_values[child_idx, actor, :].mean().item()

            # Root-actor value weighted by average policy
            actor_beliefs_avg = evaluator.beliefs_avg[root_idx, actor, :].clone()
            actor_beliefs_avg = actor_beliefs_avg.masked_fill(~allowed_hands, 0.0)
            policy_probs_avg_masked = evaluator.policy_probs_avg[child_idx].masked_fill(
                ~allowed_hands, 0.0
            )
            policy_weight = actor_beliefs_avg * policy_probs_avg_masked
            weight_sum = float(policy_weight.sum().item())
            if weight_sum > 0.0:
                value_avg_policy = (
                    evaluator.values_avg[child_idx, actor, :] * policy_weight
                ).sum().item() / weight_sum
            else:
                value_avg_policy = 0.0

        action_name = action_to_string(action_idx, bet_bins)

        print(
            f"{child_idx:>6} | {action_name:>10} | {policy_prob:7.4f} | {policy_avg_prob:7.4f} | "
            f"{regret:7.2f} [{regret_min:7.2f}, {regret_max:7.2f}] | {value:7.4f} | {value_latest:9.4f} | {value_avg_policy:9.4f}"
        )


def _print_node_line(
    evaluator: RebelCFREvaluator | RebelCFREvaluatorOld | SparseCFREvaluator,
    node_idx: int,
    depth: int,
    selected_hand_idx: Optional[int],
    bet_bins: list[float],
    action_idx: Optional[int],
    show_specific_hand: bool,
) -> None:
    policy_probs = evaluator.policy_probs[node_idx].clone()
    policy_probs_avg = evaluator.policy_probs_avg[node_idx].clone()
    actor = evaluator.prev_actor[node_idx].item()
    root_actor = evaluator.env.to_act[0].item()
    belief_weight: Optional[float] = None
    actor_str = (
        "-" if node_idx == 0 else f"*P{actor}" if actor == root_actor else f"P{actor}"
    )

    fallback_action = action_idx
    if fallback_action is None:
        offset_current = evaluator.depth_offsets[depth]
        fallback_action = (node_idx - offset_current) % evaluator.num_actions
    resolved_action = _resolve_sparse_action_idx(
        evaluator, node_idx, fallback_action if fallback_action is not None else -1
    )

    if selected_hand_idx is not None and show_specific_hand:
        policy_prob = float(policy_probs[selected_hand_idx].item())
        policy_avg_prob = float(policy_probs_avg[selected_hand_idx].item())
        regret_val = float(
            evaluator.cumulative_regrets[node_idx][selected_hand_idx].item()
        )
        regret = regret_val
        regret_min = regret_val
        regret_max = regret_val
        value = (
            evaluator.values_avg[node_idx, root_actor, selected_hand_idx].mean().item()
        )
        value_latest = (
            evaluator.latest_values[node_idx, root_actor, selected_hand_idx]
            .mean()
            .item()
        )
        value_avg_policy = value
    else:
        if depth == 0:
            parent_idx = node_idx
        else:
            offset_current = evaluator.depth_offsets[depth]
            offset_parent = evaluator.depth_offsets[depth - 1]
            parent_local_index = (node_idx - offset_current) // evaluator.num_actions
            parent_idx = offset_parent + parent_local_index

        beliefs_current = evaluator.beliefs[parent_idx, :, :]
        beliefs_avg = evaluator.beliefs_avg[parent_idx, :, :]
        weights_current = beliefs_current.sum(dim=0)
        weights_avg = beliefs_avg.sum(dim=0)

        parent_allowed = evaluator.allowed_hands[parent_idx]
        child_allowed = evaluator.allowed_hands[node_idx]
        allowed_hands = parent_allowed & child_allowed
        policy_probs = policy_probs.masked_fill(~allowed_hands, 0.0)
        policy_probs_avg = policy_probs_avg.masked_fill(~allowed_hands, 0.0)
        weights_current = weights_current.masked_fill(~allowed_hands, 0.0)
        weights_avg = weights_avg.masked_fill(~allowed_hands, 0.0)

        weights_current_sum = float(weights_current.sum().item())
        weights_avg_sum = float(weights_avg.sum().item())
        policy_prob = (
            (policy_probs * weights_current).sum().item() / weights_current_sum
            if weights_current_sum > 0.0
            else 0.0
        )
        policy_avg_prob = (
            (policy_probs_avg * weights_avg).sum().item() / weights_avg_sum
            if weights_avg_sum > 0.0
            else 0.0
        )
        regret = evaluator.cumulative_regrets[node_idx].mean().item()
        regret_max = evaluator.cumulative_regrets[node_idx].max().item()
        regret_min = evaluator.cumulative_regrets[node_idx].min().item()
        if selected_hand_idx is not None:
            value = evaluator.values_avg[node_idx, root_actor, selected_hand_idx].item()
            value_latest = evaluator.latest_values[
                node_idx, root_actor, selected_hand_idx
            ].item()
            value_avg_policy = value
        else:
            beliefs_actor_avg = evaluator.beliefs_avg[parent_idx, root_actor, :].clone()
            value = (
                (evaluator.values_avg[node_idx, root_actor, :] * beliefs_actor_avg)
                .sum()
                .item()
            )
            value_latest = (
                (evaluator.latest_values[node_idx, root_actor, :] * beliefs_actor_avg)
                .sum()
                .item()
            )
            beliefs_actor_avg = beliefs_actor_avg.masked_fill(~allowed_hands, 0.0)
            policy_weight = evaluator.policy_probs_avg[node_idx].masked_fill(
                ~allowed_hands, 0.0
            )
            policy_weight = policy_weight * beliefs_actor_avg
            weight_sum = float(policy_weight.sum().item())
            if weight_sum > 0.0:
                value_avg_policy = (
                    evaluator.values_avg[node_idx, root_actor, :] * policy_weight
                ).sum().item() / weight_sum
            else:
                value_avg_policy = 0.0

    indent = "  " * depth
    leaf_str = "L" if evaluator.leaf_mask[node_idx].item() else " "
    # Derive action if not provided
    action_name = action_to_string(int(resolved_action), bet_bins)
    if node_idx == 0:
        action_name = "ROOT"
    node_label = f"{indent}{node_idx} {action_name}"
    regret_minmax = f" [{regret_min:6.2f}, {regret_max:6.2f}]"
    belief_str = ""
    if selected_hand_idx is not None:
        hero_player = root_actor
        belief_weight = float(
            evaluator.beliefs[node_idx, hero_player, selected_hand_idx].item()
        )
        belief_str = f"{belief_weight:7.4f}"
    print(
        f"{node_label:<{NODE_COL_WIDTH - 2}} {leaf_str} | {actor_str:>5} | {policy_prob:7.4f} | {policy_avg_prob:7.4f} | "
        f"{regret:7.2f}{regret_minmax} | {belief_str:>7} | {value:7.4f} | {value_latest:9.4f} | {value_avg_policy:9.4f}"
    )


def print_nodes_depth_first(
    evaluator: RebelCFREvaluator | RebelCFREvaluatorOld | SparseCFREvaluator,
    max_depth: int,
    selected_hand_idx: Optional[int],
    bet_bins: list[float],
) -> None:
    print(f"\nDepth-first Nodes (depth 0-{max_depth})")
    regret_str = f"{'Regret [min, max]':>24}"
    belief_header = "Belief" if selected_hand_idx is not None else ""

    B = evaluator.num_actions
    root_actor = evaluator.env.to_act[0].item()

    print(
        f"{'Node':<{NODE_COL_WIDTH}} | {'Actor':>5} | {'Policy':>7} | {'PolAvg':>7} | {regret_str} | "
        f"{belief_header:>7} | {f'P{root_actor} Value':>7} | {f'ValLatest':>9} | {f'ValAvgPol':>9}"
    )
    print("-" * 110)

    def dfs_at(node_idx: int, depth: int, came_action: Optional[int]) -> None:
        if depth > max_depth:
            return
        if not evaluator.valid_mask[node_idx]:
            return
        prev_actor = evaluator.prev_actor[node_idx].item()
        _print_node_line(
            evaluator,
            node_idx,
            depth,
            selected_hand_idx,
            bet_bins,
            came_action,
            show_specific_hand=prev_actor == root_actor,
        )
        # compute children if next depth exists
        if depth < max_depth:
            if isinstance(evaluator, SparseCFREvaluator):
                child_count = int(evaluator.child_count[node_idx].item())
                if child_count == 0:
                    return
                child_offset = int(evaluator.child_offsets[node_idx].item())
                start = child_offset - child_count
                child_range = range(start, child_offset)
            else:
                offset = evaluator.depth_offsets[depth]
                offset_next = evaluator.depth_offsets[depth + 1]
                local_index = node_idx - offset
                base_child = offset_next + local_index * B
                child_range = range(base_child, base_child + B)

            for idx, child_idx in enumerate(child_range):
                fallback_action = idx
                if not isinstance(evaluator, SparseCFREvaluator):
                    fallback_action = idx
                action_id = _resolve_sparse_action_idx(
                    evaluator, child_idx, fallback_action
                )
                dfs_at(child_idx, depth + 1, action_id)

    # Start from depth 1 valid nodes in DFS order
    d0_start = evaluator.depth_offsets[0]
    d0_end = evaluator.depth_offsets[1]
    for n in range(d0_start, d0_end):
        dfs_at(n, 0, None)


def debug_cfr_depth1(
    cfg: Config,
    checkpoint_path: Optional[str] = None,
    street: Optional[str] = None,
    iterations: Optional[int] = None,
    selected_hand_idx: Optional[int] = None,
    selected_hole_str: Optional[str] = None,
    verbose: bool = False,
    random_beliefs: bool = False,
    forced_board: Optional[list[int]] = None,
    sparse: bool = False,
    old_evaluator: bool = False,
) -> None:
    """Main debugging function."""
    if sparse and old_evaluator:
        raise ValueError(
            "Cannot use both --sparse and --old-evaluator options together"
        )

    if sparse:
        evaluator_type = "Sparse CFR"
    elif old_evaluator:
        evaluator_type = "ReBeL CFR (Old)"
    else:
        evaluator_type = "ReBeL CFR"
    print(f"=== {evaluator_type} Depth-1 Debugger ===")
    street = street or "preflop"
    valid_streets = {"preflop", "flop", "turn", "river", "river_plus"}
    if street not in valid_streets:
        raise ValueError(
            f"Invalid street: {street}. Must be one of: {', '.join(sorted(valid_streets))}"
        )
    street_upper = street.upper()
    print(f"Mode: {street_upper}")

    # Create or use provided configuration
    cfr_type = cfg.search.cfr_type
    if cfg.seed is None:
        cfg.seed = random.randint(0, 1000000)
    if iterations is not None:
        # Ensure warm_start < iterations
        cfg.search.iterations = iterations
        if cfg.search.warm_start_iterations >= cfg.search.iterations:
            cfg.search.warm_start_iterations = max(0, cfg.search.iterations - 1)
    device = torch.device(cfg.device)

    print(f"Using device: {device}")
    print(f"CFR Type: {str(cfr_type)}")
    print(f"Tree depth: {cfg.search.depth}")
    print(f"Warm start iterations: {cfg.search.warm_start_iterations}")
    print(f"Total iterations: {cfg.search.iterations}")

    # Load or create model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = load_model_from_checkpoint(checkpoint_path, cfg, device)
    else:
        model = create_random_model(cfg, device)

    # Create environment
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )

    env.reset()

    if street != "preflop":
        # Advance using environment dynamics from a clean reset
        advance_to_street_with_history(
            env, street=street, button=1, forced_board=forced_board
        )
        print(f"\nStarting {street_upper} CFR Tree Construction...")
        print(f"Pot size: {int(env.pot[0].item())}")
        print(f"Street: {env.street[0].item()}")
        print(f"Actions this round: {env.actions_this_round[0].item()}")
        board_idxs = [i for i in env.board_indices[0].tolist() if i >= 0]
        if board_idxs:
            board_str = " ".join(_card_index_to_str(i) for i in board_idxs)
            print(f"Board: {board_str}")
            if forced_board is not None:
                print(f"(Forced board)")
        print(f"To act: {int(env.to_act[0].item())}")
    else:
        print("\nStarting Preflop CFR Tree Construction...")

    # Create evaluator
    if sparse:
        evaluator = SparseCFREvaluator(
            model=model,
            device=device,
            cfg=cfg,
        )
    elif old_evaluator:
        evaluator = RebelCFREvaluatorOld(
            search_batch_size=1,
            env_proto=env,
            model=model,
            bet_bins=cfg.env.bet_bins,
            max_depth=cfg.search.depth,
            cfr_iterations=cfg.search.iterations,
            device=device,
            float_dtype=torch.float32,
            warm_start_iterations=cfg.search.warm_start_iterations,
            cfr_type=cfr_type,
            cfr_avg=cfg.search.cfr_avg,
            dcfr_delay=cfg.search.dcfr_plus_delay,
        )
    else:
        evaluator = RebelCFREvaluator(
            search_batch_size=1,
            env_proto=env,
            model=model,
            bet_bins=cfg.env.bet_bins,
            max_depth=cfg.search.depth,
            cfr_iterations=cfg.search.iterations,
            device=device,
            float_dtype=torch.float32,
            warm_start_iterations=cfg.search.warm_start_iterations,
            cfr_type=cfr_type,
            cfr_avg=cfg.search.cfr_avg,
            dcfr_delay=cfg.search.dcfr_plus_delay,
        )

    # Initialize search
    roots = torch.tensor([0], device=device)
    initial_beliefs = None
    if random_beliefs:
        # Generate random beliefs: sample from Dirichlet distribution for each player
        # Shape: [1, 2, NUM_HANDS]
        initial_beliefs = torch.zeros(
            1, 2, NUM_HANDS, dtype=torch.float32, device=device
        )
        for p in range(2):
            # Sample from Dirichlet(alpha=1) which is uniform over simplex
            beliefs_p = torch.distributions.Dirichlet(torch.ones(NUM_HANDS)).sample()
            initial_beliefs[0, p, :] = beliefs_p.to(device=device, dtype=torch.float32)

    if sparse:
        evaluator.initialize_subgame(env, roots, initial_beliefs=initial_beliefs)
        evaluator.initialize_policy_and_beliefs()
    else:
        evaluator.initialize_subgame(env, roots, initial_beliefs=initial_beliefs)
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
        f"\nRunning CFR iterations {cfg.search.warm_start_iterations}-{cfg.search.iterations}..."
    )

    evaluator.t_sample = evaluator._get_sampling_schedule()

    # Run iterations 15-25, but only print 16-25
    for t in range(cfg.search.warm_start_iterations, cfg.search.iterations):
        evaluator.cfr_iteration(t)

        if (t + 1) % 10 == 0:
            print_single_iteration_data(
                evaluator=evaluator,
                iteration=t,
                bet_bins=cfg.env.bet_bins,
                selected_hand_idx=selected_hand_idx,
            )

            # Compute and display exploitability after updating averages
            exploit_stats = evaluator._compute_exploitability()
            if exploit_stats.local_exploitability.numel() > 0:
                total_expl = exploit_stats.local_exploitability.mean().item()
                print(
                    f"Exploitability (avg best-response improv): total={total_expl:.6f}"
                )

    print(f"\n{'='*80}")
    print("Debug Complete!")
    print(f"{'='*80}")

    # Depth-first listing with indentation (only if verbose)
    if verbose:
        max_depth_df = len(evaluator.depth_offsets) - 2
        if max_depth_df >= 1:
            # Reprint state summary before DFS
            board_idxs = [i for i in env.board_indices[0].tolist() if i >= 0]
            board_str = " ".join(_card_index_to_str(i) for i in board_idxs)
            print("\nState summary before DFS:")
            print(f"Pot size: {int(env.pot[0].item())}")
            print(f"Street: {env.street[0].item()}")
            print(f"Actions this round: {env.actions_this_round[0].item()}")
            print(f"Board: {board_str}")
            if selected_hole_str:
                print(f"Hole: {selected_hole_str}")
            print_nodes_depth_first(
                evaluator, max_depth_df, selected_hand_idx, bet_bins=cfg.env.bet_bins
            )


@dataclass
class TopLevel:
    checkpoint: Optional[str] = None
    street: Optional[str] = None
    iterations: Optional[int] = None
    hole: Optional[str] = None
    verbose: bool = False
    random_beliefs: bool = False
    board: Optional[str] = None
    sparse: bool = False
    old_evaluator: bool = False


cs = ConfigStore.instance()
cs.store(group="", name="debug_cfr_depth1_schema", node=TopLevel)


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    # Extract top-level script params and convert the rest into Config
    checkpoint = dict_config.get("checkpoint")
    street = dict_config.get("street")
    iterations = dict_config.get("iterations")
    hole = dict_config.get("hole")
    verbose = bool(dict_config.get("verbose", False))
    random_beliefs = bool(dict_config.get("random_beliefs", False))
    board = dict_config.get("board")
    # Check if sparse was explicitly provided at top level
    sparse_explicit = "sparse" in dict_config
    sparse = bool(dict_config.get("sparse", False)) if sparse_explicit else None
    old_evaluator = bool(dict_config.get("old_evaluator", False))

    container: dict[str, any] = OmegaConf.to_container(dict_config, resolve=True)
    # Remove our script-specific keys before constructing core Config
    for k in [
        "checkpoint",
        "street",
        "iterations",
        "hole",
        "verbose",
        "random_beliefs",
        "board",
        "sparse",
        "old_evaluator",
    ]:
        if k in container:
            container.pop(k)

    cfg = Config.from_dict(container)
    cfg.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    cfg.num_envs = 1

    # Use cfg.search.sparse if sparse wasn't explicitly provided at top level
    if sparse is None:
        sparse = cfg.search.sparse

    # Resolve checkpoint path relative to original working directory
    if isinstance(checkpoint, str) and checkpoint:
        checkpoint = to_absolute_path(checkpoint)

    # Derive selected hand index if hole provided
    selected_hand_idx: Optional[int] = None
    if hole:
        c1, c2 = _parse_hole_cards_str(hole)
        selected_hand_idx = int(card_utils.combo_index(c1, c2))

    # Parse board string if provided
    forced_board: Optional[list[int]] = None
    if board:
        forced_board = _parse_board_str(board)
        # River_plus also requires 5 board cards
        if street == "river_plus" and len(forced_board) != 5:
            raise ValueError(
                f"River_plus requires 5 board cards, got {len(forced_board)}"
            )

    debug_cfr_depth1(
        cfg=cfg,
        checkpoint_path=checkpoint,
        street=street,
        iterations=iterations,
        selected_hand_idx=selected_hand_idx,
        selected_hole_str=hole,
        verbose=verbose,
        random_beliefs=random_beliefs,
        forced_board=forced_board,
        sparse=sparse,
        old_evaluator=old_evaluator,
    )


if __name__ == "__main__":
    main()
