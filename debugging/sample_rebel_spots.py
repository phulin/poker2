#!/usr/bin/env python3
"""
Sample one example spot on each street (preflop, flop, turn, river) using the
ReBeL CFR evaluator and a provided checkpoint, configured via conf/config_rebel_cfr.yaml.

For each street, prints:
- Street name and to-act player
- Board cards
- Sampled hero hole cards (from beliefs of the acting player)
- Hero EV for the sampled combo
- Action probabilities for that combo (FOLD, CALL, bet sizes, ALLIN)
- Betting history sampled from the evaluator between streets

Usage:
  python debugging/sample_rebel_spots.py \
    --checkpoint checkpoints-rebel/rebel_latest.pt \
    --config conf/config_rebel_cfr.yaml \
    --depth 2 \
    --hand AsAh

Notes:
- Uses the same model-loading approach as other debugging scripts.
- Runs CFR self-play per decision with training_mode=False (no epsilon) and samples actions from policy.
"""

import argparse
import os
from typing import List, Optional, Tuple

import torch
from omegaconf import OmegaConf

from alphaholdem.core.structured_config import (
    CFRType,
    Config,
    EnvConfig,
    ModelConfig,
    SearchConfig,
    TrainingConfig,
)
from alphaholdem.env.card_utils import hand_combos_tensor
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.better_features import (
    context_length as better_context_length,
)
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator


RANKS = {
    c: i
    for i, c in enumerate(
        ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    )
}
SUITS = {"s": 0, "h": 1, "d": 2, "c": 3}


def card_to_string(card_idx: int) -> str:
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["♠", "♥", "♦", "♣"]
    rank = card_idx // 4
    suit = card_idx % 4
    return f"{ranks[rank]}{suits[suit]}"


def parse_card(token: str) -> int:
    token = token.strip().upper()
    if len(token) != 2:
        raise ValueError(f"Invalid card: '{token}'")
    r, s = token[0], token[1].lower()
    if r not in RANKS or s not in SUITS:
        raise ValueError(f"Invalid card: '{token}'")
    return RANKS[r] * 4 + SUITS[s]


def parse_hand_str(hand_str: str) -> Tuple[int, int]:
    s = hand_str.replace(" ", "").replace("/", "").replace("-", "").strip()
    if len(s) == 4:
        c1 = parse_card(s[:2])
        c2 = parse_card(s[2:])
        return c1, c2
    parts = hand_str.replace(",", " ").split()
    if len(parts) == 2:
        return parse_card(parts[0]), parse_card(parts[1])
    raise ValueError(f"Invalid hand string: '{hand_str}' (expected like 'AsAh')")


def hand_str_to_combo_index(hand_str: str) -> int:
    c1, c2 = parse_hand_str(hand_str)
    combos = hand_combos_tensor(device=torch.device("cpu"))
    a = torch.tensor([c1, c2], dtype=torch.long)
    match = torch.where(
        (combos[:, 0] == a[0]) & (combos[:, 1] == a[1])
        | ((combos[:, 0] == a[1]) & (combos[:, 1] == a[0]))
    )[0]
    if match.numel() == 0:
        raise ValueError(f"Hand not found in combos: '{hand_str}'")
    return int(match[0].item())


def hand_to_string(hand_idx: int, combos: torch.Tensor) -> str:
    c1, c2 = combos[hand_idx].tolist()
    return f"{card_to_string(c1)}{card_to_string(c2)}"


def action_to_string(action_idx: int, bet_bins: List[float]) -> str:
    if action_idx == 0:
        return "FOLD"
    elif action_idx == 1:
        return "CALL/CHECK"
    elif action_idx == len(bet_bins) + 2:
        return "ALLIN"
    elif 2 <= action_idx < len(bet_bins) + 2:
        return f"B{bet_bins[action_idx - 2]:.2f}x"
    return f"A{action_idx}"


def load_model_from_checkpoint(
    checkpoint_path: str, cfg: Config, device: torch.device
) -> RebelFFN | BetterFFN:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]

    # Detect BetterFFN by presence of trunk/post_norm keys
    is_better_ffn = "street_embedding.weight" in model_state
    if is_better_ffn:
        # Infer hidden sizes to match checkpoint
        if "post_norm.weight" in model_state:
            hidden_dim = model_state["post_norm.weight"].shape[0]
            cfg.model.hidden_dim = hidden_dim
            cfg.model.range_hidden_dim = 128
            cfg.model.ffn_dim = hidden_dim * 2
        # Infer num layers
        trunk_layers: set[int] = set()
        for k in model_state.keys():
            if k.startswith("trunk."):
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    trunk_layers.add(int(parts[1]))
        if trunk_layers:
            cfg.model.num_hidden_layers = max(trunk_layers) + 1
    # Use trainer to properly construct and load
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    trainer.load_checkpoint(checkpoint_path)
    model = trainer.model
    model.eval()
    return model


# Replaced deterministic call/check advance with sampling-based betting


def advance_round_with_sampling(
    env: HUNLTensorEnv,
    evaluator: RebelCFREvaluator,
    bet_bins: List[float],
    max_actions: int = 8,
    designated_combo_idx: Optional[int] = None,
) -> list[str]:
    """Advance the current betting round by sampling actions from the evaluator.

    Returns a list of pretty-printed actions for the history.
    """
    history: list[str] = []
    start_street = env.street[0].item()
    device = env.device
    combos = hand_combos_tensor(device=torch.device("cpu"))

    steps = 0
    while (
        env.street[0].item() == start_street
        and not env.done[0].item()
        and steps < max_actions
    ):
        # Build evaluator for current root and compute policy
        root_indices = torch.tensor([0], dtype=torch.long, device=device)
        evaluator.initialize_search(env, root_indices)
        evaluator.self_play_iteration(training_mode=False)

        # Pull root policy [actions, hands]
        root_policy = evaluator._pull_back(evaluator.policy_probs_avg)[0]
        to_act = evaluator.env.to_act[0].item()
        beliefs = evaluator.beliefs_avg[0, to_act]

        # Choose hand: hero (P0) fixed if designated, P1 sampled when acting
        if to_act == 0 and designated_combo_idx is not None:
            hand_idx = designated_combo_idx
        else:
            hand_idx = torch.multinomial(beliefs, num_samples=1).item()

        # Sample action (block ALLIN from sampling)
        action_probs = root_policy[:, hand_idx].clone()
        allin_idx = len(bet_bins) + 2
        if allin_idx < action_probs.numel():
            action_probs[allin_idx] = 0.0
        total = action_probs.sum().clamp(min=1e-8)
        action_probs = action_probs / total
        action_idx = torch.multinomial(action_probs, num_samples=1).item()

        # Step the environment
        bin_amounts, legal_masks = env.legal_bins_amounts_and_mask()
        env.step_bins(
            torch.tensor([action_idx], device=device),
            bin_amounts=bin_amounts,
            legal_masks=legal_masks,
        )

        # Record history line
        pot = env.pot[0].item()
        history.append(
            f"P{to_act} -> {action_to_string(action_idx, bet_bins)} | pot={pot}"
        )

        steps += 1

    return history


def print_spot(
    evaluator: RebelCFREvaluator,
    bet_bins: List[float],
    label: str,
    designated_combo_idx: Optional[int] = None,
) -> None:
    """Extract and print info for the single root state currently in the evaluator."""
    # Root info
    N = evaluator.search_batch_size
    assert N == 1, "This printer assumes a single root (N=1)."

    # Pull root policy per-hand [N, actions, hands]
    root_policy = evaluator._pull_back(evaluator.policy_probs_avg)[:N]
    # Root values per player per hand [N, 2, hands]
    root_values = evaluator.values_avg[:N]

    to_act = evaluator.env.to_act[0].item()
    street = evaluator.env.street[0].item()
    streets = ["preflop", "flop", "turn", "river", "showdown"]

    board_idx = evaluator.env.board_indices[0].tolist()
    board_str = " ".join(
        card_to_string(c) for c in board_idx if isinstance(c, int) and c >= 0
    )
    pot = evaluator.env.pot[0].item()

    # Hero is always P0
    combos = hand_combos_tensor(device=torch.device("cpu"))
    if designated_combo_idx is not None:
        hand_idx = designated_combo_idx
    else:
        # sample from P0 beliefs regardless of to_act
        beliefs_p0 = evaluator.beliefs_avg[0, 0]
        hand_idx = torch.multinomial(beliefs_p0, num_samples=1).item()

    # Hero EV is for player 0
    hero_ev = root_values[0, 0, hand_idx].item()
    # Action probs at the node (acting player's distribution) conditioned on hero hand
    probs = root_policy[0, :, hand_idx].detach().cpu()

    # Pretty print
    print(f"\n=== {label.upper()} ===")
    print(f"Street: {streets[street]} | To act: P{to_act} | Pot: {pot}")
    print(f"Board: {board_str if board_str else '-'}")
    print(f"Hero hand: {hand_to_string(hand_idx, combos)}")
    print(f"Hero EV (chips, P0): {hero_ev:.4f}")
    labels = [action_to_string(i, bet_bins) for i in range(probs.numel())]
    parts = [f"{lab}: {probs[i].item():.4f}" for i, lab in enumerate(labels)]
    print(" | ".join(parts))


def print_tree_compact(evaluator: RebelCFREvaluator) -> None:
    """Print a compact summary of valid nodes per depth in the evaluator tree."""
    max_depth = evaluator.max_depth
    lines: list[str] = []
    for d in range(0, max_depth + 1):
        start = evaluator.depth_offsets[d]
        end = evaluator.depth_offsets[d + 1]
        sl = slice(start, end)
        valid = evaluator.valid_mask[sl]
        v = int(valid.sum().item())
        if v == 0:
            lines.append(f"  D{d}: valid=0")
            continue
        leaf = evaluator.leaf_mask[sl] & valid
        l = int(leaf.sum().item())
        to_act = evaluator.env.to_act[sl]
        p0 = int(((to_act == 0) & valid).sum().item())
        p1 = int(((to_act == 1) & valid).sum().item())
        # Average legal branching among valid non-leaf nodes
        legal = evaluator.legal_mask[sl]
        nl_mask = valid & ~leaf
        if nl_mask.any():
            legal_counts = legal[nl_mask].sum(dim=-1).float()
            avg_branch = float(legal_counts.mean().item())
        else:
            avg_branch = 0.0
        lines.append(
            f"  D{d}: valid={v} leaf={l} to_act[P0={p0},P1={p1}] avg_branch={avg_branch:.2f}"
        )
    print("Tree:")
    for ln in lines:
        print(ln)


def print_tree_full(
    evaluator: RebelCFREvaluator, bet_bins: List[float], hero_hand_idx: int
) -> None:
    """Print every valid node, one per line, with node value (P0) and action prob from parent.

    For depth 0 (root), action is ROOT and prob=1.0.
    For depth > 0, action_index is computed from node index and depth; probability is
    evaluator.policy_probs_avg[node, hero_hand_idx].
    """
    max_depth = evaluator.max_depth
    B = evaluator.num_actions
    M = evaluator.total_nodes

    def parent_and_action(
        node_idx: int, depth: int
    ) -> tuple[int, int] | tuple[None, None]:
        if depth == 0:
            return None, None
        start_prev = evaluator.depth_offsets[depth]
        start_src = evaluator.depth_offsets[depth - 1]
        # node_idx in [start_prev, start_next)
        local = node_idx - start_prev
        parent_local = local // B
        action = local % B
        parent_idx = start_src + parent_local
        return parent_idx, action

    print("Full tree (post-CFR):")
    for d in range(0, max_depth + 1):
        start = evaluator.depth_offsets[d]
        end = evaluator.depth_offsets[d + 1]
        for node in range(start, end):
            if not evaluator.valid_mask[node].item():
                continue
            to_act = int(evaluator.env.to_act[node].item())
            is_leaf = bool(evaluator.leaf_mask[node].item())
            parent_idx, action = parent_and_action(node, d)
            if parent_idx is None:
                action_str = "ROOT"
                prob = 1.0
            else:
                action_str = action_to_string(action, bet_bins)
                prob = float(evaluator.policy_probs_avg[node, hero_hand_idx].item())
            value_p0 = float(evaluator.values_avg[node, 0, hero_hand_idx].item())
            print(
                f"D{d} N{node:>5} to_act=P{to_act} leaf={'Y' if is_leaf else 'N'} "
                f"act={action_str:>10} prob={prob:>7.4f} val_p0={value_p0:>+9.4f}"
            )


def print_tree_full_depth_first(
    evaluator: RebelCFREvaluator, bet_bins: List[float], hero_hand_idx: int
) -> None:
    """Print the whole tree depth-first, indenting by depth; one node per line.

    Each line: indent + act_from_parent + prob + val_p0 + to_act/leaf flag.
    Root prints as ROOT with prob=1.0.
    """
    B = evaluator.num_actions

    def node_line(
        node_idx: int, depth: int, action_from_parent: str, prob: float
    ) -> str:
        to_act = int(evaluator.env.to_act[node_idx].item())
        is_leaf = bool(evaluator.leaf_mask[node_idx].item())
        value_p0 = float(evaluator.values_avg[node_idx, 0, hero_hand_idx].item())
        indent = "  " * depth
        return (
            f"{indent}{action_from_parent} prob={prob:>7.4f} val_p0={value_p0:>+9.4f} "
            f"to_act=P{to_act} leaf={'Y' if is_leaf else 'N'}"
        )

    def dfs(node_idx: int, depth: int) -> None:
        # Iterate over children actions in order
        if depth >= evaluator.max_depth:
            return
        start = evaluator.depth_offsets[depth]
        start_next = evaluator.depth_offsets[depth + 1]
        for a in range(B):
            child_idx = start_next + (node_idx - start) * B + a
            if child_idx >= evaluator.total_nodes:
                continue
            if not evaluator.valid_mask[child_idx].item():
                continue
            action_str = action_to_string(a, bet_bins)
            prob = float(evaluator.policy_probs_avg[child_idx, hero_hand_idx].item())
            print(node_line(child_idx, depth + 1, action_str, prob))
            dfs(child_idx, depth + 1)

    # Root line
    root_idx = 0
    print("Full tree (post-CFR, depth-first):")
    print(node_line(root_idx, 0, "ROOT", 1.0))
    dfs(root_idx, 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample one spot per street and print ReBeL CFR outputs."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config_rebel_cfr.yaml",
        help="Hydra config path",
    )
    parser.add_argument(
        "--depth", type=int, default=None, help="Override search depth (optional)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override CFR iterations (optional)",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--hand",
        type=str,
        default=None,
        help="Designate hero hole hand (e.g., AsAh). Applies to the player to act.",
    )
    parser.add_argument(
        "--preflop",
        action="store_true",
        help="Only show preflop spot/output (can be combined with other street flags)",
    )
    parser.add_argument(
        "--flop",
        action="store_true",
        help="Only show flop spot/output (can be combined with other street flags)",
    )
    parser.add_argument(
        "--turn",
        action="store_true",
        help="Only show turn spot/output (can be combined with other street flags)",
    )
    parser.add_argument(
        "--river",
        action="store_true",
        help="Only show river spot/output (can be combined with other street flags)",
    )
    args = parser.parse_args()

    # Load config via OmegaConf and construct structured Config (filter out hydra defaults)
    cfg_dict = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg_dict.pop("defaults", None)

    train_config = TrainingConfig(**cfg_dict.get("train", {}))
    model_config = ModelConfig(**cfg_dict.get("model", {}))
    env_config = EnvConfig(**cfg_dict.get("env", {}))
    search_config = SearchConfig(**cfg_dict.get("search", {}))

    cfg = Config(
        train=train_config,
        model=model_config,
        env=env_config,
        search=search_config,
        **{
            k: v
            for k, v in cfg_dict.items()
            if k not in ["train", "model", "env", "search"]
        },
    )

    if args.depth is not None:
        cfg.search.depth = args.depth
    if args.iterations is not None:
        cfg.search.iterations = args.iterations

    device = torch.device(
        cfg.device
        if torch.cuda.is_available() or cfg.device == "cpu"
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, cfg, device)

    # One-env prototype for search roots
    env_proto = HUNLTensorEnv(
        num_envs=1,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )

    bet_bins = cfg.env.bet_bins
    designated_combo_idx: Optional[int] = None
    if args.hand is not None:
        designated_combo_idx = hand_str_to_combo_index(args.hand)

    # Build evaluator once and reuse
    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env_proto,
        model=model,
        bet_bins=bet_bins,
        max_depth=cfg.search.depth,
        cfr_iterations=cfg.search.iterations,
        device=device,
        float_dtype=torch.float32,
        warm_start_iterations=min(
            cfg.search.warm_start_iterations, max(0, cfg.search.iterations - 1)
        ),
        generator=torch.Generator(device=device).manual_seed(args.seed),
        cfr_type=(
            cfg.search.cfr_type
            if isinstance(cfg.search.cfr_type, CFRType)
            else CFRType[cfg.search.cfr_type]
        ),
        cfr_avg=True,
        dcfr_alpha=cfg.search.dcfr_alpha,
        dcfr_beta=cfg.search.dcfr_beta,
        dcfr_gamma=cfg.search.dcfr_gamma,
        dcfr_delay=cfg.search.dcfr_delay,
    )

    def run_one(label: str):
        root_indices = torch.tensor([0], dtype=torch.long, device=device)
        evaluator.initialize_search(env_proto, root_indices)
        evaluator.self_play_iteration(training_mode=False)
        print_tree_compact(evaluator)
        print_spot(evaluator, bet_bins, label, designated_combo_idx)
        # Determine hero hand index for full tree print (P0 designated or sample from P0 beliefs)
        if designated_combo_idx is not None:
            hero_idx = designated_combo_idx
        else:
            hero_idx = int(torch.multinomial(evaluator.beliefs_avg[0, 0], 1).item())
        print_tree_full_depth_first(evaluator, bet_bins, hero_idx)

    # Reset env and do each street, sampling actions between them
    env_proto.reset()

    # If no street filter flags are provided, show all
    filter_set = args.preflop or args.flop or args.turn or args.river

    if (not filter_set) or args.preflop:
        run_one("preflop")
        history = advance_round_with_sampling(
            env_proto, evaluator, bet_bins, designated_combo_idx=designated_combo_idx
        )
        if history:
            print("Actions:")
            for h in history:
                print(f"  {h}")

    if (not filter_set) or args.flop:
        if not ((not filter_set) or args.preflop):
            # If we skipped preflop, ensure env is at flop
            env_proto.reset()
        run_one("flop")
        history = advance_round_with_sampling(
            env_proto, evaluator, bet_bins, designated_combo_idx=designated_combo_idx
        )
        if history:
            print("Actions:")
            for h in history:
                print(f"  {h}")

    if (not filter_set) or args.turn:
        if not ((not filter_set) or args.flop):
            # If we skipped earlier streets, advance to turn
            env_proto.reset()
            # advance two rounds to reach turn
            advance_round_with_sampling(env_proto, evaluator, bet_bins)
        run_one("turn")
        history = advance_round_with_sampling(
            env_proto, evaluator, bet_bins, designated_combo_idx=designated_combo_idx
        )
        if history:
            print("Actions:")
            for h in history:
                print(f"  {h}")

    if (not filter_set) or args.river:
        if not ((not filter_set) or args.turn):
            # If we skipped to river directly, advance three rounds
            env_proto.reset()
            advance_round_with_sampling(env_proto, evaluator, bet_bins)
            advance_round_with_sampling(env_proto, evaluator, bet_bins)
        run_one("river")


if __name__ == "__main__":
    main()
