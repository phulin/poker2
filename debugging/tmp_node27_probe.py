#!/usr/bin/env python3
"""Temporary helper to inspect node 27 beliefs and values for 3s4c on the river."""

from __future__ import annotations

import pathlib
import sys

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from alphaholdem.core.structured_config import Config
from alphaholdem.env import card_utils
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rules import rank_hands
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
from debugging.debug_cfr_depth1 import (  # type: ignore  # local helper import
    _parse_hole_cards_str,
    load_model_from_checkpoint,
)


def advance_with_forced_deck(
    env: HUNLTensorEnv, street: str, forced_deck: torch.Tensor
) -> None:
    """Advance environment to the desired street using the same history as the debugger."""
    button = torch.tensor([1], device=env.device)
    env.reset(force_button=button, force_deck=forced_deck)

    if street == "preflop":
        return

    bet_bin_index = min(3, len(env.default_bet_bins) + 2)
    call_action = torch.ones(env.N, dtype=torch.long, device=env.device)

    env.step_bins(
        torch.full((env.N,), bet_bin_index, dtype=torch.long, device=env.device)
    )
    env.step_bins(call_action)

    if street == "flop":
        return

    env.step_bins(call_action)
    env.step_bins(call_action)

    if street == "turn":
        return

    env.step_bins(call_action)  # SB check
    env.step_bins(
        torch.full((env.N,), bet_bin_index, dtype=torch.long, device=env.device)
    )  # BB bet
    env.step_bins(call_action)  # SB call


def run_probe() -> None:
    overrides = [
        "+street=river",
        "search.iterations=500",
        "search.depth=3",
        "+seed=6",
    ]
    with initialize(version_base=None, config_path="../conf"):
        dict_cfg = compose(config_name="config_rebel_cfr", overrides=overrides)

    container = OmegaConf.to_container(dict_cfg, resolve=True)
    for key in [
        "checkpoint",
        "street",
        "iterations",
        "hole",
        "verbose",
        "random_beliefs",
    ]:
        container.pop(key, None)

    cfg = Config.from_dict(container)
    if cfg.seed is None:
        cfg.seed = 6
    cfg.device = "cpu"
    device = torch.device(cfg.device)

    checkpoint_path = "rebel_115_3600.pt"
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

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

    env.rng.manual_seed(int(cfg.seed))
    forced_deck = torch.tensor(
        [[1, 41, 3, 14, 38, 22, 7, 44, 13]], device=device, dtype=torch.long
    )
    advance_with_forced_deck(env, street="river", forced_deck=forced_deck)

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
        cfr_type=cfg.search.cfr_type,
        cfr_avg=cfg.search.cfr_avg,
        dcfr_delay=cfg.search.dcfr_plus_delay,
    )

    roots = torch.tensor([0], device=device)
    evaluator.initialize_search(env, roots)
    evaluator.construct_subgame()
    evaluator.initialize_policy_and_beliefs()
    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.warm_start()
    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values

    evaluator.sample_count = 0
    for t in range(cfg.search.warm_start_iterations, cfg.search.iterations):
        evaluator.cfr_iteration(t, training_mode=False)

    # Locate node 5 (at depth 1)
    node5 = 5

    # Locate villain CALL node after hero jams (node 27 in the debugger output)
    B = evaluator.num_actions
    depth1 = evaluator.depth_offsets[1]
    allin_idx = depth1 + 4  # hero ALLIN at depth 1
    call_node = evaluator.depth_offsets[2] + (allin_idx - depth1) * B + 1

    hero_hand = "3s4c"
    hero_cards = _parse_hole_cards_str(hero_hand)
    hero_idx = int(card_utils.combo_index(*hero_cards))

    board = evaluator.env.board_indices[call_node].to(device=device).unsqueeze(0)
    print("Board:", evaluator.env.board_indices[call_node].tolist())
    ranks, _ = rank_hands(board)
    hero_rank = ranks[0, hero_idx].item()

    villain_beliefs = evaluator.beliefs[call_node, 1].clone()
    villain_belief_total = villain_beliefs.sum().item()
    if villain_belief_total > 0:
        villain_beliefs /= villain_belief_total

    # Get beliefs at node 5
    node5_beliefs = evaluator.beliefs[node5, 1].clone()
    node5_belief_total = node5_beliefs.sum().item()
    if node5_belief_total > 0:
        node5_beliefs /= node5_belief_total

    beats_mask = ranks[0] > hero_rank
    tie_mask = ranks[0] == hero_rank
    beats_mass = float(villain_beliefs[beats_mask].sum().item())
    tie_mass = float(villain_beliefs[tie_mask].sum().item())
    lose_mass = 1.0 - beats_mass - tie_mass

    print("=== Node 27 (villain CALL vs hero jam) ===")
    print(f"Belief mass (villain beats 3s4c): {beats_mass:.4f}")
    print(f"Belief mass (villain ties 3s4c):  {tie_mass:.4f}")
    print(f"Belief mass (villain loses):      {lose_mass:.4f}")
    combos = card_utils.hand_combos_tensor(device=device)
    ranks_lookup = "23456789TJQKA"
    probs, indices = torch.topk(villain_beliefs, 10)
    print("Top villain combos at this node:")
    for prob, idx in zip(probs.tolist(), indices.tolist()):
        if prob < 1e-4:
            continue
        c1, c2 = combos[idx].tolist()
        label = f"{ranks_lookup[c1 % 13]}{ranks_lookup[c2 % 13]}"
        outcome = (
            "beat"
            if ranks[0, idx] > hero_rank
            else ("tie" if ranks[0, idx] == hero_rank else "lose")
        )
        print(f"  {label:<4} -> {prob:.4f} ({outcome})")

    # Print top 10 hands that beat 3s4c
    beats_beliefs = villain_beliefs[beats_mask]
    beats_indices = torch.where(beats_mask)[0]
    if len(beats_beliefs) > 0:
        beats_probs, beats_sorted_idx = torch.topk(
            beats_beliefs, min(10, len(beats_beliefs))
        )
        print("\n=== Top 10 hands that beat 3s4c ===")
        for prob, sorted_idx in zip(beats_probs.tolist(), beats_sorted_idx.tolist()):
            idx = beats_indices[sorted_idx].item()
            c1, c2 = combos[idx].tolist()
            label = f"{ranks_lookup[c1 % 13]}{ranks_lookup[c2 % 13]}"
            node5_prob = node5_beliefs[idx].item()
            print(f"  {label:<4} -> Node27: {prob:.4f}, Node5: {node5_prob:.4f}")
    else:
        print("\n=== No hands beat 3s4c ===")

    # Print top 10 hands that tie 3s4c
    tie_beliefs = villain_beliefs[tie_mask]
    tie_indices = torch.where(tie_mask)[0]
    if len(tie_beliefs) > 0:
        tie_probs, tie_sorted_idx = torch.topk(tie_beliefs, min(10, len(tie_beliefs)))
        print("\n=== Top 10 hands that tie 3s4c ===")
        for prob, sorted_idx in zip(tie_probs.tolist(), tie_sorted_idx.tolist()):
            idx = tie_indices[sorted_idx].item()
            c1, c2 = combos[idx].tolist()
            label = f"{ranks_lookup[c1 % 13]}{ranks_lookup[c2 % 13]}"
            node5_prob = node5_beliefs[idx].item()
            print(f"  {label:<4} -> Node27: {prob:.4f}, Node5: {node5_prob:.4f}")
    else:
        print("\n=== No hands tie 3s4c ===")

    # Show the value numbers that appear in the debugger table
    hero_value = evaluator.values_avg[call_node, 0, hero_idx].item()
    table_value = evaluator.values_avg[call_node, 0, :].mean().item()
    print(f"Hero-specific value (3s4c): {hero_value:.6f}")
    print(f"Average value (table entry): {table_value:.6f}")


if __name__ == "__main__":
    run_probe()
