from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from p2.env.card_utils import NUM_HANDS, combo_lookup_tensor, combo_to_onehot_tensor
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator

from split_reference import _load_split_model


DEFAULT_FORCE_DECK = [12, 25, 38, 51, 0, 13, 26, 1, 14]


def _combo_index(card_a: int, card_b: int) -> int:
    return int(combo_lookup_tensor()[card_a, card_b].item())


def _force_deck(
    hero_player: int | None,
    hero_cards: tuple[int, int] | None,
) -> list[int]:
    if hero_player is None or hero_cards is None:
        return DEFAULT_FORCE_DECK

    private_cards = DEFAULT_FORCE_DECK[:4]
    offset = hero_player * 2
    private_cards[offset] = hero_cards[0]
    private_cards[offset + 1] = hero_cards[1]

    deck: list[int] = []
    blocked: set[int] = set()
    priority = [*private_cards, *DEFAULT_FORCE_DECK]
    for card in priority:
        if card in blocked:
            continue
        deck.append(card)
        blocked.add(card)
        if len(deck) == 4:
            break
    for card in [*DEFAULT_FORCE_DECK, *range(52)]:
        if card in blocked:
            continue
        deck.append(card)
        blocked.add(card)
        if len(deck) >= 9:
            break
    return deck


def _make_env(
    stack: int,
    sb: int,
    bb: int,
    bet_bins: list[float],
    force_button: int,
    force_deck: list[int],
    flop_showdown: bool,
    device: torch.device,
) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=stack,
        sb=sb,
        bb=bb,
        default_bet_bins=bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=flop_showdown,
    )
    env.reset(
        force_button=torch.tensor([force_button], dtype=torch.long, device=device),
        force_deck=torch.tensor([force_deck], dtype=torch.long, device=device),
    )
    return env


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part]


def _public_beliefs(env: HUNLTensorEnv, mode: str, device: torch.device) -> torch.Tensor:
    board_cards = env.board_indices[0]
    board_mask = torch.zeros(52, device=device, dtype=torch.float32)
    public_cards = board_cards[board_cards >= 0]
    if public_cards.numel() > 0:
        board_mask[public_cards.long()] = 1.0
    allowed = (combo_to_onehot_tensor(device=device).float() @ board_mask) < 0.5
    beliefs = torch.zeros((1, 2, NUM_HANDS), device=device, dtype=torch.float32)
    if mode == "uniform":
        weights = allowed.float()
        weights = weights / weights.sum().clamp_min(1.0)
        beliefs[0, 0] = weights
        beliefs[0, 1] = weights
        return beliefs
    if mode == "skewed":
        hand_ids = torch.arange(NUM_HANDS, device=device, dtype=torch.float32)
        p0 = torch.where(allowed, 1.0 + (hand_ids.remainder(17.0) / 17.0), 0.0)
        p1 = torch.where(allowed, 1.0 + (hand_ids.remainder(23.0) / 23.0), 0.0)
        beliefs[0, 0] = p0 / p0.sum().clamp_min(1.0)
        beliefs[0, 1] = p1 / p1.sum().clamp_min(1.0)
        return beliefs
    raise ValueError(f"unknown belief mode {mode!r}")


def build_cfr_fixture(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu")
    model, cfg = _load_split_model(args.snapshot, device)
    cfg = copy.deepcopy(cfg)

    cfg.device = "cpu"
    cfg.num_envs = 1
    cfg.env.stack = args.stack
    cfg.env.sb = args.sb
    cfg.env.bb = args.bb
    cfg.search.depth = args.depth
    cfg.search.iterations = args.iterations
    cfg.search.iterations_final = None
    cfg.search.sparse = True
    cfg.search.sparse_fused = False
    cfg.search.cfr_avg = args.cfr_avg
    if args.disable_allin:
        cfg.search.allin_by_depth = [False] * len(cfg.search.bet_bins_by_depth)
    cfg.search.preflop_allin_table_path = str(args.preflop_allin_table)

    hero_cards = (args.hero_card0, args.hero_card1) if args.hero_player >= 0 else None
    hero_player = args.hero_player if args.hero_player >= 0 else None
    force_deck = _force_deck(hero_player, hero_cards)
    if args.force_deck:
        force_deck = _parse_int_list(args.force_deck)
    env = _make_env(
        stack=args.stack,
        sb=args.sb,
        bb=args.bb,
        bet_bins=list(cfg.env.bet_bins),
        force_button=args.force_button,
        force_deck=force_deck,
        flop_showdown=cfg.env.flop_showdown,
        device=device,
    )
    for action in _parse_int_list(args.actions):
        env.step_bins(torch.tensor([action], dtype=torch.long, device=device))
    beliefs = _public_beliefs(env, args.belief_mode, device)

    evaluator = SparseCFREvaluator(
        model=model,
        device=device,
        cfg=cfg,
        generator=torch.Generator(device=device).manual_seed(cfg.seed),
    )
    evaluator.initialize_subgame(
        env,
        torch.tensor([0], dtype=torch.long, device=device),
        beliefs,
    )
    evaluator.evaluate_cfr(training_mode=False)

    num_actions = evaluator.num_actions
    start = evaluator.depth_offsets[1]
    end = evaluator.depth_offsets[2]
    child_indices = torch.arange(start, end, device=device)
    action_ids = evaluator.action_from_parent[child_indices]
    root_policy = torch.zeros(
        num_actions,
        NUM_HANDS,
        device=device,
        dtype=torch.float32,
    )
    for child_index, action_id in zip(child_indices, action_ids):
        action = int(action_id.item())
        if 0 <= action < num_actions:
            root_policy[action] = evaluator.policy_probs_avg[child_index]
    root_policy /= root_policy.sum(dim=0, keepdim=True).clamp_min(1e-12)

    actor = int(evaluator.env.to_act[0].item())
    action_probs = (root_policy * evaluator.beliefs[0, actor][None, :]).sum(dim=1)
    beliefs_after = None
    if args.selected_action >= 0:
        action = args.selected_action
        updated = evaluator.beliefs[0].clone()
        updated_actor = updated[actor] * root_policy[action]
        denom = updated_actor.sum()
        if denom > 1.0e-12:
            updated[actor] = updated_actor / denom
        beliefs_after = updated.float().reshape(-1).tolist()
    hands = {
        "AsKd": _combo_index(51, 24),
        "AsAh": _combo_index(51, 38),
        "2c7d": _combo_index(0, 18),
        "QsJh": _combo_index(49, 35),
    }
    return {
        "schemaVersion": 1,
        "source": "p2.website.split_cfr_reference",
        "snapshot": str(args.snapshot),
        "numHands": NUM_HANDS,
        "numActions": num_actions,
        "forceButton": args.force_button,
        "forceDeck": force_deck,
        "iterations": args.iterations,
        "depth": args.depth,
        "cfrAvg": args.cfr_avg,
        "actions": _parse_int_list(args.actions),
        "beliefMode": args.belief_mode,
        "selectedAction": args.selected_action,
        "actionProbs": action_probs.float().tolist(),
        "beliefsAfter": beliefs_after,
        "hands": {
            label: {
                "index": index,
                "policy": root_policy[:, index].float().tolist(),
            }
            for label, index in hands.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--preflop-allin-table", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--force-button", type=int, default=0)
    parser.add_argument("--stack", type=int, default=400)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--hero-player", type=int, default=0)
    parser.add_argument("--hero-card0", type=int, default=51)
    parser.add_argument("--hero-card1", type=int, default=24)
    parser.add_argument("--force-deck", type=str)
    parser.add_argument("--actions", type=str)
    parser.add_argument("--belief-mode", choices=("uniform", "skewed"), default="uniform")
    parser.add_argument("--selected-action", type=int, default=-1)
    parser.add_argument("--disable-allin", action="store_true")
    parser.add_argument("--cfr-avg", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    fixture = build_cfr_fixture(args)
    text = json.dumps(fixture, separators=(",", ":"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
