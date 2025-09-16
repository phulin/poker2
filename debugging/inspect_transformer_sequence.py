"""Inspect transformer tokenization for a simple hand history.

Run with:

    python debugging/inspect_transformer_sequence.py --actions 2,1,0

This will step a single environment through the provided discrete bet bins,
print the tensor environment state after every action, and then pretty-print
the resulting transformer token sequences for both players.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder
from alphaholdem.models.transformer.tokens import Context, Special


RANK_STR = "AKQJT98765432"[::-1]  # -> "23456789TJQKA"
SUIT_STR = "shdc"


def format_card(card_idx: int) -> str:
    if card_idx < 0:
        return "--"
    rank = RANK_STR[card_idx % 13]
    suit = SUIT_STR[card_idx // 13]
    return f"{rank}{suit}"


def describe_action(action_id: int, bet_bins: List[float]) -> str:
    if action_id == 0:
        return "fold"
    if action_id == 1:
        return "check/call"
    if action_id == len(bet_bins) + 2:
        return "all-in"
    idx = action_id - 2
    if 0 <= idx < len(bet_bins):
        return f"raise x{bet_bins[idx]:.2f}"
    return f"bet_bin_{action_id}"


def print_environment_state(env: HUNLTensorEnv, step_idx: int) -> None:
    print("\n=== Environment after step", step_idx, "===")
    print("Street:", int(env.street[0].item()))
    print("To act:", int(env.to_act[0].item()))
    print(
        "Stacks:",
        [int(env.stacks[0, p].item()) for p in range(2)],
        "Committed:",
        [int(env.committed[0, p].item()) for p in range(2)],
        "Pot:",
        int(env.pot[0].item()),
    )
    hole = env.hole_indices[0]
    print(
        "Hole cards:",
        [format_card(int(hole[p, c].item())) for p in range(2) for c in range(2)],
    )
    board = [format_card(int(card.item())) for card in env.board_indices[0]]
    print("Board:", board)


def make_pretty_tokens(data, bet_bins: List[float], title: str) -> None:
    encoder = TransformerStateEncoder
    card_offset = encoder.get_card_token_offset(len(bet_bins) + 3)
    action_offset = encoder.get_action_token_offset(len(bet_bins) + 3)
    special_offset = encoder.get_special_token_offset(len(bet_bins) + 3)

    print(f"\n--- Token sequence for {title} (length={int(data.lengths[0])}) ---")
    for pos, token in enumerate(data.token_ids[0]):
        if token < 0:
            break
        token = int(token.item())
        parts: List[str] = [f"[{pos:02d}]"]
        if special_offset <= token < special_offset + Special.NUM_SPECIAL.value:
            special_name = Special(token - special_offset).name
            parts.append(f"special={special_name}")
            if special_name == "CONTEXT":
                ctx_vec = data.context_features[0, pos]
                non_zero = {
                    Context(idx).name.lower(): float(ctx_vec[idx].item())
                    for idx in range(len(Context))
                    if idx < Context.NUM_CONTEXT.value
                    and abs(float(ctx_vec[idx].item())) > 1e-6
                }
                parts.append(f"context={non_zero}")
        elif card_offset <= token < card_offset + 52:
            parts.append("type=card")
            parts.append(f"card={format_card(token - card_offset)}")
            parts.append(f"street={int(data.card_streets[0, pos].item())}")
        elif action_offset <= token < action_offset + len(bet_bins) + 3:
            parts.append("type=action")
            action_id = token - action_offset
            parts.append(f"action={describe_action(action_id, bet_bins)}")
            parts.append(f"actor={int(data.action_actors[0, pos].item())}")
            parts.append(f"street={int(data.action_streets[0, pos].item())}")
            legal_mask = data.action_legal_masks[0, pos]
            legal_bins = [i for i, flag in enumerate(legal_mask.tolist()) if flag]
            parts.append(f"legal={legal_bins}")
        else:
            parts.append(f"token_id={token}")

        print(" ".join(parts))


def parse_actions(arg: str) -> List[int]:
    if not arg:
        return []
    return [int(item.strip()) for item in arg.split(",") if item.strip()]


def main(actions: Iterable[int]) -> None:
    device = torch.device("cpu")
    bet_bins = [0.5, 1.0, 1.5, 2.0]
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=20000,
        sb=50,
        bb=100,
        bet_bins=bet_bins,
        device=device,
    )
    env.reset()

    print("Initial environment ready. Starting sequence...")
    print_environment_state(env, step_idx=0)

    for idx, action in enumerate(actions, start=1):
        current_player = int(env.to_act[0].item())
        print(f"\n--> Step {idx}: player {current_player} takes bin {action}")
        env.step_bins(torch.tensor([action], device=device))
        print_environment_state(env, step_idx=idx)

    encoder = TransformerStateEncoder(env, device)
    idx_tensor = torch.tensor([0], device=device)

    player0 = encoder.encode_tensor_states(player=0, idxs=idx_tensor)
    player1 = encoder.encode_tensor_states(player=1, idxs=idx_tensor)

    make_pretty_tokens(player0, bet_bins, title="player 0 perspective")
    make_pretty_tokens(player1, bet_bins, title="player 1 perspective")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug transformer tokenization for a scripted hand."
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="2,1",
        help="Comma-separated list of discrete bet bin indices to play sequentially.",
    )
    args = parser.parse_args()
    action_list = parse_actions(args.actions)
    main(action_list)
