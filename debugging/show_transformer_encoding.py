"""Demonstrate transformer encoding for a scripted heads-up hand.

The script plays a short sequence in the tensor environment, prints the raw
game state, encodes the observation for each player, and then inspects the
resulting token sequence using standalone decoding logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder
from alphaholdem.models.transformer.tokens import Context, Special
from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer


RANKS = "23456789TJQKA"
SUITS = "shdc"
STREET_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
    4: "showdown",
}


def card_to_str(card_idx: int) -> str:
    """Convert a 0-51 card index into rank-suit notation."""

    if card_idx < 0:
        return "--"
    rank = RANKS[card_idx % 13]
    suit = SUITS[card_idx // 13]
    return f"{rank}{suit}"


def describe_action(action_idx: int, bet_bins: List[float], amount: int | None) -> str:
    """Return a readable description for a discrete bet-bin index."""

    if action_idx == 0:
        return "fold"
    if action_idx == 1:
        return "check/call"
    if action_idx == len(bet_bins) + 2:
        return "all-in"
    preset = action_idx - 2
    if 0 <= preset < len(bet_bins):
        postfix = f" (to {amount})" if amount is not None else ""
        return f"raise x{bet_bins[preset]:.2f}{postfix}"
    return f"bin_{action_idx}"


def first_legal_raise(env: HUNLTensorEnv) -> int:
    """Return the first preset raise index that is currently legal."""

    _, mask = env.legal_bins_amounts_and_mask()
    legal_presets = mask[0, 2:-1].nonzero(as_tuple=False)
    if legal_presets.numel() == 0:
        return env.num_bet_bins - 1
    return int(legal_presets[0].item()) + 2


def scripted_hand(
    env: HUNLTensorEnv,
    buffer: VectorizedReplayBuffer | None = None,
    encoder: TransformerStateEncoder | None = None,
) -> List[Dict[str, object]]:
    """Play a short hand and record step-by-step metadata.

    If a transformer replay buffer and encoder are provided, mirror the new
    single-stream token approach by appending non-action tokens on street
    changes and action tokens (context+action for hero, single action for
    opponent) as the hand progresses.
    """

    device = env.device
    env.reset(force_button=torch.tensor([0], device=device))

    if buffer is not None and encoder is not None:
        # Initialize transformer stream with CLS + preflop markers/cards
        buffer.start_adding_trajectory_batches(1)
        init_states = encoder.encode_tensor_states(
            player=0, idxs=torch.tensor([0], device=device)
        )
        buffer.add_tokens(init_states, torch.tensor([0], device=device))
        prev_street = [env.street.clone()]

    log: List[Dict[str, object]] = []

    def take_action(action_idx: int) -> bool:
        actor = int(env.to_act[0].item())
        amounts, mask = env.legal_bins_amounts_and_mask()
        if not mask[0, action_idx].item():
            raise ValueError(
                f"action {action_idx} illegal under mask {mask[0].tolist()}"
            )

        amount_val = (
            int(amounts[0, action_idx].item()) if action_idx not in (0, 1) else None
        )
        reward, done, *_ = env.step_bins(torch.tensor([action_idx], device=device))

        if buffer is not None and encoder is not None:
            if actor == 0:
                # Hero action: append CONTEXT + ACTION and record end
                our_states = encoder.encode_tensor_states(
                    player=0, idxs=torch.tensor([0], device=device)
                )
                buffer.add_transitions(
                    embedding_data=our_states,
                    action_indices=torch.tensor([action_idx], device=device),
                    log_probs=torch.zeros(1, env.num_bet_bins, device=device),
                    rewards=reward[0:1],
                    dones=done[0:1],
                    legal_masks=mask[0:1],
                    delta2=torch.zeros(1, device=device),
                    delta3=torch.zeros(1, device=device),
                    values=torch.zeros(1, device=device),
                    trajectory_indices=torch.tensor([0], device=device),
                )
            else:
                # Opponent action: append single ACTION token without context
                buffer.add_opponent_actions(
                    trajectory_indices=torch.tensor([0], device=device),
                    action_indices=torch.tensor([action_idx], device=device),
                    legal_masks=mask[0:1],
                    streets=prev_street[0][0:1],
                )

            # Append street/card tokens if street advanced
            if (env.street > prev_street[0]).item():
                adv_states = encoder.encode_tensor_states(
                    player=0, idxs=torch.tensor([0], device=device)
                )
                buffer.add_tokens(adv_states, torch.tensor([0], device=device))
                prev_street[0] = env.street.clone()

            # Show direct encodings from both perspectives at this step
            step_idx = len(log)
            print(f"\n=== Direct encodings at step {step_idx} ===")
            for perspective in (0, 1):
                emb_now = encoder.encode_tensor_states(
                    player=perspective, idxs=torch.tensor([0], device=device)
                )
                inspect_embedding(emb_now, perspective, env)

        log.append(
            {
                "step": len(log),
                "actor": actor,
                "action_idx": action_idx,
                "action_desc": describe_action(action_idx, env.bet_bins, amount_val),
                "reward": float(reward[0].item()),
                "done": bool(done[0].item()),
                "street": int(env.street[0].item()),
                "pot": int(env.pot[0].item()),
            }
        )
        return log[-1]["done"]  # type: ignore[return-value]

    # Preflop: hero raises, villain calls.
    take_action(first_legal_raise(env))  # Player 0 raises.
    take_action(1)  # Player 1 calls.

    # Postflop through river: both players check/call to showdown
    # Keep taking the neutral action until the hand ends.
    safety = 100
    while safety > 0 and not bool(env.done[0].item()):
        take_action(1)  # check/call
        safety -= 1

    return log


def print_hand_summary(env: HUNLTensorEnv, log: List[Dict[str, object]]) -> None:
    """Pretty-print the scripted action trace and resulting board state."""

    print("\n=== Scripted hand summary ===")
    for entry in log:
        actor = "Hero (P0)" if entry["actor"] == 0 else "Villain (P1)"
        street = STREET_NAMES.get(entry["street"], str(entry["street"]))
        reward = f"{entry['reward']:+.3f}"
        print(
            f"Step {entry['step']:02d} | {actor:<11} | {entry['action_desc']:<20} "
            f"| street={street:<7} | pot={entry['pot']:4d} | reward={reward}"
        )

    hole = env.hole_indices[0]
    hero_cards = [card_to_str(int(hole[0, i].item())) for i in range(2)]
    villain_cards = [card_to_str(int(hole[1, i].item())) for i in range(2)]
    board_cards = [
        card_to_str(int(idx.item()))
        for idx in env.board_indices[0]
        if int(idx.item()) >= 0
    ]

    print("\nHero hole cards   :", " ".join(hero_cards))
    print("Villain hole cards:", " ".join(villain_cards))
    print("Board cards       :", " ".join(board_cards) if board_cards else "<none>")
    print("Final pot         :", int(env.pot[0].item()))
    print("Winner index      :", int(env.winner[0].item()))


def inspect_embedding(data, perspective: int, env: HUNLTensorEnv) -> None:
    """Decode the transformer tokens without relying on encoder internals."""

    card_offset = Special.NUM_SPECIAL.value
    action_offset = card_offset + 52
    num_bet_bins = env.num_bet_bins

    length = int(data.lengths[0].item())
    bet_bins = env.bet_bins

    print(f"\n--- Encoded tokens for player {perspective} (length={length}) ---")

    for pos in range(length):
        token = int(data.token_ids[0, pos].item())
        parts: List[str] = [f"[{pos:02d}]"]

        if token < card_offset:
            special = Special(token).name
            parts.append(f"special={special}")

            if special == "CLS":
                cls_vec = data.context_features[0, pos]
                parts.append(
                    "cls={sb:.0f} bb={bb:.0f} hero_on_button={btn:.0f}".format(
                        sb=float(cls_vec[0].item()),
                        bb=float(cls_vec[1].item()),
                        btn=float(cls_vec[2].item()),
                    )
                )
            elif special == "CONTEXT":
                ctx_vec = data.context_features[0, pos]
                non_zero = {
                    Context(idx).name.lower(): float(ctx_vec[idx].item())
                    for idx in range(Context.NUM_CONTEXT.value)
                    if abs(float(ctx_vec[idx].item())) > 1e-6
                }
                parts.append(f"context={non_zero}")

        elif token < action_offset:
            card_idx = token - card_offset
            street_idx = int(data.card_streets[0, pos].item())
            street = STREET_NAMES.get(street_idx, str(street_idx))
            parts.extend(
                [
                    "type=card",
                    f"card={card_to_str(card_idx)}",
                    f"street={street}",
                ]
            )

        elif token < action_offset + num_bet_bins:
            action_id = token - action_offset
            street_idx = int(data.action_streets[0, pos].item())
            street = STREET_NAMES.get(street_idx, str(street_idx))
            actor = int(data.action_actors[0, pos].item())
            legal_mask = data.action_legal_masks[0, pos]
            legal_bins = [i for i, flag in enumerate(legal_mask.tolist()) if flag]
            parts.extend(
                [
                    "type=action",
                    f"action={describe_action(action_id, bet_bins, None)}",
                    f"actor={actor}",
                    f"street={street}",
                    f"legal={legal_bins}",
                ]
            )

        else:
            parts.append(f"token={token}")

        print(" ".join(parts))


def main() -> None:
    device = torch.device("cpu")
    bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        bet_bins=bet_bins,
        device=device,
    )
    encoder = TransformerStateEncoder(env, device)
    # Create transformer-mode replay buffer
    seq_len = encoder.get_sequence_length()
    buffer = VectorizedReplayBuffer(
        capacity=4,
        max_trajectory_length=16,
        num_bet_bins=len(bet_bins) + 3,
        device=device,
        is_transformer=True,
        max_sequence_length=seq_len,
    )

    action_log = scripted_hand(env, buffer=buffer, encoder=encoder)
    print_hand_summary(env, action_log)

    # Finish and display the token stream and ends
    buffer.finish_adding_trajectory_batches()
    pos = int(buffer.token_positions[0].item())
    print(f"\nReplay token stream length={pos}")
    print("Token IDs:", buffer.token_ids[0, :pos].tolist())
    print(
        "Transition ends:", buffer.transition_token_ends[0, : len(action_log)].tolist()
    )

    # Show reconstructed prefix for each transition
    step_indices = torch.arange(len(action_log), device=device)
    data = buffer._sample_transformer_steps(
        torch.tensor([0] * len(action_log), device=device), step_indices
    )
    for i in range(len(action_log)):
        print(f"\n--- Tokens up to transition {i} ---")
        sub = StructuredView(data, i)
        # Reuse existing printer for a single row by wrapping
        tmp = type("obj", (), {})()
        tmp.token_ids = data.token_ids[i : i + 1]
        tmp.card_streets = data.card_streets[i : i + 1]
        tmp.action_streets = data.action_streets[i : i + 1]
        tmp.action_actors = data.action_actors[i : i + 1]
        tmp.action_legal_masks = data.action_legal_masks[i : i + 1]
        tmp.context_features = data.context_features[i : i + 1]
        tmp.lengths = data.lengths[i : i + 1]
        inspect_embedding(tmp, 0, env)

    # Also show full encoding from player 1 perspective
    print("\n=== Full encoding from player 1 perspective ===")
    emb_p1 = encoder.encode_tensor_states(
        player=1, idxs=torch.tensor([0], device=device)
    )
    inspect_embedding(emb_p1, 1, env)


class StructuredView:
    def __init__(self, data, i):
        self.data = data
        self.i = i


if __name__ == "__main__":
    main()
