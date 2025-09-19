"""Inspect transformer tokenization by hooking into SelfPlayTrainer.

Run with:

    python debugging/inspect_transformer_sequence.py --actions 2,1,0

This will run a single iteration of SelfPlayTrainer and hook into the model
to see what exactly gets passed in as input data.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.tokens import Context, Special, Cls
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.core.structured_config import (
    Config,
    TrainingConfig,
    ModelConfig,
    EnvConfig,
)


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
    """Pretty print token sequence data."""
    encoder = type(data)  # Get the class for static methods
    card_offset = encoder.get_card_token_offset(len(bet_bins) + 3)
    action_offset = encoder.get_action_token_offset(len(bet_bins) + 3)
    special_offset = encoder.get_special_token_offset(len(bet_bins) + 3)

    print(f"\n--- Token sequence for {title} (length={int(data.lengths[0])}) ---")
    for pos in range(int(data.lengths[0])):
        token = int(data.token_ids[0, pos].item())
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


def hook_model_forward(model, original_forward):
    """Hook into model forward to inspect input data."""

    def hooked_forward(embedding_data):
        print("\n=== MODEL INPUT INSPECTION ===")
        print(f"Input type: {type(embedding_data)}")
        print(
            f"Input shape: {embedding_data.token_ids.shape if hasattr(embedding_data, 'token_ids') else 'N/A'}"
        )

        if hasattr(embedding_data, "token_ids"):
            print(f"Token IDs shape: {embedding_data.token_ids.shape}")
            print(f"Lengths: {embedding_data.lengths.tolist()}")

            # Parse and display token sequences for each sample
            for batch_idx in range(
                min(3, embedding_data.token_ids.shape[0])
            ):  # Show first 3 samples
                seq_len = int(embedding_data.lengths[batch_idx])
                print(f"\n--- Sample {batch_idx} (length={seq_len}) ---")

                tokens = embedding_data.token_ids[batch_idx, :seq_len]
                for pos, token in enumerate(tokens):
                    token = int(token.item())
                    parts = [f"[{pos:02d}]"]

                    # Decode token based on type
                    if token == Special.CLS.value:
                        parts.append("CLS")
                        if hasattr(embedding_data, "context_features"):
                            ctx = embedding_data.context_features[batch_idx, pos]
                            parts.append(f"sb={ctx[Cls.SB.value]:.0f}")
                            parts.append(f"bb={ctx[Cls.BB.value]:.0f}")
                            parts.append(
                                f"hero_button={ctx[Cls.HERO_ON_BUTTON.value]:.0f}"
                            )

                    elif token == Special.CONTEXT.value:
                        parts.append("CONTEXT")
                        if hasattr(embedding_data, "context_features"):
                            ctx = embedding_data.context_features[batch_idx, pos]
                            parts.append(f"pot={ctx[Context.POT.value]:.0f}")
                            parts.append(f"stack_p0={ctx[Context.STACK_P0.value]:.0f}")
                            parts.append(f"stack_p1={ctx[Context.STACK_P1.value]:.0f}")
                            parts.append(f"street={ctx[Context.STREET.value]:.0f}")

                    elif token in [
                        Special.STREET_PREFLOP.value,
                        Special.STREET_FLOP.value,
                        Special.STREET_TURN.value,
                        Special.STREET_RIVER.value,
                    ]:
                        street_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
                        street_idx = token - Special.STREET_PREFLOP.value
                        parts.append(f"STREET_{street_names[street_idx]}")

                    elif (
                        Special.NUM_SPECIAL.value
                        <= token
                        < Special.NUM_SPECIAL.value + 52
                    ):
                        # Card token
                        card_idx = token - Special.NUM_SPECIAL.value
                        parts.append(f"CARD_{format_card(card_idx)}")
                        if hasattr(embedding_data, "card_streets"):
                            street = int(
                                embedding_data.card_streets[batch_idx, pos].item()
                            )
                            parts.append(f"street={street}")

                    elif (
                        Special.NUM_SPECIAL.value + 52
                        <= token
                        < Special.NUM_SPECIAL.value + 52 + 7
                    ):
                        # Action token
                        action_id = token - Special.NUM_SPECIAL.value - 52
                        parts.append(
                            f"ACTION_{describe_action(action_id, [0.5, 1.0, 1.5, 2.0])}"
                        )
                        if hasattr(embedding_data, "action_actors"):
                            actor = int(
                                embedding_data.action_actors[batch_idx, pos].item()
                            )
                            parts.append(f"actor={actor}")
                        if hasattr(embedding_data, "action_streets"):
                            street = int(
                                embedding_data.action_streets[batch_idx, pos].item()
                            )
                            parts.append(f"street={street}")

                    else:
                        parts.append(f"UNKNOWN_TOKEN_{token}")

                    print(" ".join(parts))

            if hasattr(embedding_data, "context_features"):
                print(
                    f"\nContext features shape: {embedding_data.context_features.shape}"
                )

        # Call original forward
        result = original_forward(embedding_data)

        print(f"\nOutput keys: {list(result.keys())}")
        if "policy_logits" in result:
            print(f"Policy logits shape: {result['policy_logits'].shape}")
        if "value" in result:
            print(f"Value shape: {result['value'].shape}")

        return result

    return hooked_forward


def parse_actions(arg: str) -> List[int]:
    if not arg:
        return []
    return [int(item.strip()) for item in arg.split(",") if item.strip()]


def main(actions: Iterable[int]) -> None:
    device = torch.device("cpu")

    # Create config manually
    config = Config(
        use_wandb=False,
        num_envs=8,
        train=TrainingConfig(
            batch_size=64,
            num_epochs=4,
            max_trajectory_length=10,
            learning_rate=1e-4,
        ),
        model=ModelConfig(
            name="poker_transformer_v1",
            kwargs={
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 4,
                "num_bet_bins": 7,  # len(bet_bins) + 3
            },
        ),
        env=EnvConfig(
            stack=20000,
            bet_bins=[0.5, 1.0, 1.5, 2.0],
            sb=50,
            bb=100,
        ),
        device="cpu",
    )

    # Add missing field that SelfPlayTrainer expects
    config.train.max_sequence_length = 50

    # Create trainer
    trainer = SelfPlayTrainer(config, device)

    # Hook into the model to see what gets passed in
    original_forward = trainer.model.forward
    trainer.model.forward = hook_model_forward(trainer.model, original_forward)

    print("Starting SelfPlayTrainer iteration...")
    print(f"Actions to take: {list(actions)}")

    # Run one iteration of trajectory collection
    try:
        # This will collect trajectories and we'll see what gets passed to the model
        trainer.collect_tensor_trajectories(min_trajectories=1)
        print("\n=== TRAJECTORY COLLECTION COMPLETED ===")
    except Exception as e:
        print(f"Error during collection: {e}")
        import traceback

        traceback.print_exc()


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
