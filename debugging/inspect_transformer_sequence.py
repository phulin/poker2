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
from alphaholdem.models.transformer.tokens import (
    Context,
    Special,
    Cls,
    get_action_token_id_offset,
    get_card_token_id_offset,
)
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
            parts.append(f"street={int(data.token_streets[0, pos].item())}")
        elif action_offset <= token < action_offset + len(bet_bins) + 3:
            parts.append("type=action")
            action_id = token - action_offset
            parts.append(f"action={describe_action(action_id, bet_bins)}")
            parts.append(f"actor={int(data.action_actors[0, pos].item())}")
            parts.append(f"street={int(data.token_streets[0, pos].item())}")
            legal_mask = data.action_legal_masks[0, pos]
            legal_bins = [i for i, flag in enumerate(legal_mask.tolist()) if flag]
            parts.append(f"legal={legal_bins}")
        else:
            parts.append(f"token_id={token}")

        print(" ".join(parts))


def hook_model_forward(model, original_forward):
    """Hook into model forward to inspect input data."""

    def hooked_forward(embedding_data, kv_cache=None):
        print("\n=== MODEL INPUT INSPECTION ===")

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
                                f"hero_position={ctx[Cls.HERO_POSITION.value]:.0f}"
                            )

                    elif token == Special.CONTEXT.value:
                        parts.append("CONTEXT")
                        if hasattr(embedding_data, "context_features"):
                            ctx = embedding_data.context_features[batch_idx, pos]
                            parts.append(f"pot={ctx[Context.POT.value]:.0f}")
                            parts.append(f"stack_p0={ctx[Context.STACK_P0.value]:.0f}")
                            parts.append(f"stack_p1={ctx[Context.STACK_P1.value]:.0f}")
                            parts.append(
                                f"bet_to_call={ctx[Context.BET_TO_CALL.value]:.0f}"
                            )

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
                        if hasattr(embedding_data, "token_streets"):
                            street = int(
                                embedding_data.token_streets[batch_idx, pos].item()
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
                        if hasattr(embedding_data, "token_streets"):
                            street = int(
                                embedding_data.token_streets[batch_idx, pos].item()
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
        if kv_cache is not None:
            result = original_forward(embedding_data, kv_cache=kv_cache)
        else:
            result = original_forward(embedding_data)

        return result

    return hooked_forward


def hook_policy_action_batch(policy, original_action_batch, forced_actions):
    """Hook into policy action_batch to inspect action selection."""

    def hooked_action_batch(logits, legal_masks=None):
        action_bins, log_probs = original_action_batch(logits, legal_masks)
        if len(forced_actions) > 0:
            action_bins[0] = forced_actions.pop(0)
        return action_bins, log_probs

    return hooked_action_batch


def hook_replay_buffer_add_transitions(replay_buffer, original_add_transitions):
    """Hook into replay buffer add_transitions to inspect transition data."""

    def hooked_add_transitions(
        embedding_data,
        action_indices,
        log_probs,
        rewards,
        dones,
        legal_masks,
        delta2,
        delta3,
        values,
        trajectory_indices,
    ):
        print("\n=== REPLAY BUFFER ADD TRANSITIONS ===")
        print(f"Batch size: {action_indices.shape[0]}")
        print(f"Action indices: {action_indices.tolist()}")
        print(f"Rewards: {rewards.tolist()}")
        print(f"Dones: {dones.tolist()}")
        print(f"Trajectory indices: {trajectory_indices.tolist()}")

        # Show embedding data info if it's structured
        if hasattr(embedding_data, "lengths"):
            print(f"Token sequence lengths: {embedding_data.lengths.tolist()}")
            print(f"Token sequence indices being updated:")
            for i, traj_idx in enumerate(trajectory_indices):
                length = int(embedding_data.lengths[i].item())
                print(
                    f"  Trajectory {traj_idx.item()}: length {length}, action {action_indices[i].item()}"
                )

        # Call original method
        return original_add_transitions(
            embedding_data,
            action_indices,
            log_probs,
            rewards,
            dones,
            legal_masks,
            delta2,
            delta3,
            values,
            trajectory_indices,
        )

    return hooked_add_transitions


def hook_replay_buffer_update_opponent_rewards(
    replay_buffer, original_update_opponent_rewards
):
    """Hook into replay buffer update_opponent_rewards to inspect reward updates."""

    def hooked_update_opponent_rewards(trajectory_indices, opponent_rewards):
        print("\n=== UPDATE LAST TRANSITION REWARDS ===")
        print(f"Trajectory indices: {trajectory_indices.tolist()}")
        formatted_rewards = [f"{r:.3f}" for r in opponent_rewards.tolist()]
        print(f"Opponent rewards: [{', '.join(formatted_rewards)}]")

        # Show current trajectory lengths before update
        if hasattr(replay_buffer, "trajectory_lengths"):
            buffer_indices = (
                trajectory_indices + replay_buffer.position
            ) % replay_buffer.capacity

            # Show step positions
            step_positions = replay_buffer.current_transition_counts[buffer_indices]
            print(f"Current step positions: {step_positions.tolist()}")

        # Call original method
        return original_update_opponent_rewards(trajectory_indices, opponent_rewards)

    return hooked_update_opponent_rewards


def print_replay_buffer_summary(replay_buffer):
    """Print a comprehensive summary of all replay buffer fields."""
    print("\n" + "=" * 60)
    print("REPLAY BUFFER SUMMARY")
    print("=" * 60)

    print(f"Buffer capacity: {replay_buffer.capacity}")
    print(f"Buffer size: {replay_buffer.size}")
    print(f"Position: {replay_buffer.position}")
    print(f"Max trajectory length: {replay_buffer.max_trajectory_length}")
    print(f"Max sequence length: {replay_buffer.max_sequence_length}")
    print(f"Is transformer: {replay_buffer.is_transformer}")
    print(f"Number of bet bins: {replay_buffer.num_bet_bins}")
    print(f"Total steps: {replay_buffer.num_steps()}")

    print(
        f"\nTrajectory lengths: {replay_buffer.trajectory_lengths[:replay_buffer.size].tolist()}"
    )
    print(
        f"Current step positions: {replay_buffer.current_transition_counts[:replay_buffer.size].tolist()}"
    )

    if replay_buffer.size > 0:
        print(f"\n--- TRANSITION DATA ---")
        valid_indices = torch.where(replay_buffer.trajectory_lengths > 0)[0]
        if len(valid_indices) > 0:
            max_length = replay_buffer.trajectory_lengths[valid_indices].max()
            print(f"Max trajectory length in buffer: {max_length}")

            # Show action indices for first few trajectories
            print(f"\nAction indices:")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                actions = replay_buffer.action_indices[traj_idx, :length].tolist()
                print(f"  Trajectory {traj_idx}: {actions}")

            # Show rewards for first few trajectories
            print(f"\nRewards:")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                rewards = replay_buffer.rewards[traj_idx, :length].tolist()
                formatted_rewards = [f"{r:.3f}" for r in rewards]
                print(f"  Trajectory {traj_idx}: [{', '.join(formatted_rewards)}]")

            # Show done flags for first few trajectories
            print(f"\nDone flags:")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                dones = replay_buffer.dones[traj_idx, :length].tolist()
                print(f"  Trajectory {traj_idx}: {dones}")

            # Show values for first few trajectories
            print(f"\nValues:")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                values = replay_buffer.values[traj_idx, :length].tolist()
                formatted_values = [f"{v:.3f}" for v in values]
                print(f"  Trajectory {traj_idx}: [{', '.join(formatted_values)}]")

            # Show advantages and returns if computed
            if replay_buffer.advantages.sum() != 0:
                print(f"\nAdvantages:")
                for i in range(min(3, len(valid_indices))):
                    traj_idx = valid_indices[i]
                    length = replay_buffer.trajectory_lengths[traj_idx]
                    advantages = replay_buffer.advantages[traj_idx, :length].tolist()
                    formatted_advantages = [f"{a:.3f}" for a in advantages]
                    print(
                        f"  Trajectory {traj_idx}: [{', '.join(formatted_advantages)}]"
                    )

                print(f"\nReturns:")
                for i in range(min(3, len(valid_indices))):
                    traj_idx = valid_indices[i]
                    length = replay_buffer.trajectory_lengths[traj_idx]
                    returns = replay_buffer.returns[traj_idx, :length].tolist()
                    formatted_returns = [f"{r:.3f}" for r in returns]
                    print(f"  Trajectory {traj_idx}: [{', '.join(formatted_returns)}]")

            # Show delta2 and delta3 for first few trajectories
            print(f"\nDelta2:")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                delta2 = replay_buffer.delta2[traj_idx, :length].tolist()
                formatted_delta2 = [f"{d:.3f}" for d in delta2]
                print(f"  Trajectory {traj_idx}: [{', '.join(formatted_delta2)}]")

            print(f"\nDelta3:")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                delta3 = replay_buffer.delta3[traj_idx, :length].tolist()
                formatted_delta3 = [f"{d:.3f}" for d in delta3]
                print(f"  Trajectory {traj_idx}: [{', '.join(formatted_delta3)}]")

            # Show legal masks for first trajectory
            print(f"\nLegal masks (first trajectory):")
            if len(valid_indices) > 0:
                traj_idx = valid_indices[0]
                length = replay_buffer.trajectory_lengths[traj_idx]
                legal_masks = replay_buffer.legal_masks[traj_idx, :length]
                for step in range(length):
                    legal_actions = torch.where(legal_masks[step])[0].tolist()
                    print(f"  Step {step}: legal actions {legal_actions}")

            # Show log probabilities for first trajectory
            print(f"\nLog probabilities (first trajectory):")
            if len(valid_indices) > 0:
                traj_idx = valid_indices[0]
                length = replay_buffer.trajectory_lengths[traj_idx]
                log_probs = replay_buffer.log_probs[traj_idx, :length]
                for step in range(length):
                    probs = log_probs[step].exp().tolist()
                    formatted_probs = [f"{p:.3f}" for p in probs]
                    print(f"  Step {step}: [{', '.join(formatted_probs)}]")

        if replay_buffer.is_transformer:
            print(f"\n--- TRANSFORMER-SPECIFIC DATA ---")
            print(
                f"Token positions: {replay_buffer.current_token_positions[:replay_buffer.size].tolist()}"
            )

            # Show transition token ends for first few trajectories
            print(f"\nTransition token ends:")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                token_ends = replay_buffer.transition_token_ends[
                    traj_idx, :length
                ].tolist()
                print(f"  Trajectory {traj_idx}: {token_ends}")

            # Show token sequence fields for valid tokens (token_ids >= 0)
            print(f"\n--- TOKEN SEQUENCE DATA ---")
            for i in range(min(3, len(valid_indices))):
                traj_idx = valid_indices[i]
                token_pos = replay_buffer.current_token_positions[traj_idx]

                if token_pos > 0:
                    print(f"\nTrajectory {traj_idx} (token_pos={token_pos}):")

                    # Get valid token indices (where token_ids >= 0)
                    valid_tokens = replay_buffer.data.token_ids[traj_idx] >= 0
                    valid_token_indices = torch.where(valid_tokens)[0]

                    if len(valid_token_indices) > 0:
                        # Token IDs with transition boundaries
                        token_ids = replay_buffer.data.token_ids[
                            traj_idx, valid_token_indices
                        ].tolist()
                        print(f"  Token IDs: {token_ids}")

                        # Show transition boundaries
                        length = replay_buffer.trajectory_lengths[traj_idx]
                        token_ends = replay_buffer.transition_token_ends[
                            traj_idx, :length
                        ].tolist()
                        print(f"  Transition token ends: {token_ends}")

                        # Create a visual representation with boundaries
                        if len(token_ends) > 0:
                            print(f"  Token sequence with transition boundaries:")
                            # Create one-line representation with | separators
                            boundary_idx = 0
                            line_parts = []
                            current_transition = []

                            for i, token_id in enumerate(token_ids):
                                current_transition.append(f"{token_id}")

                                if (
                                    boundary_idx < len(token_ends)
                                    and i == token_ends[boundary_idx]
                                ):
                                    # End of transition - add to line with | separator
                                    transition_str = " ".join(current_transition)
                                    line_parts.append(f"[{transition_str}]")
                                    current_transition = []
                                    boundary_idx += 1

                            # Add any remaining tokens
                            if current_transition:
                                transition_str = " ".join(current_transition)
                                line_parts.append(f"[{transition_str}]")

                            # Print the one-line representation
                            print(f"    {' | '.join(line_parts)}")

                        # Card information
                        card_ranks = replay_buffer.data.card_ranks[
                            traj_idx, valid_token_indices
                        ].tolist()
                        card_suits = replay_buffer.data.card_suits[
                            traj_idx, valid_token_indices
                        ].tolist()
                        token_streets = replay_buffer.data.token_streets[
                            traj_idx, valid_token_indices
                        ].tolist()
                        print(f"  Card ranks: {card_ranks}")
                        print(f"  Card suits: {card_suits}")
                        print(f"  Token streets: {token_streets}")

                        # Action information
                        action_actors = replay_buffer.data.action_actors[
                            traj_idx, valid_token_indices
                        ].tolist()
                        print(f"  Action actors: {action_actors}")

                        # Context features (show for context tokens only)
                        context_features = replay_buffer.data.context_features[
                            traj_idx, valid_token_indices
                        ]
                        print(f"  Context features shape: {context_features.shape}")

                        # Show context features for context tokens (token_id == 1) as a table
                        context_token_mask = (
                            torch.tensor(token_ids) == Special.CONTEXT.value
                        )
                        if context_token_mask.any():
                            context_token_indices = torch.where(context_token_mask)[0]
                            print(
                                f"  Context tokens found at positions: {context_token_indices.tolist()}"
                            )

                            # Create table header
                            print(f"  Context features table:")
                            print(
                                f"    {'Pos':<4} {'POT':<8} {'STACK_P0':<8} {'STACK_P1':<8} {'COMM_P0':<8} {'COMM_P1':<8} {'POS':<4} {'ACT_ROUND':<9} {'MIN_RAISE':<9} {'BET_CALL':<8}"
                            )
                            print(
                                f"    {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4} {'-'*9} {'-'*9} {'-'*8}"
                            )

                            for j in context_token_indices:
                                ctx = context_features[j].tolist()
                                print(
                                    f"    {j:<4} "
                                    f"{ctx[Context.POT.value]:<8.3f} "
                                    f"{ctx[Context.STACK_P0.value]:<8.3f} "
                                    f"{ctx[Context.STACK_P1.value]:<8.3f} "
                                    f"{ctx[Context.COMMITTED_P0.value]:<8.3f} "
                                    f"{ctx[Context.COMMITTED_P1.value]:<8.3f} "
                                    f"{ctx[Context.POSITION.value]:<4.0f} "
                                    f"{ctx[Context.ACTIONS_ROUND.value]:<9.0f} "
                                    f"{ctx[Context.MIN_RAISE.value]:<9.3f} "
                                    f"{ctx[Context.BET_TO_CALL.value]:<8.3f}"
                                )
                        else:
                            print(f"  No context tokens found in this trajectory")

                    else:
                        print(f"  No valid tokens found")

    print("=" * 60)


def parse_actions(arg: str) -> List[int]:
    if not arg:
        return []
    return [int(item.strip()) for item in arg.split(",") if item.strip()]


def main(actions: Iterable[int]) -> None:
    device = torch.device("cpu")

    # Create config manually
    config = Config(
        use_wandb=False,
        num_envs=1,
        train=TrainingConfig(
            batch_size=64,
            num_epochs=4,
            max_trajectory_length=12,
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
            stack=1000,
            bet_bins=[0.5, 1.0, 1.5, 2.0],
            sb=5,
            bb=10,
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

    # Hook into the policy to see action selection
    original_action_batch = trainer.policy.action_batch
    trainer.policy.action_batch = hook_policy_action_batch(
        trainer.policy, original_action_batch, actions
    )

    # Hook into replay buffer methods to see transition data
    original_add_transitions = trainer.replay_buffer.add_transitions
    trainer.replay_buffer.add_transitions = hook_replay_buffer_add_transitions(
        trainer.replay_buffer, original_add_transitions
    )

    original_update_opponent_rewards = (
        trainer.replay_buffer.update_last_transition_rewards
    )
    trainer.replay_buffer.update_last_transition_rewards = (
        hook_replay_buffer_update_opponent_rewards(
            trainer.replay_buffer, original_update_opponent_rewards
        )
    )

    print("Starting SelfPlayTrainer iteration...")
    print(f"Actions to take: {list(actions)}")

    # Run one iteration of trajectory collection
    try:
        # This will collect trajectories and we'll see what gets passed to the model
        trainer.collect_tensor_trajectories(min_trajectories=1)
        print("\n=== TRAJECTORY COLLECTION COMPLETED ===")
        trainer.replay_buffer.compute_gae_returns(
            gamma=trainer.gamma, lambda_=trainer.gae_lambda
        )

        # Print comprehensive replay buffer summary
        print_replay_buffer_summary(trainer.replay_buffer)

        # Sample a batch and show the resulting transition info
        print("\n" + "=" * 60)
        print("SAMPLE BATCH ANALYSIS")
        print("=" * 60)

        try:
            # Create a random generator for sampling
            rng = torch.Generator(device=trainer.device)
            rng.manual_seed(42)  # For reproducible results

            # Sample a small batch
            batch_size = min(3, trainer.replay_buffer.num_steps())
            if batch_size > 0:
                # Let's manually inspect what's being sampled
                print(f"\n--- SAMPLING DEBUG INFO ---")
                valid_indices = torch.where(
                    trainer.replay_buffer.trajectory_lengths > 0
                )[0]
                print(f"Valid trajectory indices: {valid_indices.tolist()}")
                print(
                    f"Trajectory lengths: {trainer.replay_buffer.trajectory_lengths[valid_indices].tolist()}"
                )
                print(
                    f"Transition token ends: {trainer.replay_buffer.transition_token_ends[valid_indices].tolist()}"
                )

                # Let's manually inspect the token sequences in the replay buffer
                print(f"\n--- REPLAY BUFFER TOKEN SEQUENCES ---")
                for i, traj_idx in enumerate(valid_indices):
                    print(f"Trajectory {traj_idx}:")
                    token_pos = trainer.replay_buffer.current_token_positions[traj_idx]
                    print(f"  Token position: {token_pos}")
                    if token_pos > 0:
                        tokens = trainer.replay_buffer.data.token_ids[
                            traj_idx, :token_pos
                        ].tolist()
                        print(f"  Token IDs: {tokens}")

                        # Show the actual token sequence that should be sampled
                        print(f"  Decoded sequence:")
                        for pos, token in enumerate(tokens):
                            if token >= 0:  # Only show valid tokens
                                parts = [f"[{pos:02d}]"]
                                if token == 0:  # CLS
                                    parts.append("CLS")
                                elif token == 1:  # CONTEXT
                                    parts.append("CONTEXT")
                                elif token == 2:  # STREET_PREFLOP
                                    parts.append("STREET_PREFLOP")
                                elif token == 3:  # STREET_FLOP
                                    parts.append("STREET_FLOP")
                                elif token == 4:  # STREET_TURN
                                    parts.append("STREET_TURN")
                                elif token == 5:  # STREET_RIVER
                                    parts.append("STREET_RIVER")
                                elif (
                                    get_card_token_id_offset()
                                    <= token
                                    < get_action_token_id_offset()
                                ):  # Card tokens
                                    card_idx = token - get_card_token_id_offset()
                                    card_str = format_card(card_idx)
                                    card_rank = RANK_STR[
                                        trainer.replay_buffer.data.card_ranks[
                                            traj_idx, pos
                                        ].item()
                                    ]
                                    card_suit = SUIT_STR[
                                        trainer.replay_buffer.data.card_suits[
                                            traj_idx, pos
                                        ].item()
                                    ]
                                    parts.append(
                                        f"CARD_{card_str} (rank={card_rank}, suit={card_suit})"
                                    )
                                elif (
                                    token >= get_action_token_id_offset()
                                ):  # Action tokens
                                    action_idx = token - get_action_token_id_offset()
                                    parts.append(f"ACTION_{action_idx}")
                                print("    " + " ".join(parts))

                batch = trainer.replay_buffer.sample_batch(rng, batch_size)

                print(f"Sampled batch size: {batch_size}")
                print(f"Batch keys: {list(batch.keys())}")

                # Show embedding data info
                if "embedding_data" in batch:
                    embedding_data = batch["embedding_data"]
                    print(f"\nEmbedding data type: {type(embedding_data)}")
                    if hasattr(embedding_data, "token_ids"):
                        print(f"Lengths: {embedding_data.lengths.tolist()}")

                        # Show token sequences for sampled transitions
                        for i in range(min(3, embedding_data.token_ids.shape[0])):
                            seq_len = int(embedding_data.lengths[i])
                            print(
                                f"\n--- Sampled Transition {i} (length={seq_len}) ---"
                            )
                            tokens = embedding_data.token_ids[i, :seq_len]
                            for pos, token in enumerate(tokens):
                                token = int(token.item())
                                parts = [f"[{pos:02d}]"]

                                # Decode token based on type
                                if token == Special.CLS.value:
                                    parts.append("CLS")
                                    if hasattr(embedding_data, "context_features"):
                                        ctx = embedding_data.context_features[i, pos]
                                        parts.append(f"sb={ctx[Cls.SB.value]:.3f}")
                                        parts.append(f"bb={ctx[Cls.BB.value]:.3f}")
                                        parts.append(
                                            f"hero_position={ctx[Cls.HERO_POSITION.value]:.0f}"
                                        )

                                elif token == Special.CONTEXT.value:
                                    parts.append("CONTEXT")
                                    if hasattr(embedding_data, "context_features"):
                                        ctx = embedding_data.context_features[i, pos]
                                        parts.append(
                                            f"pot={ctx[Context.POT.value]:.3f}"
                                        )
                                        parts.append(
                                            f"stack_p0={ctx[Context.STACK_P0.value]:.3f}"
                                        )
                                        parts.append(
                                            f"stack_p1={ctx[Context.STACK_P1.value]:.3f}"
                                        )
                                        parts.append(
                                            f"bet_to_call={ctx[Context.BET_TO_CALL.value]:.3f}"
                                        )

                                elif token in [
                                    Special.STREET_PREFLOP.value,
                                    Special.STREET_FLOP.value,
                                    Special.STREET_TURN.value,
                                    Special.STREET_RIVER.value,
                                ]:
                                    street_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
                                    street_idx = token - Special.STREET_PREFLOP.value
                                    if 0 <= street_idx < len(street_names):
                                        parts.append(
                                            f"STREET_{street_names[street_idx]}"
                                        )

                                elif (
                                    get_card_token_id_offset()
                                    <= token
                                    <= get_card_token_id_offset() + 52
                                ):  # Card tokens
                                    card_idx = token - get_card_token_id_offset()
                                    card_str = format_card(card_idx)
                                    card_rank = RANK_STR[
                                        trainer.replay_buffer.data.card_ranks[
                                            traj_idx, pos
                                        ].item()
                                    ]
                                    card_suit = SUIT_STR[
                                        trainer.replay_buffer.data.card_suits[
                                            traj_idx, pos
                                        ].item()
                                    ]
                                    parts.append(
                                        f"CARD_{card_str} (rank={card_rank}, suit={card_suit})"
                                    )

                                elif (
                                    token >= get_action_token_id_offset()
                                ):  # Action tokens
                                    action_idx = token - get_action_token_id_offset()
                                    parts.append(f"ACTION_{action_idx}")

                                print(" ".join(parts))

                # Show other batch information
                print(f"\nAction indices: {batch['action_indices'].tolist()}")
                print(f"Old log probs: {batch['log_probs_old'].tolist()}")
                print(f"Advantages: {batch['advantages'].tolist()}")
                print(f"Returns: {batch['returns'].tolist()}")
                print(f"Delta2: {batch['delta2'].tolist()}")
                print(f"Delta3: {batch['delta3'].tolist()}")

                # Show legal masks
                legal_masks = batch["legal_masks"]
                print(f"\nLegal masks shape: {legal_masks.shape}")
                for i in range(min(3, legal_masks.shape[0])):
                    legal_actions = torch.where(legal_masks[i])[0].tolist()
                    print(f"  Sample {i}: legal actions {legal_actions}")

            else:
                print("No trajectories available for sampling")

        except Exception as e:
            print(f"Error during batch sampling: {e}")
            import traceback

            traceback.print_exc()

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
