"""Inspect transformer tokenization by hooking into SelfPlayTrainer.

Run with:

    python debugging/inspect_transformer_sequence.py --actions 2,1,0

This will run a single iteration of SelfPlayTrainer and hook into the model
to see what exactly gets passed in as input data.

Use ``--trace-suit-permute`` to dump before/after token sequences when
``permute_suits`` is invoked on structured transformer batches.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List

import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.tokens import (
    Context,
    Game,
    Special,
    get_action_token_id_offset,
    get_card_token_id_offset,
)
from alphaholdem.rl.self_play import SelfPlayTrainer

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
    """Pretty print token sequence data using the unified decoder."""
    length = (
        int(data.lengths[0]) if hasattr(data, "lengths") else data.token_ids.shape[1]
    )
    print(f"\n--- Token sequence for {title} (length={length}) ---")
    # Provide bet_bins to the decoder via a temporary attribute if missing
    needs_cleanup = False
    if not hasattr(data, "bet_bins"):
        try:
            setattr(data, "bet_bins", bet_bins)
            needs_cleanup = True
        except Exception:
            pass
    print_decoded_token_sequence(data, 0, length)
    if needs_cleanup:
        try:
            delattr(data, "bet_bins")
        except Exception:
            pass


def print_decoded_token_sequence(data, row_idx: int, length: int) -> None:
    """Unified, detailed token sequence printer used across the script.
    data: StructuredEmbeddingData-like object with token_ids and context/token metadata
    row_idx: which row/trajectory/sample to render
    length: number of valid tokens to display from the start
    """
    tokens = data.token_ids[row_idx, :length]
    for pos, token_t in enumerate(tokens):
        token = int(token_t.item())
        parts = [f"[{pos:02d}]"]

        if token == Special.CLS.value:
            parts.append("CLS")
        elif token == Special.GAME.value:
            parts.append("GAME")
            parts.append(f"sb={data.context_features[row_idx, pos, Game.SB.value]:.0f}")
            parts.append(f"bb={data.context_features[row_idx, pos, Game.BB.value]:.0f}")
            parts.append(
                f"hero_position={data.context_features[row_idx, pos, Game.HERO_POSITION.value]:.0f}"
            )
        elif token == Special.CONTEXT.value:
            parts.append("CONTEXT")
        elif token in [
            Special.STREET_PREFLOP.value,
            Special.STREET_FLOP.value,
            Special.STREET_TURN.value,
            Special.STREET_RIVER.value,
        ]:
            street_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
            street_idx = token - Special.STREET_PREFLOP.value
            if 0 <= street_idx < len(street_names):
                parts.append(f"STREET_{street_names[street_idx]}")
        elif get_card_token_id_offset() <= token < get_action_token_id_offset():
            card_idx = token - get_card_token_id_offset()
            parts.append(f"CARD_{format_card(card_idx)}")
        elif token >= get_action_token_id_offset():
            action_idx = token - get_action_token_id_offset()
            # Use env bet_bins if present for nicer labeling
            bet_bins = None
            if hasattr(data, "bet_bins"):
                bet_bins = data.bet_bins
            # Fallback bet_bins used elsewhere in this script (env config)
            # EnvConfig is available at module import time; no local import needed
            bet_bins = bet_bins or []
            parts.append(
                f"ACTION_{describe_action(action_idx, bet_bins if bet_bins else [0.5, 1.0, 1.5, 2.0])}"
            )
            if hasattr(data, "action_actors"):
                parts.append(f"actor={int(data.action_actors[row_idx, pos].item())}")
            if hasattr(data, "action_legal_masks"):
                legal_mask = data.action_legal_masks[row_idx, pos]
                if legal_mask.dim() > 0:
                    legal_bins = [
                        i for i, flag in enumerate(legal_mask.tolist()) if flag
                    ]
                    parts.append(f"legal={legal_bins}")

        print(" ".join(parts))


def _print_card_token_table(
    data,
    row_idx: int,
    length: int,
    title: str,
) -> None:
    """Render a per-token card table so suit permutations are easy to spot."""
    offset = get_card_token_id_offset()
    tokens = data.token_ids[row_idx, :length]
    mask = (tokens >= offset) & (tokens < offset + 52)
    card_positions = torch.where(mask)[0]
    print(f"  Card tokens for {title}:")
    if card_positions.numel() == 0:
        print("    (none)")
        return
    header = "    pos token card rank suit"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for pos in card_positions.tolist():
        token = int(tokens[pos].item())
        card_idx = token - offset
        rank = int(data.card_ranks[row_idx, pos].item())
        suit = int(data.card_suits[row_idx, pos].item())
        print(f"    {pos:3d} {token:5d} {format_card(card_idx):>4} {rank:4d} {suit:4d}")


def print_token_sequence_data(
    data,
    indices: List[int],
    current_token_positions: torch.Tensor,
    transition_token_ends: torch.Tensor,
    trajectory_lengths: torch.Tensor,
) -> None:
    """Print the TOKEN SEQUENCE DATA section for given indices."""
    print(f"\n--- TOKEN SEQUENCE DATA ---")
    for traj_idx in indices:
        token_pos = int(current_token_positions[traj_idx])
        if token_pos <= 0:
            continue
        print(f"\nTrajectory {traj_idx} (token_pos={token_pos}):")

        # Valid tokens and their indices
        valid_tokens = data.token_ids[traj_idx] >= 0
        valid_token_indices = torch.where(valid_tokens)[0]
        if len(valid_token_indices) == 0:
            print("  No valid tokens found")
            continue

        print(f"  Attention mask: {data.attention_mask[traj_idx].int().tolist()}")

        # Token IDs with transition boundaries
        token_ids = data.token_ids[traj_idx, valid_token_indices].tolist()
        print(f"  Token IDs: {token_ids}")

        length = int(trajectory_lengths[traj_idx])
        token_ends = transition_token_ends[traj_idx, :length].tolist()
        print(f"  Transition token ends: {token_ends}")

        if len(token_ends) > 0:
            print(f"  Token sequence with transition boundaries:")
            boundary_idx = 0
            line_parts = []
            current_transition = []
            for i, token_id in enumerate(token_ids):
                current_transition.append(f"{token_id}")
                if boundary_idx < len(token_ends) and i == token_ends[boundary_idx]:
                    transition_str = " ".join(current_transition)
                    line_parts.append(f"[{transition_str}]")
                    current_transition = []
                    boundary_idx += 1
            if current_transition:
                transition_str = " ".join(current_transition)
                line_parts.append(f"[{transition_str}]")
            print(f"    {' | '.join(line_parts)}")

        # Card / street info
        card_ranks = data.card_ranks[traj_idx, valid_token_indices].tolist()
        card_suits = data.card_suits[traj_idx, valid_token_indices].tolist()
        token_streets = data.token_streets[traj_idx, valid_token_indices].tolist()
        print(f"  Card ranks: {card_ranks}")
        print(f"  Card suits: {card_suits}")
        print(f"  Token streets: {token_streets}")

        # Action info
        action_actors = data.action_actors[traj_idx, valid_token_indices].tolist()
        print(f"  Action actors: {action_actors}")

        # Context features
        context_features = data.context_features[traj_idx, valid_token_indices]
        print(f"  Context features shape: {context_features.shape}")
        context_token_mask = torch.tensor(token_ids) == Special.CONTEXT.value
        if context_token_mask.any():
            context_token_indices = torch.where(context_token_mask)[0]
            print(
                f"  Context tokens found at positions: {context_token_indices.tolist()}"
            )
            print(f"  Context features table:")
            print(
                f"    {'Pos':>3} {'POT':>4} {'STACK_P0':>8} {'STACK_P1':>8} {'COMM_P0':>7} {'COMM_P1':>7} {'POS':>3} {'ACT_ROUND':>9} {'MIN_RAISE':>9} {'BET_CALL':>8}"
            )
            print(
                f"    {'-'*3} {'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*3} {'-'*9} {'-'*9} {'-'*8}"
            )
            for j in context_token_indices:
                ctx = context_features[j].tolist()
                print(
                    f"    {j:>3} "
                    f"{ctx[Context.POT.value]:>4} "
                    f"{ctx[Context.STACK_P0.value]:>8} "
                    f"{ctx[Context.STACK_P1.value]:>8} "
                    f"{ctx[Context.COMMITTED_P0.value]:>7} "
                    f"{ctx[Context.COMMITTED_P1.value]:>7} "
                    f"{ctx[Context.POSITION.value]:>3} "
                    f"{ctx[Context.ACTIONS_ROUND.value]:>9} "
                    f"{ctx[Context.MIN_RAISE.value]:>9} "
                    f"{ctx[Context.BET_TO_CALL.value]:>8}"
                )
        else:
            print(f"  No context tokens found in this trajectory")


def hook_model_forward(model, original_forward, trainer: SelfPlayTrainer | None = None):
    """Hook into model forward to inspect input data."""

    def hooked_forward(embedding_data, kv_cache=None):
        print("\n=== MODEL INPUT INSPECTION ===")

        if hasattr(embedding_data, "token_ids"):
            print(f"Token IDs shape: {embedding_data.token_ids.shape}")
            print(f"Lengths: {embedding_data.lengths.tolist()}")

            # Parse and display token sequences for each sample (unified decoder)
            max_samples = min(3, embedding_data.token_ids.shape[0])
            for batch_idx in range(max_samples):
                seq_len = int(embedding_data.lengths[batch_idx])
                print(f"\n--- Sample {batch_idx} (length={seq_len}) ---")
                print_decoded_token_sequence(embedding_data, batch_idx, seq_len)

            if hasattr(embedding_data, "context_features"):
                print(
                    f"\nContext features shape: {embedding_data.context_features.shape}"
                )

            # Print the same TOKEN SEQUENCE DATA section for inputs
            indices = list(range(min(3, embedding_data.token_ids.shape[0])))
            # For model input, we don't have transition boundaries or positions; synthesize minimal arrays
            masks = [embedding_data.token_ids[i] < 0 for i in indices]
            seq_pos = torch.tensor(
                [
                    (masks[i].int().argmax() if masks[i].any() else masks[i].shape[0])
                    for i in indices
                ],
                dtype=torch.long,
            )
            # Build placeholders matching expected shapes
            max_len = int(embedding_data.token_ids.shape[1])
            trans_ends = torch.zeros((len(indices), max_len), dtype=torch.long)
            traj_lens = torch.tensor(
                [int(embedding_data.lengths[i]) for i in indices], dtype=torch.long
            )
            print_token_sequence_data(
                data=embedding_data,
                indices=indices,
                current_token_positions=seq_pos,
                transition_token_ends=trans_ends,
                trajectory_lengths=traj_lens,
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
        logits,
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
        print(f"Logits shape: {list(logits.shape)}")
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
            logits,
            rewards,
            dones,
            legal_masks,
            delta2,
            delta3,
            values,
            trajectory_indices,
        )

    return hooked_add_transitions


def hook_replay_buffer_update_last_transition_rewards(
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


def enable_permute_suit_trace(max_samples: int = 2) -> None:
    """Monkey-patch StructuredEmbeddingData.permute_suits to show before/after dumps."""

    try:
        from alphaholdem.models.transformer.structured_embedding_data import (
            StructuredEmbeddingData,
        )
    except ImportError:
        print("Could not import StructuredEmbeddingData; suit trace disabled.")
        return

    original = StructuredEmbeddingData.permute_suits

    def describe_state(data, phase: str) -> None:
        batch = data.token_ids.shape[0]
        if batch == 0:
            print(f"\n=== Suit permutation {phase}: empty batch ===")
            return

        indices = list(range(min(max_samples, batch)))
        lengths = data.lengths.to(torch.long).cpu()
        # Build helper tensors on CPU for printing utilities
        current_positions = lengths.clone()
        transition_token_ends = torch.zeros(
            (batch, data.token_ids.shape[1]), dtype=torch.long
        )

        print(f"\n=== SUIT PERMUTATION {phase} ===")
        for idx in indices:
            length = int(lengths[idx].item())
            print(f"\n--- Sample {idx} (length={length}) ---")
            print_decoded_token_sequence(data, idx, length)
            _print_card_token_table(data, idx, length, title=phase)

        print_token_sequence_data(
            data=data,
            indices=indices,
            current_token_positions=current_positions,
            transition_token_ends=transition_token_ends,
            trajectory_lengths=lengths,
        )

    def wrapped(self, generator: torch.Generator) -> None:
        describe_state(self, "BEFORE")
        original(self, generator)
        describe_state(self, "AFTER")

    StructuredEmbeddingData.permute_suits = wrapped


def print_sampling_debug_info(trainer: SelfPlayTrainer) -> None:
    """Print high-level sampling debug info for trajectories and token ends."""
    print(f"\n--- SAMPLING DEBUG INFO ---")
    valid_indices = torch.where(trainer.replay_buffer.trajectory_lengths > 0)[0]
    print(f"Valid trajectory indices: {valid_indices.tolist()}")
    print(
        f"Trajectory lengths: {trainer.replay_buffer.trajectory_lengths[valid_indices].tolist()}"
    )
    print(
        f"Transition token ends: {trainer.replay_buffer.transition_token_ends[valid_indices].tolist()}"
    )


def print_replay_buffer_token_sequences(trainer: SelfPlayTrainer) -> None:
    """Print decoded token sequences for valid trajectories in the replay buffer."""
    print(f"\n--- REPLAY BUFFER TOKEN SEQUENCES ---")
    valid_indices = torch.where(trainer.replay_buffer.trajectory_lengths > 0)[0]
    for _, traj_idx in enumerate(valid_indices):
        print(f"Trajectory {traj_idx}:")
        token_pos = trainer.replay_buffer.current_token_positions[traj_idx]
        print(f"  Token position: {token_pos}")
        if token_pos > 0:
            # Use the detailed batch-style decoder for consistency
            print_decoded_token_sequence(
                trainer.replay_buffer.data, int(traj_idx), int(token_pos)
            )


def print_sampled_batch_analysis(trainer: SelfPlayTrainer, batch) -> None:
    """Print detailed analysis of a sampled training batch."""
    print(f"Sampled batch size: {min(3, trainer.replay_buffer.num_steps())}")

    embedding_data = batch.embedding_data
    print(f"\nEmbedding data type: {type(embedding_data)}")
    if hasattr(embedding_data, "token_ids"):
        print(f"Lengths: {embedding_data.lengths.tolist()}")

        # Show token sequences for sampled transitions (detailed unified printer)
        for i in range(min(3, embedding_data.token_ids.shape[0])):
            seq_len = int(embedding_data.lengths[i])
            print(f"\n--- Sampled Transition {i} (length={seq_len}) ---")
            print_decoded_token_sequence(embedding_data, i, seq_len)

    # Other batch fields
    action_indices = batch.action_indices
    selected_log_probs = getattr(batch, "selected_log_probs", None)
    if selected_log_probs is None:
        selected_log_probs = batch.frozen_selected_log_probs
        print("Using frozen_selected_log_probs as selection log-probs snapshot.")
    if hasattr(batch, "step_selected_log_probs"):
        print(f"Current-policy log probs: {batch.step_selected_log_probs.tolist()}")
    advantages = batch.advantages
    returns = batch.returns
    delta2 = batch.delta2
    delta3 = batch.delta3
    legal_masks = batch.legal_masks

    print(f"\nAction indices: {action_indices.tolist()}")
    print(f"Reference log probs: {selected_log_probs.tolist()}")
    print(f"Advantages: {advantages.tolist()}")
    print(f"Returns: {returns.tolist()}")
    print(f"Delta2: {delta2.tolist()}")
    print(f"Delta3: {delta3.tolist()}")
    print(f"\nLegal masks shape: {legal_masks.shape}")
    for i in range(min(3, legal_masks.shape[0])):
        legal_actions = torch.where(legal_masks[i])[0].tolist()
        print(f"  Sample {i}: legal actions {legal_actions}")


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

            # Group info for each trajectory together
            num_show = min(3, len(valid_indices))
            for i in range(num_show):
                traj_idx = valid_indices[i]
                length = replay_buffer.trajectory_lengths[traj_idx]
                print(f"\nTrajectory {traj_idx} (length={length}):")

                # Action indices
                actions = replay_buffer.action_indices[traj_idx, :length].tolist()
                print(f"  Action indices: {actions}")

                # Rewards
                rewards = replay_buffer.rewards[traj_idx, :length].tolist()
                formatted_rewards = [f"{r:.3f}" for r in rewards]
                print(f"  Rewards: [{', '.join(formatted_rewards)}]")

                # Done flags
                dones = replay_buffer.dones[traj_idx, :length].tolist()
                print(f"  Done flags: {dones}")

                # Values
                values = replay_buffer.values[traj_idx, :length].tolist()
                formatted_values = [f"{v:.3f}" for v in values]
                print(f"  Values: [{', '.join(formatted_values)}]")

                # Advantages and returns if computed
                if replay_buffer.advantages.sum() != 0:
                    advantages = replay_buffer.advantages[traj_idx, :length].tolist()
                    formatted_advantages = [f"{a:.3f}" for a in advantages]
                    print(f"  Advantages: [{', '.join(formatted_advantages)}]")

                    returns = replay_buffer.returns[traj_idx, :length].tolist()
                    formatted_returns = [f"{r:.3f}" for r in returns]
                    print(f"  Returns: [{', '.join(formatted_returns)}]")

                # Delta2
                delta2 = replay_buffer.delta2[traj_idx, :length].tolist()
                formatted_delta2 = [f"{d:.3f}" for d in delta2]
                print(f"  Delta2: [{', '.join(formatted_delta2)}]")

                # Delta3
                delta3 = replay_buffer.delta3[traj_idx, :length].tolist()
                formatted_delta3 = [f"{d:.3f}" for d in delta3]
                print(f"  Delta3: [{', '.join(formatted_delta3)}]")

                # Legal masks (only for first trajectory)
                if i == 0:
                    print(f"  Legal masks:")
                    legal_masks = replay_buffer.legal_masks[traj_idx, :length]
                    for step in range(length):
                        legal_actions = torch.where(legal_masks[step])[0].tolist()
                        print(f"    Step {step}: legal actions {legal_actions}")

                    # Log probabilities derived from stored logits for completeness
                    print(f"  Log probabilities (from logits):")
                    for step in range(length):
                        logits = replay_buffer.logits[traj_idx, step]
                        legal_mask = replay_buffer.legal_masks[traj_idx, step]
                        safe_logits = torch.where(
                            legal_mask,
                            logits,
                            torch.full_like(logits, -1e9),
                        )
                        log_probs = torch.log_softmax(safe_logits, dim=-1)
                        probs = log_probs.exp().tolist()
                        formatted_probs = [f"{p:.3f}" for p in probs]
                        print(f"    Step {step}: [{', '.join(formatted_probs)}]")

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
            print_token_sequence_data(
                data=replay_buffer.data,
                indices=[
                    int(idx)
                    for idx in valid_indices[: min(3, len(valid_indices))].tolist()
                ],
                current_token_positions=replay_buffer.current_token_positions,
                transition_token_ends=replay_buffer.transition_token_ends,
                trajectory_lengths=replay_buffer.trajectory_lengths,
            )

    print("=" * 60)


def parse_actions(arg: str) -> List[int]:
    if not arg:
        return []
    return [int(item.strip()) for item in arg.split(",") if item.strip()]


def main(
    actions: Iterable[int], trace_suit_permute: bool, trace_suit_max_samples: int
) -> None:
    device = torch.device("cpu")

    # Create config manually
    config = Config(
        use_wandb=False,
        num_envs=1,
        train=TrainingConfig(
            batch_size=64,
            episodes_per_step=4,
            max_trajectory_length=12,
            max_sequence_length=47,
            learning_rate=1e-4,
        ),
        model=ModelConfig(
            name="poker_transformer_v1",
            kwargs={
                "max_sequence_length": 47,
                "d_model": 128,
                "n_layers": 2,
                "n_heads": 2,
                "dropout": 0.1,
                "use_gradient_checkpointing": False,
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

    if trace_suit_permute:
        enable_permute_suit_trace(max_samples=trace_suit_max_samples)

    # Create trainer
    trainer = SelfPlayTrainer(config, device, rng_seed=42)

    # Hook into the model to see what gets passed in
    original_forward = trainer.model.forward
    trainer.model.forward = hook_model_forward(trainer.model, original_forward, trainer)

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

    original_update_last_transition_rewards = (
        trainer.replay_buffer.update_last_transition_rewards
    )
    trainer.replay_buffer.update_last_transition_rewards = (
        hook_replay_buffer_update_last_transition_rewards(
            trainer.replay_buffer, original_update_last_transition_rewards
        )
    )

    print("Starting SelfPlayTrainer iteration...")
    print(f"Actions to take: {list(actions)}")

    # Run one iteration of trajectory collection
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

    # Sample a small batch
    batch_size = min(3, trainer.replay_buffer.num_steps())
    if batch_size > 0:
        print_sampling_debug_info(trainer)
        print_replay_buffer_token_sequences(trainer)

        batch = trainer.replay_buffer.sample_batch(trainer.rng, batch_size)
        print_sampled_batch_analysis(trainer, batch)

        if trace_suit_permute and hasattr(batch.embedding_data, "permute_suits"):
            print("\n=== MANUAL SUIT PERMUTATION TRACE ON SAMPLED BATCH ===")
            batch.embedding_data.permute_suits(trainer.rng)
    else:
        print("No trajectories available for sampling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug transformer tokenization for a scripted hand."
    )
    parser.add_argument(
        "--actions",
        type=str,
        default="3,1",
        help="Comma-separated list of discrete bet bin indices to play sequentially.",
    )
    parser.add_argument(
        "--trace-suit-permute",
        action="store_true",
        help="Print before/after token dumps when permute_suits is invoked.",
    )
    parser.add_argument(
        "--trace-suit-max-samples",
        type=int,
        default=2,
        help="Number of samples to display when tracing suit permutations.",
    )
    args = parser.parse_args()
    action_list = parse_actions(args.actions)
    main(action_list, args.trace_suit_permute, args.trace_suit_max_samples)
