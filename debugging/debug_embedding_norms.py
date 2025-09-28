#!/usr/bin/env python3
"""
Debugging script to analyze embedding norms in a trained transformer model.

This script:
1. Creates a SelfPlayTrainer
2. Loads a checkpoint using load_checkpoint
3. Extracts card ID, rank, and suit embeddings
4. Calculates and prints average norms for each embedding type
"""

import argparse
import itertools
import os
import sys

import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ExploiterConfig,
    ModelConfig,
    StateEncoderConfig,
    TrainingConfig,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder
from alphaholdem.models.transformer.tokens import (
    Special,
    get_card_token_id_offset,
    get_special_token_id_offset,
)
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.utils.config_loader import load_config_from_checkpoint
from alphaholdem.utils.model_context import model_eval
from alphaholdem.utils.model_utils import get_probs_and_values

RANKS = "23456789TJQKA"
SUITS = "shdc"


def analyze_embedding_norms(trainer: SelfPlayTrainer) -> None:
    """Analyze and print embedding norms for card ID, rank, and suit embeddings."""

    # Get the transformer model
    model = trainer.model

    # Access the embedding module
    embedding_module = model.embedding

    # Get token offsets
    special_offset = get_special_token_id_offset()
    card_offset = get_card_token_id_offset()

    print(f"Token offsets:")
    print(f"  Special tokens: {special_offset}")
    print(f"  Card tokens: {card_offset}")
    print(f"  Number of special tokens: {Special.NUM_SPECIAL.value}")
    print()

    # Extract base embedding table
    base_embedding = embedding_module.base_embedding.weight.data

    # Extract card ID embeddings (token IDs 7-58, which are cards 0-51)
    card_id_embeddings = base_embedding[card_offset : card_offset + 52]

    # Extract rank and suit embeddings
    rank_embeddings = embedding_module.card_rank_emb.weight.data  # 13 ranks
    suit_embeddings = embedding_module.card_suit_emb.weight.data  # 4 suits

    # Calculate norms
    card_id_norm = torch.norm(card_id_embeddings, dim=1).mean().item()
    rank_norm = torch.norm(rank_embeddings, dim=1).mean().item()
    suit_norm = torch.norm(suit_embeddings, dim=1).mean().item()

    # Print results
    print("Embedding Norms:")
    print(f"  Card ID embeddings (avg norm): {card_id_norm:.6f}")
    print(f"  Rank embeddings (avg norm):    {rank_norm:.6f}")
    print(f"  Suit embeddings (avg norm):    {suit_norm:.6f}")
    print()

    # Additional statistics
    print("Additional Statistics:")
    print(f"  Card ID embedding shape: {card_id_embeddings.shape}")
    print(f"  Rank embedding shape:    {rank_embeddings.shape}")
    print(f"  Suit embedding shape:    {suit_embeddings.shape}")
    print()

    # Show individual card norms
    card_norms = torch.norm(card_id_embeddings, dim=1)
    print("Individual Card ID Norms:")
    for i, norm in enumerate(card_norms):
        print(f"  Card {i:2d}: {norm:.6f}")
    print()

    # Show rank norms
    rank_norms = torch.norm(rank_embeddings, dim=1)
    rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    print("Individual Rank Norms:")
    for i, norm in enumerate(rank_norms):
        print(f"  {rank_names[i]:2s}: {norm:.6f}")
    print()

    # Show suit norms
    suit_norms = torch.norm(suit_embeddings, dim=1)
    suit_names = ["♠", "♥", "♦", "♣"]
    print("Individual Suit Norms:")
    for i, norm in enumerate(suit_norms):
        print(f"  {suit_names[i]:2s}: {norm:.6f}")


def collect_trajectory_with_transitions(
    trainer: SelfPlayTrainer, min_transitions: int = 3
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Collect a trajectory until we find one with at least min_transitions."""
    print(f"Collecting trajectory with at least {min_transitions} transitions...")

    # Clear the replay buffer to start fresh
    trainer.replay_buffer.clear()

    # Ensure all components are on the same device
    print(f"Trainer device: {trainer.device}")
    print(f"Model device: {next(trainer.model.parameters()).device}")
    print(f"Tensor env device: {trainer.tensor_env.device}")
    print(f"Replay buffer device: {trainer.replay_buffer.device}")

    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        attempts += 1

        # Collect trajectories using tensor environment
        trainer.collect_tensor_trajectories(
            min_trajectories=1, add_to_replay_buffer=True
        )

        # Get the most recent trajectory from replay buffer
        if trainer.replay_buffer.size > 0:
            # Get the last trajectory from the replay buffer
            last_trajectory_idx = (
                trainer.replay_buffer.position - 1
            ) % trainer.replay_buffer.capacity

            # Get the number of transitions in this trajectory
            # Count non-zero transitions by checking where dones are True
            trajectory_dones = trainer.replay_buffer.dones[last_trajectory_idx]
            num_transitions = trajectory_dones.sum().item()

            if num_transitions >= min_transitions:
                print(
                    f"✅ Found trajectory with {num_transitions} transitions (attempt {attempts})"
                )

                # Get the last state from StructuredEmbeddingData
                embedding_data = trainer.replay_buffer.data[last_trajectory_idx]
                # Get the last legal mask
                legal_mask = trainer.replay_buffer.legal_masks[
                    last_trajectory_idx, num_transitions - 1
                ]

                return embedding_data, legal_mask, {"num_transitions": num_transitions}

        if attempts % 10 == 0:
            print(f"  Attempt {attempts}: Continuing...")

    raise RuntimeError(
        f"Could not find trajectory with {min_transitions}+ transitions after {max_attempts} attempts"
    )


def get_action_predictions(
    trainer: SelfPlayTrainer,
    embedding_data,
    legal_mask: torch.Tensor,
    device: torch.device,
):
    """Get action predictions from the model for the current state."""
    from alphaholdem.utils import model_eval

    with model_eval(trainer.model), torch.no_grad():
        # Convert to batch format - StructuredEmbeddingData needs to be batched
        from alphaholdem.models.transformer.structured_embedding_data import (
            StructuredEmbeddingData,
        )

        if isinstance(embedding_data, StructuredEmbeddingData):
            # Create a batch with a single trajectory
            batched_embedding_data = StructuredEmbeddingData(
                token_ids=embedding_data.token_ids.unsqueeze(0),  # Add batch dimension
                token_streets=embedding_data.token_streets.unsqueeze(0),
                card_ranks=embedding_data.card_ranks.unsqueeze(0),
                card_suits=embedding_data.card_suits.unsqueeze(0),
                action_actors=embedding_data.action_actors.unsqueeze(0),
                action_legal_masks=embedding_data.action_legal_masks.unsqueeze(0),
                context_features=embedding_data.context_features.unsqueeze(0),
                lengths=embedding_data.lengths.unsqueeze(0),
            )
        else:
            # Fallback for other types
            batched_embedding_data = embedding_data

        # Get model predictions
        outputs = trainer.model(batched_embedding_data)
        logits = outputs.policy_logits.squeeze(0)  # [num_bet_bins]
        value = outputs.value.squeeze(0).item()

        # Apply legal mask
        masked_logits = torch.where(legal_mask == 0, -1e9, logits)

        # Convert to probabilities
        probs = torch.softmax(masked_logits, dim=-1)

        return probs, value, legal_mask


def permute_suit(card: int, suit_perm: list[int]) -> int:
    """Permute the suit of a card."""
    suit = card // 13
    new_suit = suit_perm[suit]
    return new_suit * 13 + (card % 13)


def test_suit_permutations(trainer: SelfPlayTrainer, device: torch.device) -> None:
    """Test how suit permutations affect river action predictions."""

    print("=" * 80)
    print("RIVER ACTION PREDICTION TEST WITH SUIT PERMUTATIONS")
    print("=" * 80)

    # Use the existing trainer - collect trajectories and inspect final decisions
    print(f"Using existing trainer with {trainer.tensor_env.N} environments")
    print(f"Environment device: {trainer.tensor_env.device}")
    print()

    # Collect trajectories until we find one with at least 3 transitions
    print("Collecting trajectory with at least 3 transitions...")

    suit_permutations = torch.tensor(
        list(itertools.permutations([0, 1, 2, 3])), device=device
    )  # Shape: [6, 4]

    # Clear the replay buffer to start fresh
    trainer.replay_buffer.clear()

    attempts = 0
    max_attempts = 100

    while attempts < max_attempts:
        attempts += 1

        # Collect trajectories using tensor environment
        trainer.collect_tensor_trajectories(
            min_trajectories=1, add_to_replay_buffer=True
        )

        # longest trajectory
        longest_trajectory_idx = (
            trainer.replay_buffer.trajectory_lengths.argmax().item()
        )
        if trainer.replay_buffer.trajectory_lengths[longest_trajectory_idx] >= 4:
            data = StructuredEmbeddingData.empty(
                batch_size=24,
                seq_len=trainer.cfg.train.max_sequence_length,
                num_bet_bins=trainer.num_bet_bins,
                device=device,
            )

            # Copy the trajectory data from the replay buffer
            data.token_ids[:] = trainer.replay_buffer.data.token_ids[
                longest_trajectory_idx
            ][None, :]
            data.token_streets[:] = trainer.replay_buffer.data.token_streets[
                longest_trajectory_idx
            ][None, :]
            data.card_ranks[:] = trainer.replay_buffer.data.card_ranks[
                longest_trajectory_idx
            ][None, :]
            data.card_suits[:] = trainer.replay_buffer.data.card_suits[
                longest_trajectory_idx
            ][None, :]
            data.action_actors[:] = trainer.replay_buffer.data.action_actors[
                longest_trajectory_idx
            ][None, :]
            data.action_legal_masks[:] = trainer.replay_buffer.data.action_legal_masks[
                longest_trajectory_idx
            ][None, :]
            data.context_features[:] = trainer.replay_buffer.data.context_features[
                longest_trajectory_idx
            ][None, :]
            data.lengths[:] = trainer.replay_buffer.data.lengths[longest_trajectory_idx]

            for i in range(trainer.cfg.train.max_sequence_length):
                if (
                    data.token_ids[0, i] >= get_card_token_id_offset()
                    and data.token_ids[0, i] < get_card_token_id_offset() + 52
                ):
                    data.card_suits[:, i] = suit_permutations[
                        torch.arange(24, device=device), data.card_suits[:, i].long()
                    ]
                    data.token_ids[:, i] = (
                        data.card_suits[:, i] * 13
                        + data.card_ranks[:, i]
                        + get_card_token_id_offset()
                    )

            print(
                f"  Found trajectory with {trainer.replay_buffer.trajectory_lengths[longest_trajectory_idx]} transitions."
            )
            print(f"  Created batch with 24 permuted suit variations.")

            # Show the sequence of actions in the trajectory
            print("\nTrajectory Action Sequence:")
            print("-" * 60)
            trajectory_length = trainer.replay_buffer.trajectory_lengths[
                longest_trajectory_idx
            ]
            for step in range(trajectory_length):
                action = trainer.replay_buffer.action_indices[
                    longest_trajectory_idx, step
                ].item()
                reward = trainer.replay_buffer.rewards[
                    longest_trajectory_idx, step
                ].item()
                done = trainer.replay_buffer.dones[longest_trajectory_idx, step].item()
                actor = trainer.replay_buffer.data.action_actors[
                    longest_trajectory_idx, step
                ].item()
                street = trainer.replay_buffer.data.token_streets[
                    longest_trajectory_idx, step
                ].item()
                print(
                    f"  Step {step+1}: Player {actor} on Street {street} -> Action {action}, Reward {reward:.4f}, Done {done}"
                )
            print("-" * 60)
            print()

            # Run the model on the batch with permuted suits
            with torch.no_grad(), model_eval(trainer.model):
                # Get legal masks for the last step
                legal_masks = trainer.replay_buffer.legal_masks[
                    longest_trajectory_idx, -1
                ]

                # Get model outputs for all 24 permutations
                action_probs, value_preds = get_probs_and_values(
                    trainer.model, data, legal_masks
                )

            # Compare results across permutations
            print("=" * 80)
            print("SUIT PERMUTATION RESULTS")
            print("=" * 80)

            suit_names = ["♠", "♥", "♦", "♣"]
            suit_permutations_list = list(itertools.permutations([0, 1, 2, 3]))

            # Find the most different predictions
            max_value_diff = 0
            max_value_pair = None
            max_prob_diff = 0
            max_prob_pair = None

            for i in range(24):
                perm_idx = i // 4
                suit_idx = i % 4

                print(
                    f"Permutation {perm_idx+1}, Suit {suit_idx+1}: {[suit_names[s] for s in suit_permutations_list[perm_idx]]}"
                )
                print(f"  Value: {value_preds[i].item():.4f}")
                print(f"  Action probs: {action_probs[i].cpu().numpy()}")
                print()

                # Compare with other permutations
                for j in range(i + 1, 24):
                    value_diff = abs(value_preds[i].item() - value_preds[j].item())
                    prob_diff = (
                        torch.abs(action_probs[i] - action_probs[j]).max().item()
                    )

                    if value_diff > max_value_diff:
                        max_value_diff = value_diff
                        max_value_pair = (i, j)

                    if prob_diff > max_prob_diff:
                        max_prob_diff = prob_diff
                        max_prob_pair = (i, j)

            print("=" * 80)
            print("COMPARISON SUMMARY")
            print("=" * 80)
            print(f"Maximum value difference: {max_value_diff:.6f}")
            if max_value_pair:
                print(
                    f"  Between permutations {max_value_pair[0]//4 + 1} and {max_value_pair[1]//4 + 1}"
                )
            print(f"Maximum probability difference: {max_prob_diff:.6f}")
            if max_prob_pair:
                print(
                    f"  Between permutations {max_prob_pair[0]//4 + 1} and {max_prob_pair[1]//4 + 1}"
                )

            if max_prob_diff < 1e-6:
                print("✅ All suit permutations give identical action probabilities!")
                print("The policy head appears to be suit-invariant.")
            else:
                print("❌ Suit permutations give different action probabilities!")
                print("The policy head is NOT suit-invariant.")

            if max_value_diff < 1e-6:
                print("✅ All suit permutations give identical value estimates!")
                print("The value head appears to be suit-invariant.")
            else:
                print("❌ Suit permutations give different value estimates!")
                print("The value head is NOT suit-invariant.")

                print("\n" + "=" * 80)
            return

        print(
            f"  Attempt {attempts}: No trajectory with 4 transitions found yet. Retrying..."
        )

    print(
        f"Failed to collect a trajectory with at least 1 transition after {max_attempts} attempts."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze embedding norms in a trained transformer model"
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the checkpoint file to load"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu, cuda, mps)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict model loading (default: False)",
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint_path}")

    # Set up device
    device_obj = torch.device(args.device)

    # Create a minimal CLI config for overrides
    # Only override specific fields, let checkpoint config handle model/env details
    cli_config = Config(
        device=args.device,
        use_wandb=False,  # Disable wandb for debugging
        strict_model_loading=args.strict,
        num_envs=24,
    )

    # Load config from checkpoint with CLI overrides
    config = load_config_from_checkpoint(args.checkpoint_path, cli_config)

    print(f"Creating SelfPlayTrainer...")

    # Create trainer
    trainer = SelfPlayTrainer(config, device_obj)

    print(f"Loading checkpoint into trainer...")

    # Use the trainer's load_checkpoint method (strict=False by default)
    step, _ = trainer.load_checkpoint(args.checkpoint_path)

    # Recreate replay buffer if it was loaded from a different device to avoid device mismatch
    if trainer.replay_buffer.device != device_obj:
        print(
            f"Recreating replay buffer (was on {trainer.replay_buffer.device}, need {device_obj})"
        )
        from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer

        sequence_length = (
            trainer.state_encoder.sequence_length if trainer.is_transformer else -1
        )
        trainer.replay_buffer = VectorizedReplayBuffer(
            capacity=trainer.batch_size * max(1, 1 + trainer.replay_buffer_batches),
            max_trajectory_length=trainer.max_trajectory_length,
            num_bet_bins=trainer.num_bet_bins,
            device=device_obj,
            float_dtype=trainer.float_dtype,
            is_transformer=trainer.is_transformer,
            max_sequence_length=sequence_length,
        )

    print(f"✅ Checkpoint loaded successfully")
    print(f"   Step: {step}")
    print(f"   ELO: {trainer.opponent_pool.current_elo:.1f}")
    print()

    # Analyze embedding norms
    analyze_embedding_norms(trainer)

    # Test suit permutations on river
    test_suit_permutations(trainer, device_obj)


if __name__ == "__main__":
    main()
