from __future__ import annotations

from typing import Dict, Union

import torch

from ..models.cnn_embedding_data import CNNEmbeddingData
from ..models.transformer.structured_embedding_data import StructuredEmbeddingData
from ..models.transformer.tokens import Special


class VectorizedReplayBuffer:
    """
    Vectorized replay buffer that stores trajectories in a 2D structure.
    Each row represents a trajectory, each column represents a step within that trajectory.
    Uses vectorized operations throughout - no Python loops in core operations.
    """

    def __init__(
        self,
        capacity: int,  # Number of trajectories (not steps)
        max_trajectory_length: int,  # Maximum steps per trajectory
        num_bet_bins: int,  # Batch size for tensor operations
        device: torch.device,
        float_dtype: torch.dtype = torch.float32,  # Dtype for float tensors
        is_transformer: bool = False,  # Whether this buffer is for transformer models
        max_sequence_length: int = 50,  # Sequence length for transformer models
    ):
        self.capacity = capacity  # Number of trajectories
        self.max_trajectory_length = max_trajectory_length
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.float_dtype = float_dtype  # Store float dtype for use in methods
        self.is_transformer = is_transformer  # Store transformer flag
        self.num_bet_bins = num_bet_bins
        self.position = 0  # Next trajectory write position (end of ring buffer)
        self.size = 0  # Total number of valid trajectories

        self.open_batch = (
            -1
        )  # -1 if no batch is open, otherwise the nominal size of the open batch

        # T is the maximum number of transitions per trajectory
        # L is the maximum number of tokens per trajectory (should be > T)
        C, T, L = capacity, max_trajectory_length, max_sequence_length

        # Pre-allocate tensors for all transition fields: (capacity, max_trajectory_length, ...)
        if not is_transformer:
            # CNN model tensors
            # Cards features tensor: (capacity, max_trajectory_length, 6, 4, 13) - bool dtype for memory efficiency
            self.cards_features = torch.zeros(
                C,
                T,
                6,
                4,
                13,  # Fixed cards shape
                dtype=torch.bool,
                device=device,
            )

            # Actions features tensor: (capacity, max_trajectory_length, 24, 4, num_bet_bins) - bool dtype for memory efficiency
            # 4 slots: p1, p2, sum, legal
            self.actions_features = torch.zeros(
                C,
                T,
                24,
                4,
                num_bet_bins,
                dtype=torch.bool,
                device=device,
            )
        else:
            # Transformer structured embedding fields: maintain a single growing token stream per trajectory
            self.data = StructuredEmbeddingData.empty(
                C, L, num_bet_bins, float_dtype, device
            )
            self.current_token_positions = torch.zeros(
                C,
                dtype=torch.uint8,
                device=device,
            )
            self.transition_token_ends = torch.zeros(
                C,
                T,
                dtype=torch.uint8,
                device=device,
            )

        self.action_indices = torch.zeros(C, T, dtype=torch.long, device=device)
        # Store full log-prob distributions per step for exact KL and stable ratio
        self.log_probs = torch.zeros(
            C, T, num_bet_bins, dtype=float_dtype, device=device
        )
        self.rewards = torch.zeros(C, T, dtype=float_dtype, device=device)
        self.dones = torch.zeros(C, T, dtype=torch.bool, device=device)
        self.legal_masks = torch.zeros(
            C,
            T,
            num_bet_bins,
            dtype=torch.bool,  # Changed to bool
            device=device,
        )
        self.delta2 = torch.zeros(C, T, dtype=float_dtype, device=device)
        self.delta3 = torch.zeros(C, T, dtype=float_dtype, device=device)
        self.values = torch.zeros(C, T, dtype=float_dtype, device=device)
        self.advantages = torch.zeros(C, T, dtype=float_dtype, device=device)
        self.returns = torch.zeros(C, T, dtype=float_dtype, device=device)
        # No separate logits tensor; full distributions are stored in log_probs

        # Track trajectory metadata
        self.trajectory_lengths = torch.zeros(capacity, dtype=torch.long, device=device)

        # Track current step position within each trajectory
        self.current_transition_counts = torch.zeros(
            capacity, dtype=torch.long, device=device
        )

    def start_adding_trajectory_batches(self, num_trajectories: int) -> None:
        """
        Mark the start of new trajectories being added to the buffer.
        Clear out rows up to position + num_trajectories to prepare for new data.

        Args:
            num_trajectories: Number of trajectories that will be added
        """

        self.open_batch = num_trajectories

        # Clear out rows that will be used for new trajectories
        end_position = (self.position + num_trajectories) % self.capacity

        if end_position > self.position:
            # No wraparound - clear contiguous block
            clear_indices = torch.arange(
                self.position, end_position, device=self.device
            )
        else:
            # Wraparound - clear from position to end, then from start to end_position
            clear_indices = torch.cat(
                [
                    torch.arange(self.position, self.capacity, device=self.device),
                    torch.arange(0, end_position, device=self.device),
                ]
            )

        # Clear trajectory metadata for these rows
        self.trajectory_lengths[clear_indices] = 0
        self.current_transition_counts[clear_indices] = 0

        # Clear the actual data tensors for these rows
        if not self.is_transformer:
            # CNN model tensors
            self.cards_features[clear_indices] = False
            self.actions_features[clear_indices] = False
        else:
            # Transformer structured embedding tensors
            self.data.token_ids[clear_indices] = -1
            self.data.token_streets[clear_indices] = 0
            self.data.card_ranks[clear_indices] = 0
            self.data.card_suits[clear_indices] = 0
            self.data.action_actors[clear_indices] = 0
            self.data.action_legal_masks[clear_indices] = False
            self.data.context_features[clear_indices] = 0
            self.current_token_positions[clear_indices] = 0
            self.transition_token_ends[clear_indices] = 0

        # Common tensors for both model types
        self.action_indices[clear_indices] = 0
        self.log_probs[clear_indices] = 0
        self.rewards[clear_indices] = 0
        self.dones[clear_indices] = False
        self.legal_masks[clear_indices] = False
        self.delta2[clear_indices] = 0
        self.delta3[clear_indices] = 0
        self.values[clear_indices] = 0
        self.advantages[clear_indices] = 0
        self.returns[clear_indices] = 0

    def finish_adding_trajectory_batches(self) -> tuple[int, int]:
        """
        Finish adding trajectories and advance the ring buffer position.
        This should be called when environments are reset.

        Args:
            num_trajectories: Number of trajectories that were added

        Returns:
            Number of trajectories of nonzero length that were added (this is the change in self.size)
        """

        assert (
            self.open_batch > 0
        ), "Must call start_adding_trajectories before finishing a batch"

        num_trajectories = self.open_batch
        self.open_batch = -1

        # Compute the indices in the ring buffer window [position, position + num_trajectories)
        start = self.position
        end = (self.position + num_trajectories) % self.capacity

        if num_trajectories == 0:
            return 0, 0

        if end > start:
            indices = torch.arange(start, end, device=self.device)
        else:
            # Wraparound
            indices = torch.cat(
                [
                    torch.arange(start, self.capacity, device=self.device),
                    torch.arange(0, end, device=self.device),
                ]
            )

        # Find which indices have 0-length trajectories
        zero_length_mask = self.trajectory_lengths[indices] == 0

        # Remove (compact out) zero-length trajectories by shifting nonzero-length ones forward
        nonzero_indices = indices[~zero_length_mask]
        num_valid = nonzero_indices.numel()
        # If all are zero-length, just skip
        if num_valid == 0:
            # Advance position and update size as if nothing was added
            return 0, 0

        compacted_indices = indices[:num_valid]
        unused_indices = indices[num_valid:]

        # Compact nonzero-length trajectories to the front of the window
        # For each field, move data from nonzero_indices to the front of indices
        common_fields = [
            "trajectory_lengths",
            "current_transition_counts",
            "action_indices",
            "log_probs",
            "rewards",
            "dones",
            "legal_masks",
            "delta2",
            "delta3",
            "values",
            "advantages",
            "returns",
        ]

        data_fields = []
        if not self.is_transformer:
            # CNN model fields
            cnn_fields = ["cards_features", "actions_features"]
            all_fields = common_fields + cnn_fields
        else:
            # Transformer model fields
            transformer_fields = [
                "transition_token_ends",
                "current_token_positions",
            ]
            all_fields = common_fields + transformer_fields
            data_fields = [
                "token_ids",
                "token_streets",
                "card_ranks",
                "card_suits",
                "action_actors",
                "action_legal_masks",
                "context_features",
                "lengths",
            ]

        for attr in all_fields:
            buf = getattr(self, attr)
            buf[compacted_indices] = buf[nonzero_indices]
            # Zero out the now-unused slots at the end of the window
            buf[unused_indices] = 0

        for attr in data_fields:
            buf = getattr(self.data, attr)
            buf[compacted_indices] = buf[nonzero_indices]
            # Zero out the now-unused slots at the end of the window
            buf[unused_indices] = 0

        # Only advance position and size by the number of valid (nonzero-length) trajectories
        trajectories_added = num_valid

        # Advance the ring buffer position
        self.position = (self.position + trajectories_added) % self.capacity

        # Update buffer size
        self.size = min(self.size + trajectories_added, self.capacity)

        steps_added = self.trajectory_lengths[compacted_indices].sum().item()

        return trajectories_added, steps_added

    def add_transitions(
        self,
        embedding_data: Union[CNNEmbeddingData, StructuredEmbeddingData],
        action_indices: torch.Tensor,  # [batch_size] - long
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        legal_masks: torch.Tensor,  # [batch_size, legal_mask_dim] - bool
        delta2: torch.Tensor,
        delta3: torch.Tensor,
        values: torch.Tensor,
        trajectory_indices: torch.Tensor,  # Which trajectory each transition belongs to
    ) -> None:
        """
        Add a batch of transitions to the buffer using embedding data.

        Args:
            embedding_data: Either CNNEmbeddingData or StructuredEmbeddingData containing embedding components
            action_indices: [batch_size] - long dtype
            log_probs: [batch_size] - float dtype
            rewards: [batch_size] - float dtype
            dones: [batch_size] - bool dtype
            legal_masks: [batch_size, legal_mask_dim] - bool dtype
            delta2: [batch_size] - float dtype
            delta3: [batch_size] - float dtype
            values: [batch_size] - float dtype
            trajectory_indices: [batch_size] - long dtype
        """

        if self.open_batch <= 0:
            raise RuntimeError(
                "Must call start_adding_trajectory_batches before adding data"
            )

        # Convert source trajectory indices to buffer indices by adding current position
        buffer_trajectory_indices = (trajectory_indices + self.position) % self.capacity

        # Get step positions within trajectories
        transition_counts = self.current_transition_counts[buffer_trajectory_indices]

        # Check if any trajectory would exceed max length
        if (transition_counts >= self.max_trajectory_length).any():
            raise ValueError("Some trajectories would exceed max_trajectory_length")

        # Handle different embedding data types
        if isinstance(embedding_data, CNNEmbeddingData):
            # Store CNN features - convert from float to bool for memory efficiency
            self.cards_features[buffer_trajectory_indices, transition_counts] = (
                embedding_data.cards.to(torch.bool)
            )
            self.actions_features[buffer_trajectory_indices, transition_counts] = (
                embedding_data.actions.to(torch.bool)
            )
        elif isinstance(embedding_data, StructuredEmbeddingData):
            if not self.is_transformer:
                raise TypeError(
                    "StructuredEmbeddingData provided but buffer is not in transformer mode"
                )
            # Append any newly provided tokens to the sequence (vectorized)
            self._append_from_embedding(buffer_trajectory_indices, embedding_data)
            # Record transition end pointer for this step
            self.transition_token_ends[buffer_trajectory_indices, transition_counts] = (
                self.current_token_positions[buffer_trajectory_indices]
            )
        else:
            raise TypeError("Unsupported embedding_data type")

        # Store common transition data
        self.action_indices[buffer_trajectory_indices, transition_counts] = (
            action_indices
        )
        # Expect log_probs shape [batch_size, num_bet_bins]
        self.log_probs[buffer_trajectory_indices, transition_counts, :] = log_probs
        self.rewards[buffer_trajectory_indices, transition_counts] = rewards
        self.dones[buffer_trajectory_indices, transition_counts] = dones
        self.legal_masks[buffer_trajectory_indices, transition_counts, :] = legal_masks
        self.delta2[buffer_trajectory_indices, transition_counts] = delta2
        self.delta3[buffer_trajectory_indices, transition_counts] = delta3
        self.values[buffer_trajectory_indices, transition_counts] = values

        # Advance step positions
        self.current_transition_counts[buffer_trajectory_indices] += 1

        # Handle trajectory completion
        completed_trajectories = buffer_trajectory_indices[dones]
        if len(completed_trajectories) > 0:
            # Mark completed trajectories as valid
            self.trajectory_lengths[completed_trajectories] = (
                self.current_transition_counts[completed_trajectories]
            )

    def update_last_transition_rewards(
        self, trajectory_indices: torch.Tensor, opponent_rewards: torch.Tensor
    ) -> None:
        """
        Update the last transition's reward for trajectories where opponent ended the hand.

        Args:
            trajectory_indices: [K] - trajectory indices to update
            opponent_rewards: [K] - rewards from our perspective returned by the environment
        """
        if trajectory_indices.numel() == 0:
            return

        # Convert source trajectory indices to buffer indices by adding current position
        buffer_trajectory_indices = (trajectory_indices + self.position) % self.capacity

        # Get current step positions for each trajectory
        transition_counts = self.current_transition_counts[buffer_trajectory_indices]

        # Only update trajectories that have at least one transition (not e.g. opponent folded immediately)
        valid_mask = transition_counts > 0
        if not valid_mask.any():
            return

        valid_trajectories = buffer_trajectory_indices[valid_mask]
        valid_transition_counts = transition_counts[valid_mask]
        valid_transition_positions = valid_transition_counts - 1
        valid_rewards = opponent_rewards[valid_mask]

        # Vectorized update of rewards and done flags
        self.trajectory_lengths[valid_trajectories] = valid_transition_counts
        self.rewards[valid_trajectories, valid_transition_positions] = valid_rewards
        self.dones[valid_trajectories, valid_transition_positions] = True

    def compute_gae_returns(self, gamma: float = 0.999, lambda_: float = 0.95) -> None:
        """Compute GAE advantages and returns for all stored trajectories using vectorized operations."""
        if self.size == 0:
            return

        # Get valid trajectory indices
        valid_indices = torch.where(self.trajectory_lengths > 0)[0]

        if len(valid_indices) == 0:
            return

        # Process trajectories in batches for vectorized GAE computation
        batch_size = len(valid_indices)
        max_length = self.trajectory_lengths[valid_indices].max()

        if max_length == 0:
            return

        # Extract batch data: [batch_size, max_length]
        batch_rewards = self.rewards[valid_indices, :max_length]
        batch_values = self.values[valid_indices, :max_length]
        batch_dones = self.dones[valid_indices, :max_length].float()

        # Add terminal values (zeros) for GAE computation
        terminal_values = torch.zeros(
            batch_size, 1, device=self.device, dtype=batch_values.dtype
        )
        batch_values_with_terminal = torch.cat([batch_values, terminal_values], dim=1)

        # Compute GAE using vectorized implementation
        advantages, returns = self._compute_gae_vectorized(
            batch_rewards, batch_values_with_terminal, batch_dones, gamma, lambda_
        )

        # Store computed values back to the buffer (vectorized)
        self.advantages[valid_indices, :max_length] = advantages
        self.returns[valid_indices, :max_length] = returns

    def _compute_gae_vectorized(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """
        Vectorized GAE computation for multiple trajectories.

        Args:
            rewards: [B, T] - rewards for each trajectory
            values: [B, T] - values including terminal values
            dones: [B, T] - done flags (0/1)
            gamma: discount factor
            lam: GAE lambda parameter

        Returns:
            advantages: [B, T] - computed advantages
            returns: [B, T] - computed returns
        """
        # rewards: [B, T], values: [B, T+1], dones: [B, T] (0/1)
        B, T = rewards.shape
        device, dtype = rewards.device, rewards.dtype

        # Compute lengths from dones: length is the first True in each row, or T if no True, all in torch
        # Find first done (==1) in each row. There should always be one.
        first_done_idx = dones.long().argmax(dim=1)
        lengths = first_done_idx + 1
        valid = torch.arange(T, device=device)[None, :] < lengths[:, None]
        next_valid = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
        nonterminal = (1.0 - dones.to(dtype)) * next_valid.to(dtype)

        # Mask deltas outside valid region
        deltas = (
            rewards + gamma * values[:, 1:] * (1.0 - dones.to(dtype)) - values[:, :-1]
        )
        deltas = deltas * valid.to(dtype)

        adv = torch.zeros(B, T, device=device, dtype=dtype)
        gae = torch.zeros(B, device=device, dtype=dtype)
        for t in reversed(range(T)):
            gae = deltas[:, t] + gamma * lam * nonterminal[:, t] * gae
            adv[:, t] = gae

        returns = adv + values[:, :-1]
        # Optional: zero out padding positions
        adv = adv * valid.to(dtype)
        returns = returns * valid.to(dtype) + values[:, :-1] * (~valid).to(dtype)
        return adv, returns

    def sample_trajectories(self, num_trajectories: int) -> Dict[str, torch.Tensor]:
        """Sample complete trajectories for PPO updates. Not currently used."""
        if self.size == 0:
            raise ValueError("No trajectories available")

        # Get valid trajectory indices
        valid_indices = torch.where(self.trajectory_lengths > 0)[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid trajectories available")

        # Sample trajectory indices (with replacement)
        traj_sample_indices = torch.randint(
            0, len(valid_indices), (num_trajectories,), device=self.device
        )
        traj_indices = valid_indices[traj_sample_indices]

        return {
            "embedding_data": self.data[traj_indices],
            "action_indices": self.action_indices[traj_indices],
            "log_probs_old": self.log_probs[traj_indices],
            "log_probs_old_full": self.log_probs[traj_indices],
            "advantages": self.advantages[traj_indices],
            "returns": self.returns[traj_indices],
            "legal_masks": self.legal_masks[traj_indices],
            "delta2": self.delta2[traj_indices],
            "delta3": self.delta3[traj_indices],
        }

    def sample_batch(
        self, rng: torch.Generator, batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Sample individual transitions for training (vectorized)."""
        if self.size == 0:
            raise ValueError("No trajectories available")

        # Get valid trajectory indices (this will only be nonzero in valid positions inside the buffer)
        valid_indices = torch.where(self.trajectory_lengths > 0)[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid trajectories available")

        # Get lengths of valid trajectories
        traj_lengths = self.trajectory_lengths[valid_indices]

        # Sample random trajectories
        # Sample trajectories weighted by their length (longer trajectories more likely)
        traj_weights = traj_lengths.float()
        traj_probs = traj_weights / traj_weights.sum()
        traj_sample_indices = torch.multinomial(
            traj_probs, batch_size, replacement=True, generator=rng
        )
        traj_indices = valid_indices[traj_sample_indices]  # [batch_size]
        traj_lengths_sampled = traj_lengths[traj_sample_indices]  # [batch_size]

        # For each sampled trajectory, sample a random step within its length (vectorized)
        transition_indices = torch.floor(
            torch.rand(batch_size, device=self.device, generator=rng)
            * traj_lengths_sampled.float()
        ).long()

        # Vectorized extraction (inlined variables)
        if self.is_transformer:
            # Use decision-end prefixes for training/inference consistency
            data = self._sample_transformer_steps(traj_indices, transition_indices)
        else:
            data = CNNEmbeddingData(
                cards=self.cards_features[traj_indices, transition_indices].to(
                    self.float_dtype
                ),
                actions=self.actions_features[traj_indices, transition_indices].to(
                    self.float_dtype
                ),
            )

        action_indices_sel = self.action_indices[traj_indices, transition_indices]
        full_log_probs = self.log_probs[traj_indices, transition_indices]
        action_log_probs = full_log_probs.gather(
            1, action_indices_sel.unsqueeze(1)
        ).squeeze(1)

        batch = {
            "embedding_data": data,
            "action_indices": action_indices_sel,
            # Keep scalar old log-probs for losses/ratios
            "log_probs_old": action_log_probs,
            # Provide full old distribution for exact KL
            "log_probs_old_full": full_log_probs,
            "advantages": self.advantages[traj_indices, transition_indices],
            "returns": self.returns[traj_indices, transition_indices],
            "legal_masks": self.legal_masks[traj_indices, transition_indices],
            "delta2": self.delta2[traj_indices, transition_indices],
            "delta3": self.delta3[traj_indices, transition_indices],
        }

        return batch

    def clear(self) -> None:
        """Clear all stored trajectories."""
        self.position = 0
        self.size = 0
        self.trajectory_lengths.zero_()
        self.current_transition_counts.zero_()

    def num_steps(self) -> int:
        """Total number of transitions (steps) stored across all valid trajectories."""
        return (self.trajectory_lengths).sum().item()

    def trim_to_steps(self, max_steps: int) -> None:
        """Trim oldest trajectories until removing the next traj would make total steps <= max_steps."""
        while (
            self.num_steps()
            - self.trajectory_lengths[(self.position - self.size) % self.capacity]
            > max_steps
            and self.size > 0
        ):
            oldest_idx = (self.position - self.size) % self.capacity
            self.trajectory_lengths[oldest_idx] = 0
            self.current_transition_counts[oldest_idx] = 0
            self.size -= 1

    def add_trajectory_legacy(self, trajectory) -> None:
        """Legacy method no longer supported - use add_batch with embedding data instead."""
        raise NotImplementedError(
            "add_trajectory_legacy is no longer supported. Use add_batch with CNNEmbeddingData or StructuredEmbeddingData instead."
        )

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------ Transformer helpers

    def _append_from_embedding(
        self,
        buffer_indices: torch.Tensor,
        embedding_data: StructuredEmbeddingData,
    ) -> None:
        """Append new tokens described by embedding_data to the per-trajectory streams."""

        if buffer_indices.numel() == 0:
            return

        M = buffer_indices.shape[0]
        L = self.max_sequence_length

        new_lens = embedding_data.lengths
        prev_lens = self.current_token_positions[buffer_indices]

        if (new_lens < prev_lens).any():
            raise ValueError("Embedding length cannot decrease when appending")

        if (new_lens > L).any():
            raise ValueError("Token sequence exceeds configured sequence length")

        # Build mask for positions to append: prev_len <= pos < new_len per row
        pos = torch.arange(L, device=self.device).unsqueeze(0).expand(M, L)
        mask = (pos >= prev_lens.unsqueeze(1)) & (pos < new_lens.unsqueeze(1))
        rows, cols = torch.where(mask)
        buffer_rows = buffer_indices[rows]

        if not mask.any():
            return

        self.data.token_ids[buffer_rows, cols] = embedding_data.token_ids[rows, cols]
        self.data.token_streets[buffer_rows, cols] = embedding_data.token_streets[
            rows, cols
        ]
        self.data.card_ranks[buffer_rows, cols] = embedding_data.card_ranks[rows, cols]
        self.data.card_suits[buffer_rows, cols] = embedding_data.card_suits[rows, cols]
        self.data.action_actors[buffer_rows, cols] = embedding_data.action_actors[
            rows, cols
        ]
        self.data.action_legal_masks[buffer_rows, cols] = (
            embedding_data.action_legal_masks[rows, cols]
        )
        self.data.context_features[buffer_rows, cols] = embedding_data.context_features[
            rows, cols
        ].to(self.float_dtype)

        # Update positions to new_lens
        self.current_token_positions[buffer_rows] = new_lens[rows]

    def add_tokens(
        self,
        embedding_data: StructuredEmbeddingData,
        trajectory_indices: torch.Tensor,
    ) -> None:
        """Append street markers and dealt cards to token streams.

        Use when the game state changes without taking an action (e.g., new street, cards dealt).
        The provided embedding_data should reflect the full sequence up to the new point; only the
        newly appended portion will be copied.
        """

        if self.open_batch <= 0:
            raise RuntimeError(
                "Must call start_adding_trajectory_batches before adding data"
            )

        buffer_trajectory_indices = (trajectory_indices + self.position) % self.capacity
        if not self.is_transformer:
            raise TypeError("add_tokens is only valid in transformer mode")
        # Append full new suffix from embedding_data to token streams
        self._append_from_embedding(buffer_trajectory_indices, embedding_data)

    def add_opponent_actions(
        self,
        trajectory_indices: torch.Tensor,
        action_indices: torch.Tensor,
        legal_masks: torch.Tensor,
        streets: torch.Tensor,
    ) -> None:
        """Add opponent actions to the token streams.

        Args:
            trajectory_indices: Which trajectories to update
            action_indices: Action IDs to add
            legal_masks: Legal action masks
            streets: Street IDs for the actions
        """
        if self.open_batch <= 0:
            raise RuntimeError(
                "Must call start_adding_trajectory_batches before adding data"
            )

        if not self.is_transformer:
            raise TypeError("add_opponent_actions is only valid in transformer mode")

        buffer_trajectory_indices = (trajectory_indices + self.position) % self.capacity
        batch_size = trajectory_indices.shape[0]
        seq_len = self.max_sequence_length

        token_ids = torch.full(
            (batch_size, seq_len), -1, dtype=torch.long, device=self.device
        )
        token_streets = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        card_ranks = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        card_suits = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        action_actors = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        action_legal_masks_tensor = torch.zeros(
            batch_size, seq_len, self.num_bet_bins, dtype=torch.bool, device=self.device
        )
        context_features = torch.zeros(
            batch_size, seq_len, 10, dtype=self.float_dtype, device=self.device
        )

        # Add action tokens (opponent = actor 1)
        for i in range(batch_size):
            pos = self.current_token_positions[buffer_trajectory_indices[i]].item()
            if pos < seq_len:
                token_ids[i, pos] = (
                    Special.NUM_SPECIAL.value + 52 + action_indices[i].item()
                )
                action_actors[i, pos] = 1  # Opponent
                token_streets[i, pos] = streets[i].item()
                action_legal_masks_tensor[i, pos] = legal_masks[i]

        lengths = self.current_token_positions[buffer_trajectory_indices] + 1

        embedding_data = StructuredEmbeddingData(
            token_ids=token_ids,
            token_streets=token_streets,
            card_ranks=card_ranks,
            card_suits=card_suits,
            action_actors=action_actors,
            action_legal_masks=action_legal_masks_tensor,
            context_features=context_features,
            lengths=lengths,
        )

        self._append_from_embedding(buffer_trajectory_indices, embedding_data)

    def get_current_transition_counts(
        self, trajectory_indices: torch.Tensor
    ) -> torch.Tensor:
        """Get the current transition counts for the given trajectory indices."""
        if self.open_batch <= 0:
            raise RuntimeError("No open batch, cannot get current transition counts")

        buffer_trajectory_indices = (trajectory_indices + self.position) % self.capacity
        return self.current_transition_counts[buffer_trajectory_indices]

    def _sample_transformer_steps(
        self,
        traj_indices: torch.Tensor,
        step_indices: torch.Tensor,
    ) -> StructuredEmbeddingData:
        """Materialize transformer sequences up to each step's end token.

        If use_decision_end is True, cut at decision_end (after CONTEXT), otherwise
        cut at transition_end (after ACTION).
        """

        ends = self.transition_token_ends[traj_indices, step_indices]

        result = self.data[traj_indices]
        result.lengths = ends

        range_tensor = torch.arange(self.max_sequence_length, device=self.device)
        mask = range_tensor.unsqueeze(0) >= ends.unsqueeze(1)

        result.token_ids[mask] = -1
        result.token_streets[mask] = 0
        result.card_ranks[mask] = 0
        result.card_suits[mask] = 0
        result.action_actors[mask] = 0
        result.action_legal_masks[mask] = False
        result.context_features[mask] = 0

        return result
