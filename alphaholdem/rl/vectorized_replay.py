from __future__ import annotations

from typing import Dict, Union

import torch

# Import embedding data classes
from ..models.cnn_embedding_data import CNNEmbeddingData
from ..models.transformer.embedding_data import StructuredEmbeddingData


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
        sequence_length: int = 50,  # Sequence length for transformer models
    ):
        self.capacity = capacity  # Number of trajectories
        self.max_trajectory_length = max_trajectory_length
        self.device = device
        self.float_dtype = float_dtype  # Store float dtype for use in methods
        self.is_transformer = is_transformer  # Store transformer flag
        self.sequence_length = (
            sequence_length  # Store sequence length for transformer models
        )
        self.position = 0  # Next trajectory write position
        self.size = 0  # Total number of valid trajectories

        self.open_batch = (
            -1
        )  # -1 if no batch is open, otherwise the nominal size of the open batch

        C, T, L = capacity, max_trajectory_length, sequence_length

        # Pre-allocate tensors for all transition fields: (capacity, max_trajectory_length, ...)
        if not is_transformer:
            # CNN model tensors
            # Cards features tensor: (capacity, max_trajectory_length, 6, 4, 13) - bool dtype
            self.cards_features = torch.zeros(
                C,
                T,
                6,
                4,
                13,  # Fixed cards shape
                dtype=torch.bool,
                device=device,
            )

            # Actions features tensor: (capacity, max_trajectory_length, 24, 4, num_bet_bins) - bool dtype
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
            # Transformer structured embedding fields: (capacity, max_trajectory_length, ...)
            self.token_ids = torch.full(
                (C, T, L),
                -1,
                dtype=torch.int8,
                device=device,
            )
            self.card_ranks = torch.zeros(
                C,
                T,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.card_suits = torch.zeros(
                C,
                T,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.card_streets = torch.zeros(
                C,
                T,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.action_actors = torch.zeros(
                C,
                T,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.action_streets = torch.zeros(
                C,
                T,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.action_legal_masks = torch.zeros(
                C,
                T,
                L,
                8,
                dtype=torch.bool,
                device=device,
            )
            # Consolidated context tensor [capacity, max_trajectory_length, sequence_length, 10]
            self.context_features = torch.zeros(
                C,
                T,
                L,
                10,
                dtype=float_dtype,  # Use float_dtype for context features
                device=device,
            )

        self.action_indices = torch.zeros(C, T, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(C, T, dtype=float_dtype, device=device)
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

        # Track trajectory metadata
        self.trajectory_lengths = torch.zeros(capacity, dtype=torch.long, device=device)
        self.valid_trajectories = torch.zeros(capacity, dtype=torch.bool, device=device)

        # Track current step position within each trajectory (relative to ring buffer start)
        self.current_step_positions = torch.zeros(
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
        self.valid_trajectories[clear_indices] = False
        self.current_step_positions[clear_indices] = 0

        # Clear the actual data tensors for these rows
        if not self.is_transformer:
            # CNN model tensors
            self.cards_features[clear_indices] = False
            self.actions_features[clear_indices] = False
        else:
            # Transformer structured embedding tensors
            self.token_ids[clear_indices] = -1
            self.card_ranks[clear_indices] = 0
            self.card_suits[clear_indices] = 0
            self.card_streets[clear_indices] = 0
            self.action_actors[clear_indices] = 0
            self.action_streets[clear_indices] = 0
            self.action_legal_masks[clear_indices] = False
            self.context_features[clear_indices] = 0

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
            "valid_trajectories",
            "current_step_positions",
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

        if not self.is_transformer:
            # CNN model fields
            cnn_fields = ["cards_features", "actions_features"]
            all_fields = common_fields + cnn_fields
        else:
            # Transformer model fields
            transformer_fields = [
                "token_ids",
                "card_ranks",
                "card_suits",
                "card_streets",
                "action_actors",
                "action_streets",
                "action_legal_masks",
                "context_features",
            ]
            all_fields = common_fields + transformer_fields

        for attr in all_fields:
            buf = getattr(self, attr)
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

    def add_batch(
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
        step_positions = self.current_step_positions[buffer_trajectory_indices]

        # Check if any trajectory would exceed max length
        if (step_positions >= self.max_trajectory_length).any():
            raise ValueError("Some trajectories would exceed max_trajectory_length")

        # Clamp step positions to be within bounds
        step_positions = torch.clamp(step_positions, 0, self.max_trajectory_length - 1)

        # Handle different embedding data types
        if isinstance(embedding_data, CNNEmbeddingData):
            # Store CNN features - vectorized assignment using advanced indexing
            self.cards_features[buffer_trajectory_indices, step_positions] = (
                embedding_data.cards
            )
            self.actions_features[buffer_trajectory_indices, step_positions] = (
                embedding_data.actions
            )
        elif isinstance(embedding_data, StructuredEmbeddingData):
            # Store structured embedding data - vectorized assignment using advanced indexing
            # Convert to buffer dtypes for memory efficiency
            self.token_ids[buffer_trajectory_indices, step_positions] = (
                embedding_data.token_ids.to(torch.int8)
            )
            self.card_ranks[buffer_trajectory_indices, step_positions] = (
                embedding_data.card_ranks.to(torch.uint8)
            )
            self.card_suits[buffer_trajectory_indices, step_positions] = (
                embedding_data.card_suits.to(torch.uint8)
            )
            self.card_streets[buffer_trajectory_indices, step_positions] = (
                embedding_data.card_streets.to(torch.uint8)
            )
            self.action_actors[buffer_trajectory_indices, step_positions] = (
                embedding_data.action_actors.to(torch.uint8)
            )
            self.action_streets[buffer_trajectory_indices, step_positions] = (
                embedding_data.action_streets.to(torch.uint8)
            )
            self.action_legal_masks[buffer_trajectory_indices, step_positions] = (
                embedding_data.action_legal_masks.to(torch.bool)
            )
            self.context_features[buffer_trajectory_indices, step_positions] = (
                embedding_data.context_features.to(self.float_dtype)
            )

        # Store common transition data
        self.action_indices[buffer_trajectory_indices, step_positions] = action_indices
        self.log_probs[buffer_trajectory_indices, step_positions] = log_probs
        self.rewards[buffer_trajectory_indices, step_positions] = rewards
        self.dones[buffer_trajectory_indices, step_positions] = dones
        self.legal_masks[buffer_trajectory_indices, step_positions, :] = legal_masks
        self.delta2[buffer_trajectory_indices, step_positions] = delta2
        self.delta3[buffer_trajectory_indices, step_positions] = delta3
        self.values[buffer_trajectory_indices, step_positions] = values

        # Advance step positions
        self.current_step_positions[buffer_trajectory_indices] += 1

        # Handle trajectory completion
        completed_trajectories = trajectory_indices[dones]
        if len(completed_trajectories) > 0:
            # Convert completed trajectory indices to buffer indices
            completed_buffer_indices = (
                completed_trajectories + self.position
            ) % self.capacity

            # Mark completed trajectories as valid
            self.valid_trajectories[completed_buffer_indices] = True
            self.trajectory_lengths[completed_buffer_indices] = (
                self.current_step_positions[completed_buffer_indices]
            )

            # Reset step positions for completed trajectories
            self.current_step_positions[completed_buffer_indices] = 0

    def compute_gae_returns(self, gamma: float = 0.999, lambda_: float = 0.95) -> None:
        """Compute GAE advantages and returns for all stored trajectories using vectorized operations."""
        if self.size == 0:
            return

        # Get valid trajectory indices
        valid_indices = torch.where(self.valid_trajectories)[0]

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

    def update_opponent_rewards(
        self, trajectory_indices: torch.Tensor, opponent_rewards: torch.Tensor
    ) -> None:
        """
        Update the last transition's reward for trajectories where opponent ended the hand.

        Args:
            trajectory_indices: [K] - trajectory indices to update
            opponent_rewards: [K] - rewards from opponent's perspective (will be negated)
        """
        if trajectory_indices.numel() == 0:
            return

        # Get current step positions for each trajectory
        step_positions = self.current_step_positions[trajectory_indices]

        # Only update trajectories that have at least one step
        valid_mask = step_positions > 0
        if not valid_mask.any():
            return

        valid_trajectories = trajectory_indices[valid_mask]
        valid_step_positions = step_positions[valid_mask] - 1  # Last step index
        valid_rewards = -opponent_rewards[valid_mask]  # Negate for our perspective

        # Vectorized update of rewards and done flags
        self.rewards[valid_trajectories, valid_step_positions] = valid_rewards
        self.dones[valid_trajectories, valid_step_positions] = True

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
        valid_indices = torch.where(self.valid_trajectories)[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid trajectories available")

        # Sample trajectory indices
        traj_indices = valid_indices[
            torch.randint(
                0, len(valid_indices), (num_trajectories,), device=self.device
            )
        ]

        # Vectorized collection of all transitions from sampled trajectories
        lengths = self.trajectory_lengths[traj_indices]
        max_len = lengths.max()
        mask = torch.arange(max_len, device=self.device).unsqueeze(
            0
        ) < lengths.unsqueeze(
            1
        )  # [num_traj, max_len]

        # Gather all data for sampled trajectories up to their respective lengths
        if not self.is_transformer:
            # CNN model fields
            all_cards_features = self.cards_features[traj_indices, :max_len]
            all_actions_features = self.actions_features[traj_indices, :max_len]
        else:
            # Transformer model fields
            all_token_ids = self.token_ids[traj_indices, :max_len]
            all_card_ranks = self.card_ranks[traj_indices, :max_len]
            all_card_suits = self.card_suits[traj_indices, :max_len]
            all_card_streets = self.card_streets[traj_indices, :max_len]
            all_action_actors = self.action_actors[traj_indices, :max_len]
            all_action_streets = self.action_streets[traj_indices, :max_len]
            all_action_legal_masks = self.action_legal_masks[traj_indices, :max_len]
            all_context_features = self.context_features[traj_indices, :max_len]
        action_indices = self.action_indices[traj_indices, :max_len]
        logps = self.log_probs[traj_indices, :max_len]
        advs = self.advantages[traj_indices, :max_len]
        rets = self.returns[traj_indices, :max_len]
        legal = self.legal_masks[traj_indices, :max_len]
        d2 = self.delta2[traj_indices, :max_len]
        d3 = self.delta3[traj_indices, :max_len]

        # Flatten only valid steps (mask out padding)
        mask_flat = mask.flatten()
        # Flatten the data for return
        action_indices_flat = action_indices.reshape(-1)[mask_flat]
        logps_flat = logps.reshape(-1)[mask_flat]
        advs_flat = advs.reshape(-1)[mask_flat]
        rets_flat = rets.reshape(-1)[mask_flat]
        legal_flat = legal.reshape(-1, legal.shape[-1])[mask_flat]

        # Return the flattened data directly
        if not self.is_transformer:
            # CNN model return
            if all_cards_features.numel() > 0:
                # Flatten CNN features
                cards_flat = all_cards_features.reshape(
                    -1, *all_cards_features.shape[2:]
                )[mask_flat]
                actions_flat = all_actions_features.reshape(
                    -1, *all_actions_features.shape[2:]
                )[mask_flat]
                return {
                    "cards_features": cards_flat,
                    "actions_features": actions_flat,
                    "action_indices": action_indices_flat,
                    "log_probs": logps_flat,
                    "advantages": advs_flat,
                    "returns": rets_flat,
                    "legal_masks": legal_flat,
                }
            else:
                raise ValueError("No valid trajectories found")
        else:
            # Transformer model return
            if all_token_ids.numel() > 0:
                # Flatten transformer features
                token_ids_flat = all_token_ids.reshape(-1, *all_token_ids.shape[2:])[
                    mask_flat
                ]
                card_ranks_flat = all_card_ranks.reshape(-1, *all_card_ranks.shape[2:])[
                    mask_flat
                ]
                card_suits_flat = all_card_suits.reshape(-1, *all_card_suits.shape[2:])[
                    mask_flat
                ]
                card_streets_flat = all_card_streets.reshape(
                    -1, *all_card_streets.shape[2:]
                )[mask_flat]
                action_actors_flat = all_action_actors.reshape(
                    -1, *all_action_actors.shape[2:]
                )[mask_flat]
                action_streets_flat = all_action_streets.reshape(
                    -1, *all_action_streets.shape[2:]
                )[mask_flat]
                action_legal_masks_flat = all_action_legal_masks.reshape(
                    -1, *all_action_legal_masks.shape[2:]
                )[mask_flat]
                context_features_flat = all_context_features.reshape(
                    -1, *all_context_features.shape[2:]
                )[mask_flat]
                return {
                    "token_ids": token_ids_flat,
                    "card_ranks": card_ranks_flat,
                    "card_suits": card_suits_flat,
                    "card_streets": card_streets_flat,
                    "action_actors": action_actors_flat,
                    "action_streets": action_streets_flat,
                    "action_legal_masks": action_legal_masks_flat,
                    "context_features": context_features_flat,
                    "action_indices": action_indices_flat,
                    "log_probs": logps_flat,
                    "advantages": advs_flat,
                    "returns": rets_flat,
                    "legal_masks": legal_flat,
                }
            else:
                raise ValueError("No valid trajectories found")

    def sample_batch(
        self, rng: torch.Generator, batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Sample individual transitions for training (vectorized)."""
        if self.size == 0:
            raise ValueError("No trajectories available")

        # Get valid trajectory indices
        valid_indices = torch.where(self.valid_trajectories)[0]

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
        step_indices = torch.floor(
            torch.rand(batch_size, device=self.device, generator=rng)
            * traj_lengths_sampled.float()
        ).long()

        # Vectorized extraction (inlined variables)
        if self.is_transformer:
            data = StructuredEmbeddingData(
                token_ids=self.token_ids[traj_indices, step_indices],
                card_ranks=self.card_ranks[traj_indices, step_indices],
                card_suits=self.card_suits[traj_indices, step_indices],
                card_streets=self.card_streets[traj_indices, step_indices],
                action_actors=self.action_actors[traj_indices, step_indices],
                action_streets=self.action_streets[traj_indices, step_indices],
                action_legal_masks=self.action_legal_masks[traj_indices, step_indices],
                context_features=self.context_features[traj_indices, step_indices],
            )
        else:
            data = CNNEmbeddingData(
                cards=self.cards_features[traj_indices, step_indices],
                actions=self.actions_features[traj_indices, step_indices],
            )

        # CNN model fields
        batch = {
            "embedding_data": data,
            "action_indices": self.action_indices[traj_indices, step_indices],
            "log_probs_old": self.log_probs[traj_indices, step_indices],
            "advantages": self.advantages[traj_indices, step_indices],
            "returns": self.returns[traj_indices, step_indices],
            "legal_masks": self.legal_masks[traj_indices, step_indices],
            "delta2": self.delta2[traj_indices, step_indices],
            "delta3": self.delta3[traj_indices, step_indices],
        }

        return batch

    def clear(self) -> None:
        """Clear all stored trajectories."""
        self.position = 0
        self.size = 0
        self.trajectory_lengths.zero_()
        self.valid_trajectories.zero_()
        self.current_step_positions.zero_()

    def num_steps(self) -> int:
        """Total number of transitions (steps) stored across all valid trajectories."""
        return (self.trajectory_lengths * self.valid_trajectories).sum().item()

    def trim_to_steps(self, max_steps: int) -> None:
        """Trim oldest trajectories until removing the next traj would make total steps <= max_steps."""
        # Remove oldest trajectories until removing the next would make total steps <= max_steps
        while (
            self.num_steps()
            - self.trajectory_lengths[(self.position - self.size) % self.capacity]
            > max_steps
            and self.size > 0
        ):
            oldest_idx = (self.position - self.size) % self.capacity
            self.valid_trajectories[oldest_idx] = False
            self.trajectory_lengths[oldest_idx] = 0
            self.current_step_positions[oldest_idx] = 0
            self.size -= 1

    def add_trajectory_legacy(self, trajectory) -> None:
        """Legacy method no longer supported - use add_batch with embedding data instead."""
        raise NotImplementedError(
            "add_trajectory_legacy is no longer supported. Use add_batch with CNNEmbeddingData or StructuredEmbeddingData instead."
        )

    def __len__(self) -> int:
        return self.size
