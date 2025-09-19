from __future__ import annotations

from typing import Dict, Union

import torch

from ..env.hunl_tensor_env import HUNLTensorEnv

# Import embedding data classes
from ..models.cnn_embedding_data import CNNEmbeddingData
from ..models.transformer.embedding_data import StructuredEmbeddingData
from ..models.transformer.tokens import Cls, Special, Context


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
        self.num_bet_bins = num_bet_bins
        self.position = 0  # Next trajectory write position (end of ring buffer)
        self.size = 0  # Total number of valid trajectories

        self.open_batch = (
            -1
        )  # -1 if no batch is open, otherwise the nominal size of the open batch

        C, T, L = capacity, max_trajectory_length, sequence_length

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
            # Token id offset layout matches TransformerStateEncoder
            self.special_offset = 0
            self.card_offset = Special.NUM_SPECIAL.value
            self.action_offset = self.card_offset + 52
            self.token_ids = torch.full(
                (C, L),
                -1,
                dtype=torch.int8,
                device=device,
            )
            self.card_ranks = torch.zeros(
                C,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.card_suits = torch.zeros(
                C,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.card_streets = torch.zeros(
                C,
                L,
                dtype=torch.uint8,
                device=device,
            )
            self.action_actors = torch.full(
                (C, L),
                -1,
                dtype=torch.uint8,
                device=device,
            )
            self.action_streets = torch.full(
                (C, L),
                -1,
                dtype=torch.uint8,
                device=device,
            )
            self.action_legal_masks = torch.zeros(
                C,
                L,
                num_bet_bins,
                dtype=torch.bool,
                device=device,
            )
            self.context_features = torch.zeros(
                C,
                L,
                10,
                dtype=float_dtype,
                device=device,
            )
            self.token_positions = torch.zeros(
                C,
                dtype=torch.long,
                device=device,
            )
            self.decision_token_ends = torch.zeros(
                C,
                T,
                dtype=torch.long,
                device=device,
            )
            self.transition_token_ends = torch.zeros(
                C,
                T,
                dtype=torch.long,
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

        # Track current step position within each trajectory (relative to ring buffer start)
        self.current_step_positions = torch.zeros(
            capacity, dtype=torch.long, device=device
        )

        # Track positions of the first two hole cards (preflop) per trajectory row
        # These are buffer-row-local positions, used to temporarily swap to opponent
        # hole cards for self-play opponent action tokens.
        self.hole_pos0 = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.hole_pos1 = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.hero_hole0 = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.hero_hole1 = torch.full((capacity,), -1, dtype=torch.long, device=device)

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
            self.action_actors[clear_indices] = -1
            self.action_streets[clear_indices] = -1
            self.action_legal_masks[clear_indices] = False
            self.context_features[clear_indices] = 0
            self.token_positions[clear_indices] = 0
            self.decision_token_ends[clear_indices] = 0
            self.transition_token_ends[clear_indices] = 0
            # Reset hole tracking for these rows
            self.hole_pos0[clear_indices] = -1
            self.hole_pos1[clear_indices] = -1
            self.hero_hole0[clear_indices] = -1
            self.hero_hole1[clear_indices] = -1

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
                "decision_token_ends",
                "transition_token_ends",
                "token_positions",
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
        step_positions = self.current_step_positions[buffer_trajectory_indices]

        # Check if any trajectory would exceed max length
        if (step_positions >= self.max_trajectory_length).any():
            raise ValueError("Some trajectories would exceed max_trajectory_length")

        # Clamp step positions to be within bounds
        step_positions = torch.clamp(step_positions, 0, self.max_trajectory_length - 1)

        # Handle different embedding data types
        if isinstance(embedding_data, CNNEmbeddingData):
            # Store CNN features - convert from float to bool for memory efficiency
            self.cards_features[buffer_trajectory_indices, step_positions] = (
                embedding_data.cards.to(torch.bool)
            )
            self.actions_features[buffer_trajectory_indices, step_positions] = (
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
            self.transition_token_ends[buffer_trajectory_indices, step_positions] = (
                self.token_positions[buffer_trajectory_indices]
            )
        else:
            raise TypeError("Unsupported embedding_data type")

        # Store common transition data
        self.action_indices[buffer_trajectory_indices, step_positions] = action_indices
        # Expect log_probs shape [batch_size, num_bet_bins]
        self.log_probs[buffer_trajectory_indices, step_positions, :] = log_probs
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
            self.trajectory_lengths[completed_buffer_indices] = (
                self.current_step_positions[completed_buffer_indices]
            )

            # Reset step positions for completed trajectories
            self.current_step_positions[completed_buffer_indices] = 0

    def update_opponent_rewards(
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
        step_positions = self.current_step_positions[buffer_trajectory_indices]

        # Only update trajectories that have at least one step (not e.g. opponent folded immediately)
        valid_mask = step_positions > 0
        if not valid_mask.any():
            return

        valid_trajectories = buffer_trajectory_indices[valid_mask]
        valid_step_positions = step_positions[valid_mask] - 1  # Last step index
        valid_rewards = opponent_rewards[valid_mask]

        # Vectorized update of rewards and done flags
        self.trajectory_lengths[valid_trajectories] = valid_step_positions + 1
        self.rewards[valid_trajectories, valid_step_positions] = valid_rewards
        self.dones[valid_trajectories, valid_step_positions] = True

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

        C, T, L = self.capacity, self.max_trajectory_length, self.sequence_length

        # Get valid trajectory indices
        valid_indices = torch.where(self.trajectory_lengths > 0)[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid trajectories available")

        # Sample trajectory indices (with replacement)
        traj_sample_indices = torch.randint(
            0, len(valid_indices), (num_trajectories,), device=self.device
        )
        traj_indices = valid_indices[traj_sample_indices]

        # Get lengths of sampled trajectories
        traj_lengths = self.trajectory_lengths[traj_indices]
        max_len = traj_lengths.max().item()

        # Build mask for valid steps in each trajectory
        mask = torch.arange(max_len, device=self.device).unsqueeze(
            0
        ) < traj_lengths.unsqueeze(
            1
        )  # [num_traj, max_len]
        mask_flat = mask.flatten()

        # Gather all data for sampled trajectories up to their respective lengths
        if not self.is_transformer:
            # CNN model fields
            all_cards_features = self.cards_features[traj_indices, :max_len]
            all_actions_features = self.actions_features[traj_indices, :max_len]

        action_indices = self.action_indices[
            traj_indices, :max_len
        ]  # [num_traj, max_len]
        logps_full = self.log_probs[
            traj_indices, :max_len
        ]  # [num_traj, max_len, num_bet_bins]
        num_traj = traj_indices.shape[0]
        flat_logps_full = logps_full.reshape(
            -1, logps_full.shape[-1]
        )  # [num_traj*max_len, num_bet_bins]
        flat_action_indices = action_indices.reshape(-1)  # [num_traj*max_len]
        logps = flat_logps_full[
            torch.arange(
                flat_action_indices.shape[0], device=flat_action_indices.device
            ),
            flat_action_indices,
        ]
        logps = logps.reshape(num_traj, max_len)  # [num_traj, max_len]
        advs = self.advantages[traj_indices, :max_len]
        rets = self.returns[traj_indices, :max_len]
        legal = self.legal_masks[traj_indices, :max_len]
        d2 = self.delta2[traj_indices, :max_len]
        d3 = self.delta3[traj_indices, :max_len]

        # Flatten only valid steps (mask out padding)
        action_indices_flat = action_indices.reshape(-1)[mask_flat]
        logps_full_flat = logps_full.reshape(num_traj * max_len, -1)[mask_flat]
        logps_flat = logps.reshape(-1)[mask_flat]
        advs_flat = advs.reshape(-1)[mask_flat]
        rets_flat = rets.reshape(-1)[mask_flat]
        legal_flat = legal.reshape(-1, legal.shape[-1])[mask_flat]
        d2_flat = d2.reshape(-1)[mask_flat]
        d3_flat = d3.reshape(-1)[mask_flat]

        if not self.is_transformer:
            if all_cards_features.numel() == 0:
                raise ValueError("No valid trajectories found")
            cards_flat = all_cards_features.reshape(-1, *all_cards_features.shape[2:])[
                mask_flat
            ]
            actions_flat = all_actions_features.reshape(
                -1, *all_actions_features.shape[2:]
            )[mask_flat]
            return {
                "cards_features": cards_flat,
                "actions_features": actions_flat,
                "action_indices": action_indices_flat,
                "log_probs_old": logps_flat,
                "log_probs_old_full": logps_full_flat,
                "advantages": advs_flat,
                "returns": rets_flat,
                "legal_masks": legal_flat,
            }
        else:
            # Transformer: materialize per-step prefixes using transition_token_ends
            # Build per-sample (traj, step) lists
            traj_list = []
            step_list = []
            for i in range(traj_indices.shape[0]):
                length_i = int(traj_lengths[i].item())
                if length_i > 0:
                    traj_list.append(traj_indices[i].repeat(length_i))
                    step_list.append(torch.arange(length_i, device=self.device))
            if len(traj_list) == 0:
                raise ValueError("No valid trajectories found")
            traj_idx_flat = torch.cat(traj_list, dim=0)
            step_idx_flat = torch.cat(step_list, dim=0)

            data = self._sample_transformer_steps(traj_idx_flat, step_idx_flat)

            return {
                "embedding_data": data,
                "action_indices": action_indices_flat,
                "log_probs_old": logps_flat,
                "log_probs_old_full": logps_full_flat,
                "advantages": advs_flat,
                "returns": rets_flat,
                "legal_masks": legal_flat,
                "delta2": d2_flat,
                "delta3": d3_flat,
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
        step_indices = torch.floor(
            torch.rand(batch_size, device=self.device, generator=rng)
            * traj_lengths_sampled.float()
        ).long()

        # Vectorized extraction (inlined variables)
        if self.is_transformer:
            # Use decision-end prefixes for training/inference consistency
            data = self._sample_transformer_steps(
                traj_indices, step_indices, use_decision_end=True
            )
        else:
            data = CNNEmbeddingData(
                cards=self.cards_features[traj_indices, step_indices].to(
                    self.float_dtype
                ),
                actions=self.actions_features[traj_indices, step_indices].to(
                    self.float_dtype
                ),
            )

        # CNN model fields
        action_indices_sel = self.action_indices[traj_indices, step_indices]
        full_log_probs = self.log_probs[traj_indices, step_indices]
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
            "advantages": self.advantages[traj_indices, step_indices],
            "returns": self.returns[traj_indices, step_indices],
            "legal_masks": self.legal_masks[traj_indices, step_indices],
            "delta2": self.delta2[traj_indices, step_indices],
            "delta3": self.delta3[traj_indices, step_indices],
        }

        return batch

    def record_step_metadata(
        self,
        trajectory_indices: torch.Tensor,
        decision_ends: torch.Tensor,
        transition_ends: torch.Tensor,
        action_indices: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        legal_masks: torch.Tensor,
        delta2: torch.Tensor,
        delta3: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Store per-step scalars and end indices for transformer steps.

        Assumes tokens have already been appended via add_tokens and token_positions updated.
        """
        if not self.is_transformer:
            raise TypeError("record_step_metadata is only valid in transformer mode")
        if trajectory_indices.numel() == 0:
            return

        bidx = (trajectory_indices + self.position) % self.capacity
        step_positions = self.current_step_positions[bidx]

        # Record ends
        self.decision_token_ends[bidx, step_positions] = decision_ends
        self.transition_token_ends[bidx, step_positions] = transition_ends

        # Store step scalars
        self.action_indices[bidx, step_positions] = action_indices
        self.log_probs[bidx, step_positions, :] = log_probs
        self.rewards[bidx, step_positions] = rewards
        self.dones[bidx, step_positions] = dones
        self.legal_masks[bidx, step_positions, :] = legal_masks
        self.delta2[bidx, step_positions] = delta2
        self.delta3[bidx, step_positions] = delta3
        self.values[bidx, step_positions] = values

        # Advance step positions
        self.current_step_positions[bidx] += 1

        # Handle completed
        completed = trajectory_indices[dones]
        if len(completed) > 0:
            completed_b = (completed + self.position) % self.capacity
            self.trajectory_lengths[completed_b] = self.current_step_positions[
                completed_b
            ]
            self.current_step_positions[completed_b] = 0

    def clear(self) -> None:
        """Clear all stored trajectories."""
        self.position = 0
        self.size = 0
        self.trajectory_lengths.zero_()
        self.current_step_positions.zero_()

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
            self.current_step_positions[oldest_idx] = 0
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

        bidx = buffer_indices
        batch = bidx.shape[0]
        seq_len = self.sequence_length

        new_lens = embedding_data.lengths[:batch].long()
        prev_lens = self.token_positions[bidx]

        if (new_lens < prev_lens).any():
            raise ValueError("Embedding length cannot decrease when appending")

        if (new_lens > seq_len).any():
            raise ValueError("Token sequence exceeds configured sequence length")

        # Build mask for positions to append: prev_len <= pos < new_len per row
        pos = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch, -1)
        mask = (pos >= prev_lens.unsqueeze(1)) & (pos < new_lens.unsqueeze(1))

        if not mask.any():
            return

        # Row-wise views
        token_ids_src = embedding_data.token_ids[:batch]
        card_ranks_src = embedding_data.card_ranks[:batch]
        card_suits_src = embedding_data.card_suits[:batch]
        card_streets_src = embedding_data.card_streets[:batch]
        action_actors_src = embedding_data.action_actors[:batch]
        action_streets_src = embedding_data.action_streets[:batch]
        action_legal_src = embedding_data.action_legal_masks[:batch]
        context_src = embedding_data.context_features[:batch]

        self.token_ids[bidx][mask] = token_ids_src[mask].to(torch.int8)
        self.card_ranks[bidx][mask] = card_ranks_src[mask].to(torch.uint8)
        self.card_suits[bidx][mask] = card_suits_src[mask].to(torch.uint8)
        self.card_streets[bidx][mask] = card_streets_src[mask].to(torch.uint8)
        self.action_actors[bidx][mask] = action_actors_src[mask].to(torch.uint8)
        self.action_streets[bidx][mask] = action_streets_src[mask].to(torch.uint8)
        # For 3D, expand mask to match action_legal_masks shape
        mask3 = mask.unsqueeze(-1).expand(-1, -1, self.num_bet_bins)
        self.action_legal_masks[bidx][mask3] = action_legal_src[mask3].to(torch.bool)
        # For context features, use a different mask expansion
        mask3_context = mask.unsqueeze(-1).expand(
            -1, -1, self.context_features.shape[-1]
        )
        self.context_features[bidx][mask3_context] = context_src[mask3_context].to(
            self.float_dtype
        )

        # Update positions to new_lens
        self.token_positions[bidx] = new_lens

    def set_hole_cards(
        self, traj_indices: torch.Tensor, card0: torch.Tensor, card1: torch.Tensor
    ) -> None:
        """Temporarily overwrite preflop hole card tokens to given cards for trajectories.

        This enables opponent-perspective appends during self-play without permanently
        altering stored hero-perspective tokens (call restore_hero_holes afterward).
        """
        if not self.is_transformer or traj_indices.numel() == 0:
            return
        bidx = (traj_indices + self.position) % self.capacity
        p0 = self.hole_pos0[bidx]
        p1 = self.hole_pos1[bidx]
        m0 = p0 >= 0
        m1 = p1 >= 0
        if m0.any():
            ii0 = bidx[m0]
            pp0 = p0[m0]
            c0 = card0[m0]
            self.token_ids[ii0, pp0] = (Special.NUM_SPECIAL.value + c0).to(torch.int8)
            self.card_ranks[ii0, pp0] = (c0 % 13).to(torch.uint8)
            self.card_suits[ii0, pp0] = (c0 // 13).to(torch.uint8)
            self.card_streets[ii0, pp0] = 0
        if m1.any():
            ii1 = bidx[m1]
            pp1 = p1[m1]
            c1 = card1[m1]
            self.token_ids[ii1, pp1] = (Special.NUM_SPECIAL.value + c1).to(torch.int8)
            self.card_ranks[ii1, pp1] = (c1 % 13).to(torch.uint8)
            self.card_suits[ii1, pp1] = (c1 // 13).to(torch.uint8)
            self.card_streets[ii1, pp1] = 0

    def restore_hero_holes(self, traj_indices: torch.Tensor) -> None:
        """Restore originally recorded hero hole cards for given trajectories."""
        if not self.is_transformer or traj_indices.numel() == 0:
            return
        bidx = (traj_indices + self.position) % self.capacity
        p0 = self.hole_pos0[bidx]
        p1 = self.hole_pos1[bidx]
        m0 = p0 >= 0
        m1 = p1 >= 0
        if m0.any():
            ii0 = bidx[m0]
            pp0 = p0[m0]
            c0 = self.hero_hole0[ii0]
            self.token_ids[ii0, pp0] = (Special.NUM_SPECIAL.value + c0).to(torch.int8)
            self.card_ranks[ii0, pp0] = (c0 % 13).to(torch.uint8)
            self.card_suits[ii0, pp0] = (c0 // 13).to(torch.uint8)
            self.card_streets[ii0, pp0] = 0
        if m1.any():
            ii1 = bidx[m1]
            pp1 = p1[m1]
            c1 = self.hero_hole1[ii1]
            self.token_ids[ii1, pp1] = (Special.NUM_SPECIAL.value + c1).to(torch.int8)
            self.card_ranks[ii1, pp1] = (c1 % 13).to(torch.uint8)
            self.card_suits[ii1, pp1] = (c1 // 13).to(torch.uint8)
            self.card_streets[ii1, pp1] = 0

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

    def add_token(
        self,
        embedding_data: StructuredEmbeddingData,
        trajectory_indices: torch.Tensor,
    ) -> None:
        """Alias for add_tokens for API parity with single-token events."""
        self.add_tokens(embedding_data, trajectory_indices)

    def _sample_transformer_steps(
        self,
        traj_indices: torch.Tensor,
        step_indices: torch.Tensor,
        use_decision_end: bool = False,
    ) -> StructuredEmbeddingData:
        """Materialize transformer sequences up to each step's end token.

        If use_decision_end is True, cut at decision_end (after CONTEXT), otherwise
        cut at transition_end (after ACTION).
        """

        ends = (
            self.decision_token_ends[traj_indices, step_indices]
            if use_decision_end
            else self.transition_token_ends[traj_indices, step_indices]
        )
        batch_size = traj_indices.shape[0]
        seq_len = self.sequence_length

        token_ids_src = self.token_ids[traj_indices]
        card_ranks_src = self.card_ranks[traj_indices]
        card_suits_src = self.card_suits[traj_indices]
        card_streets_src = self.card_streets[traj_indices]
        action_actors_src = self.action_actors[traj_indices]
        action_streets_src = self.action_streets[traj_indices]
        action_legal_src = self.action_legal_masks[traj_indices]
        context_src = self.context_features[traj_indices]

        token_ids = torch.full(
            (batch_size, seq_len), -1, dtype=torch.int8, device=self.device
        )
        card_ranks = torch.zeros(
            (batch_size, seq_len), dtype=torch.uint8, device=self.device
        )
        card_suits = torch.zeros(
            (batch_size, seq_len), dtype=torch.uint8, device=self.device
        )
        card_streets = torch.zeros(
            (batch_size, seq_len), dtype=torch.uint8, device=self.device
        )
        action_actors = torch.full(
            (batch_size, seq_len), -1, dtype=torch.int16, device=self.device
        )
        action_streets = torch.full(
            (batch_size, seq_len), -1, dtype=torch.int16, device=self.device
        )
        action_legal_masks = torch.zeros(
            batch_size, seq_len, self.num_bet_bins, dtype=torch.bool, device=self.device
        )
        context_features = torch.zeros(
            batch_size, seq_len, 10, dtype=self.float_dtype, device=self.device
        )

        range_tensor = torch.arange(seq_len, device=self.device)
        mask = range_tensor.unsqueeze(0) < ends.unsqueeze(1)
        expanded_mask = mask.unsqueeze(-1)

        token_ids[mask] = token_ids_src[mask]
        card_ranks[mask] = card_ranks_src[mask]
        card_suits[mask] = card_suits_src[mask]
        card_streets[mask] = card_streets_src[mask]
        action_actors[mask] = action_actors_src[mask]
        action_streets[mask] = action_streets_src[mask]
        # Copy full legal mask rows for valid token positions
        action_legal_masks[mask] = action_legal_src[mask]
        # Copy context per-token rows by 2D boolean mask across the first two dims
        context_features[mask] = context_src[mask]

        return StructuredEmbeddingData(
            token_ids=token_ids,
            card_ranks=card_ranks,
            card_suits=card_suits,
            card_streets=card_streets,
            action_actors=action_actors,
            action_streets=action_streets,
            action_legal_masks=action_legal_masks,
            context_features=context_features,
            lengths=ends,
        )

    def snapshot_current_prefixes(
        self, traj_indices: torch.Tensor
    ) -> StructuredEmbeddingData:
        """Return StructuredEmbeddingData cut at current token_positions for each trajectory."""
        if not self.is_transformer:
            raise TypeError(
                "snapshot_current_prefixes is only valid in transformer mode"
            )
        if traj_indices.numel() == 0:
            return StructuredEmbeddingData(
                token_ids=self.token_ids[:0].clone(),
                card_ranks=self.card_ranks[:0].clone(),
                card_suits=self.card_suits[:0].clone(),
                card_streets=self.card_streets[:0].clone(),
                action_actors=self.action_actors[:0].clone(),
                action_streets=self.action_streets[:0].clone(),
                action_legal_masks=self.action_legal_masks[:0].clone(),
                context_features=self.context_features[:0].clone(),
                lengths=torch.zeros(0, dtype=torch.long, device=self.device),
            )

        bidx = (traj_indices + self.position) % self.capacity
        ends = self.token_positions[bidx]

        # Reuse _sample_transformer_steps by building synthetic step indices with ends
        # We can materialize directly here for efficiency
        batch_size = bidx.shape[0]
        seq_len = self.sequence_length

        token_ids_src = self.token_ids[bidx]
        card_ranks_src = self.card_ranks[bidx]
        card_suits_src = self.card_suits[bidx]
        card_streets_src = self.card_streets[bidx]
        action_actors_src = self.action_actors[bidx]
        action_streets_src = self.action_streets[bidx]
        action_legal_src = self.action_legal_masks[bidx]
        context_src = self.context_features[bidx]

        token_ids = torch.full(
            (batch_size, seq_len), -1, dtype=torch.int8, device=self.device
        )
        card_ranks = torch.zeros(
            (batch_size, seq_len), dtype=torch.uint8, device=self.device
        )
        card_suits = torch.zeros(
            (batch_size, seq_len), dtype=torch.uint8, device=self.device
        )
        card_streets = torch.zeros(
            (batch_size, seq_len), dtype=torch.uint8, device=self.device
        )
        action_actors = torch.full(
            (batch_size, seq_len), -1, dtype=torch.int16, device=self.device
        )
        action_streets = torch.full(
            (batch_size, seq_len), -1, dtype=torch.int16, device=self.device
        )
        action_legal_masks = torch.zeros(
            batch_size, seq_len, self.num_bet_bins, dtype=torch.bool, device=self.device
        )
        context_features = torch.zeros(
            batch_size, seq_len, 10, dtype=self.float_dtype, device=self.device
        )

        range_tensor = torch.arange(seq_len, device=self.device)
        mask = range_tensor.unsqueeze(0) < ends.unsqueeze(1)

        token_ids[mask] = token_ids_src[mask]
        card_ranks[mask] = card_ranks_src[mask]
        card_suits[mask] = card_suits_src[mask]
        card_streets[mask] = card_streets_src[mask]
        action_actors[mask] = action_actors_src[mask]
        action_streets[mask] = action_streets_src[mask]
        action_legal_masks[mask] = action_legal_src[mask]
        context_features[mask] = context_src[mask]

        return StructuredEmbeddingData(
            token_ids=token_ids,
            card_ranks=card_ranks,
            card_suits=card_suits,
            card_streets=card_streets,
            action_actors=action_actors,
            action_streets=action_streets,
            action_legal_masks=action_legal_masks,
            context_features=context_features,
            lengths=ends,
        )

    # add_hero_action_step removed in favor of TokenSequenceBuilder + record_step_metadata

    def snapshot_decision_prefixes(
        self, traj_indices: torch.Tensor
    ) -> StructuredEmbeddingData:
        """Return StructuredEmbeddingData cut at decision_end for latest steps.

        Only valid for transformer mode and trajectories with at least one step.
        """
        if not self.is_transformer:
            raise TypeError(
                "snapshot_decision_prefixes is only valid in transformer mode"
            )
        if traj_indices.numel() == 0:
            return StructuredEmbeddingData(
                token_ids=self.token_ids[:0].clone(),
                card_ranks=self.card_ranks[:0].clone(),
                card_suits=self.card_suits[:0].clone(),
                card_streets=self.card_streets[:0].clone(),
                action_actors=self.action_actors[:0].clone(),
                action_streets=self.action_streets[:0].clone(),
                action_legal_masks=self.action_legal_masks[:0].clone(),
                context_features=self.context_features[:0].clone(),
                lengths=torch.zeros(0, dtype=torch.long, device=self.device),
            )
        bidx = (traj_indices + self.position) % self.capacity
        step_indices = (self.current_step_positions[bidx] - 1).clamp_min(0)
        return self._sample_transformer_steps(bidx, step_indices, use_decision_end=True)
