from __future__ import annotations

import torch
from typing import Dict


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
    ):
        self.capacity = capacity  # Number of trajectories
        self.max_trajectory_length = max_trajectory_length
        self.device = device
        self.float_dtype = float_dtype  # Store float dtype for use in methods
        self.position = 0  # Next trajectory write position
        self.size = 0  # Total number of valid trajectories

        self.open_batch = (
            -1
        )  # -1 if no batch is open, otherwise the nominal size of the open batch

        # Pre-allocate tensors for all transition fields: (capacity, max_trajectory_length, ...)
        # Cards features tensor: (capacity, max_trajectory_length, 6, 4, 13) - bool dtype
        self.cards_features = torch.zeros(
            capacity,
            max_trajectory_length,
            6,
            4,
            13,  # Fixed cards shape
            dtype=torch.bool,
            device=device,
        )

        # Actions features tensor: (capacity, max_trajectory_length, 24, 4, num_bet_bins) - bool dtype
        # 4 slots: p1, p2, sum, legal
        self.actions_features = torch.zeros(
            capacity,
            max_trajectory_length,
            24,
            4,
            num_bet_bins,
            dtype=torch.bool,
            device=device,
        )

        # Transformer structured embedding fields: (capacity, max_trajectory_length, ...)
        self.card_indices = torch.full(
            (capacity, max_trajectory_length, 50), -1, dtype=torch.long, device=device
        )
        self.card_stages = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.card_visibility = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.card_order = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.action_actors = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.action_types = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.action_streets = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.action_size_bins = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.action_size_features = torch.zeros(
            capacity, max_trajectory_length, 50, 3, dtype=torch.float, device=device
        )
        self.context_pot_sizes = torch.zeros(
            capacity, max_trajectory_length, 50, 1, dtype=torch.float, device=device
        )
        self.context_stack_sizes = torch.zeros(
            capacity, max_trajectory_length, 50, 2, dtype=torch.float, device=device
        )
        self.context_positions = torch.zeros(
            capacity, max_trajectory_length, 50, dtype=torch.long, device=device
        )
        self.context_street_context = torch.zeros(
            capacity, max_trajectory_length, 50, 4, dtype=torch.float, device=device
        )
        self.action_indices = torch.zeros(
            capacity, max_trajectory_length, dtype=torch.long, device=device
        )
        self.log_probs = torch.zeros(
            capacity, max_trajectory_length, dtype=float_dtype, device=device
        )
        self.rewards = torch.zeros(
            capacity, max_trajectory_length, dtype=float_dtype, device=device
        )
        self.dones = torch.zeros(
            capacity, max_trajectory_length, dtype=torch.bool, device=device
        )
        self.legal_masks = torch.zeros(
            capacity,
            max_trajectory_length,
            num_bet_bins,
            dtype=torch.bool,  # Changed to bool
            device=device,
        )
        self.delta2 = torch.zeros(
            capacity, max_trajectory_length, dtype=float_dtype, device=device
        )
        self.delta3 = torch.zeros(
            capacity, max_trajectory_length, dtype=float_dtype, device=device
        )
        self.values = torch.zeros(
            capacity, max_trajectory_length, dtype=float_dtype, device=device
        )
        self.advantages = torch.zeros(
            capacity, max_trajectory_length, dtype=float_dtype, device=device
        )
        self.returns = torch.zeros(
            capacity, max_trajectory_length, dtype=float_dtype, device=device
        )

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
        self.cards_features[clear_indices] = False
        self.actions_features[clear_indices] = False
        self.card_indices[clear_indices] = -1
        self.card_stages[clear_indices] = 0
        self.card_visibility[clear_indices] = 0
        self.card_order[clear_indices] = 0
        self.action_actors[clear_indices] = 0
        self.action_types[clear_indices] = 0
        self.action_streets[clear_indices] = 0
        self.action_size_bins[clear_indices] = 0
        self.action_size_features[clear_indices] = 0
        self.context_pot_sizes[clear_indices] = 0
        self.context_stack_sizes[clear_indices] = 0
        self.context_positions[clear_indices] = 0
        self.context_street_context[clear_indices] = 0
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
        for attr in [
            "trajectory_lengths",
            "valid_trajectories",
            "current_step_positions",
            "cards_features",
            "actions_features",
            "card_indices",
            "card_stages",
            "card_visibility",
            "card_order",
            "action_actors",
            "action_types",
            "action_streets",
            "action_size_bins",
            "action_size_features",
            "context_pot_sizes",
            "context_stack_sizes",
            "context_positions",
            "context_street_context",
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
        ]:
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
        cards_features: torch.Tensor = None,  # [batch_size, 6, 4, 13] - bool
        actions_features: torch.Tensor = None,  # [batch_size, 24, 4, num_bet_bins] - bool
        # Structured embedding fields
        card_indices: torch.Tensor = None,  # [batch_size, seq_len] - long
        card_stages: torch.Tensor = None,  # [batch_size, seq_len] - long
        card_visibility: torch.Tensor = None,  # [batch_size, seq_len] - long
        card_order: torch.Tensor = None,  # [batch_size, seq_len] - long
        action_actors: torch.Tensor = None,  # [batch_size, seq_len] - long
        action_types: torch.Tensor = None,  # [batch_size, seq_len] - long
        action_streets: torch.Tensor = None,  # [batch_size, seq_len] - long
        action_size_bins: torch.Tensor = None,  # [batch_size, seq_len] - long
        action_size_features: torch.Tensor = None,  # [batch_size, seq_len, 3] - float
        context_pot_sizes: torch.Tensor = None,  # [batch_size, seq_len, 1] - float
        context_stack_sizes: torch.Tensor = None,  # [batch_size, seq_len, 2] - float
        context_positions: torch.Tensor = None,  # [batch_size, seq_len] - long
        context_street_context: torch.Tensor = None,  # [batch_size, seq_len, 4] - float
        action_indices: torch.Tensor = None,  # [batch_size] - long
        log_probs: torch.Tensor = None,
        rewards: torch.Tensor = None,
        dones: torch.Tensor = None,
        legal_masks: torch.Tensor = None,  # [batch_size, legal_mask_dim] - bool
        delta2: torch.Tensor = None,
        delta3: torch.Tensor = None,
        values: torch.Tensor = None,
        trajectory_indices: torch.Tensor = None,  # Which trajectory each transition belongs to
    ) -> None:
        """
        Add a batch of transitions to the buffer using vectorized operations.

        Args:
            cards_features: [batch_size, 6, 4, 13] - bool dtype
            actions_features: [batch_size, 24, 4, num_bet_bins] - bool dtype
            card_indices: [batch_size, seq_len] - long dtype
            card_stages: [batch_size, seq_len] - long dtype
            card_visibility: [batch_size, seq_len] - long dtype
            card_order: [batch_size, seq_len] - long dtype
            action_actors: [batch_size, seq_len] - long dtype
            action_types: [batch_size, seq_len] - long dtype
            action_streets: [batch_size, seq_len] - long dtype
            action_size_bins: [batch_size, seq_len] - long dtype
            action_size_features: [batch_size, seq_len, 3] - float dtype
            context_pot_sizes: [batch_size, seq_len, 1] - float dtype
            context_stack_sizes: [batch_size, seq_len, 2] - float dtype
            context_positions: [batch_size, seq_len] - long dtype
            context_street_context: [batch_size, seq_len, 4] - float dtype
            action_indices: [batch_size] - long dtype
            log_probs: [batch_size]
            rewards: [batch_size]
            dones: [batch_size]
            legal_masks: [batch_size, legal_mask_dim] - bool dtype
            delta2: [batch_size]
            delta3: [batch_size]
            values: [batch_size]
            trajectory_indices: [batch_size] - which trajectory each transition belongs to (in source space)
        """

        assert (
            self.open_batch > 0
        ), "Must call start_adding_trajectories before adding a batch"

        # Convert user's source trajectory indices to ring buffer indices
        buffer_trajectory_indices = (trajectory_indices + self.position) % self.capacity

        # Get current step positions for each trajectory using buffer indices
        step_positions = self.current_step_positions[buffer_trajectory_indices]

        # Check if any trajectories would exceed max length
        if (step_positions >= self.max_trajectory_length).any():
            raise ValueError("Some trajectories would exceed max_trajectory_length")

        # Use vectorized indexing to add transitions
        # This is the key vectorized operation - no loops!
        if cards_features is not None:
            self.cards_features[buffer_trajectory_indices, step_positions] = (
                cards_features
            )
        if actions_features is not None:
            self.actions_features[buffer_trajectory_indices, step_positions] = (
                actions_features
            )
        # Structured embedding fields
        if card_indices is not None:
            self.card_indices[buffer_trajectory_indices, step_positions] = card_indices
        if card_stages is not None:
            self.card_stages[buffer_trajectory_indices, step_positions] = card_stages
        if card_visibility is not None:
            self.card_visibility[buffer_trajectory_indices, step_positions] = (
                card_visibility
            )
        if card_order is not None:
            self.card_order[buffer_trajectory_indices, step_positions] = card_order
        if action_actors is not None:
            self.action_actors[buffer_trajectory_indices, step_positions] = (
                action_actors
            )
        if action_types is not None:
            self.action_types[buffer_trajectory_indices, step_positions] = action_types
        if action_streets is not None:
            self.action_streets[buffer_trajectory_indices, step_positions] = (
                action_streets
            )
        if action_size_bins is not None:
            self.action_size_bins[buffer_trajectory_indices, step_positions] = (
                action_size_bins
            )
        if action_size_features is not None:
            self.action_size_features[buffer_trajectory_indices, step_positions] = (
                action_size_features
            )
        if context_pot_sizes is not None:
            self.context_pot_sizes[buffer_trajectory_indices, step_positions] = (
                context_pot_sizes
            )
        if context_stack_sizes is not None:
            self.context_stack_sizes[buffer_trajectory_indices, step_positions] = (
                context_stack_sizes
            )
        if context_positions is not None:
            self.context_positions[buffer_trajectory_indices, step_positions] = (
                context_positions
            )
        if context_street_context is not None:
            self.context_street_context[buffer_trajectory_indices, step_positions] = (
                context_street_context
            )
        self.action_indices[buffer_trajectory_indices, step_positions] = action_indices
        self.log_probs[buffer_trajectory_indices, step_positions] = log_probs
        self.rewards[buffer_trajectory_indices, step_positions] = rewards
        self.dones[buffer_trajectory_indices, step_positions] = dones
        self.legal_masks[buffer_trajectory_indices, step_positions] = legal_masks
        self.delta2[buffer_trajectory_indices, step_positions] = delta2
        self.delta3[buffer_trajectory_indices, step_positions] = delta3
        self.values[buffer_trajectory_indices, step_positions] = values

        # Update trajectory step positions
        self.current_step_positions[buffer_trajectory_indices] += 1

        # Handle trajectory completion
        completed_trajectories = buffer_trajectory_indices[dones]
        if len(completed_trajectories) > 0:
            # Mark completed trajectories as valid
            self.valid_trajectories[completed_trajectories] = True
            self.trajectory_lengths[completed_trajectories] = (
                self.current_step_positions[completed_trajectories]
            )

            # Reset step positions for completed trajectories
            self.current_step_positions[completed_trajectories] = 0

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
        """Sample complete trajectories for PPO updates."""
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
        cards_features = self.cards_features[
            traj_indices, :max_len
        ]  # [num_traj, max_len, 6, 4, 13]
        actions_features = self.actions_features[
            traj_indices, :max_len
        ]  # [num_traj, max_len, 24, 4, num_bet_bins]
        # Structured embedding fields
        card_indices = self.card_indices[traj_indices, :max_len]
        card_stages = self.card_stages[traj_indices, :max_len]
        card_visibility = self.card_visibility[traj_indices, :max_len]
        card_order = self.card_order[traj_indices, :max_len]
        action_actors = self.action_actors[traj_indices, :max_len]
        action_types = self.action_types[traj_indices, :max_len]
        action_streets = self.action_streets[traj_indices, :max_len]
        action_size_bins = self.action_size_bins[traj_indices, :max_len]
        action_size_features = self.action_size_features[traj_indices, :max_len]
        context_pot_sizes = self.context_pot_sizes[traj_indices, :max_len]
        context_stack_sizes = self.context_stack_sizes[traj_indices, :max_len]
        context_positions = self.context_positions[traj_indices, :max_len]
        context_street_context = self.context_street_context[traj_indices, :max_len]
        action_indices = self.action_indices[traj_indices, :max_len]
        logps = self.log_probs[traj_indices, :max_len]
        advs = self.advantages[traj_indices, :max_len]
        rets = self.returns[traj_indices, :max_len]
        legal = self.legal_masks[traj_indices, :max_len]
        d2 = self.delta2[traj_indices, :max_len]
        d3 = self.delta3[traj_indices, :max_len]

        # Flatten only valid steps (mask out padding)
        mask_flat = mask.flatten()
        all_cards_features = cards_features.reshape(-1, *cards_features.shape[2:])[
            mask_flat
        ]  # [valid_steps, 6, 4, 13]
        all_actions_features = actions_features.reshape(
            -1, *actions_features.shape[2:]
        )[
            mask_flat
        ]  # [valid_steps, 24, 4, num_bet_bins]
        # Structured embedding fields
        all_card_indices = card_indices.reshape(-1, *card_indices.shape[2:])[mask_flat]
        all_card_stages = card_stages.reshape(-1, *card_stages.shape[2:])[mask_flat]
        all_card_visibility = card_visibility.reshape(-1, *card_visibility.shape[2:])[
            mask_flat
        ]
        all_card_order = card_order.reshape(-1, *card_order.shape[2:])[mask_flat]
        all_action_actors = action_actors.reshape(-1, *action_actors.shape[2:])[
            mask_flat
        ]
        all_action_types = action_types.reshape(-1, *action_types.shape[2:])[mask_flat]
        all_action_streets = action_streets.reshape(-1, *action_streets.shape[2:])[
            mask_flat
        ]
        all_action_size_bins = action_size_bins.reshape(
            -1, *action_size_bins.shape[2:]
        )[mask_flat]
        all_action_size_features = action_size_features.reshape(
            -1, *action_size_features.shape[2:]
        )[mask_flat]
        all_context_pot_sizes = context_pot_sizes.reshape(
            -1, *context_pot_sizes.shape[2:]
        )[mask_flat]
        all_context_stack_sizes = context_stack_sizes.reshape(
            -1, *context_stack_sizes.shape[2:]
        )[mask_flat]
        all_context_positions = context_positions.reshape(
            -1, *context_positions.shape[2:]
        )[mask_flat]
        all_context_street_context = context_street_context.reshape(
            -1, *context_street_context.shape[2:]
        )[mask_flat]
        all_action_indices = action_indices.reshape(-1)[mask_flat]
        all_log_probs = logps.reshape(-1)[mask_flat]
        all_advantages = advs.reshape(-1)[mask_flat]
        all_returns = rets.reshape(-1)[mask_flat]
        all_legal_masks = legal.reshape(-1, legal.shape[-1])[mask_flat]
        all_delta2 = d2.reshape(-1)[mask_flat]
        all_delta3 = d3.reshape(-1)[mask_flat]

        # Return the flattened data directly
        if all_cards_features.numel() > 0:
            return {
                "cards_features": all_cards_features,
                "actions_features": all_actions_features,
                # Structured embedding fields
                "card_indices": all_card_indices,
                "card_stages": all_card_stages,
                "card_visibility": all_card_visibility,
                "card_order": all_card_order,
                "action_actors": all_action_actors,
                "action_types": all_action_types,
                "action_streets": all_action_streets,
                "action_size_bins": all_action_size_bins,
                "action_size_features": all_action_size_features,
                "context_pot_sizes": all_context_pot_sizes,
                "context_stack_sizes": all_context_stack_sizes,
                "context_positions": all_context_positions,
                "context_street_context": all_context_street_context,
                "action_indices": all_action_indices,
                "log_probs_old": all_log_probs,
                "advantages": all_advantages,
                "returns": all_returns,
                "legal_masks": all_legal_masks,
                "delta2": all_delta2,
                "delta3": all_delta3,
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
        batch = {
            "cards_features": self.cards_features[traj_indices, step_indices],
            "actions_features": self.actions_features[traj_indices, step_indices],
            # Structured embedding fields
            "card_indices": self.card_indices[traj_indices, step_indices],
            "card_stages": self.card_stages[traj_indices, step_indices],
            "card_visibility": self.card_visibility[traj_indices, step_indices],
            "card_order": self.card_order[traj_indices, step_indices],
            "action_actors": self.action_actors[traj_indices, step_indices],
            "action_types": self.action_types[traj_indices, step_indices],
            "action_streets": self.action_streets[traj_indices, step_indices],
            "action_size_bins": self.action_size_bins[traj_indices, step_indices],
            "action_size_features": self.action_size_features[
                traj_indices, step_indices
            ],
            "context_pot_sizes": self.context_pot_sizes[traj_indices, step_indices],
            "context_stack_sizes": self.context_stack_sizes[traj_indices, step_indices],
            "context_positions": self.context_positions[traj_indices, step_indices],
            "context_street_context": self.context_street_context[
                traj_indices, step_indices
            ],
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
        """Add a trajectory for backward compatibility with scalar environment."""
        if trajectory.transitions:
            self.start_adding_trajectory_batches(1)

            # Iterate over transitions directly
            for t in trajectory.transitions:
                # Extract cards and actions from the observation tensor
                # Assuming observation is [cards_features, actions_features] flattened
                cards_features = t.observation[:312]  # First 312 features are cards
                actions_features = t.observation[312:]  # Next 768 features are actions

                # Reshape cards: (312,) -> (6, 4, 13)
                cards_features = cards_features.reshape(6, 4, 13).bool()

                # Reshape actions: (768,) -> (24, 4, num_bet_bins)
                num_bet_bins = actions_features.shape[0] // (24 * 4)
                actions_features = actions_features.reshape(24, 4, num_bet_bins).bool()

                self.add_batch(
                    cards_features=cards_features.unsqueeze(0),
                    actions_features=actions_features.unsqueeze(0),
                    action_indices=torch.tensor(
                        [t.action], dtype=torch.long, device=self.device
                    ),
                    log_probs=torch.tensor(
                        [t.log_prob], dtype=self.float_dtype, device=self.device
                    ),
                    rewards=torch.tensor(
                        [t.reward], dtype=self.float_dtype, device=self.device
                    ),
                    dones=torch.tensor([t.done], dtype=torch.bool, device=self.device),
                    legal_masks=t.legal_mask.unsqueeze(0).bool(),
                    delta2=torch.tensor(
                        [t.delta2], dtype=self.float_dtype, device=self.device
                    ),
                    delta3=torch.tensor(
                        [t.delta3], dtype=self.float_dtype, device=self.device
                    ),
                    values=torch.tensor(
                        [t.value], dtype=self.float_dtype, device=self.device
                    ),
                    trajectory_indices=torch.zeros(
                        1, dtype=torch.long, device=self.device
                    ),
                )

            self.finish_adding_trajectory_batches()

    def __len__(self) -> int:
        return self.size
