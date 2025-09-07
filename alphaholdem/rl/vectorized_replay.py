from __future__ import annotations

import torch
from typing import Optional, Dict, Any
import numpy as np


class VectorizedReplayBuffer:
    """
    Vectorized replay buffer that stores all transition data in separate tensors
    for efficient vectorized operations and sampling.
    """

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        legal_mask_dim: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device
        self.position = 0  # Next write position
        self.size = 0  # Total number of valid entries
        self.effective_start = 0  # Start of valid data in ring buffer

        # Pre-allocate tensors for all transition fields
        self.observations = torch.zeros(
            capacity, observation_dim, dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.legal_masks = torch.zeros(
            capacity, legal_mask_dim, dtype=torch.float32, device=device
        )
        self.chips_placed = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.delta2 = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.delta3 = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.returns = torch.zeros(capacity, dtype=torch.float32, device=device)

        # Trajectory boundaries are computed from dones tensor when needed

    def add_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        legal_masks: torch.Tensor,
        chips_placed: torch.Tensor,
        delta2: torch.Tensor,
        delta3: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Add a batch of transitions to the buffer.

        Args:
            observations: [batch_size, observation_dim]
            actions: [batch_size]
            log_probs: [batch_size]
            rewards: [batch_size]
            dones: [batch_size]
            legal_masks: [batch_size, legal_mask_dim]
            chips_placed: [batch_size]
            delta2: [batch_size]
            delta3: [batch_size]
            values: [batch_size]
        """
        batch_size = observations.shape[0]

        # Handle buffer overflow
        if self.position + batch_size > self.capacity:
            # Split the batch if it would overflow
            remaining_space = self.capacity - self.position
            self._add_partial_batch(
                observations[:remaining_space],
                actions[:remaining_space],
                log_probs[:remaining_space],
                rewards[:remaining_space],
                dones[:remaining_space],
                legal_masks[:remaining_space],
                chips_placed[:remaining_space],
                delta2[:remaining_space],
                delta3[:remaining_space],
                values[:remaining_space],
            )

            # Add the remaining part at the beginning
            overflow_size = batch_size - remaining_space
            self._add_partial_batch(
                observations[remaining_space:],
                actions[remaining_space:],
                log_probs[remaining_space:],
                rewards[remaining_space:],
                dones[remaining_space:],
                legal_masks[remaining_space:],
                chips_placed[remaining_space:],
                delta2[remaining_space:],
                delta3[remaining_space:],
                values[remaining_space:],
            )
        else:
            self._add_partial_batch(
                observations,
                actions,
                log_probs,
                rewards,
                dones,
                legal_masks,
                chips_placed,
                delta2,
                delta3,
                values,
            )

    def _add_partial_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        legal_masks: torch.Tensor,
        chips_placed: torch.Tensor,
        delta2: torch.Tensor,
        delta3: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """Add a partial batch that fits within the buffer."""
        batch_size = observations.shape[0]
        end_pos = self.position + batch_size

        # Store the data
        self.observations[self.position : end_pos] = observations
        self.actions[self.position : end_pos] = actions
        self.log_probs[self.position : end_pos] = log_probs
        self.rewards[self.position : end_pos] = rewards
        self.dones[self.position : end_pos] = dones
        self.legal_masks[self.position : end_pos] = legal_masks
        self.chips_placed[self.position : end_pos] = chips_placed
        self.delta2[self.position : end_pos] = delta2
        self.delta3[self.position : end_pos] = delta3
        self.values[self.position : end_pos] = values

        # Trajectory boundaries are computed from dones tensor when needed

        # Update position and size
        self.position = end_pos % self.capacity

        # Handle wraparound: if we're overwriting data, update effective_start
        if self.size + batch_size > self.capacity:
            # We're overwriting old data, so move effective_start forward
            overwritten = (self.size + batch_size) - self.capacity
            self.effective_start = (self.effective_start + overwritten) % self.capacity

        self.size = min(self.size + batch_size, self.capacity)

    def _compute_trajectory_boundaries(
        self, dones: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute trajectory starts and lengths from dones tensor."""
        T = len(dones)
        device = dones.device

        # A trajectory starts at t=0 or immediately after a done=True
        is_start = torch.zeros_like(dones, dtype=torch.bool)
        is_start[0] = True
        is_start[1:] = dones[:-1]
        trajectory_starts = torch.where(is_start)[0]

        # Compute lengths as the difference between consecutive starts, plus the last segment
        next_starts = torch.cat(
            [trajectory_starts[1:], torch.tensor([T], device=device)]
        )
        trajectory_lengths = next_starts - trajectory_starts

        return trajectory_starts, trajectory_lengths

    def compute_gae_returns(self, gamma: float = 0.999, lambda_: float = 0.95) -> None:
        """Compute GAE advantages and returns for all stored trajectories using no-loop approach."""
        if self.size == 0:
            return

        # Create a rectified buffer to handle wraparound
        rewards_rectified, values_rectified, dones_rectified = (
            self._create_rectified_buffer()
        )

        # Compute GAE for all trajectories at once using no-loop approach
        advantages_rectified = self._compute_gae_batch_vectorized(
            rewards_rectified, values_rectified, dones_rectified, gamma, lambda_
        )

        # Store results back to original buffer
        self._store_rectified_results(advantages_rectified, values_rectified)

    def _create_rectified_buffer(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a rectified buffer by copying data to handle wraparound."""
        if self.effective_start + self.size <= self.capacity:
            # No wraparound case - just return the data directly
            rewards = self.rewards[
                self.effective_start : self.effective_start + self.size
            ]
            values = self.values[
                self.effective_start : self.effective_start + self.size
            ]
            dones = self.dones[self.effective_start : self.effective_start + self.size]
            return rewards, values, dones
        else:
            # Wraparound case - copy data to contiguous tensor
            rewards = torch.zeros(
                self.size, device=self.device, dtype=self.rewards.dtype
            )
            values = torch.zeros(self.size, device=self.device, dtype=self.values.dtype)
            dones = torch.zeros(self.size, device=self.device, dtype=self.dones.dtype)

            # Copy first segment
            first_segment_size = self.capacity - self.effective_start
            rewards[:first_segment_size] = self.rewards[self.effective_start :]
            values[:first_segment_size] = self.values[self.effective_start :]
            dones[:first_segment_size] = self.dones[self.effective_start :]

            # Copy wraparound segment
            remaining_size = self.size - first_segment_size
            if remaining_size > 0:
                rewards[first_segment_size:] = self.rewards[:remaining_size]
                values[first_segment_size:] = self.values[:remaining_size]
                dones[first_segment_size:] = self.dones[:remaining_size]

            return rewards, values, dones

    def _store_rectified_results(
        self, advantages_rectified: torch.Tensor, values_rectified: torch.Tensor
    ) -> None:
        """Store results from rectified buffer back to original buffer."""
        if self.effective_start + self.size <= self.capacity:
            # No wraparound case - store directly
            self.advantages[self.effective_start : self.effective_start + self.size] = (
                advantages_rectified
            )
            self.returns[self.effective_start : self.effective_start + self.size] = (
                advantages_rectified + values_rectified
            )
        else:
            # Wraparound case - store back in segments
            first_segment_size = self.capacity - self.effective_start
            self.advantages[self.effective_start :] = advantages_rectified[
                :first_segment_size
            ]
            self.returns[self.effective_start :] = (
                advantages_rectified[:first_segment_size]
                + values_rectified[:first_segment_size]
            )

            remaining_size = self.size - first_segment_size
            if remaining_size > 0:
                self.advantages[:remaining_size] = advantages_rectified[
                    first_segment_size:
                ]
                self.returns[:remaining_size] = (
                    advantages_rectified[first_segment_size:]
                    + values_rectified[first_segment_size:]
                )

    def _compute_gae_batch_vectorized(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        lambda_: float,
    ) -> torch.Tensor:
        """
        Fully vectorized GAE computation using scan algorithm across all trajectories.
        This collapses the previous helper functions into a single function.
        """
        T = len(rewards)
        if T == 0:
            return torch.zeros_like(rewards)

        # Compute deltas for all transitions
        deltas = torch.zeros_like(rewards)

        # Non-terminal states: δ_t = r_t + γ * V_{t+1} - V_t
        if T > 1:
            deltas[:-1] = rewards[:-1] + gamma * values[1:] - values[:-1]

        # Terminal states: δ_t = r_t - V_t
        deltas = torch.where(dones, rewards - values, deltas)
        deltas[-1] = rewards[-1] - values[-1]  # Last timestep is always terminal

        # Use torch.where to find trajectory starts and lengths
        trajectory_starts, trajectory_lengths = self._compute_trajectory_boundaries(
            dones
        )

        # Pad trajectories to same length for vectorized processing
        max_length = trajectory_lengths.max().item()
        num_trajectories = len(trajectory_starts)

        device = deltas.device
        dtype = deltas.dtype

        # Create padded deltas and dones tensors [num_trajectories, max_length]
        padded_deltas = torch.zeros(
            num_trajectories, max_length, device=device, dtype=dtype
        )
        padded_dones = torch.zeros(
            num_trajectories, max_length, device=device, dtype=torch.bool
        )

        # Vectorized padding
        for i, (start, length) in enumerate(zip(trajectory_starts, trajectory_lengths)):
            padded_deltas[i, :length] = deltas[start : start + length]
            padded_dones[i, :length] = dones[start : start + length]

        B, T_pad = padded_deltas.shape

        c = gamma * lambda_
        rev_deltas = torch.flip(padded_deltas, dims=[1])  # [B, T_pad]
        rev_notdone = torch.flip(1.0 - padded_dones.float(), dims=[1])  # [B, T_pad]

        # Build the per-step multiplier m_t = c * (1 - done_t) in reversed order
        m = c * rev_notdone  # [B, T_pad]

        # We want: x_t = rev_deltas_t + m_t * x_{t-1}   (a scan)
        # Turn this linear recurrence into a weighted cumulative sum using cumulative products.
        # Compute cumulative product of m with an initial 1 to align offsets.
        one = torch.ones(B, 1, device=device, dtype=dtype)
        m_pad = torch.cat([one, m[:, :-1]], dim=1)  # shift-right for exclusive product
        cumprod = torch.cumprod(m_pad, dim=1)  # [B, T_pad]

        # The solution to x = rev_deltas ⊙ cumprod  scanned, then re-normalize:
        # Let y = rev_deltas * cumprod; then x = cumsum(y, dim=1) / cumprod
        y = rev_deltas * cumprod
        x = torch.cumsum(y, dim=1) / torch.clamp_min(cumprod, 1e-8)  # [B, T_pad]

        padded_advantages = torch.flip(x, dims=[1])  # back to forward time

        # Extract advantages back to original format
        advantages = torch.zeros_like(deltas)
        for i, (start, length) in enumerate(zip(trajectory_starts, trajectory_lengths)):
            advantages[start : start + length] = padded_advantages[i, :length]

        return advantages

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Sample random indices within the valid range
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Convert to actual buffer indices (handle wraparound)
        actual_indices = (self.effective_start + indices) % self.capacity

        return {
            "observations": self.observations[actual_indices],
            "actions": self.actions[actual_indices],
            "log_probs_old": self.log_probs[actual_indices],
            "advantages": self.advantages[actual_indices],
            "returns": self.returns[actual_indices],
            "legal_masks": self.legal_masks[actual_indices],
            "delta2": self.delta2[actual_indices],
            "delta3": self.delta3[actual_indices],
        }

    def sample_trajectories(self, num_trajectories: int) -> Dict[str, torch.Tensor]:
        """Sample complete trajectories for PPO updates."""
        if len(self.trajectory_starts) == 0:
            raise ValueError("No trajectories available")

        # Sample trajectory indices
        traj_indices = torch.randint(
            0, len(self.trajectory_starts), (num_trajectories,), device=self.device
        )

        # Collect all transitions from sampled trajectories
        all_observations = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_legal_masks = []
        all_delta2 = []
        all_delta3 = []

        for traj_idx in traj_indices:
            start_idx = self.trajectory_starts[traj_idx]
            length = self.trajectory_lengths[traj_idx]

            if start_idx + length <= self.size:
                # No wraparound
                end_idx = start_idx + length
                all_observations.append(self.observations[start_idx:end_idx])
                all_actions.append(self.actions[start_idx:end_idx])
                all_log_probs.append(self.log_probs[start_idx:end_idx])
                all_advantages.append(self.advantages[start_idx:end_idx])
                all_returns.append(self.returns[start_idx:end_idx])
                all_legal_masks.append(self.legal_masks[start_idx:end_idx])
                all_delta2.append(self.delta2[start_idx:end_idx])
                all_delta3.append(self.delta3[start_idx:end_idx])
            else:
                # Handle wraparound
                remaining_length = length - (self.size - start_idx)
                all_observations.append(
                    torch.cat(
                        [
                            self.observations[start_idx:],
                            self.observations[:remaining_length],
                        ]
                    )
                )
                all_actions.append(
                    torch.cat(
                        [self.actions[start_idx:], self.actions[:remaining_length]]
                    )
                )
                all_log_probs.append(
                    torch.cat(
                        [self.log_probs[start_idx:], self.log_probs[:remaining_length]]
                    )
                )
                all_advantages.append(
                    torch.cat(
                        [
                            self.advantages[start_idx:],
                            self.advantages[:remaining_length],
                        ]
                    )
                )
                all_returns.append(
                    torch.cat(
                        [self.returns[start_idx:], self.returns[:remaining_length]]
                    )
                )
                all_legal_masks.append(
                    torch.cat(
                        [
                            self.legal_masks[start_idx:],
                            self.legal_masks[:remaining_length],
                        ]
                    )
                )
                all_delta2.append(
                    torch.cat([self.delta2[start_idx:], self.delta2[:remaining_length]])
                )
                all_delta3.append(
                    torch.cat([self.delta3[start_idx:], self.delta3[:remaining_length]])
                )

        return {
            "observations": torch.cat(all_observations),
            "actions": torch.cat(all_actions),
            "log_probs_old": torch.cat(all_log_probs),
            "advantages": torch.cat(all_advantages),
            "returns": torch.cat(all_returns),
            "legal_masks": torch.cat(all_legal_masks),
            "delta2": torch.cat(all_delta2),
            "delta3": torch.cat(all_delta3),
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        self.effective_start = 0
        self.trajectory_starts.clear()
        self.trajectory_lengths.clear()

    def num_steps(self) -> int:
        """Total number of transitions stored."""
        return self.size

    def trim_to_steps(self, max_steps: int) -> None:
        """Trim oldest trajectories until total steps <= max_steps."""
        if self.size <= max_steps:
            return

        # Calculate how many steps to remove
        steps_to_remove = self.size - max_steps

        # Remove trajectories from the beginning until we've removed enough steps
        removed_steps = 0
        while removed_steps < steps_to_remove and len(self.trajectory_starts) > 0:
            oldest_length = self.trajectory_lengths[0]

            if removed_steps + oldest_length <= steps_to_remove:
                # Remove entire trajectory
                self.trajectory_starts.pop(0)
                self.trajectory_lengths.pop(0)
                removed_steps += oldest_length
                self.size -= oldest_length
                self.effective_start = (
                    self.effective_start + oldest_length
                ) % self.capacity
            else:
                # Remove partial trajectory
                partial_remove = steps_to_remove - removed_steps
                self.trajectory_lengths[0] -= partial_remove
                self.trajectory_starts[0] += partial_remove
                removed_steps += partial_remove
                self.size -= partial_remove
                self.effective_start = (
                    self.effective_start + partial_remove
                ) % self.capacity

    def add_trajectory(self, trajectory) -> None:
        """Add a trajectory for backward compatibility with scalar environment."""
        # Convert trajectory to batch format
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        legal_masks = []
        chips_placed = []
        delta2 = []
        delta3 = []
        values = []

        for transition in trajectory.transitions:
            observations.append(transition.observation)
            actions.append(transition.action)
            log_probs.append(transition.log_prob)
            rewards.append(transition.reward)
            dones.append(transition.done)
            legal_masks.append(transition.legal_mask)
            chips_placed.append(transition.chips_placed)
            delta2.append(transition.delta2)
            delta3.append(transition.delta3)
            values.append(transition.value)

        # Convert to tensors and add as batch
        if observations:
            batch = {
                "observations": torch.stack(observations),
                "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
                "log_probs": torch.tensor(
                    log_probs, dtype=torch.float32, device=self.device
                ),
                "rewards": torch.tensor(
                    rewards, dtype=torch.float32, device=self.device
                ),
                "dones": torch.tensor(dones, dtype=torch.bool, device=self.device),
                "legal_masks": torch.stack(legal_masks),
                "chips_placed": torch.tensor(
                    chips_placed, dtype=torch.float32, device=self.device
                ),
                "delta2": torch.tensor(delta2, dtype=torch.float32, device=self.device),
                "delta3": torch.tensor(delta3, dtype=torch.float32, device=self.device),
                "values": torch.tensor(values, dtype=torch.float32, device=self.device),
            }
            self.add_batch(**batch)

    def __len__(self) -> int:
        return self.size
