import pytest
import torch
from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer


class TestVectorizedReplayBuffer:
    """Test suite for VectorizedReplayBuffer with 2D trajectory-based implementation."""

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        device = torch.device("cpu")  # Use CPU for testing
        return VectorizedReplayBuffer(
            capacity=10,  # Number of trajectories
            max_trajectory_length=20,  # Max steps per trajectory
            observation_dim=10,
            legal_mask_dim=5,
            device=device,
        )

    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.capacity == 10
        assert buffer.max_trajectory_length == 20
        assert buffer.size == 0
        assert buffer.position == 0

        # Check tensor shapes - now 2D: (capacity, max_trajectory_length, ...)
        assert buffer.observations.shape == (10, 20, 10)
        assert buffer.actions.shape == (10, 20)
        assert buffer.log_probs.shape == (10, 20)
        assert buffer.rewards.shape == (10, 20)
        assert buffer.dones.shape == (10, 20)
        assert buffer.legal_masks.shape == (10, 20, 5)
        assert buffer.chips_placed.shape == (10, 20)
        assert buffer.delta2.shape == (10, 20)
        assert buffer.delta3.shape == (10, 20)
        assert buffer.values.shape == (10, 20)
        assert buffer.advantages.shape == (10, 20)
        assert buffer.returns.shape == (10, 20)

        # Check trajectory tracking tensors
        assert buffer.valid_trajectories.shape == (10,)
        assert buffer.trajectory_lengths.shape == (10,)
        assert buffer.current_step_positions.shape == (10,)

    def test_add_single_transition(self, buffer):
        """Test adding a single transition to a trajectory."""
        batch = self._create_test_batch(1, buffer.device)
        trajectory_indices = torch.tensor([0], device=buffer.device)

        buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Should not be valid yet (no done flag)
        assert buffer.size == 0
        assert buffer.valid_trajectories[0] == False
        assert buffer.current_step_positions[0] == 1

    def test_add_complete_trajectory(self, buffer):
        """Test adding a complete trajectory by adding transitions one by one."""
        # Add transitions one by one (as intended by the implementation)
        for i in range(3):
            batch = self._create_test_batch(1, buffer.device)
            trajectory_indices = torch.tensor([0], device=buffer.device)
            batch["dones"][0] = i == 2  # Only last transition is done
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Should be valid now
        assert buffer.size == 1
        assert buffer.valid_trajectories[0] == True
        assert buffer.trajectory_lengths[0] == 3
        assert buffer.current_step_positions[0] == 0  # Reset after completion

    def test_add_multiple_trajectories(self, buffer):
        """Test adding multiple trajectories."""
        # First trajectory - add transitions one by one
        for i in range(2):
            batch = self._create_test_batch(1, buffer.device)
            trajectory_indices = torch.tensor([0], device=buffer.device)
            batch["dones"][0] = i == 1  # Only last transition is done
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Second trajectory - add transitions one by one
        for i in range(3):
            batch = self._create_test_batch(1, buffer.device)
            trajectory_indices = torch.tensor([1], device=buffer.device)
            batch["dones"][0] = i == 2  # Only last transition is done
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        assert buffer.size == 2
        assert buffer.valid_trajectories[0] == True
        assert buffer.valid_trajectories[1] == True
        assert buffer.trajectory_lengths[0] == 2
        assert buffer.trajectory_lengths[1] == 3

    def test_trajectory_wraparound(self, buffer):
        """Test trajectory wraparound when capacity is exceeded."""
        # Fill buffer with trajectories
        for i in range(12):  # More than capacity
            batch = self._create_test_batch(1, buffer.device)
            trajectory_indices = torch.tensor(
                [i % buffer.capacity], device=buffer.device
            )
            batch["dones"][0] = True
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)
            # Call finish_adding_trajectories to update position
            buffer.finish_adding_trajectories(1)

        # Should be at capacity
        assert buffer.size == buffer.capacity
        assert buffer.position == 2  # 12 % 10 = 2

    def test_max_trajectory_length_exceeded(self, buffer):
        """Test error when trajectory exceeds max length."""
        batch = self._create_test_batch(1, buffer.device)
        trajectory_indices = torch.tensor([0], device=buffer.device)

        # Fill trajectory to max length
        for _ in range(buffer.max_trajectory_length):
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Next addition should fail
        with pytest.raises(
            ValueError, match="Some trajectories would exceed max_trajectory_length"
        ):
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

    def test_sample_trajectories(self, buffer):
        """Test sampling complete trajectories."""
        # Add multiple trajectories - one transition at a time
        for i in range(3):
            for j in range(2):
                batch = self._create_test_batch(1, buffer.device)
                trajectory_indices = torch.tensor([i], device=buffer.device)
                batch["dones"][0] = j == 1  # Only last transition is done
                buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Sample trajectories
        sampled = buffer.sample_trajectories(2)

        # Check that all required keys are present
        expected_keys = {
            "observations",
            "actions",
            "log_probs_old",
            "advantages",
            "returns",
            "legal_masks",
            "delta2",
            "delta3",
        }
        assert set(sampled.keys()) == expected_keys

        # Check that we get data from 2 trajectories (4 total steps)
        # Each trajectory has 2 steps, so 2 trajectories = 4 total steps
        total_steps = sampled["observations"].shape[0]
        assert total_steps == 4
        assert sampled["actions"].shape[0] == total_steps
        assert sampled["log_probs_old"].shape[0] == total_steps
        assert sampled["advantages"].shape[0] == total_steps
        assert sampled["returns"].shape[0] == total_steps
        assert sampled["legal_masks"].shape[0] == total_steps
        assert sampled["delta2"].shape[0] == total_steps
        assert sampled["delta3"].shape[0] == total_steps

    def test_sample_empty_buffer(self, buffer):
        """Test sampling from empty buffer raises error."""
        with pytest.raises(ValueError, match="No trajectories available"):
            buffer.sample_trajectories(1)

    def test_sample_no_valid_trajectories(self, buffer):
        """Test sampling when no trajectories are completed."""
        # Add incomplete trajectory
        batch = self._create_test_batch(2, buffer.device)
        trajectory_indices = torch.tensor([0, 0], device=buffer.device)
        # No done flags - trajectory not completed
        buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        with pytest.raises(ValueError, match="No trajectories available"):
            buffer.sample_trajectories(1)

    def test_gae_computation(self, buffer):
        """Test vectorized GAE computation."""
        # Add a complete trajectory
        batch = self._create_test_batch(3, buffer.device)
        trajectory_indices = torch.tensor([0, 0, 0], device=buffer.device)
        batch["dones"][2] = True
        batch["rewards"] = torch.tensor([0.0, 0.0, 1.0], device=buffer.device)
        buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Compute GAE
        buffer.compute_gae_returns(gamma=0.99, lambda_=0.95)

        # Check that advantages and returns are computed
        assert not torch.all(buffer.advantages[0, :3] == 0)
        assert not torch.all(buffer.returns[0, :3] == 0)

        # Check that returns = advantages + values
        expected_returns = buffer.advantages[0, :3] + buffer.values[0, :3]
        assert torch.allclose(buffer.returns[0, :3], expected_returns, atol=1e-6)

    def test_gae_with_multiple_trajectories(self, buffer):
        """Test GAE computation with multiple trajectories."""
        # Add two trajectories
        for i in range(2):
            batch = self._create_test_batch(2, buffer.device)
            trajectory_indices = torch.tensor([i, i], device=buffer.device)
            batch["dones"][1] = True
            batch["rewards"] = torch.tensor([0.0, 1.0], device=buffer.device)
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Compute GAE
        buffer.compute_gae_returns(gamma=0.99, lambda_=0.95)

        # Check both trajectories
        for i in range(2):
            assert not torch.all(buffer.advantages[i, :2] == 0)
            assert not torch.all(buffer.returns[i, :2] == 0)

    def test_clear(self, buffer):
        """Test clearing buffer."""
        # Add some trajectories
        for i in range(3):
            batch = self._create_test_batch(1, buffer.device)
            trajectory_indices = torch.tensor([i], device=buffer.device)
            batch["dones"][0] = True
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        assert buffer.size == 3

        # Clear buffer
        buffer.clear()

        assert buffer.size == 0
        assert buffer.position == 0
        assert not buffer.valid_trajectories.any()
        assert buffer.trajectory_lengths.sum() == 0
        assert buffer.current_step_positions.sum() == 0

    def test_finish_adding_trajectories(self, buffer):
        """Test finish_adding_trajectories method."""
        # Add some trajectories
        for i in range(3):
            batch = self._create_test_batch(1, buffer.device)
            trajectory_indices = torch.tensor([i], device=buffer.device)
            batch["dones"][0] = True
            buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        initial_size = buffer.size
        initial_position = buffer.position

        # Finish adding trajectories
        buffer.finish_adding_trajectories(3)

        # Should advance position and update size
        assert buffer.position == (initial_position + 3) % buffer.capacity
        assert buffer.size == min(initial_size + 3, buffer.capacity)

    def test_device_consistency(self, buffer):
        """Test that all tensors are on the correct device."""
        batch = self._create_test_batch(1, buffer.device)
        trajectory_indices = torch.tensor([0], device=buffer.device)
        batch["dones"][0] = True
        buffer.add_batch(trajectory_indices=trajectory_indices, **batch)

        # Check that all buffer tensors are on the correct device
        for attr_name in [
            "observations",
            "actions",
            "log_probs",
            "rewards",
            "dones",
            "legal_masks",
            "chips_placed",
            "delta2",
            "delta3",
            "values",
            "advantages",
            "returns",
            "valid_trajectories",
            "trajectory_lengths",
            "current_step_positions",
        ]:
            tensor = getattr(buffer, attr_name)
            assert tensor.device == buffer.device

    def test_trajectory_reuse_after_completion(self, buffer):
        """Test that trajectories can be reused after completion."""
        # Complete a trajectory
        batch1 = self._create_test_batch(2, buffer.device)
        trajectory_indices1 = torch.tensor([0, 0], device=buffer.device)
        batch1["dones"][1] = True
        buffer.add_batch(trajectory_indices=trajectory_indices1, **batch1)

        assert buffer.size == 1
        assert buffer.valid_trajectories[0] == True
        assert buffer.current_step_positions[0] == 0  # Reset

        # Reuse the same trajectory slot
        batch2 = self._create_test_batch(1, buffer.device)
        trajectory_indices2 = torch.tensor([0], device=buffer.device)
        batch2["dones"][0] = True
        buffer.add_batch(trajectory_indices=trajectory_indices2, **batch2)

        # Should still be valid and position should be reset
        # Note: Reusing a trajectory slot creates a new valid trajectory
        assert buffer.size == 2  # Now 2 valid trajectories (original + reused)
        assert buffer.valid_trajectories[0] == True
        assert buffer.current_step_positions[0] == 0

    def _create_test_batch(self, batch_size: int, device: torch.device) -> dict:
        """Create a test batch with random data."""
        return {
            "observations": torch.randn(batch_size, 10, device=device),
            "actions": torch.randint(0, 5, (batch_size,), device=device),
            "log_probs": torch.randn(batch_size, device=device),
            "rewards": torch.randn(batch_size, device=device),
            "dones": torch.zeros(batch_size, dtype=torch.bool, device=device),
            "legal_masks": torch.ones(batch_size, 5, device=device),
            "chips_placed": torch.randn(batch_size, device=device),
            "delta2": torch.randn(batch_size, device=device),
            "delta3": torch.randn(batch_size, device=device),
            "values": torch.randn(batch_size, device=device),
        }


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
