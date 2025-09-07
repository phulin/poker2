import pytest
import torch
from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer


class TestVectorizedReplayBuffer:
    """Test suite for VectorizedReplayBuffer with ring buffer implementation."""

    @pytest.fixture
    def buffer(self):
        """Create a test buffer."""
        device = torch.device("cpu")  # Use CPU for testing
        return VectorizedReplayBuffer(
            capacity=100, observation_dim=10, legal_mask_dim=5, device=device
        )

    def test_initialization(self, buffer):
        """Test buffer initialization."""
        assert buffer.capacity == 100
        assert buffer.size == 0
        assert buffer.position == 0
        assert buffer.effective_start == 0
        # Note: trajectory_starts and trajectory_lengths are computed on-demand, not stored as attributes

        # Check tensor shapes
        assert buffer.observations.shape == (100, 10)
        assert buffer.actions.shape == (100,)
        assert buffer.log_probs.shape == (100,)
        assert buffer.rewards.shape == (100,)
        assert buffer.dones.shape == (100,)
        assert buffer.legal_masks.shape == (100, 5)
        assert buffer.chips_placed.shape == (100,)
        assert buffer.delta2.shape == (100,)
        assert buffer.delta3.shape == (100,)
        assert buffer.values.shape == (100,)
        assert buffer.advantages.shape == (100,)
        assert buffer.returns.shape == (100,)

    def test_add_single_batch(self, buffer):
        """Test adding a single batch."""
        batch_size = 20
        batch = self._create_test_batch(batch_size, buffer.device)

        buffer.add_batch(**batch)

        assert buffer.size == batch_size
        assert buffer.position == batch_size
        assert buffer.effective_start == 0
        # Trajectory boundaries are computed on-demand from dones tensor

    def test_add_multiple_batches(self, buffer):
        """Test adding multiple batches."""
        batch_sizes = [15, 25, 20]
        total_size = 0

        for i, batch_size in enumerate(batch_sizes):
            batch = self._create_test_batch(batch_size, buffer.device)
            buffer.add_batch(**batch)
            total_size += batch_size

            assert buffer.size == total_size
            assert buffer.position == total_size % buffer.capacity
            assert buffer.effective_start == 0  # No wraparound yet

    def test_wraparound(self, buffer):
        """Test buffer wraparound when capacity is exceeded."""
        # Fill buffer beyond capacity
        batch_size = 60
        batch = self._create_test_batch(batch_size, buffer.device)
        buffer.add_batch(**batch)

        # Add another batch that causes wraparound
        batch2 = self._create_test_batch(50, buffer.device)
        buffer.add_batch(**batch2)

        # Check wraparound behavior
        assert buffer.size == 100  # Should be at capacity
        assert buffer.position == 10  # 60 + 50 - 100 = 10
        assert buffer.effective_start == 10  # Should have moved forward

    def test_sample_batch(self, buffer):
        """Test sampling from buffer."""
        batch_size = 30
        batch = self._create_test_batch(batch_size, buffer.device)
        buffer.add_batch(**batch)

        # Sample a batch
        sampled = buffer.sample_batch(10)

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

        # Check shapes
        assert sampled["observations"].shape == (10, 10)
        assert sampled["actions"].shape == (10,)
        assert sampled["log_probs_old"].shape == (10,)
        assert sampled["advantages"].shape == (10,)
        assert sampled["returns"].shape == (10,)
        assert sampled["legal_masks"].shape == (10, 5)
        assert sampled["delta2"].shape == (10,)
        assert sampled["delta3"].shape == (10,)

    def test_sample_empty_buffer(self, buffer):
        """Test sampling from empty buffer raises error."""
        with pytest.raises(ValueError, match="Cannot sample from empty buffer"):
            buffer.sample_batch(10)

    def test_gae_computation(self, buffer):
        """Test vectorized GAE computation."""
        batch_size = 20
        batch = self._create_test_batch(batch_size, buffer.device)
        buffer.add_batch(**batch)

        # Compute GAE
        buffer.compute_gae_returns(gamma=0.99, lambda_=0.95)

        # Check that advantages and returns are computed
        assert not torch.all(buffer.advantages == 0)
        assert not torch.all(buffer.returns == 0)

        # Check that returns = advantages + values
        expected_returns = buffer.advantages + buffer.values
        assert torch.allclose(buffer.returns, expected_returns, atol=1e-6)

    def test_gae_with_terminal_states(self, buffer):
        """Test GAE computation with terminal states."""
        batch_size = 10
        batch = self._create_test_batch(batch_size, buffer.device)

        # Set some terminal states
        batch["dones"][3] = True
        batch["dones"][7] = True
        batch["rewards"] = torch.tensor(
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0], device=buffer.device
        )

        buffer.add_batch(**batch)

        # Compute GAE
        buffer.compute_gae_returns(gamma=0.99, lambda_=0.95)

        # Check that terminal states are handled correctly
        assert not torch.all(buffer.advantages == 0)
        assert not torch.all(buffer.returns == 0)

        # Check that returns = advantages + values
        expected_returns = buffer.advantages + buffer.values
        assert torch.allclose(buffer.returns, expected_returns, atol=1e-6)

    def test_trim_to_steps(self, buffer):
        """Test trimming buffer to specified number of steps."""
        # Add multiple batches
        batch_sizes = [30, 40, 20]
        for batch_size in batch_sizes:
            batch = self._create_test_batch(batch_size, buffer.device)
            buffer.add_batch(**batch)

        total_size = sum(batch_sizes)
        assert buffer.size == total_size

        # Trim to 50 steps
        buffer.trim_to_steps(50)

        assert buffer.size == 50
        assert buffer.effective_start == total_size - 50  # Should move forward

    def test_trim_with_wraparound(self, buffer):
        """Test trimming when buffer has wraparound."""
        # Fill buffer and cause wraparound
        batch1 = self._create_test_batch(60, buffer.device)
        buffer.add_batch(**batch1)

        batch2 = self._create_test_batch(50, buffer.device)
        buffer.add_batch(**batch2)

        assert buffer.size == 100
        assert buffer.effective_start == 10

        # Trim to 30 steps
        buffer.trim_to_steps(30)

        assert buffer.size == 30
        assert buffer.effective_start == 80  # 10 + (100 - 30) = 80

    def test_clear(self, buffer):
        """Test clearing buffer."""
        # Add some data
        batch = self._create_test_batch(30, buffer.device)
        buffer.add_batch(**batch)

        assert buffer.size == 30

        # Clear buffer
        buffer.clear()

        assert buffer.size == 0
        assert buffer.position == 0
        assert buffer.effective_start == 0

    def test_trajectory_boundaries(self, buffer):
        """Test trajectory boundary computation with done flags."""
        # Create batch with some done flags
        batch_size = 20
        batch = self._create_test_batch(batch_size, buffer.device)

        # Set some done flags to create multiple trajectories
        batch["dones"][5] = True
        batch["dones"][12] = True

        buffer.add_batch(**batch)

        # Test that trajectory boundaries can be computed
        trajectory_starts, trajectory_lengths = buffer._compute_trajectory_boundaries(
            batch["dones"]
        )

        # Should have 3 trajectories: [0:6], [6:13], [13:20]
        assert len(trajectory_starts) == 3
        assert len(trajectory_lengths) == 3
        assert trajectory_starts.tolist() == [0, 6, 13]
        assert trajectory_lengths.tolist() == [6, 7, 7]

    def test_ring_buffer_semantics(self, buffer):
        """Test that ring buffer maintains correct semantics."""
        # Fill buffer completely
        batch = self._create_test_batch(100, buffer.device)
        buffer.add_batch(**batch)

        assert buffer.size == 100
        assert buffer.position == 0  # Wrapped around
        assert buffer.effective_start == 0

        # Add one more batch to test overwriting
        batch2 = self._create_test_batch(30, buffer.device)
        buffer.add_batch(**batch2)

        assert buffer.size == 100  # Still at capacity
        assert buffer.position == 30
        assert buffer.effective_start == 30  # Moved forward

        # Verify that old data is overwritten
        # The first 30 entries should be from batch2, not batch1
        assert torch.allclose(buffer.observations[:30], batch2["observations"])

    def test_sample_trajectories(self, buffer):
        """Test sampling complete trajectories."""
        # Add multiple trajectories
        batch1 = self._create_test_batch(20, buffer.device)
        batch1["dones"][10] = True  # Split into 2 trajectories
        buffer.add_batch(**batch1)

        batch2 = self._create_test_batch(15, buffer.device)
        buffer.add_batch(**batch2)

        # Sample trajectories
        sampled = buffer.sample_trajectories(2)

        # Check that we get concatenated data from multiple trajectories
        assert sampled["observations"].shape[0] > 0
        assert sampled["actions"].shape[0] > 0
        assert sampled["observations"].shape[0] == sampled["actions"].shape[0]

    def test_device_consistency(self, buffer):
        """Test that all tensors are on the correct device."""
        batch = self._create_test_batch(10, buffer.device)
        buffer.add_batch(**batch)

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
        ]:
            tensor = getattr(buffer, attr_name)
            assert tensor.device == buffer.device

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
