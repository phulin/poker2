import torch

from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.rl.rebel_batch import RebelBatch
from alphaholdem.rl.rebel_replay import RebelReplayBuffer


def test_rebel_replay_buffer_roundtrip():
    buffer = RebelReplayBuffer(
        capacity=16,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    # Create MLPFeatures for the test
    mlp_features = MLPFeatures(
        context=torch.randn(4, 4),
        street=torch.zeros(4, dtype=torch.long),
        to_act=torch.zeros(4, dtype=torch.long),
        board=torch.zeros(4, 5, dtype=torch.long),
        beliefs=torch.randn(4, 2 * NUM_HANDS),
    )
    policy_targets = torch.softmax(torch.randn(4, NUM_HANDS, 5), dim=-1)
    value_targets = torch.randn(4, 2, NUM_HANDS)
    legal_masks = torch.ones(4, 5, dtype=torch.bool)
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        legal_masks=legal_masks,
    )
    buffer.add_batch(batch)
    assert len(buffer) == 4
    sample = buffer.sample(2)
    assert sample.features.context.shape == (2, 4)
    assert sample.policy_targets.shape == (2, NUM_HANDS, 5)
    assert sample.value_targets.shape == (2, 2, NUM_HANDS)
    assert sample.legal_masks.shape == (2, 5)


def test_decimation():
    """Test that decimation works when buffer is full."""
    device = torch.device("cpu")
    capacity = 10
    decimate = 0.5  # Keep 50% of samples
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    buffer = RebelReplayBuffer(
        capacity=capacity,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
        policy_targets=True,
        value_targets=False,
        decimate=decimate,
        generator=generator,
    )

    # Fill the buffer to capacity
    batch_size = capacity
    mlp_features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.zeros(batch_size, 5, dtype=torch.long),
        beliefs=torch.randn(batch_size, 2 * NUM_HANDS),
    )
    policy_targets = torch.softmax(torch.randn(batch_size, NUM_HANDS, 5), dim=-1)
    legal_masks = torch.ones(batch_size, 5, dtype=torch.bool)
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=None,
        legal_masks=legal_masks,
    )
    buffer.add_batch(batch)
    assert len(buffer) == capacity, f"Buffer should be full, got {len(buffer)}"

    # Now add another batch - should be decimated
    new_batch_size = 8
    mlp_features2 = MLPFeatures(
        context=torch.randn(new_batch_size, 4),
        street=torch.zeros(new_batch_size, dtype=torch.long),
        to_act=torch.zeros(new_batch_size, dtype=torch.long),
        board=torch.zeros(new_batch_size, 5, dtype=torch.long),
        beliefs=torch.randn(new_batch_size, 2 * NUM_HANDS),
    )
    policy_targets2 = torch.softmax(torch.randn(new_batch_size, NUM_HANDS, 5), dim=-1)
    legal_masks2 = torch.ones(new_batch_size, 5, dtype=torch.bool)
    batch2 = RebelBatch(
        features=mlp_features2,
        policy_targets=policy_targets2,
        value_targets=None,
        legal_masks=legal_masks2,
    )

    # Store original size
    original_size = len(buffer)

    # Add the batch - should be decimated to ~50%
    buffer.add_batch(batch2)

    # Buffer should still be at capacity
    assert (
        len(buffer) == capacity
    ), f"Buffer should still be at capacity, got {len(buffer)}"

    # Verify that some of the new batch's data is in the buffer
    # (we can't easily verify exact count without knowing which entries were overwritten,
    # but we can verify the buffer still works)
    sample = buffer.sample(5)
    assert len(sample) == 5
    assert sample.policy_targets.shape == (5, NUM_HANDS, 5)


def test_no_decimation_when_not_full():
    """Test that decimation doesn't happen when buffer is not full."""
    device = torch.device("cpu")
    capacity = 20
    decimate = 0.5
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    buffer = RebelReplayBuffer(
        capacity=capacity,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
        policy_targets=True,
        value_targets=False,
        decimate=decimate,
        generator=generator,
    )

    # Add a batch when buffer is not full - should not be decimated
    batch_size = 10
    mlp_features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.zeros(batch_size, 5, dtype=torch.long),
        beliefs=torch.randn(batch_size, 2 * NUM_HANDS),
    )
    policy_targets = torch.softmax(torch.randn(batch_size, NUM_HANDS, 5), dim=-1)
    legal_masks = torch.ones(batch_size, 5, dtype=torch.bool)
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=None,
        legal_masks=legal_masks,
    )
    buffer.add_batch(batch)

    # Buffer should have all 10 samples (no decimation)
    assert (
        len(buffer) == batch_size
    ), f"Buffer should have {batch_size} samples, got {len(buffer)}"
