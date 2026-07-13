import torch
import pytest

from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import (
    RebelPolicyBuffer,
    RebelReplayBuffer,
    StreamingEpochValueBuffer,
)


def _make_policy_batch(batch_size: int, num_actions: int, depths: torch.Tensor):
    return RebelBatch(
        features=MLPFeatures(
            context=torch.arange(batch_size, dtype=torch.float32)[:, None].expand(
                -1, 4
            ),
            street=torch.zeros(batch_size, dtype=torch.long),
            to_act=torch.zeros(batch_size, dtype=torch.long),
            board=torch.zeros(batch_size, 5, dtype=torch.long),
            beliefs=torch.randn(batch_size, 2 * NUM_HANDS),
        ),
        policy_targets=torch.softmax(
            torch.randn(batch_size, NUM_HANDS, num_actions), dim=-1
        ),
        value_targets=None,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
        statistics={"node_depth": depths},
    )


def _make_value_batch(ids: torch.Tensor, num_actions: int = 3) -> RebelBatch:
    count = int(ids.shape[0])
    return RebelBatch(
        features=MLPFeatures(
            context=ids.float()[:, None],
            street=torch.full((count,), 2, dtype=torch.long),
            to_act=torch.zeros(count, dtype=torch.long),
            board=torch.zeros(count, 5, dtype=torch.long),
            beliefs=torch.ones(count, 2 * NUM_HANDS),
        ),
        policy_targets=None,
        value_targets=torch.zeros(count, 2, NUM_HANDS),
        legal_masks=torch.ones(count, num_actions, dtype=torch.bool),
        statistics={"sample_id": ids.clone()},
    )


def test_streaming_epoch_value_buffer_exact_coverage_and_swap() -> None:
    buffer = StreamingEpochValueBuffer(
        block_capacity=8,
        epochs=3,
        num_actions=3,
        num_players=2,
        num_context_features=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(17),
    )
    buffer.add_batch(_make_value_batch(torch.arange(3)))
    assert len(buffer) == 0
    buffer.add_batch(_make_value_batch(torch.arange(3, 8)))
    assert len(buffer) == 8

    buffer.add_batch(_make_value_batch(torch.arange(8, 16)))
    seen = []
    for _ in range(12):
        seen.extend(buffer.sample(2).statistics["sample_id"].tolist())

    assert sorted(seen) == sorted(list(range(8)) * 3)
    assert buffer.read_block == 1
    assert buffer.write_size == 0
    assert set(buffer.sample(2).statistics["sample_id"].tolist()).issubset(
        set(range(8, 16))
    )


def test_streaming_epoch_value_buffer_checkpoint_roundtrip() -> None:
    source = StreamingEpochValueBuffer(
        block_capacity=8,
        epochs=3,
        num_actions=3,
        num_players=2,
        num_context_features=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(19),
    )
    source.add_batch(_make_value_batch(torch.arange(8)))
    source.add_batch(_make_value_batch(torch.arange(8, 12)))
    source.sample(2)

    restored = StreamingEpochValueBuffer(
        block_capacity=8,
        epochs=3,
        num_actions=3,
        num_players=2,
        num_context_features=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(19),
    )
    restored.load_state_dict(source.state_dict())

    assert restored.read_block == source.read_block
    assert restored.write_block == source.write_block
    assert restored.write_size == source.write_size
    assert restored.read_epoch == source.read_epoch
    assert restored.read_cursor == source.read_cursor
    torch.testing.assert_close(restored.read_order, source.read_order)
    torch.testing.assert_close(
        restored.sample(2).statistics["sample_id"],
        source.sample(2).statistics["sample_id"],
    )


def test_streaming_epoch_value_buffer_carries_generation_overflow() -> None:
    buffer = StreamingEpochValueBuffer(
        block_capacity=8,
        epochs=1,
        num_actions=3,
        num_players=2,
        num_context_features=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(23),
    )
    buffer.add_batch(_make_value_batch(torch.arange(10)))

    assert len(buffer) == 8
    assert buffer.write_size == 2
    assert not buffer.pending_batches

    buffer.add_batch(_make_value_batch(torch.arange(10, 18)))
    assert buffer.write_size == 8
    assert len(buffer.pending_batches) == 1
    assert len(buffer.pending_batches[0]) == 2

    for _ in range(4):
        buffer.sample(2)

    assert buffer.write_size == 2
    assert not buffer.pending_batches


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


def test_rebel_replay_buffer_allocates_multiway_beliefs():
    buffer = RebelReplayBuffer(
        capacity=16,
        num_actions=5,
        num_players=3,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    assert buffer.features.beliefs.shape == (16, 3 * NUM_HANDS)
    assert buffer.value_targets.shape == (16, 3, NUM_HANDS)


def test_rebel_replay_buffer_state_dict_roundtrip():
    device = torch.device("cpu")
    source = RebelReplayBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
    )
    batch = RebelBatch(
        features=MLPFeatures(
            context=torch.arange(24, dtype=torch.float32).view(6, 4),
            street=torch.arange(6, dtype=torch.long) % 4,
            to_act=torch.arange(6, dtype=torch.long) % 2,
            board=torch.arange(30, dtype=torch.long).view(6, 5),
            beliefs=torch.randn(6, 2 * NUM_HANDS),
        ),
        policy_targets=torch.softmax(torch.randn(6, NUM_HANDS, 5), dim=-1),
        value_targets=torch.randn(6, 2, NUM_HANDS),
        legal_masks=torch.ones(6, 5, dtype=torch.bool),
        statistics={"node_depth": torch.arange(6, dtype=torch.long)},
    )
    source.add_batch(batch)
    source.sample(3)

    restored = RebelReplayBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
    )
    restored.load_state_dict(source.state_dict())

    assert restored.position == source.position
    assert restored.size == source.size
    torch.testing.assert_close(restored.features.context, source.features.context)
    torch.testing.assert_close(restored.policy_targets, source.policy_targets)
    torch.testing.assert_close(restored.value_targets, source.value_targets)
    torch.testing.assert_close(restored.legal_masks, source.legal_masks)
    torch.testing.assert_close(restored.sample_count, source.sample_count)
    torch.testing.assert_close(
        restored.statistics["node_depth"], source.statistics["node_depth"]
    )


def test_rebel_replay_buffer_allows_mixed_statistics_keys():
    buffer = RebelReplayBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
        policy_targets=False,
    )
    features = MLPFeatures(
        context=torch.arange(16, dtype=torch.float32).view(4, 4),
        street=torch.zeros(4, dtype=torch.long),
        to_act=torch.zeros(4, dtype=torch.long),
        board=torch.zeros(4, 5, dtype=torch.long),
        beliefs=torch.randn(4, 2 * NUM_HANDS),
    )
    first = RebelBatch(
        features=features[:2],
        policy_targets=None,
        value_targets=torch.randn(2, 2, NUM_HANDS),
        legal_masks=torch.ones(2, 5, dtype=torch.bool),
        statistics={"node_depth": torch.tensor([1, 2])},
    )
    second = RebelBatch(
        features=features[2:],
        policy_targets=None,
        value_targets=torch.randn(2, 2, NUM_HANDS),
        legal_masks=torch.ones(2, 5, dtype=torch.bool),
        statistics={"target_source": torch.tensor([3, 4])},
    )

    buffer.add_batch(first)
    buffer.add_batch(second)

    valid = buffer._valid_physical_indices()
    assert set(buffer.statistics) == {"node_depth", "target_source"}
    torch.testing.assert_close(
        buffer.statistics["node_depth"][valid], torch.tensor([1, 2, 0, 0])
    )
    torch.testing.assert_close(
        buffer.statistics["target_source"][valid], torch.tensor([0, 0, 3, 4])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rebel_replay_buffer_cuda_roundtrip():
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    buffer = RebelReplayBuffer(
        capacity=16,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
        generator=generator,
    )
    batch_size = 4
    batch = RebelBatch(
        features=MLPFeatures(
            context=torch.randn(batch_size, 4, device=device),
            street=torch.zeros(batch_size, dtype=torch.long, device=device),
            to_act=torch.zeros(batch_size, dtype=torch.long, device=device),
            board=torch.zeros(batch_size, 5, dtype=torch.long, device=device),
            beliefs=torch.randn(batch_size, 2 * NUM_HANDS, device=device),
        ),
        policy_targets=torch.softmax(
            torch.randn(batch_size, NUM_HANDS, 5, device=device), dim=-1
        ),
        value_targets=torch.randn(batch_size, 2, NUM_HANDS, device=device),
        legal_masks=torch.ones(batch_size, 5, dtype=torch.bool, device=device),
    )

    buffer.add_batch(batch)
    sample = buffer.sample(2)

    assert len(buffer) == batch_size
    assert sample.features.context.device.type == "cuda"
    assert sample.policy_targets is not None
    assert sample.policy_targets.device.type == "cuda"
    assert sample.value_targets is not None
    assert sample.value_targets.device.type == "cuda"


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

    # Add the batch - should be decimated to ~50%
    buffer.add_batch(batch2)

    # Buffer should still be at capacity
    assert len(buffer) == capacity, (
        f"Buffer should still be at capacity, got {len(buffer)}"
    )

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
    assert len(buffer) == batch_size, (
        f"Buffer should have {batch_size} samples, got {len(buffer)}"
    )


def test_underfull_eviction_removes_oldest_entries():
    device = torch.device("cpu")
    buffer = RebelPolicyBuffer(
        capacity=20,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
        underfull_evict_fraction=0.25,
    )

    buffer.add_batch(_make_policy_batch(10, 5, torch.arange(10)))
    buffer.add_batch(_make_policy_batch(8, 5, torch.arange(100, 108)))

    assert len(buffer) == 16
    assert buffer.start == 2
    assert buffer.position == 18
    torch.testing.assert_close(
        buffer.statistics["node_depth"][buffer._valid_physical_indices()],
        torch.cat([torch.arange(2, 10), torch.arange(100, 108)]),
    )


def test_policy_buffer_depth_stratified_decimation_keeps_roots():
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    buffer = RebelPolicyBuffer(
        capacity=100,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
        decimate=0.2,
        generator=generator,
        depth_stratify_decimate=True,
        depth_stratify_buckets=4,
    )

    buffer.add_batch(_make_policy_batch(100, 5, torch.full((100,), 3)))
    incoming_depths = torch.cat(
        [
            torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            torch.full((46,), 3, dtype=torch.long),
        ]
    )
    buffer.add_batch(_make_policy_batch(50, 5, incoming_depths))

    stored_depths = buffer.statistics["node_depth"][: len(buffer)]
    assert (stored_depths == 0).sum() == 2
    assert (stored_depths == 1).sum() >= 1


def test_policy_buffer_depth_stratified_sampling_balances_depths():
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(1)
    buffer = RebelPolicyBuffer(
        capacity=100,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=device,
        generator=generator,
        depth_stratify_sample=True,
        depth_stratify_buckets=4,
    )
    depths = torch.cat(
        [
            torch.zeros(5, dtype=torch.long),
            torch.ones(5, dtype=torch.long),
            torch.full((5,), 2, dtype=torch.long),
            torch.full((85,), 3, dtype=torch.long),
        ]
    )
    buffer.add_batch(_make_policy_batch(100, 5, depths))

    sample = buffer.sample(80)
    sampled_depths = sample.statistics["node_depth"]
    counts = torch.bincount(sampled_depths, minlength=4).float()
    proportions = counts / counts.sum()

    assert torch.all(proportions[:3] > 0.15)
    assert proportions[3] < 0.45
