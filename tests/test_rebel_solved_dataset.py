from __future__ import annotations

import json

import pytest
import torch

from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.search.rebel_solved_dataset import (
    MANIFEST_NAME,
    RebelSolvedDataset,
    rebel_batch_from_tensors,
    rebel_batch_to_tensors,
    write_rebel_solved_dataset,
)


def _features(start: int, count: int, *, num_actions: int = 5) -> MLPFeatures:
    rows = torch.arange(start, start + count)
    beliefs = torch.zeros(count, 2, NUM_HANDS)
    beliefs[:, :, rows.remainder(NUM_HANDS)] = 1.0
    return MLPFeatures(
        context=torch.stack(
            [
                rows.to(torch.float32),
                rows.to(torch.float32) + 0.5,
                torch.full((count,), float(num_actions)),
            ],
            dim=1,
        ),
        street=rows.remainder(4).to(torch.long),
        to_act=rows.remainder(2).to(torch.long),
        board=torch.full((count, 5), -1, dtype=torch.long),
        beliefs=beliefs.reshape(count, -1),
    )


def _value_batch(start: int, count: int) -> RebelBatch:
    features = _features(start, count)
    value_targets = torch.arange(
        start * 2 * NUM_HANDS,
        (start + count) * 2 * NUM_HANDS,
        dtype=torch.float32,
    ).reshape(count, 2, NUM_HANDS)
    return RebelBatch(
        features=features,
        legal_masks=torch.ones(count, 5, dtype=torch.bool),
        value_targets=value_targets,
        statistics={"node_depth": torch.arange(start, start + count)},
    )


def _policy_batch(start: int, count: int) -> RebelBatch:
    features = _features(start, count)
    policy_targets = torch.zeros(count, NUM_HANDS, 5)
    policy_targets[..., 1] = 1.0
    return RebelBatch(
        features=features,
        legal_masks=torch.ones(count, 5, dtype=torch.bool),
        policy_targets=policy_targets,
        statistics={"node_depth": torch.arange(start, start + count)},
    )


def test_rebel_batch_tensor_serialization_round_trips():
    batch = _value_batch(0, 3)

    restored = rebel_batch_from_tensors(rebel_batch_to_tensors(batch))

    torch.testing.assert_close(restored.features.context, batch.features.context)
    torch.testing.assert_close(restored.features.beliefs, batch.features.beliefs)
    assert torch.equal(restored.legal_masks, batch.legal_masks)
    torch.testing.assert_close(restored.value_targets, batch.value_targets)
    assert restored.policy_targets is None
    assert torch.equal(restored.statistics["node_depth"], batch.statistics["node_depth"])


def test_rebel_solved_dataset_reads_wrapped_batches(tmp_path):
    value_batches = [_value_batch(0, 2), _value_batch(2, 3)]
    policy_batches = [_policy_batch(10, 2)]

    manifest = write_rebel_solved_dataset(
        tmp_path,
        value_batches=value_batches,
        policy_batches=policy_batches,
        metadata={"stage": "river"},
    )

    assert manifest["value_examples"] == 5
    assert manifest["policy_examples"] == 2
    assert manifest["street_support"] == [0, 1, 2, 3]
    dataset = RebelSolvedDataset(
        tmp_path,
        num_players=2,
        num_actions=5,
        context_length=3,
        street_support=[0, 1, 2, 3],
    )

    value = dataset.get_batch("value", 4, 3, wrap=True)
    expected = RebelBatch.cat([_value_batch(4, 1), _value_batch(0, 2)])
    torch.testing.assert_close(value.features.context, expected.features.context)
    torch.testing.assert_close(value.value_targets, expected.value_targets)
    assert torch.equal(value.statistics["node_depth"], expected.statistics["node_depth"])

    policy = dataset.get_batch("policy", 0, 2)
    assert policy.value_targets is None
    assert policy.policy_targets is not None
    torch.testing.assert_close(policy.features.context, _policy_batch(10, 2).features.context)


def test_rebel_solved_dataset_samples_random_rows(tmp_path):
    write_rebel_solved_dataset(tmp_path, value_batches=[_value_batch(0, 6)])
    dataset = RebelSolvedDataset(tmp_path)

    batch = dataset.sample_batch(
        "value",
        4,
        generator=torch.Generator().manual_seed(123),
    )

    assert len(batch) == 4
    assert batch.value_targets is not None
    assert batch.policy_targets is None
    assert (batch.features.context[:, 0] >= 0).all()
    assert (batch.features.context[:, 0] < 6).all()


def test_rebel_solved_dataset_prefetched_reads_match_plain_reads(tmp_path):
    write_rebel_solved_dataset(
        tmp_path,
        value_batches=[_value_batch(0, 2), _value_batch(2, 2)],
        policy_batches=[_policy_batch(10, 2), _policy_batch(12, 2)],
    )
    plain = RebelSolvedDataset(tmp_path)
    prefetched = RebelSolvedDataset(
        tmp_path,
        pin_memory=True,
        async_shard_prefetch=True,
    )

    plain_first = plain.get_batch("value", 1, 3, wrap=True)
    prefetched.prefetch_shard_for_row("value", 1)
    prefetched_first = prefetched.get_batch("value", 1, 3, wrap=True)
    torch.testing.assert_close(
        prefetched_first.features.context, plain_first.features.context
    )
    torch.testing.assert_close(prefetched_first.value_targets, plain_first.value_targets)

    plain_policy = plain.get_batch("policy", 1, 3, wrap=True)
    prefetched.prefetch_shard_for_row("policy", 1)
    prefetched_policy = prefetched.get_batch("policy", 1, 3, wrap=True)
    torch.testing.assert_close(
        prefetched_policy.policy_targets, plain_policy.policy_targets
    )
    prefetched.close()


def test_rebel_solved_dataset_rejects_manifest_mismatch(tmp_path):
    write_rebel_solved_dataset(
        tmp_path,
        value_batches=[_value_batch(0, 2)],
        metadata={
            "model_family": "BetterFFN",
            "action_schedule": {
                "bet_bins": [0.5, 1.0],
                "bet_bins_by_depth": None,
                "allin_by_depth": None,
            },
        },
    )

    with pytest.raises(ValueError, match="num_actions mismatch"):
        RebelSolvedDataset(tmp_path, num_actions=9)
    with pytest.raises(ValueError, match="model mismatch"):
        RebelSolvedDataset(tmp_path, model_name="OtherModel")
    with pytest.raises(ValueError, match="action_schedule mismatch"):
        RebelSolvedDataset(
            tmp_path,
            action_schedule={
                "bet_bins": [1.0],
                "bet_bins_by_depth": None,
                "allin_by_depth": None,
            },
        )
    with pytest.raises(ValueError, match="street_support mismatch"):
        RebelSolvedDataset(tmp_path, street_support=[3])

    manifest_path = tmp_path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported solved dataset format"):
        RebelSolvedDataset(tmp_path)
