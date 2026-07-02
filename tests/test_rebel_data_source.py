from __future__ import annotations

import pytest
import torch

from p2.core.structured_config import PregeneratedDatasetConfig
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import RebelPolicyBuffer, RebelValueBuffer
from p2.search.rebel_data_source import (
    BootstrapPregeneratedRebelDataSource,
    HybridRebelDataSource,
    LiveRebelDataSource,
    PregeneratedRebelDataSource,
)
from p2.search.rebel_solved_dataset import write_rebel_solved_dataset


class _FakeBuffer:
    def __init__(self, size: int = 0) -> None:
        self.size = size
        self.samples = []

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, stratify_streets):
        self.samples.append((batch_size, stratify_streets))
        return ("batch", batch_size, stratify_streets)


class _StagingOnlyBuffer:
    def __init__(self) -> None:
        self.size = 0
        self.sample_calls = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, batch: RebelBatch) -> None:
        self.size += len(batch)

    def sample(self, batch_size: int, stratify_streets):
        del batch_size, stratify_streets
        self.sample_calls += 1
        raise AssertionError("direct pregenerated sampling should not hit buffer")


class _FakeGenerator:
    def __init__(self, value_buffer: _FakeBuffer, policy_buffer: _FakeBuffer) -> None:
        self.value_buffer = value_buffer
        self.policy_buffer = policy_buffer
        self.calls = []
        self.loaded_state = None

    def generate_data(self, value_sample_count: int, **kwargs):
        self.calls.append((value_sample_count, kwargs))
        self.value_buffer.size += value_sample_count
        self.policy_buffer.size += value_sample_count * 2
        if kwargs.get("return_value_batch", True) or kwargs.get(
            "return_policy_batch", True
        ):
            return "value", "policy"
        return None, None

    def state_dict(self):
        return {"cursor": 3}

    def load_state_dict(self, state):
        self.loaded_state = state


def test_live_rebel_data_source_delegates_generation_sampling_and_state():
    value_buffer = _FakeBuffer()
    policy_buffer = _FakeBuffer()
    generator = _FakeGenerator(value_buffer, policy_buffer)
    source = LiveRebelDataSource(
        generator,
        value_buffer,
        policy_buffer,
        value_sample_count=5,
        max_return_policy_samples=7,
    )

    assert source.prepare_step(11) == ("value", "policy")
    assert generator.calls[-1] == (
        5,
        {
            "return_policy_batch": True,
            "max_return_policy_samples": 7,
        },
    )

    source.ensure_min_samples(value_samples=12, policy_samples=20)
    assert len(value_buffer) >= 12
    assert len(policy_buffer) >= 20
    assert source.sample_value(4, [0.1, 0.2, 0.3, 0.4]) == (
        "batch",
        4,
        [0.1, 0.2, 0.3, 0.4],
    )
    assert source.sample_policy(6, None) == ("batch", 6, None)

    assert source.state_dict() == {"cursor": 3}
    source.load_state_dict({"cursor": 4})
    assert generator.loaded_state == {"cursor": 4}


def test_hybrid_rebel_data_source_trains_live_and_returns_holdout_metrics():
    live_value_buffer = _FakeBuffer()
    live_policy_buffer = _FakeBuffer()
    holdout_value_buffer = _FakeBuffer()
    holdout_policy_buffer = _FakeBuffer()
    live = LiveRebelDataSource(
        _FakeGenerator(live_value_buffer, live_policy_buffer),
        live_value_buffer,
        live_policy_buffer,
        value_sample_count=5,
        max_return_policy_samples=7,
    )
    holdout = LiveRebelDataSource(
        _FakeGenerator(holdout_value_buffer, holdout_policy_buffer),
        holdout_value_buffer,
        holdout_policy_buffer,
        value_sample_count=2,
        max_return_policy_samples=3,
    )
    source = HybridRebelDataSource(live, holdout)

    assert source.prepare_step(4) == ("value", "policy")
    assert len(live_value_buffer) == 5
    assert len(holdout_value_buffer) == 2
    source.ensure_min_samples(10, 20)
    assert len(live_value_buffer) >= 10
    assert source.sample_value(4, None) == ("batch", 4, None)
    assert source.sample_policy(6, [0.25, 0.25, 0.25, 0.25]) == (
        "batch",
        6,
        [0.25, 0.25, 0.25, 0.25],
    )
    assert source.state_dict() == {
        "live": {"cursor": 3},
        "holdout": {"cursor": 3},
    }


def _batch(stream: str, start: int, count: int) -> RebelBatch:
    rows = torch.arange(start, start + count)
    features = MLPFeatures(
        context=torch.stack(
            [rows.float(), rows.float() + 1.0, rows.float() + 2.0, rows.float() + 3.0],
            dim=1,
        ),
        street=rows.remainder(4).long(),
        to_act=rows.remainder(2).long(),
        board=torch.full((count, 5), -1, dtype=torch.long),
        beliefs=torch.full((count, 2 * NUM_HANDS), 1.0 / NUM_HANDS),
    )
    kwargs = {}
    if stream == "value":
        kwargs["value_targets"] = torch.full((count, 2, NUM_HANDS), float(start))
    else:
        policy_targets = torch.zeros(count, NUM_HANDS, 5)
        policy_targets[..., 1] = 1.0
        kwargs["policy_targets"] = policy_targets
    return RebelBatch(
        features=features,
        legal_masks=torch.ones(count, 5, dtype=torch.bool),
        statistics={"node_depth": rows},
        **kwargs,
    )


def test_pregenerated_rebel_data_source_stages_batches_and_state(tmp_path):
    write_rebel_solved_dataset(
        tmp_path,
        value_batches=[_batch("value", 0, 3)],
        policy_batches=[_batch("policy", 10, 3)],
    )
    value_buffer = RebelValueBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    policy_buffer = RebelPolicyBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    source = PregeneratedRebelDataSource(
        [PregeneratedDatasetConfig(path=str(tmp_path))],
        value_buffer,
        policy_buffer,
        value_sample_count=2,
        policy_sample_count=2,
        num_players=2,
        num_actions=5,
        context_length=4,
        generator=torch.Generator().manual_seed(7),
        shuffle=False,
    )

    fresh_value, fresh_policy = source.prepare_step(0)
    assert len(fresh_value) == 2
    assert len(fresh_policy) == 2
    assert len(value_buffer) == 2
    assert len(policy_buffer) == 2

    source.ensure_min_samples(value_samples=4, policy_samples=4)
    assert len(value_buffer) >= 4
    assert len(policy_buffer) >= 4
    assert len(source.sample_value(2, None)) == 2
    assert len(source.sample_policy(2, None)) == 2

    state = source.state_dict()
    assert state["value_cursors"] == [1]
    assert state["policy_cursors"] == [1]
    assert state["datasets"][0]["path"] == str(tmp_path)
    assert state["datasets"][0]["manifest"]["value_examples"] == 3
    assert state["datasets"][0]["manifest"]["policy_examples"] == 3
    assert state["datasets"][0]["value_cursor"] == 1
    assert state["datasets"][0]["policy_cursor"] == 1
    source.load_state_dict(
        {"value_cursors": [2], "policy_cursors": [2], "current_step": 13}
    )
    assert source.state_dict()["value_cursors"] == [2]
    assert source.state_dict()["policy_cursors"] == [2]
    assert source.state_dict()["current_step"] == 13


def test_pregenerated_rebel_data_source_rejects_manifest_mismatch(tmp_path):
    write_rebel_solved_dataset(
        tmp_path,
        value_batches=[_batch("value", 0, 3)],
        policy_batches=[_batch("policy", 10, 3)],
    )
    value_buffer = RebelValueBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    policy_buffer = RebelPolicyBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    source = PregeneratedRebelDataSource(
        [PregeneratedDatasetConfig(path=str(tmp_path))],
        value_buffer,
        policy_buffer,
        value_sample_count=2,
        policy_sample_count=2,
        num_players=2,
        num_actions=5,
        context_length=4,
        generator=torch.Generator().manual_seed(7),
        shuffle=False,
    )
    state = source.state_dict()
    state["datasets"][0]["manifest"]["value_examples"] = 999

    with pytest.raises(ValueError, match="checkpoint manifest mismatch"):
        source.load_state_dict(state)


def test_pregenerated_rebel_data_source_honors_step_windows_and_shuffle(tmp_path):
    early = tmp_path / "early"
    late = tmp_path / "late"
    write_rebel_solved_dataset(
        early,
        value_batches=[_batch("value", 0, 5)],
        policy_batches=[_batch("policy", 100, 5)],
    )
    write_rebel_solved_dataset(
        late,
        value_batches=[_batch("value", 50, 5)],
        policy_batches=[_batch("policy", 150, 5)],
    )
    value_buffer = RebelValueBuffer(
        capacity=16,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    policy_buffer = RebelPolicyBuffer(
        capacity=16,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    source = PregeneratedRebelDataSource(
        [
            PregeneratedDatasetConfig(path=str(early), min_step=0, max_step=10),
            PregeneratedDatasetConfig(path=str(late), min_step=10),
        ],
        value_buffer,
        policy_buffer,
        value_sample_count=3,
        policy_sample_count=3,
        num_players=2,
        num_actions=5,
        context_length=4,
        generator=torch.Generator().manual_seed(11),
        shuffle=True,
    )

    early_value, early_policy = source.prepare_step(9)
    assert (early_value.features.context[:, 0] < 5).all()
    assert (early_policy.features.context[:, 0] >= 100).all()
    assert (early_policy.features.context[:, 0] < 105).all()
    # Shuffled pregenerated sampling should not advance sequential cursors.
    assert source.state_dict()["value_cursors"] == [0, 0]
    assert source.state_dict()["policy_cursors"] == [0, 0]

    late_value, late_policy = source.prepare_step(10)
    assert (late_value.features.context[:, 0] >= 50).all()
    assert (late_value.features.context[:, 0] < 55).all()
    assert (late_policy.features.context[:, 0] >= 150).all()


def test_pregenerated_rebel_data_source_can_sample_directly_from_dataset(tmp_path):
    write_rebel_solved_dataset(
        tmp_path,
        value_batches=[_batch("value", 0, 5)],
        policy_batches=[_batch("policy", 100, 5)],
    )
    value_buffer = _StagingOnlyBuffer()
    policy_buffer = _StagingOnlyBuffer()
    source = PregeneratedRebelDataSource(
        [PregeneratedDatasetConfig(path=str(tmp_path))],
        value_buffer,
        policy_buffer,
        value_sample_count=1,
        policy_sample_count=1,
        num_players=2,
        num_actions=5,
        context_length=4,
        generator=torch.Generator().manual_seed(13),
        shuffle=False,
        direct_sample=True,
    )

    source.prepare_step(0)
    value_batch = source.sample_value(2, stratify_streets=None)
    policy_batch = source.sample_policy(2, stratify_streets=None)

    torch.testing.assert_close(
        value_batch.features.context[:, 0], torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(
        policy_batch.features.context[:, 0], torch.tensor([101.0, 102.0])
    )
    assert value_buffer.sample_calls == 0
    assert policy_buffer.sample_calls == 0


def test_bootstrap_pregenerated_value_stages_into_live_replay_buffer(tmp_path):
    write_rebel_solved_dataset(
        tmp_path,
        value_batches=[_batch("value", 0, 5)],
        policy_batches=[_batch("policy", 100, 5)],
    )
    value_buffer = RebelValueBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    policy_buffer = RebelPolicyBuffer(
        capacity=8,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    pregenerated = PregeneratedRebelDataSource(
        [PregeneratedDatasetConfig(path=str(tmp_path))],
        value_buffer,
        policy_buffer,
        value_sample_count=2,
        policy_sample_count=0,
        num_players=2,
        num_actions=5,
        context_length=4,
        generator=torch.Generator().manual_seed(17),
        shuffle=False,
    )
    live = LiveRebelDataSource(
        _FakeGenerator(value_buffer, policy_buffer),
        value_buffer,
        policy_buffer,
        value_sample_count=2,
        max_return_policy_samples=2,
    )
    source = BootstrapPregeneratedRebelDataSource(pregenerated, live)

    fresh = source.prepare_value_bootstrap_step(0)

    assert fresh is not None
    assert len(fresh) == 2
    assert len(value_buffer) == 2
    assert source.bootstrap_value_available() == 2
    assert source.bootstrap_value_remaining() == 3
    assert source._resident_value is None

    source.ensure_min_value_samples(4)

    assert len(value_buffer) == 4
    assert source.bootstrap_value_available() == 4
    assert source.bootstrap_value_remaining() == 1
    sampled = source.sample_value(2, stratify_streets=None)
    assert len(sampled) == 2
    assert int(value_buffer.sample_count.sum().item()) == 2
