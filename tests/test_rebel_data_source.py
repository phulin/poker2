from __future__ import annotations

import torch

from p2.core.structured_config import PregeneratedDatasetConfig
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import RebelPolicyBuffer, RebelValueBuffer
from p2.search.rebel_data_source import LiveRebelDataSource
from p2.search.rebel_data_source import PregeneratedRebelDataSource
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
    source.load_state_dict({"value_cursors": [2], "policy_cursors": [2]})
    assert source.state_dict()["value_cursors"] == [2]
    assert source.state_dict()["policy_cursors"] == [2]
