from __future__ import annotations

from p2.search.rebel_data_source import LiveRebelDataSource


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
