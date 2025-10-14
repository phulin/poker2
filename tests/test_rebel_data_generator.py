from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.search.rebel_cfr_evaluator import PublicBeliefState
from alphaholdem.search.rebel_data_generator import NUM_HANDS, RebelDataGenerator
from alphaholdem.rl.rebel_replay import RebelBatch


class DummyEnv:
    """Lightweight stand-in for HUNLTensorEnv tracking copy/reset usage."""

    def __init__(self, num_envs: int, num_actions: int, base_state: int = 0):
        self.N = int(num_envs)
        self.num_actions = int(num_actions)
        self.states = torch.arange(base_state, base_state + self.N, dtype=torch.float32)
        rows = torch.arange(self.N, dtype=torch.long).unsqueeze(1)
        cols = torch.arange(self.num_actions, dtype=torch.long).unsqueeze(0)
        self._legal_mask = (rows + cols) % 2 == 0
        self.copy_history: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.reset_history: list[torch.Tensor] = []

    def copy_state_from(
        self,
        src_env: DummyEnv,
        src_indices: torch.Tensor,
        dest_indices: torch.Tensor,
        copy_deck: bool = False,
    ) -> None:
        src = src_indices.to(dtype=torch.long)
        dest = dest_indices.to(dtype=torch.long)
        if dest.numel() == 0:
            return
        self.copy_history.append((src.clone(), dest.clone()))
        self.states[dest] = src_env.states[src]

    def reset(self, indices: torch.Tensor | None = None) -> None:
        if indices is None:
            ids = torch.arange(self.N, dtype=torch.long)
        else:
            ids = indices.to(dtype=torch.long)
        if ids.numel() == 0:
            return
        self.reset_history.append(ids.clone())
        self.states[ids] = -1.0

    def legal_bins_mask(self) -> torch.Tensor:
        return self._legal_mask.clone()


class DummyBuffer:
    """Replay buffer stub that records appended batches."""

    def __init__(self) -> None:
        self.batches: list[RebelBatch] = []

    def add_batch(self, batch: RebelBatch) -> None:
        self.batches.append(batch)

    @property
    def total_rows(self) -> int:
        return sum(len(batch) for batch in self.batches)


class DummyEvaluator:
    """Minimal evaluator exposing the surface used by RebelDataGenerator."""

    def __init__(
        self,
        env_proto: SimpleNamespace,
        search_batch_size: int,
        total_nodes: int,
        num_players: int,
        num_actions: int,
        feature_dim: int = 3,
    ):
        self.device = torch.device("cpu")
        self.search_batch_size = search_batch_size
        self.total_nodes = total_nodes
        self.num_players = num_players
        self.num_actions = num_actions
        self.feature_dim = feature_dim

        self.env = HUNLTensorEnv.from_proto(env_proto, num_envs=total_nodes)
        self.feature_matrix = torch.arange(
            total_nodes * feature_dim, dtype=torch.float32
        ).view(total_nodes, feature_dim)
        self.values = torch.stack(
            [
                torch.full(
                    (num_players, NUM_HANDS),
                    float(i),
                    dtype=torch.float32,
                )
                for i in range(total_nodes)
            ],
            dim=0,
        )
        self.policy_probs_avg = torch.stack(
            [
                torch.full(
                    (NUM_HANDS, num_actions),
                    float(i + 1),
                    dtype=torch.float32,
                )
                for i in range(total_nodes)
            ],
            dim=0,
        )

        self.initialize_args: list[torch.Tensor] = []
        self.last_initialized_env: DummyEnv | None = None
        self.last_beliefs: torch.Tensor | None = None
        self.future_pbs: list[PublicBeliefState] = []
        self.self_play_calls = 0
        self.sample_calls = 0

    def initialize_search(
        self,
        env: DummyEnv,
        indices: torch.Tensor,
        initial_beliefs: torch.Tensor,
    ) -> None:
        self.last_initialized_env = env
        self.initialize_args.append(indices.clone())
        self.last_beliefs = initial_beliefs.clone()

    def self_play_iteration(self) -> PublicBeliefState:
        self.self_play_calls += 1
        if self.future_pbs:
            return self.future_pbs.pop(0)
        env = DummyEnv(self.search_batch_size, self.num_actions, base_state=100)
        beliefs = torch.full(
            (self.search_batch_size, self.num_players, NUM_HANDS),
            1.0 / NUM_HANDS,
            dtype=torch.float32,
        )
        return PublicBeliefState(env=env, beliefs=beliefs)

    def encode_current_states(self, indices: torch.Tensor) -> torch.Tensor:
        return self.feature_matrix[indices]

    def sample_data(self) -> RebelBatch:
        self.sample_calls += 1
        count = self.search_batch_size
        indices = torch.arange(count, dtype=torch.long)
        acting_players = torch.arange(count, dtype=torch.long) % self.num_players
        return RebelBatch(
            features=self.encode_current_states(indices),
            policy_targets=self.policy_probs_avg[:count],
            value_targets=self.values[:count],
            legal_masks=self.env.legal_bins_mask()[indices],
            acting_players=acting_players,
        )


def fake_from_proto(proto: SimpleNamespace, num_envs: int | None = None) -> DummyEnv:
    count = num_envs or proto.num_envs
    return DummyEnv(count, proto.num_actions)


@pytest.fixture
def env_proto() -> SimpleNamespace:
    return SimpleNamespace(num_envs=2, num_actions=5)


def test_rebel_data_generator_collects_training_data(monkeypatch, env_proto):
    monkeypatch.setattr(HUNLTensorEnv, "from_proto", fake_from_proto)
    evaluator = DummyEvaluator(
        env_proto=env_proto,
        search_batch_size=2,
        total_nodes=4,
        num_players=2,
        num_actions=env_proto.num_actions,
    )
    future_env = DummyEnv(2, env_proto.num_actions, base_state=50)
    future_beliefs = torch.full(
        (2, evaluator.num_players, NUM_HANDS), 0.25, dtype=torch.float32
    )
    evaluator.future_pbs.append(
        PublicBeliefState(env=future_env, beliefs=future_beliefs)
    )

    buffer = DummyBuffer()
    generator = RebelDataGenerator(
        env_proto=env_proto, evaluator=evaluator, buffer=buffer
    )
    batch = generator.generate_data()

    assert isinstance(batch, RebelBatch)
    torch.testing.assert_close(
        batch.features, evaluator.feature_matrix[: evaluator.search_batch_size]
    )
    assert torch.equal(
        batch.legal_masks,
        evaluator.env.legal_bins_mask()[: evaluator.search_batch_size],
    )
    torch.testing.assert_close(
        batch.value_targets, evaluator.values[: evaluator.search_batch_size]
    )
    torch.testing.assert_close(
        batch.policy_targets, evaluator.policy_probs_avg[: evaluator.search_batch_size]
    )
    assert evaluator.initialize_args
    torch.testing.assert_close(
        evaluator.initialize_args[-1],
        torch.arange(evaluator.search_batch_size),
    )
    assert evaluator.self_play_calls == 1
    assert evaluator.sample_calls >= 1
    assert buffer.total_rows == evaluator.search_batch_size
    assert generator.next_pbs_idx == 0
    assert len(generator.pbs_queue) == 1
    assert generator.pbs_queue[0].env is future_env
    torch.testing.assert_close(generator.pbs_queue[0].beliefs, future_beliefs)


def test_rebel_data_generator_does_not_reset_when_queue_suffices(
    monkeypatch, env_proto
):
    monkeypatch.setattr(HUNLTensorEnv, "from_proto", fake_from_proto)
    evaluator = DummyEvaluator(
        env_proto=env_proto,
        search_batch_size=2,
        total_nodes=3,
        num_players=2,
        num_actions=env_proto.num_actions,
    )
    buffer = DummyBuffer()
    generator = RebelDataGenerator(
        env_proto=env_proto, evaluator=evaluator, buffer=buffer
    )
    generator.next_pbs_idx = 0

    seeded_env = DummyEnv(2, env_proto.num_actions, base_state=10)
    seeded_beliefs = torch.full(
        (2, evaluator.num_players, NUM_HANDS), 0.3, dtype=torch.float32
    )
    generator.pbs_queue = [PublicBeliefState(env=seeded_env, beliefs=seeded_beliefs)]

    next_env = DummyEnv(2, env_proto.num_actions, base_state=20)
    next_beliefs = torch.full(
        (2, evaluator.num_players, NUM_HANDS), 0.4, dtype=torch.float32
    )
    evaluator.future_pbs = [PublicBeliefState(env=next_env, beliefs=next_beliefs)]

    batch = generator.generate_data()

    assert isinstance(batch, RebelBatch)
    env = evaluator.last_initialized_env
    assert env is not None
    assert env.reset_history == []
    torch.testing.assert_close(env.states, seeded_env.states)
    assert evaluator.self_play_calls == 1
    assert evaluator.sample_calls >= 1
    assert buffer.total_rows == evaluator.search_batch_size
    assert generator.next_pbs_idx == 0
    assert len(generator.pbs_queue) == 1
    assert generator.pbs_queue[0].env is next_env
    torch.testing.assert_close(generator.pbs_queue[0].beliefs, next_beliefs)


def test_rebel_data_generator_reuses_appended_pbs(monkeypatch, env_proto):
    monkeypatch.setattr(HUNLTensorEnv, "from_proto", fake_from_proto)
    evaluator = DummyEvaluator(
        env_proto=env_proto,
        search_batch_size=2,
        total_nodes=4,
        num_players=2,
        num_actions=env_proto.num_actions,
    )
    buffer = DummyBuffer()
    generator = RebelDataGenerator(
        env_proto=env_proto, evaluator=evaluator, buffer=buffer
    )

    generator.generate_data()
    assert len(generator.pbs_queue) == 1
    queued_pbs = generator.pbs_queue[0]
    queued_env = queued_pbs.env
    queued_beliefs = queued_pbs.beliefs.clone()
    buffer_total_before = buffer.total_rows

    evaluator.future_pbs = []
    evaluator.self_play_calls = 0
    evaluator.sample_calls = 0

    generator.generate_data()

    assert evaluator.last_initialized_env is queued_env
    torch.testing.assert_close(evaluator.last_beliefs, queued_beliefs)
    assert evaluator.self_play_calls >= 1
    assert buffer.total_rows == buffer_total_before + evaluator.search_batch_size
    assert generator.next_pbs_idx == 0


def test_rebel_data_generator_pads_smaller_pbs(monkeypatch, env_proto):
    monkeypatch.setattr(HUNLTensorEnv, "from_proto", fake_from_proto)
    evaluator = DummyEvaluator(
        env_proto=env_proto,
        search_batch_size=2,
        total_nodes=3,
        num_players=2,
        num_actions=env_proto.num_actions,
    )
    buffer = DummyBuffer()
    generator = RebelDataGenerator(
        env_proto=env_proto, evaluator=evaluator, buffer=buffer
    )

    small_env = DummyEnv(1, env_proto.num_actions, base_state=30)
    small_beliefs = torch.full((1, evaluator.num_players, NUM_HANDS), 0.6)
    generator.pbs_queue = [PublicBeliefState(env=small_env, beliefs=small_beliefs)]

    batch = generator.generate_data()
    assert isinstance(batch, RebelBatch)
    assert batch.features.shape[0] == evaluator.search_batch_size
    assert generator.pbs_queue[0].env.N == evaluator.search_batch_size
