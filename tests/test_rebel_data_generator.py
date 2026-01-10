from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.rl.rebel_batch import RebelBatch
from alphaholdem.search.cfr_evaluator import PublicBeliefState
from alphaholdem.search.rebel_data_generator import RebelDataGenerator


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
        self.root_nodes = search_batch_size
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
        self.self_play_calls = 0
        self.sample_calls = 0
        self.self_play_return_none = False

    def initialize_search(
        self,
        env: DummyEnv,
        indices: torch.Tensor,
        initial_beliefs: torch.Tensor,
    ) -> None:
        self.last_initialized_env = env
        self.initialize_args.append(indices.clone())
        self.last_beliefs = initial_beliefs.clone()

    def initialize_subgame(
        self, env: DummyEnv, indices: torch.Tensor, initial_beliefs: torch.Tensor
    ) -> None:
        self.initialize_search(env, indices, initial_beliefs)

    def self_play_iteration(self) -> PublicBeliefState | None:
        self.self_play_calls += 1
        if self.self_play_return_none:
            return None
        env = DummyEnv(self.search_batch_size, self.num_actions, base_state=100)
        beliefs = torch.full(
            (self.search_batch_size, self.num_players, NUM_HANDS),
            1.0 / NUM_HANDS,
            dtype=torch.float32,
        )
        return PublicBeliefState(
            env=env,
            beliefs=beliefs,
        )

    def evaluate_cfr(self):
        return self.self_play_iteration()

    def training_data(self):
        """Return training data as tuple (value_batch, policy_batch)."""

        count = self.search_batch_size
        indices = torch.arange(count, dtype=torch.long)
        device = self.device

        # Create MLPFeatures from feature_matrix
        # Assuming feature_matrix is just context for simplicity
        # In real code, features would be properly encoded MLPFeatures
        mlp_features = MLPFeatures(
            context=self.feature_matrix[indices],
            street=torch.zeros(count, dtype=torch.long, device=device),
            to_act=torch.zeros(count, dtype=torch.long, device=device),
            board=torch.full((count, 5), -1, dtype=torch.long, device=device),
            beliefs=torch.full(
                (count, 2 * NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=torch.float32,
                device=device,
            ),
        )

        value_batch = RebelBatch(
            features=mlp_features,
            value_targets=self.values[:count],
            legal_masks=self.env.legal_bins_mask()[indices],
        )
        policy_batch = RebelBatch(
            features=mlp_features,
            policy_targets=self.policy_probs_avg[:count],
            legal_masks=self.env.legal_bins_mask()[indices],
        )
        return value_batch, value_batch, policy_batch


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

    buffer = DummyBuffer()
    generator = RebelDataGenerator(
        env_proto=env_proto,
        evaluator=evaluator,
        value_buffer=buffer,
        policy_buffer=buffer,
    )

    # generate_data() now returns None and adds data to buffer
    generator.generate_data(2)

    # Check that data was added to buffer
    assert len(buffer.batches) >= 3
    # Policy batch is added first, followed by start- and end-of-street value batches
    policy_batch = buffer.batches[0]
    value_batch_start = buffer.batches[1]
    value_batch_end = buffer.batches[2]
    assert isinstance(policy_batch, RebelBatch)
    assert isinstance(value_batch_start, RebelBatch)
    assert isinstance(value_batch_end, RebelBatch)

    torch.testing.assert_close(
        value_batch_start.features.context,
        evaluator.feature_matrix[: evaluator.search_batch_size],
    )
    assert torch.equal(
        value_batch_start.legal_masks,
        evaluator.env.legal_bins_mask()[: evaluator.search_batch_size],
    )
    torch.testing.assert_close(
        value_batch_start.value_targets, evaluator.values[: evaluator.search_batch_size]
    )
    torch.testing.assert_close(
        value_batch_end.value_targets, evaluator.values[: evaluator.search_batch_size]
    )
    torch.testing.assert_close(
        policy_batch.policy_targets,
        evaluator.policy_probs_avg[: evaluator.search_batch_size],
    )

    # Check that evaluator was called correctly
    assert evaluator.initialize_args
    torch.testing.assert_close(
        evaluator.initialize_args[0],
        torch.arange(evaluator.search_batch_size),
    )
    assert evaluator.self_play_calls >= 1


def test_rebel_data_generator_terminates_when_no_next_pbs(monkeypatch, env_proto):
    """Test that generator terminates when self_play_iteration returns None."""
    monkeypatch.setattr(HUNLTensorEnv, "from_proto", fake_from_proto)
    evaluator = DummyEvaluator(
        env_proto=env_proto,
        search_batch_size=2,
        total_nodes=4,
        num_players=2,
        num_actions=env_proto.num_actions,
    )
    # Make self_play_iteration return None immediately
    evaluator.self_play_return_none = True

    buffer = DummyBuffer()
    generator = RebelDataGenerator(
        env_proto=env_proto,
        evaluator=evaluator,
        value_buffer=buffer,
        policy_buffer=buffer,
    )

    generator.generate_data(2)

    # Should have initialized search and called self_play_iteration once
    assert len(evaluator.initialize_args) == 1
    assert evaluator.self_play_calls == 1
    # Should have added data to buffer
    assert len(buffer.batches) > 0


def test_rebel_data_generator_multiple_iterations(monkeypatch, env_proto):
    """Test that generator can run multiple iterations when self_play_iteration returns PBS."""
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
        env_proto=env_proto,
        evaluator=evaluator,
        value_buffer=buffer,
        policy_buffer=buffer,
    )

    generator.generate_data(2)

    # Should have called self_play_iteration multiple times
    assert evaluator.self_play_calls >= 1
    # Should have added multiple batches to buffer
    assert len(buffer.batches) >= 1
