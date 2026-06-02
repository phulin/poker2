from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from p2.core.structured_config import PregeneratedDatasetConfig
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import RebelReplayBuffer
from p2.search.rebel_data_generator import RebelDataGenerator
from p2.search.rebel_solved_dataset import RebelSolvedDataset


class RebelDataSource(ABC):
    """Training data boundary for live and pregenerated ReBeL examples."""

    @abstractmethod
    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        """Prepare fresh data for one trainer step and return metrics batches."""

    @abstractmethod
    def ensure_min_samples(self, value_samples: int, policy_samples: int) -> None:
        """Fill backing storage until both sample counts are available."""

    @abstractmethod
    def sample_value(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        """Sample a value batch."""

    @abstractmethod
    def sample_policy(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        """Sample a policy batch."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable data-source cursor/generator state."""

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore serializable data-source cursor/generator state."""


class LiveRebelDataSource(RebelDataSource):
    """Live CFR data source that streams the current generated batches."""

    def __init__(
        self,
        generator: RebelDataGenerator,
        *,
        value_sample_count: int,
        max_return_policy_samples: int,
    ) -> None:
        self.generator = generator
        self.value_sample_count = int(value_sample_count)
        self.max_return_policy_samples = int(max_return_policy_samples)
        self._value_batch: RebelBatch | None = None
        self._policy_batch: RebelBatch | None = None

    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        del step
        self._value_batch, self._policy_batch = self.generator.generate_data(
            self.value_sample_count,
            return_policy_batch=True,
            max_return_policy_samples=self.max_return_policy_samples,
        )
        return self._value_batch, self._policy_batch

    def ensure_min_samples(self, value_samples: int, policy_samples: int) -> None:
        if self._value_batch is None or len(self._value_batch) < value_samples:
            self._value_batch, self._policy_batch = self.generator.generate_data(
                max(self.value_sample_count, value_samples),
                return_policy_batch=True,
                max_return_policy_samples=max(
                    self.max_return_policy_samples, policy_samples
                ),
            )
        if self._policy_batch is None or len(self._policy_batch) < policy_samples:
            _, self._policy_batch = self.generator.generate_data(
                max(self.value_sample_count, value_samples),
                return_value_batch=False,
                return_policy_batch=True,
                max_return_policy_samples=max(
                    self.max_return_policy_samples, policy_samples
                ),
            )

    def sample_value(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        del stratify_streets
        if self._value_batch is None or len(self._value_batch) < batch_size:
            self.ensure_min_samples(batch_size, 0)
        if self._value_batch is None:
            raise RuntimeError("Live data source has no generated value batch")
        return self._value_batch[:batch_size]

    def sample_policy(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        del stratify_streets
        if self._policy_batch is None or len(self._policy_batch) < batch_size:
            self.ensure_min_samples(0, batch_size)
        if self._policy_batch is None:
            raise RuntimeError("Live data source has no generated policy batch")
        return self._policy_batch[:batch_size]

    def state_dict(self) -> dict:
        return self.generator.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.generator.load_state_dict(state)


class HybridRebelDataSource(RebelDataSource):
    """Live training source with pregenerated holdout batches for metrics."""

    def __init__(
        self,
        live_source: RebelDataSource,
        holdout_source: RebelDataSource,
    ) -> None:
        self.live_source = live_source
        self.holdout_source = holdout_source

    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        self.live_source.prepare_step(step)
        return self.holdout_source.prepare_step(step)

    def ensure_min_samples(self, value_samples: int, policy_samples: int) -> None:
        self.live_source.ensure_min_samples(value_samples, policy_samples)

    def sample_value(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        return self.live_source.sample_value(
            batch_size, stratify_streets=stratify_streets
        )

    def sample_policy(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        return self.live_source.sample_policy(
            batch_size, stratify_streets=stratify_streets
        )

    def state_dict(self) -> dict:
        return {
            "live": self.live_source.state_dict(),
            "holdout": self.holdout_source.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if "live" in state:
            self.live_source.load_state_dict(state["live"])
        if "holdout" in state:
            self.holdout_source.load_state_dict(state["holdout"])


@dataclass
class _PregeneratedDatasetState:
    dataset: RebelSolvedDataset
    value_weight: float
    policy_weight: float
    min_step: int = 0
    max_step: int | None = None
    value_cursor: int = 0
    policy_cursor: int = 0

    def active_at(self, step: int) -> bool:
        return step >= self.min_step and (
            self.max_step is None or step < self.max_step
        )


class PregeneratedRebelDataSource(RebelDataSource):
    """Bounded solved-example source staged through existing replay buffers."""

    def __init__(
        self,
        dataset_configs: list[PregeneratedDatasetConfig],
        value_buffer: RebelReplayBuffer,
        policy_buffer: RebelReplayBuffer,
        *,
        value_sample_count: int,
        policy_sample_count: int,
        num_players: int,
        num_actions: int,
        context_length: int,
        model_name: str | None = None,
        action_schedule: dict[str, Any] | None = None,
        street_support: list[int] | None = None,
        generator: torch.Generator | None = None,
        shuffle: bool = True,
        direct_sample: bool = False,
        pin_memory: bool = False,
        async_shard_prefetch: bool = False,
    ) -> None:
        if not dataset_configs:
            raise ValueError("data.pregenerated.datasets must list at least one dataset")

        self.value_buffer = value_buffer
        self.policy_buffer = policy_buffer
        self.value_sample_count = int(value_sample_count)
        self.policy_sample_count = int(policy_sample_count)
        self.generator = generator or torch.Generator(device="cpu")
        self.shuffle = bool(shuffle)
        self.direct_sample = bool(direct_sample)
        self.current_step = 0
        self.datasets = [
            _PregeneratedDatasetState(
                dataset=RebelSolvedDataset(
                    dataset_cfg.path,
                    num_players=num_players,
                    num_actions=num_actions,
                    context_length=context_length,
                    model_name=model_name,
                    action_schedule=action_schedule,
                    street_support=street_support,
                    pin_memory=pin_memory,
                    async_shard_prefetch=async_shard_prefetch,
                ),
                value_weight=float(dataset_cfg.value_weight),
                policy_weight=float(dataset_cfg.policy_weight),
                min_step=int(dataset_cfg.min_step),
                max_step=(
                    int(dataset_cfg.max_step)
                    if dataset_cfg.max_step is not None
                    else None
                ),
            )
            for dataset_cfg in dataset_configs
        ]

        if not any(
            state.dataset.stream_len("value") > 0 and state.value_weight > 0.0
            for state in self.datasets
        ):
            raise ValueError("pregenerated data has no positive-weight value examples")
        if not any(
            state.dataset.stream_len("policy") > 0 and state.policy_weight > 0.0
            for state in self.datasets
        ):
            raise ValueError("pregenerated data has no positive-weight policy examples")

    def _choose_dataset(self, stream: str) -> int:
        weights = []
        for state in self.datasets:
            weight = state.value_weight if stream == "value" else state.policy_weight
            available = (
                state.active_at(self.current_step)
                and state.dataset.stream_len(stream) > 0
            )
            weights.append(max(0.0, weight) if available else 0.0)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if weights_tensor.sum() <= 0:
            raise ValueError(
                f"pregenerated {stream} stream has no active examples at "
                f"step {self.current_step}"
            )
        return int(torch.multinomial(weights_tensor, 1, generator=self.generator).item())

    def _next_batch(self, stream: str, count: int) -> RebelBatch:
        index = self._choose_dataset(stream)
        state = self.datasets[index]
        if self.shuffle:
            return state.dataset.sample_batch(
                stream,
                count,
                generator=self.generator,
            )
        cursor_attr = "value_cursor" if stream == "value" else "policy_cursor"
        cursor = int(getattr(state, cursor_attr))
        batch = state.dataset.get_batch(stream, cursor, count, wrap=True)
        total = state.dataset.stream_len(stream)
        setattr(state, cursor_attr, (cursor + count) % total)
        return batch

    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        self.current_step = int(step)
        value_batch = self._next_batch("value", self.value_sample_count)
        policy_batch = self._next_batch("policy", self.policy_sample_count)
        self.value_buffer.add_batch(value_batch)
        self.policy_buffer.add_batch(policy_batch)
        return value_batch, policy_batch

    def ensure_min_samples(self, value_samples: int, policy_samples: int) -> None:
        while (
            len(self.value_buffer) < value_samples
            or len(self.policy_buffer) < policy_samples
        ):
            value_batch = self._next_batch("value", self.value_sample_count)
            policy_batch = self._next_batch("policy", self.policy_sample_count)
            self.value_buffer.add_batch(value_batch)
            self.policy_buffer.add_batch(policy_batch)

    def sample_value(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        if self.direct_sample and stratify_streets is None:
            return self._next_batch("value", batch_size)
        return self.value_buffer.sample(batch_size, stratify_streets=stratify_streets)

    def sample_policy(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        if self.direct_sample and stratify_streets is None:
            return self._next_batch("policy", batch_size)
        return self.policy_buffer.sample(batch_size, stratify_streets=stratify_streets)

    def state_dict(self) -> dict:
        return {
            "value_cursors": [state.value_cursor for state in self.datasets],
            "policy_cursors": [state.policy_cursor for state in self.datasets],
            "datasets": [
                {
                    "path": str(state.dataset.path),
                    "manifest": copy.deepcopy(state.dataset.manifest),
                    "value_cursor": state.value_cursor,
                    "policy_cursor": state.policy_cursor,
                }
                for state in self.datasets
            ],
            "generator_state": self.generator.get_state(),
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict) -> None:
        dataset_states = state.get("datasets")
        if dataset_states is not None:
            if len(dataset_states) != len(self.datasets):
                raise ValueError(
                    "pregenerated checkpoint dataset count mismatch: "
                    f"expected {len(self.datasets)}, got {len(dataset_states)}"
                )
            for index, (dataset_state, checkpoint_state) in enumerate(
                zip(self.datasets, dataset_states, strict=True)
            ):
                checkpoint_manifest = checkpoint_state.get("manifest")
                if checkpoint_manifest != dataset_state.dataset.manifest:
                    raise ValueError(
                        "pregenerated checkpoint manifest mismatch for "
                        f"dataset {index}: {dataset_state.dataset.path}"
                    )
        self.current_step = int(state.get("current_step", self.current_step))
        if "value_cursors" not in state and dataset_states is not None:
            state["value_cursors"] = [
                dataset_state.get("value_cursor", 0)
                for dataset_state in dataset_states
            ]
        if "policy_cursors" not in state and dataset_states is not None:
            state["policy_cursors"] = [
                dataset_state.get("policy_cursor", 0)
                for dataset_state in dataset_states
            ]
        for dataset_state, cursor in zip(
            self.datasets, state.get("value_cursors", []), strict=False
        ):
            dataset_state.value_cursor = int(cursor)
        for dataset_state, cursor in zip(
            self.datasets, state.get("policy_cursors", []), strict=False
        ):
            dataset_state.policy_cursor = int(cursor)
        if "generator_state" in state:
            self.generator.set_state(state["generator_state"])
