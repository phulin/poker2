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
from p2.utils.rng import generator_state_for_set_state


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
    """Live CFR data source that wraps the existing ReBeL data generator."""

    def __init__(
        self,
        generator: RebelDataGenerator,
        value_buffer: RebelReplayBuffer,
        policy_buffer: RebelReplayBuffer,
        *,
        value_sample_count: int,
        max_return_policy_samples: int,
    ) -> None:
        self.generator = generator
        self.value_buffer = value_buffer
        self.policy_buffer = policy_buffer
        self.value_sample_count = int(value_sample_count)
        self.max_return_policy_samples = int(max_return_policy_samples)

    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        del step
        return self.generator.generate_data(
            self.value_sample_count,
            return_policy_batch=True,
            max_return_policy_samples=self.max_return_policy_samples,
        )

    def prepare_policy_step(self, step: int) -> RebelBatch | None:
        del step
        old_store_replay = self.generator.store_replay
        self.generator.store_replay = False
        try:
            _, policy_batch = self.generator.generate_data(
                self.value_sample_count,
                return_value_batch=False,
                return_policy_batch=True,
                max_return_policy_samples=self.max_return_policy_samples,
            )
        finally:
            self.generator.store_replay = old_store_replay
        if policy_batch is not None:
            self.policy_buffer.add_batch(policy_batch)
        return policy_batch

    def ensure_min_policy_samples(self, policy_samples: int) -> None:
        while len(self.policy_buffer) < policy_samples:
            self.prepare_policy_step(0)

    def ensure_min_samples(self, value_samples: int, policy_samples: int) -> None:
        while (
            len(self.value_buffer) < value_samples
            or len(self.policy_buffer) < policy_samples
        ):
            self.generator.generate_data(
                self.value_sample_count,
                return_value_batch=False,
                return_policy_batch=False,
            )

    def sample_value(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        return self.value_buffer.sample(batch_size, stratify_streets=stratify_streets)

    def sample_policy(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        return self.policy_buffer.sample(batch_size, stratify_streets=stratify_streets)

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
    value_order: torch.Tensor | None = None
    policy_order: torch.Tensor | None = None

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
        ) and self.policy_sample_count > 0:
            raise ValueError("pregenerated data has no positive-weight policy examples")

    def stream_enabled(self, stream: str) -> bool:
        sample_count = (
            self.value_sample_count if stream == "value" else self.policy_sample_count
        )
        if sample_count <= 0:
            return False
        for state in self.datasets:
            weight = state.value_weight if stream == "value" else state.policy_weight
            if state.dataset.stream_len(stream) > 0 and weight > 0.0:
                return True
        return False

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

    def _stream_cursor(self, state: _PregeneratedDatasetState, stream: str) -> int:
        return int(state.value_cursor if stream == "value" else state.policy_cursor)

    def _set_stream_cursor(
        self,
        state: _PregeneratedDatasetState,
        stream: str,
        cursor: int,
    ) -> None:
        if stream == "value":
            state.value_cursor = int(cursor)
        else:
            state.policy_cursor = int(cursor)

    def _stream_order(
        self,
        state: _PregeneratedDatasetState,
        stream: str,
    ) -> torch.Tensor | None:
        if not self.shuffle:
            return None
        attr = "value_order" if stream == "value" else "policy_order"
        order = getattr(state, attr)
        if order is None:
            order = torch.randperm(
                state.dataset.stream_len(stream),
                generator=self.generator,
            )
            setattr(state, attr, order)
        return order

    def remaining(self, stream: str) -> int:
        total = 0
        for state in self.datasets:
            weight = state.value_weight if stream == "value" else state.policy_weight
            if (
                weight <= 0.0
                or not state.active_at(self.current_step)
                or state.dataset.stream_len(stream) <= 0
            ):
                continue
            total += max(0, state.dataset.stream_len(stream) - self._stream_cursor(state, stream))
        return int(total)

    def _choose_finite_dataset(self, stream: str) -> int | None:
        weights = []
        for state in self.datasets:
            weight = state.value_weight if stream == "value" else state.policy_weight
            remaining = state.dataset.stream_len(stream) - self._stream_cursor(
                state, stream
            )
            available = (
                state.active_at(self.current_step)
                and state.dataset.stream_len(stream) > 0
                and remaining > 0
            )
            weights.append(max(0.0, weight) if available else 0.0)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        if weights_tensor.sum() <= 0:
            return None
        return int(torch.multinomial(weights_tensor, 1, generator=self.generator).item())

    def _next_finite_batch(self, stream: str, max_count: int) -> RebelBatch | None:
        if max_count <= 0:
            return None
        batches: list[RebelBatch] = []
        remaining = int(max_count)
        while remaining > 0:
            index = self._choose_finite_dataset(stream)
            if index is None:
                break
            state = self.datasets[index]
            cursor = self._stream_cursor(state, stream)
            available = state.dataset.stream_len(stream) - cursor
            take = min(remaining, available)
            order = self._stream_order(state, stream)
            if order is None:
                batch = state.dataset.get_batch(stream, cursor, take, wrap=False)
            else:
                batch = state.dataset.get_indexed_batch(
                    stream,
                    order[cursor : cursor + take],
                )
            batches.append(batch)
            self._set_stream_cursor(state, stream, cursor + take)
            remaining -= take
        if not batches:
            return None
        return batches[0] if len(batches) == 1 else RebelBatch.cat(batches)

    def prepare_finite_step(
        self,
        step: int,
        *,
        include_value: bool = True,
        include_policy: bool = True,
    ) -> tuple[RebelBatch | None, RebelBatch | None]:
        self.current_step = int(step)
        value_batch = (
            self._next_finite_batch("value", self.value_sample_count)
            if include_value and self.value_sample_count > 0
            else None
        )
        policy_batch = (
            self._next_finite_batch("policy", self.policy_sample_count)
            if include_policy and self.policy_sample_count > 0
            else None
        )
        if value_batch is not None:
            self.value_buffer.add_batch(value_batch)
        if policy_batch is not None:
            self.policy_buffer.add_batch(policy_batch)
        return value_batch, policy_batch

    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        self.current_step = int(step)
        value_batch = self._next_batch("value", self.value_sample_count)
        policy_batch = (
            self._next_batch("policy", self.policy_sample_count)
            if self.policy_sample_count > 0
            else None
        )
        self.value_buffer.add_batch(value_batch)
        if policy_batch is not None:
            self.policy_buffer.add_batch(policy_batch)
        return value_batch, policy_batch

    def ensure_min_samples(self, value_samples: int, policy_samples: int) -> None:
        while (
            len(self.value_buffer) < value_samples
            or (
                self.policy_sample_count > 0
                and len(self.policy_buffer) < policy_samples
            )
        ):
            value_batch = self._next_batch("value", self.value_sample_count)
            self.value_buffer.add_batch(value_batch)
            if self.policy_sample_count > 0:
                policy_batch = self._next_batch("policy", self.policy_sample_count)
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
        if self.policy_sample_count <= 0:
            raise RuntimeError("pregenerated policy stream is disabled")
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
                    "value_order": state.value_order,
                    "policy_order": state.policy_order,
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
        if dataset_states is not None:
            for dataset_state, checkpoint_state in zip(
                self.datasets, dataset_states, strict=True
            ):
                dataset_state.value_order = checkpoint_state.get("value_order")
                dataset_state.policy_order = checkpoint_state.get("policy_order")
        for dataset_state, cursor in zip(
            self.datasets, state.get("value_cursors", []), strict=False
        ):
            dataset_state.value_cursor = int(cursor)
        for dataset_state, cursor in zip(
            self.datasets, state.get("policy_cursors", []), strict=False
        ):
            dataset_state.policy_cursor = int(cursor)
        if "generator_state" in state:
            self.generator.set_state(
                generator_state_for_set_state(state["generator_state"])
            )


class BootstrapPregeneratedRebelDataSource(RebelDataSource):
    """Bootstrap replay buffers from finite pregenerated data, then use live CFR."""

    def __init__(
        self,
        pregenerated_source: PregeneratedRebelDataSource,
        live_source: LiveRebelDataSource,
    ) -> None:
        self.pregenerated_source = pregenerated_source
        self.live_source = live_source
        self.live_active = False
        self._resident_value: RebelBatch | None = None
        self._resident_value_order: torch.Tensor | None = None
        self._resident_value_cursor = 0

    def _bootstrap_value_active(self) -> bool:
        if self._resident_value is not None:
            return self._resident_value_cursor < len(self._resident_value)
        return (
            self.pregenerated_source.stream_enabled("value")
            and self.pregenerated_source.remaining("value") > 0
        )

    def _bootstrap_policy_active(self) -> bool:
        return (
            self.pregenerated_source.stream_enabled("policy")
            and self.pregenerated_source.remaining("policy") > 0
        )

    def value_bootstrap_active(self) -> bool:
        return not self.live_active and self._bootstrap_value_active()

    def _ensure_resident_value_loaded(self) -> None:
        if self._resident_value is not None:
            return
        batches: list[RebelBatch] = []
        device = self.live_source.value_buffer.device
        for state in self.pregenerated_source.datasets:
            if state.value_weight <= 0.0 or state.dataset.stream_len("value") <= 0:
                continue
            batches.append(
                state.dataset.get_batch(
                    "value",
                    0,
                    state.dataset.stream_len("value"),
                    device=device,
                    float_dtype=torch.float32,
                    wrap=False,
                )
            )
        if not batches:
            raise RuntimeError("bootstrap pregenerated value stream is empty")
        self._resident_value = batches[0] if len(batches) == 1 else RebelBatch.cat(batches)
        if self.pregenerated_source.shuffle:
            if self._resident_value_order is None:
                order = torch.randperm(
                    len(self._resident_value),
                    generator=self.pregenerated_source.generator,
                )
                self._resident_value_order = order.to(device, non_blocking=True)
            else:
                self._resident_value_order = self._resident_value_order.to(
                    device,
                    non_blocking=True,
                )
        else:
            self._resident_value_order = torch.arange(
                len(self._resident_value),
                device=device,
            )
        self._resident_value_cursor = 0

    def _resident_value_indices(self, start: int, end: int) -> torch.Tensor:
        self._ensure_resident_value_loaded()
        assert self._resident_value_order is not None
        return self._resident_value_order[start:end]

    def bootstrap_value_available(self) -> int:
        self._ensure_resident_value_loaded()
        return int(self._resident_value_cursor)

    def bootstrap_value_capacity(self) -> int:
        return int(self.live_source.value_buffer.capacity)

    def bootstrap_value_remaining(self) -> int:
        self._ensure_resident_value_loaded()
        assert self._resident_value is not None
        return max(0, len(self._resident_value) - self._resident_value_cursor)

    def prepare_value_bootstrap_step(self, step: int) -> RebelBatch | None:
        del step
        if not self.value_bootstrap_active():
            return None
        self._ensure_resident_value_loaded()
        assert self._resident_value is not None
        start = self._resident_value_cursor
        end = min(
            len(self._resident_value),
            start + int(self.pregenerated_source.value_sample_count),
        )
        self._resident_value_cursor = end
        return self._resident_value[self._resident_value_indices(start, end)]

    def ensure_min_value_samples(self, value_samples: int) -> None:
        if not self.value_bootstrap_active():
            return
        self._ensure_resident_value_loaded()
        assert self._resident_value is not None
        self._resident_value_cursor = min(
            len(self._resident_value),
            max(self._resident_value_cursor, int(value_samples)),
        )

    def sample_value_bootstrap(
        self,
        batch_size: int,
        stratify_streets: list[float] | None = None,
    ) -> RebelBatch:
        if stratify_streets is not None:
            raise NotImplementedError(
                "bootstrap pregenerated direct sampling does not support "
                "street stratification"
            )
        self._ensure_resident_value_loaded()
        assert self._resident_value is not None
        if self._resident_value_cursor <= 0:
            raise ValueError("bootstrap pregenerated value prefix is empty")
        sample_count = min(int(self._resident_value_cursor), len(self._resident_value))
        if batch_size > sample_count:
            raise ValueError(
                "not enough bootstrap pregenerated value examples: "
                f"{sample_count} < {batch_size}"
            )
        device = self.live_source.value_buffer.device
        idxs = torch.randint(
            0,
            sample_count,
            (batch_size,),
            generator=self.live_source.value_buffer.generator,
            device=device,
        )
        idxs = self._resident_value_indices(0, sample_count).index_select(0, idxs)
        return self._resident_value[idxs]

    def prepare_step(self, step: int) -> tuple[RebelBatch | None, RebelBatch | None]:
        if not self.live_active and (
            self._bootstrap_value_active() or self._bootstrap_policy_active()
        ):
            value_batch, policy_batch = self.pregenerated_source.prepare_finite_step(
                step,
                include_value=self._bootstrap_value_active(),
                include_policy=self._bootstrap_policy_active(),
            )
            return value_batch, policy_batch
        self.live_active = True
        return self.live_source.prepare_step(step)

    def ensure_min_samples(self, value_samples: int, policy_samples: int) -> None:
        while not self.live_active and self._bootstrap_value_active():
            if len(self.pregenerated_source.value_buffer) >= value_samples:
                break
            self.pregenerated_source.prepare_finite_step(
                self.pregenerated_source.current_step,
                include_value=True,
                include_policy=False,
            )
        if len(self.live_source.policy_buffer) < policy_samples:
            self.live_source.ensure_min_policy_samples(policy_samples)
        if len(self.pregenerated_source.value_buffer) < value_samples:
            self.live_active = True
            self.live_source.ensure_min_samples(value_samples, policy_samples)

    def sample_value(
        self, batch_size: int, stratify_streets: list[float] | None
    ) -> RebelBatch:
        if (
            not self.live_active
            and self._resident_value is not None
            and self._resident_value_cursor > 0
        ):
            return self.sample_value_bootstrap(
                batch_size,
                stratify_streets=stratify_streets,
            )
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
            "live_active": self.live_active,
            "resident_value_cursor": self._resident_value_cursor,
            "resident_value_order": (
                self._resident_value_order.detach().cpu()
                if self._resident_value_order is not None
                else None
            ),
            "pregenerated": self.pregenerated_source.state_dict(),
            "live": self.live_source.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.live_active = bool(state.get("live_active", self.live_active))
        self._resident_value_cursor = int(
            state.get("resident_value_cursor", self._resident_value_cursor)
        )
        resident_value_order = state.get("resident_value_order")
        if resident_value_order is not None:
            self._resident_value_order = resident_value_order.to(torch.long)
        if "pregenerated" in state:
            self.pregenerated_source.load_state_dict(state["pregenerated"])
        if "live" in state:
            self.live_source.load_state_dict(state["live"])
