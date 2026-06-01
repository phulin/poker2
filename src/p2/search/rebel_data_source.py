from __future__ import annotations

from abc import ABC, abstractmethod

from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import RebelReplayBuffer
from p2.search.rebel_data_generator import RebelDataGenerator


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
