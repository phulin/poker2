from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T", bound=float)


@dataclass
class ExponentialController(Generic[T]):
    """Generic controller that adapts a coefficient exponentially based on a target value.

    This controller can be used for various adaptive parameters like KL penalty coefficients,
    learning rates, or other hyperparameters that need exponential adjustment.
    """

    target_value: T
    init_value: T
    min_value: T
    max_value: T
    increase_factor: float = 2.0
    decrease_factor: float = 1.0 / 2.0
    upper_threshold: float = 2.0
    lower_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._current_value = float(self.init_value)
        self._current_value = max(
            self.min_value, min(self._current_value, self.max_value)
        )
        self.last_measured_value: Optional[float] = None

    @property
    def current_value(self) -> float:
        """Get the current controlled value."""
        return self._current_value

    def update(self, measured_value: float) -> None:
        """Update the controlled value based on the measured value for the step."""
        if measured_value is None:
            return

        self.last_measured_value = float(measured_value)

        if self.last_measured_value > self.upper_threshold * self.target_value:
            self._current_value = min(
                self._current_value * self.increase_factor, self.max_value
            )
        elif self.last_measured_value < self.lower_threshold * self.target_value:
            self._current_value = max(
                self._current_value * self.decrease_factor, self.min_value
            )

    def state_dict(self) -> dict:
        """Return the current state for checkpointing."""
        return {
            "current_value": self._current_value,
            "last_measured_value": self.last_measured_value,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint."""
        if "current_value" in state:
            self._current_value = float(state["current_value"])
        self.last_measured_value = state.get("last_measured_value")
        self._current_value = max(
            self.min_value, min(self._current_value, self.max_value)
        )
