from typing import Optional


class EMA:
    """Lightweight Exponential Moving Average calculator."""

    def __init__(self, decay: float = 0.99, initial_value: Optional[float] = None):
        """
        Initialize EMA calculator.

        Args:
            decay: EMA decay factor (0 < decay < 1)
            initial_value: Initial value for the EMA
        """
        self.decay = decay
        self.reset(initial_value)

    def reset(self, initial_value: float = None) -> None:
        """Reset EMA to initial value."""
        self.value = initial_value if initial_value is not None else 0.0
        self.initialized = initial_value is not None

    def update(self, new_value: float) -> float:
        """
        Update EMA with a new value.

        Args:
            new_value: New value to incorporate into the EMA

        Returns:
            Updated EMA value
        """
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value

        return self.value

    def value_or_none(self) -> float | None:
        """
        Get the current EMA value if initialized, otherwise return None.

        Returns:
            Current EMA value if initialized, None otherwise
        """
        return self.value if self.initialized else None
