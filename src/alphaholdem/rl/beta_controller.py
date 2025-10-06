from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BetaController:
    """Controller that adapts the KL penalty coefficient β for KL-PPO."""

    target_kl: float
    beta_init: float = 1.0
    beta_min: float = 1e-4
    beta_max: float = 1e4
    increase_factor: float = 2.0
    decrease_factor: float = 1.0 / 2.0
    upper_threshold: float = 2.0
    lower_threshold: float = 0.5

    def __post_init__(self) -> None:
        self._beta = float(self.beta_init)
        self._beta = max(self.beta_min, min(self._beta, self.beta_max))
        self.last_kl: Optional[float] = None

    @property
    def beta(self) -> float:
        return self._beta

    def update(self, kl_value: float) -> None:
        """Update β based on the measured KL divergence for the step."""
        if kl_value is None:
            return

        self.last_kl = float(kl_value)

        if self.last_kl > self.upper_threshold * self.target_kl:
            self._beta = min(self._beta * self.increase_factor, self.beta_max)
        elif self.last_kl < self.lower_threshold * self.target_kl:
            self._beta = max(self._beta * self.decrease_factor, self.beta_min)

    def state_dict(self) -> dict:
        return {"beta": self._beta, "last_kl": self.last_kl}

    def load_state_dict(self, state: dict) -> None:
        if "beta" in state:
            self._beta = float(state["beta"])
        self.last_kl = state.get("last_kl")
        self._beta = max(self.beta_min, min(self._beta, self.beta_max))
