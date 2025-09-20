from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


class Env(ABC):
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:  # Observation structure
        ...

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]: ...

    @abstractmethod
    def legal_actions(self) -> Any: ...


class Encoder(ABC):
    @abstractmethod
    def encode_cards(self, game_state: Any, seat: int) -> Any: ...

    @abstractmethod
    def encode_actions(self, game_state: Any, seat: int, num_bet_bins: int) -> Any: ...


class Model(ABC):
    @abstractmethod
    def forward(self, cards_tensor: Any, actions_tensor: Any) -> Tuple[Any, Any]:
        """Returns (policy_logits, value)."""
        ...


class Policy(ABC):
    @abstractmethod
    def action(
        self, logits: Any, legal_mask: Optional[Any] = None
    ) -> Tuple[int, float]:
        """Select an action id and return (action_id, log_prob)."""
        ...


class League(ABC):
    @abstractmethod
    def sample_lineup(self, num_seats: int) -> Tuple[Any, ...]: ...

    @abstractmethod
    def update_meta_mix(self, scores: Mapping[Tuple[int, int], float]) -> None: ...

    @abstractmethod
    def maybe_add_snapshot(self, agent: Any) -> None: ...

    @abstractmethod
    def spawn_best_response(self, base_agent: Any, steps: int) -> None: ...


@dataclass
class BuildContext:
    config: Dict[str, Any]
    registry: Dict[str, Any]
