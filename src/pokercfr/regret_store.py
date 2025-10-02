"""Regret storage backends supporting vectorized CFR updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .information_set import InformationSetKey

KeyType = Tuple[int, int]


@dataclass
class RegretStoreConfig:
    num_actions: int
    device: torch.device
    dtype: torch.dtype = torch.float32


class TensorRegretStore:
    """Simple dictionary-backed regret and strategy storage."""

    def __init__(self, config: RegretStoreConfig) -> None:
        self.config = config
        self.regrets: Dict[KeyType, torch.Tensor] = {}
        self.strategy_sum: Dict[KeyType, torch.Tensor] = {}

    def get_regrets(self, key: InformationSetKey) -> torch.Tensor:
        tensors = [self._regret_tensor(k) for k in self._iter_keys(key)]
        return torch.stack(tensors, dim=0)

    def update_regrets(self, key: InformationSetKey, delta: torch.Tensor) -> None:
        for per_env_delta, table_key in zip(delta, self._iter_keys(key)):
            self._regret_tensor(table_key).add_(per_env_delta)

    def get_strategy_sum(self, key: InformationSetKey) -> torch.Tensor:
        tensors = [self._strategy_tensor(k) for k in self._iter_keys(key)]
        return torch.stack(tensors, dim=0)

    def update_strategy_sum(
        self, key: InformationSetKey, strategy: torch.Tensor, weight: torch.Tensor
    ) -> None:
        for probs, w, table_key in zip(strategy, weight, self._iter_keys(key)):
            self._strategy_tensor(table_key).add_(probs * w)

    def _iter_keys(self, key: InformationSetKey):
        hashes = key.observation_hash.tolist()
        players = key.player.tolist()
        for obs, player in zip(hashes, players):
            yield (int(obs), int(player))

    def _regret_tensor(self, table_key: KeyType) -> torch.Tensor:
        if table_key not in self.regrets:
            self.regrets[table_key] = torch.zeros(
                self.config.num_actions,
                device=self.config.device,
                dtype=self.config.dtype,
            )
        return self.regrets[table_key]

    def _strategy_tensor(self, table_key: KeyType) -> torch.Tensor:
        if table_key not in self.strategy_sum:
            self.strategy_sum[table_key] = torch.zeros(
                self.config.num_actions,
                device=self.config.device,
                dtype=self.config.dtype,
            )
        return self.strategy_sum[table_key]


__all__ = ["RegretStoreConfig", "TensorRegretStore"]
