"""Utilities for information-set encoding in vectorized CFR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class InformationSetKey:
    observation_hash: torch.Tensor
    player: torch.Tensor


class InformationSetEncoder(Protocol):
    def encode(
        self, observations: torch.Tensor, player: torch.Tensor
    ) -> InformationSetKey:  # pragma: no cover - protocol
        ...


class TensorHasher:
    """Default encoder that derives stable hashes from batched tensors."""

    def __init__(self, dtype: torch.dtype = torch.int64) -> None:
        self.dtype = dtype

    def encode(
        self, observations: torch.Tensor, player: torch.Tensor
    ) -> InformationSetKey:
        flat = observations.reshape(observations.shape[0], -1).to(torch.float64)
        weights = torch.linspace(
            1, flat.shape[1], flat.shape[1], device=flat.device, dtype=flat.dtype
        )
        hashed = torch.matmul(flat, weights)
        hashed = (hashed * 1_000_000).to(self.dtype)
        return InformationSetKey(observation_hash=hashed, player=player.to(self.dtype))


__all__ = ["InformationSetKey", "InformationSetEncoder", "TensorHasher"]
