"""Utilities for managing per-layer transformer KV caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch


@dataclass
class LayerKVCache:
    """Simple container for a layer's key/value tensors and valid lengths."""

    keys: torch.Tensor
    values: torch.Tensor
    lengths: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.lengths.device

    def to_device_dtype(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> "LayerKVCache":
        """Return cache tensors moved to the requested device/dtype."""

        return LayerKVCache(
            keys=self.keys.to(device=device, dtype=dtype),
            values=self.values.to(device=device, dtype=dtype),
            lengths=self.lengths.to(device=device),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Serialize into the legacy dictionary representation."""

        return {"k": self.keys, "v": self.values, "lengths": self.lengths}

    @classmethod
    def from_dict(
        cls, data: Optional[dict[str, torch.Tensor]]
    ) -> Optional["LayerKVCache"]:
        """Convert a legacy dictionary cache into a ``LayerKVCache``."""

        if data is None:
            return None
        keys = data.get("k")
        values = data.get("v")
        lengths = data.get("lengths")
        if keys is None or values is None or lengths is None:
            return None
        return cls(keys=keys, values=values, lengths=lengths)


class CacheManager:
    """Helper to marshal caches per transformer layer."""

    def __init__(self, caches: List[Optional[LayerKVCache]]) -> None:
        self._caches = caches

    @classmethod
    def from_input(
        cls,
        cache_input: Optional[Iterable[Optional[dict[str, torch.Tensor]]]],
        num_layers: int,
    ) -> "CacheManager":
        """Construct manager from optional legacy cache list."""

        caches: List[Optional[LayerKVCache]] = [None] * num_layers
        if cache_input is not None:
            source = list(cache_input)
            for idx in range(min(num_layers, len(source))):
                caches[idx] = LayerKVCache.from_dict(source[idx])
        return cls(caches)

    def get(self, layer_idx: int) -> Optional[LayerKVCache]:
        return self._caches[layer_idx]

    def set(self, layer_idx: int, cache: LayerKVCache) -> None:
        self._caches[layer_idx] = cache

    def as_output(self) -> List[Optional[dict[str, torch.Tensor]]]:
        return [
            cache.as_dict() if cache is not None else None for cache in self._caches
        ]
