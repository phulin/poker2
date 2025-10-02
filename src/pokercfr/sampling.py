"""Sampling utilities for chance nodes in vectorized CFR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ChanceSampler:
    generator: Optional[torch.Generator] = None

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        if self.generator is None:
            self.generator = torch.Generator(device=probs.device)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample(generator=self.generator)


__all__ = ["ChanceSampler"]
