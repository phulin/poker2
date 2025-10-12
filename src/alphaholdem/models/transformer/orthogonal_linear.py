"""Custom Linear variants with orthogonal initialization and optional output scaling."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class OrthogonalLinear(nn.Linear):
    """Linear layer with orthogonal initialization and optional output scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        gain: float = math.sqrt(2.0),
        output_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self._gain = gain
        self._output_scale = output_scale

        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def init_weights(self, rng: Optional[torch.Generator] = None) -> None:  # type: ignore[override]
        nn.init.orthogonal_(self.weight, gain=self._gain, generator=rng)
        if self._output_scale is not None:
            self.weight.data.mul_(self._output_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        return super().forward(input)


class OrthogonalLinearNoScale(OrthogonalLinear):
    """Orthogonal linear layer without scaling, for clarity in call sites."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        gain: float = math.sqrt(2.0),
        rng: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            gain=gain,
            output_scale=1.0,
            rng=rng,
            device=device,
            dtype=dtype,
        )
