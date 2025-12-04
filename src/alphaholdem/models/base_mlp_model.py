from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseMLPModel(nn.Module, ABC):
    """Common interface for MLP poker models."""

    @abstractmethod
    def forward(
        self,
        features,
        include_policy: bool = True,
        include_value: bool = True,
        latent=None,
    ): ...

    @abstractmethod
    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Factory for the feature encoder associated with this model."""
        ...

    @abstractmethod
    def repeat(
        self,
        features,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
    ): ...
