from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from p2.env.card_utils import NUM_HANDS


class BaseMLPModel(nn.Module, ABC):
    """Common interface for MLP poker models."""

    hand_dim: int = NUM_HANDS
    hidden_dim: int
    num_actions: int
    num_players: int
    enforce_zero_sum: bool

    def compile_forward_modes(self, **kwargs):
        """Compile fixed-mode forwards without compiling boolean dispatch."""
        self._compiled_forward_policy = torch.compile(self.forward_policy, **kwargs)
        self._compiled_forward_value = torch.compile(self.forward_value, **kwargs)
        self._compiled_forward_both = torch.compile(self.forward_both, **kwargs)
        static_value_fn = getattr(self, "forward_value_static_base", None)
        if static_value_fn is not None:
            self._compiled_forward_value_static_base = torch.compile(
                static_value_fn, **kwargs
            )
        return self

    def _call_forward_policy(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_policy", None)
        if fn is None:
            fn = self.forward_policy
        return fn(*args, **kwargs)

    def _call_forward_value(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_value", None)
        if fn is None:
            fn = self.forward_value
        return fn(*args, **kwargs)

    def _call_forward_value_static_base(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_value_static_base", None)
        if fn is None:
            fn = getattr(self, "forward_value_static_base")
        return fn(*args, **kwargs)

    def _call_forward_both(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_both", None)
        if fn is None:
            fn = self.forward_both
        return fn(*args, **kwargs)

    @abstractmethod
    def forward_policy(self, features, latent=None): ...

    @abstractmethod
    def forward_value(self, features, latent=None): ...

    @abstractmethod
    def forward_both(self, features, latent=None): ...

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
