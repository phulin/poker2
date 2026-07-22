from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

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
        kwargs = dict(kwargs)
        compile_policy = bool(kwargs.pop("policy_compile", True))
        policy_dynamic = bool(
            kwargs.pop("policy_dynamic", kwargs.get("dynamic", False))
        )
        dynamic_batch = bool(kwargs.get("dynamic", False))
        policy_kwargs = dict(kwargs)
        policy_kwargs["dynamic"] = policy_dynamic
        self._compiled_forward_dynamic_batch = dynamic_batch
        self._compiled_forward_policy_dynamic_batch = (
            policy_dynamic if compile_policy else False
        )
        self._compiled_forward_value_dynamic_batch = dynamic_batch
        self._compiled_forward_value_static_base_dynamic_batch = False
        self._compiled_forward_both_dynamic_batch = dynamic_batch
        if compile_policy:
            self._compiled_forward_policy = torch.compile(
                self.forward_policy,
                **policy_kwargs,
            )
        else:
            self._compiled_forward_policy = None
        self._compiled_forward_value = torch.compile(self.forward_value, **kwargs)
        self._compiled_forward_both = torch.compile(self.forward_both, **kwargs)
        static_value_fn = getattr(self, "forward_value_static_base", None)
        if static_value_fn is not None:
            self._compiled_forward_value_static_base = torch.compile(
                static_value_fn, **kwargs
            )
        return self

    @staticmethod
    def _mark_dynamic_batch(tensor: torch.Tensor | None) -> None:
        if tensor is None or tensor.dim() == 0:
            return
        # maybe_mark_dynamic (a hint), not mark_dynamic (a hard constraint):
        # keep the batch dim dynamic when possible, but let the compiler
        # specialize+recompile for a rare shape (e.g. a subgame with exactly 4
        # model-leaf nodes) instead of raising ConstraintViolationError and
        # killing the run.
        torch._dynamo.maybe_mark_dynamic(tensor, 0)

    @classmethod
    def _mark_feature_batch_dynamic(cls, features) -> None:
        for name in ("context", "street", "to_act", "board", "beliefs"):
            value = getattr(features, name, None)
            if isinstance(value, torch.Tensor):
                cls._mark_dynamic_batch(value)

    @classmethod
    def _mark_compiled_forward_dynamic(cls, args, kwargs) -> None:
        batch_size = None
        if args:
            cls._mark_feature_batch_dynamic(args[0])
            context = getattr(args[0], "context", None)
            batch_size = getattr(context, "shape", (None,))[0]
            for value in args[1:]:
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    if batch_size is None or value.shape[0] == batch_size:
                        cls._mark_dynamic_batch(value)
                elif isinstance(value, Iterable) and not isinstance(
                    value, (str, bytes)
                ):
                    for item in value:
                        if isinstance(item, torch.Tensor) and item.dim() > 0:
                            if batch_size is None or item.shape[0] == batch_size:
                                cls._mark_dynamic_batch(item)
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                if batch_size is None or value.shape[0] == batch_size:
                    cls._mark_dynamic_batch(value)

    def _call_forward_policy(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_policy", None)
        if fn is None:
            fn = self.forward_policy
        elif getattr(
            self,
            "_compiled_forward_policy_dynamic_batch",
            getattr(self, "_compiled_forward_dynamic_batch", True),
        ):
            self._mark_compiled_forward_dynamic(args, kwargs)
        return fn(*args, **kwargs)

    def _call_forward_value(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_value", None)
        if fn is None:
            fn = self.forward_value
        elif getattr(
            self,
            "_compiled_forward_value_dynamic_batch",
            getattr(self, "_compiled_forward_dynamic_batch", True),
        ):
            self._mark_compiled_forward_dynamic(args, kwargs)
        return fn(*args, **kwargs)

    def _call_forward_value_static_base(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_value_static_base", None)
        if fn is None:
            fn = getattr(self, "forward_value_static_base")
        elif getattr(
            self,
            "_compiled_forward_value_static_base_dynamic_batch",
            getattr(self, "_compiled_forward_dynamic_batch", True),
        ):
            self._mark_compiled_forward_dynamic(args, kwargs)
        return fn(*args, **kwargs)

    def _call_forward_both(self, *args, **kwargs):
        fn = getattr(self, "_compiled_forward_both", None)
        if fn is None:
            fn = self.forward_both
        elif getattr(
            self,
            "_compiled_forward_both_dynamic_batch",
            getattr(self, "_compiled_forward_dynamic_batch", True),
        ):
            self._mark_compiled_forward_dynamic(args, kwargs)
        return fn(*args, **kwargs)

    @abstractmethod
    def forward_policy(self, features, latent=None): ...

    @abstractmethod
    def forward_value(self, features, latent=None, apply_zero_sum: bool = True): ...

    @abstractmethod
    def forward_both(self, features, latent=None, apply_zero_sum: bool = True): ...

    @abstractmethod
    def forward(
        self,
        features,
        include_policy: bool = True,
        include_value: bool = True,
        latent=None,
        apply_zero_sum: bool = True,
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
        apply_zero_sum: bool = True,
    ): ...
