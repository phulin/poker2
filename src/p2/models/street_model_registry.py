from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput, TRMLatent

STREET_NAME_TO_INDEX = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
}
STREET_INDEX_TO_NAME = {index: name for name, index in STREET_NAME_TO_INDEX.items()}


def street_index(street: int | str) -> int:
    if isinstance(street, str):
        key = street.lower()
        if key not in STREET_NAME_TO_INDEX:
            raise ValueError(f"Unknown street name: {street!r}")
        return STREET_NAME_TO_INDEX[key]
    index = int(street)
    if index not in STREET_INDEX_TO_NAME:
        raise ValueError(f"Unknown street index: {street!r}")
    return index


class StreetModelRegistry(BaseMLPModel):
    """Dispatch model calls to street-specific frozen nets.

    The curriculum promotes separate ``S_flop``, ``S_turn``, and ``S_river``
    checkpoints. This wrapper lets search receive one model-like object while
    preserving per-street dispatch at resolve time.
    """

    def __init__(
        self,
        models: Mapping[int | str, BaseMLPModel],
        *,
        default_street: int | str | None = None,
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("StreetModelRegistry requires at least one model")

        self._street_to_key: dict[int, str] = {}
        modules: dict[str, BaseMLPModel] = {}
        for street, model in models.items():
            index = street_index(street)
            key = STREET_INDEX_TO_NAME[index]
            if index in self._street_to_key:
                raise ValueError(f"Duplicate model for street {key!r}")
            self._street_to_key[index] = key
            modules[key] = model
        self.street_models = nn.ModuleDict(modules)

        self.default_street = (
            street_index(default_street) if default_street is not None else None
        )
        if self.default_street is not None and self.default_street not in self._street_to_key:
            raise ValueError("default_street must refer to a registered street")

        first = next(iter(self.street_models.values()))
        self.hidden_dim = first.hidden_dim
        self.hand_dim = first.hand_dim
        self.num_players = first.num_players
        self.num_actions = first.num_actions
        self.enforce_zero_sum = first.enforce_zero_sum
        for name, model in self.street_models.items():
            if model.hidden_dim != self.hidden_dim:
                raise ValueError(
                    f"Street model {name!r} has hidden_dim={model.hidden_dim}, expected {self.hidden_dim}"
                )
            if model.hand_dim != self.hand_dim:
                raise ValueError(
                    f"Street model {name!r} has hand_dim={model.hand_dim}, expected {self.hand_dim}"
                )
            if model.num_players != self.num_players:
                raise ValueError(
                    f"Street model {name!r} has num_players={model.num_players}, expected {self.num_players}"
                )
            if model.num_actions != self.num_actions:
                raise ValueError(
                    f"Street model {name!r} has num_actions={model.num_actions}, expected {self.num_actions}"
                )

    def _model_for_index(self, index: int) -> BaseMLPModel:
        key = self._street_to_key[index]
        return self.street_models[key]

    def _dispatch_outputs(
        self,
        features: MLPFeatures,
        call: Callable[[BaseMLPModel, MLPFeatures], ModelOutput],
    ) -> ModelOutput:
        N = len(features)
        if N == 0:
            model = self._model_for_index(next(iter(self._street_to_key)))
            return call(model, features)

        result = _OutputBuilder(batch_size=N, device=features.context.device)
        handled = torch.zeros(N, dtype=torch.bool, device=features.street.device)
        for index in sorted(self._street_to_key):
            mask = features.street == index
            if mask.any():
                output = call(self._model_for_index(index), features[mask])
                result.put(mask, output)
                handled |= mask

        remaining = ~handled
        if remaining.any():
            if self.default_street is None:
                streets = torch.unique(features.street[remaining]).detach().cpu().tolist()
                raise ValueError(f"No model registered for street(s): {streets}")
            output = call(self._model_for_index(self.default_street), features[remaining])
            result.put(remaining, output)
        return result.build()

    def _dispatch_tensors(
        self,
        features: MLPFeatures,
        call: Callable[[BaseMLPModel, MLPFeatures], torch.Tensor],
    ) -> torch.Tensor:
        N = len(features)
        result: torch.Tensor | None = None
        handled = torch.zeros(N, dtype=torch.bool, device=features.street.device)
        for index in sorted(self._street_to_key):
            mask = features.street == index
            if mask.any():
                value = call(self._model_for_index(index), features[mask])
                if result is None:
                    result = torch.empty(
                        (N, *value.shape[1:]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                result[mask] = value
                handled |= mask

        remaining = ~handled
        if remaining.any():
            if self.default_street is None:
                streets = torch.unique(features.street[remaining]).detach().cpu().tolist()
                raise ValueError(f"No model registered for street(s): {streets}")
            value = call(self._model_for_index(self.default_street), features[remaining])
            if result is None:
                result = torch.empty(
                    (N, *value.shape[1:]),
                    device=value.device,
                    dtype=value.dtype,
                )
            result[remaining] = value
        if result is None:
            raise ValueError("Cannot dispatch an empty tensor batch")
        return result

    def compile_forward_modes(self, **kwargs):
        for model in self.street_models.values():
            model.compile_forward_modes(**kwargs)
        return self

    def forward_policy(self, features: MLPFeatures, latent=None) -> ModelOutput:
        return self._dispatch_outputs(
            features,
            lambda model, item: _as_model_output(model.forward_policy(item)),
        )

    def forward_value(self, features: MLPFeatures, latent=None, **kwargs) -> ModelOutput:
        return self._dispatch_outputs(
            features,
            lambda model, item: model.forward_value(item, **kwargs),
        )

    def forward_pre(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        return self._dispatch_tensors(
            features,
            lambda model, item: model.forward_pre(item, **kwargs),
        )

    def forward_post(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        return self._dispatch_tensors(
            features,
            lambda model, item: model.forward_post(item, **kwargs),
        )

    def forward_both(self, features: MLPFeatures, latent=None, **kwargs) -> ModelOutput:
        return self(
            features,
            include_policy=True,
            include_value=True,
            latent=latent,
            **kwargs,
        )

    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        latent=None,
        **kwargs,
    ) -> ModelOutput:
        if not include_policy and not include_value:
            raise ValueError("At least one of include_policy/include_value must be true")
        return self._dispatch_outputs(
            features,
            lambda model, item: model(
                item,
                include_policy=include_policy,
                include_value=include_value,
                **kwargs,
            ),
        )

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        model = self._model_for_index(next(iter(self._street_to_key)))
        return model.create_feature_encoder(env, device=device, dtype=dtype)

    def repeat(
        self,
        features: MLPFeatures,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
    ) -> ModelOutput:
        return self(
            features,
            include_policy=include_policy,
            include_value=include_value,
        )


class _OutputBuilder:
    def __init__(self, *, batch_size: int, device: torch.device) -> None:
        self.batch_size = batch_size
        self.device = device
        self.value: torch.Tensor | None = None
        self.policy_logits: torch.Tensor | None = None
        self.value_quantiles: torch.Tensor | None = None
        self.hand_values: torch.Tensor | None = None
        self.latent_y: torch.Tensor | None = None
        self.latent_z: torch.Tensor | None = None

    def put(self, mask: torch.Tensor, output: ModelOutput) -> None:
        self.value = _put_optional(self.value, mask, output.value, self.batch_size)
        self.policy_logits = _put_optional(
            self.policy_logits, mask, output.policy_logits, self.batch_size
        )
        self.value_quantiles = _put_optional(
            self.value_quantiles, mask, output.value_quantiles, self.batch_size
        )
        self.hand_values = _put_optional(
            self.hand_values, mask, output.hand_values, self.batch_size
        )
        if output.latent is not None:
            self.latent_y = _put_optional(
                self.latent_y, mask, output.latent.y, self.batch_size
            )
            self.latent_z = _put_optional(
                self.latent_z, mask, output.latent.z, self.batch_size
            )

    def build(self) -> ModelOutput:
        latent = None
        if self.latent_y is not None and self.latent_z is not None:
            latent = TRMLatent(y=self.latent_y, z=self.latent_z)
        return ModelOutput(
            value=self.value,
            policy_logits=self.policy_logits,
            value_quantiles=self.value_quantiles,
            hand_values=self.hand_values,
            latent=latent,
        )


def _put_optional(
    target: torch.Tensor | None,
    mask: torch.Tensor,
    value: torch.Tensor | None,
    batch_size: int,
) -> torch.Tensor | None:
    if value is None:
        return target
    if target is None:
        target = torch.empty(
            (batch_size, *value.shape[1:]),
            device=value.device,
            dtype=value.dtype,
        )
    target[mask] = value
    return target


def _as_model_output(output: ModelOutput | torch.Tensor) -> ModelOutput:
    if isinstance(output, ModelOutput):
        return output
    return ModelOutput(policy_logits=output)
