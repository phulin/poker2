from __future__ import annotations

import pytest
import torch

from p2.env.card_utils import NUM_HANDS
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.models.street_model_registry import StreetModelRegistry, street_index


class _StreetConstantModel(BaseMLPModel):
    def __init__(self, *, value: float, logit: float, num_actions: int = 5) -> None:
        super().__init__()
        self.value = float(value)
        self.logit = float(logit)
        self.num_actions = num_actions
        self.num_players = 2
        self.hidden_dim = 1
        self.enforce_zero_sum = False

    def _output(
        self,
        features: MLPFeatures,
        *,
        include_policy: bool,
        include_value: bool,
    ) -> ModelOutput:
        batch = len(features)
        device = features.context.device
        dtype = features.context.dtype
        policy_logits = None
        if include_policy:
            policy_logits = torch.full(
                (batch, NUM_HANDS, self.num_actions),
                self.logit,
                device=device,
                dtype=dtype,
            )
        value = None
        hand_values = None
        if include_value:
            hand_values = torch.full(
                (batch, self.num_players, NUM_HANDS),
                self.value,
                device=device,
                dtype=dtype,
            )
            value = hand_values.mean(dim=-1)
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
        )

    def forward_policy(self, features: MLPFeatures, latent=None) -> ModelOutput:
        return self._output(features, include_policy=True, include_value=False)

    def forward_value(self, features: MLPFeatures, latent=None, **kwargs) -> ModelOutput:
        return self._output(features, include_policy=False, include_value=True)

    def forward_pre(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        return torch.full(
            (len(features), self.num_players, NUM_HANDS),
            self.value + 10.0,
            device=features.context.device,
            dtype=features.context.dtype,
        )

    def forward_both(self, features: MLPFeatures, latent=None) -> ModelOutput:
        return self._output(features, include_policy=True, include_value=True)

    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        latent=None,
        **kwargs,
    ) -> ModelOutput:
        return self._output(
            features,
            include_policy=include_policy,
            include_value=include_value,
        )

    def create_feature_encoder(self, env, device=None, dtype=None):
        raise NotImplementedError

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


def _features(streets: list[int]) -> MLPFeatures:
    batch = len(streets)
    return MLPFeatures(
        context=torch.arange(batch, dtype=torch.float32).view(batch, 1),
        street=torch.tensor(streets, dtype=torch.long),
        to_act=torch.zeros(batch, dtype=torch.long),
        board=torch.full((batch, 5), -1, dtype=torch.long),
        beliefs=torch.zeros(batch, 2 * NUM_HANDS),
    )


def test_street_model_registry_routes_policy_and_value_by_feature_street() -> None:
    registry = StreetModelRegistry(
        {
            "flop": _StreetConstantModel(value=1.0, logit=11.0),
            "turn": _StreetConstantModel(value=2.0, logit=22.0),
            "river": _StreetConstantModel(value=3.0, logit=33.0),
        }
    )

    output = registry(_features([3, 1, 2, 1]), include_policy=True, include_value=True)

    assert output.policy_logits is not None
    assert output.hand_values is not None
    torch.testing.assert_close(
        output.policy_logits[:, 0, 0],
        torch.tensor([33.0, 11.0, 22.0, 11.0]),
    )
    torch.testing.assert_close(
        output.hand_values[:, 0, 0],
        torch.tensor([3.0, 1.0, 2.0, 1.0]),
    )


def test_street_model_registry_routes_pre_value_heads() -> None:
    registry = StreetModelRegistry(
        {
            "turn": _StreetConstantModel(value=2.0, logit=22.0),
            "river": _StreetConstantModel(value=3.0, logit=33.0),
        }
    )

    values = registry.forward_pre(_features([2, 3, 2]))

    torch.testing.assert_close(
        values[:, 0, 0],
        torch.tensor([12.0, 13.0, 12.0]),
    )


def test_street_model_registry_rejects_unregistered_street_without_default() -> None:
    registry = StreetModelRegistry(
        {"river": _StreetConstantModel(value=3.0, logit=33.0)}
    )

    with pytest.raises(ValueError, match="No model registered"):
        registry(_features([2]), include_policy=False, include_value=True)


def test_street_model_registry_can_default_missing_streets() -> None:
    registry = StreetModelRegistry(
        {"river": _StreetConstantModel(value=3.0, logit=33.0)},
        default_street="river",
    )

    output = registry(_features([2]), include_policy=False, include_value=True)

    assert output.hand_values is not None
    torch.testing.assert_close(output.hand_values[:, 0, 0], torch.tensor([3.0]))


def test_street_index_accepts_names_and_indices() -> None:
    assert street_index("flop") == 1
    assert street_index(3) == 3
    with pytest.raises(ValueError, match="Unknown street"):
        street_index("showdown")
