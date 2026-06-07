from __future__ import annotations

import pytest
import torch

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    board_allowed_hands,
    canonical_flops_with_weights,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.better_feature_encoder import (
    BetterPreflopValueFeatureEncoder,
    BetterStreetValueFeatureEncoder,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.rl.target_provenance import TARGET_SOURCE_CHANCE_EXPECTATION
from p2.search.end_of_street_distillation import build_end_of_street_value_batch
from p2.search.postflop_spot_sampler import sample_end_of_street_chance_roots


class BoardCardCountModel(torch.nn.Module):
    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterStreetValueFeatureEncoder:
        return BetterStreetValueFeatureEncoder(env, device=device, dtype=dtype)

    def forward_post(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del static_base_features
        counts = (features.board >= 0).sum(dim=1).to(features.beliefs.dtype)
        return counts[:, None, None].expand(-1, 2, NUM_HANDS).clone()


class ForwardOnlyBoardCardCountModel(torch.nn.Module):
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = False,
        include_value: bool = True,
        latent=None,
    ) -> ModelOutput:
        del include_policy, include_value, latent
        counts = (features.board >= 0).sum(dim=1).to(features.beliefs.dtype)
        hand_values = counts[:, None, None].expand(-1, 2, NUM_HANDS).clone()
        return ModelOutput(hand_values=hand_values)


def _env() -> HUNLTensorEnv:
    return HUNLTensorEnv(
        num_envs=2,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
    )


def test_canonical_flops_with_weights_covers_raw_flops():
    flops, weights = canonical_flops_with_weights()

    assert flops.shape == (1755, 3)
    assert weights.shape == (1755,)
    assert int(weights.sum().item()) == 22100


def test_build_end_of_street_value_batch_uses_pre_chance_features_and_targets():
    env = _env()
    generator = torch.Generator(device=env.device).manual_seed(11)
    target_model = BoardCardCountModel()

    for closed_street, expected_post_cards in ((1, 4), (2, 5)):
        sample = sample_end_of_street_chance_roots(
            env,
            batch_size=3,
            closed_street=closed_street,
            generator=generator,
        )
        encoder = BetterStreetValueFeatureEncoder(
            sample.pbs.env, device=env.device, dtype=torch.float32
        )

        batch = build_end_of_street_value_batch(
            sample,
            value_encoder=encoder,
            target_model=target_model,
            generator=generator,
        )

        assert batch.policy_targets is None
        assert batch.value_targets is not None
        assert torch.equal(batch.features.board, sample.pbs.env.last_board_indices)
        assert torch.equal(
            batch.features.street,
            torch.full((3,), closed_street, dtype=batch.features.street.dtype),
        )
        assert torch.equal(batch.statistics["street"], torch.full((3,), closed_street))
        assert torch.equal(
            batch.statistics["stage"], torch.full((3,), 2 * closed_street + 1)
        )
        assert torch.equal(
            batch.statistics["target_source"],
            torch.full((3,), TARGET_SOURCE_CHANCE_EXPECTATION),
        )

        previous_allowed = board_allowed_hands(sample.pbs.env.last_board_indices)
        expected_targets = torch.where(
            previous_allowed[:, None, :],
            torch.full((3, 2, NUM_HANDS), float(expected_post_cards)),
            torch.zeros(3, 2, NUM_HANDS),
        )
        torch.testing.assert_close(batch.value_targets, expected_targets)


def test_build_end_of_street_value_batch_supports_sampled_flop_targets():
    env = _env()
    generator = torch.Generator(device=env.device).manual_seed(22)
    sample = sample_end_of_street_chance_roots(
        env,
        batch_size=2,
        closed_street=0,
        generator=generator,
        compact_preflop_beliefs=True,
    )
    assert sample.pre_chance_beliefs.shape == (2, 2, PREFLOP_HANDS)
    encoder = BetterPreflopValueFeatureEncoder(
        sample.pbs.env, device=env.device, dtype=torch.float32
    )

    batch = build_end_of_street_value_batch(
        sample,
        value_encoder=encoder,
        target_model=BoardCardCountModel(),
        generator=generator,
    )

    assert torch.equal(batch.features.board, sample.pbs.env.last_board_indices)
    assert torch.equal(
        batch.features.street, torch.zeros(2, dtype=batch.features.street.dtype)
    )
    torch.testing.assert_close(
        batch.value_targets,
        torch.full((2, 2, PREFLOP_HANDS), 3.0),
    )


def test_build_end_of_street_value_batch_supports_canonical_flop_targets():
    env = _env()
    generator = torch.Generator(device=env.device).manual_seed(23)
    sample = sample_end_of_street_chance_roots(
        env,
        batch_size=2,
        closed_street=0,
        generator=generator,
    )
    encoder = BetterStreetValueFeatureEncoder(
        sample.pbs.env, device=env.device, dtype=torch.float32
    )

    batch = build_end_of_street_value_batch(
        sample,
        value_encoder=encoder,
        target_model=BoardCardCountModel(),
        chance="canonical_flops",
        generator=generator,
    )

    assert torch.equal(batch.features.board, sample.pbs.env.last_board_indices)
    assert torch.equal(
        batch.features.street, torch.zeros(2, dtype=batch.features.street.dtype)
    )
    torch.testing.assert_close(
        batch.value_targets,
        torch.full((2, 2, NUM_HANDS), 3.0),
    )


def test_build_preflop_distillation_batch_is_compact_169() -> None:
    env = _env()
    generator = torch.Generator(device=env.device).manual_seed(24)
    sample = sample_end_of_street_chance_roots(
        env,
        batch_size=2,
        closed_street=0,
        generator=generator,
    )
    encoder = BetterPreflopValueFeatureEncoder(
        sample.pbs.env, device=env.device, dtype=torch.float32
    )

    batch = build_end_of_street_value_batch(
        sample,
        value_encoder=encoder,
        target_model=BoardCardCountModel(),
        generator=generator,
    )

    assert batch.features.hand_dim == PREFLOP_HANDS
    assert batch.features.beliefs.shape == (2, 2 * PREFLOP_HANDS)
    assert batch.value_targets is not None
    assert batch.value_targets.shape == (2, 2, PREFLOP_HANDS)
    torch.testing.assert_close(
        batch.value_targets,
        torch.full((2, 2, PREFLOP_HANDS), 3.0),
    )


def test_build_end_of_street_value_batch_rejects_wrong_chance_mode():
    env = _env()
    sample = sample_end_of_street_chance_roots(
        env,
        batch_size=1,
        closed_street=0,
        generator=torch.Generator(device=env.device).manual_seed(33),
    )
    encoder = BetterStreetValueFeatureEncoder(
        sample.pbs.env, device=env.device, dtype=torch.float32
    )

    with pytest.raises(ValueError, match="requires chance in"):
        build_end_of_street_value_batch(
            sample,
            value_encoder=encoder,
            target_model=BoardCardCountModel(),
            chance="single_card",
        )


def test_build_end_of_street_value_batch_supports_forward_only_value_models():
    env = _env()
    sample = sample_end_of_street_chance_roots(
        env,
        batch_size=1,
        closed_street=2,
        generator=torch.Generator(device=env.device).manual_seed(44),
    )
    encoder = BetterStreetValueFeatureEncoder(
        sample.pbs.env, device=env.device, dtype=torch.float32
    )

    batch = build_end_of_street_value_batch(
        sample,
        value_encoder=encoder,
        target_model=ForwardOnlyBoardCardCountModel(),
    )

    previous_allowed = board_allowed_hands(sample.pbs.env.last_board_indices)
    expected_targets = torch.where(
        previous_allowed[:, None, :],
        torch.full((1, 2, NUM_HANDS), 5.0),
        torch.zeros(1, 2, NUM_HANDS),
    )
    torch.testing.assert_close(batch.value_targets, expected_targets)
