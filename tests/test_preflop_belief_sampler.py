import math

import pytest
import torch

from p2.env.card_utils import NUM_HANDS, PREFLOP_HANDS
from p2.search.preflop_belief_sampler import (
    DEFAULT_QUANTILES,
    OBSERVED_CASCADE_PROFILES,
    BeliefShapeProfile,
    belief_row_statistics,
    sample_preflop_beliefs,
)


def _generator(seed: int = 1234) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _observed_values(points: tuple[tuple[float, float], ...]) -> torch.Tensor:
    return torch.tensor([value for _, value in points[1:]], dtype=torch.float32)


def _sample_histogram_stats(
    profile_name: str,
    *,
    rows: int = 12_000,
    seed: int = 11,
) -> tuple[BeliefShapeProfile, dict[str, torch.Tensor]]:
    profile = OBSERVED_CASCADE_PROFILES[profile_name]
    beliefs = sample_preflop_beliefs(
        rows,
        num_players=1,
        profile=profile_name,
        mode="histogram",
        generator=_generator(seed),
    )
    return profile, belief_row_statistics(beliefs)


def _expected_uniform_entropy_share(profile: BeliefShapeProfile) -> float:
    uniform_entropy = math.log(PREFLOP_HANDS)
    for quantile, value in profile.entropy_quantiles:
        if value >= uniform_entropy - 1.0e-5:
            return 1.0 - quantile
    return 0.0


@pytest.mark.parametrize("num_players", [2, 6])
@pytest.mark.parametrize("hand_dim", [PREFLOP_HANDS, NUM_HANDS])
def test_preflop_belief_sampler_shape_and_normalization(
    num_players: int,
    hand_dim: int,
) -> None:
    beliefs = sample_preflop_beliefs(
        128,
        num_players=num_players,
        profile="actions_8_11",
        mode="mixed",
        hand_dim=hand_dim,
        generator=_generator(),
    )

    assert beliefs.shape == (128, num_players, hand_dim)
    assert beliefs.isfinite().all()
    assert torch.allclose(
        beliefs.sum(dim=-1),
        torch.ones(128, num_players),
        atol=1.0e-5,
    )
    assert (beliefs >= 0).all()


@pytest.mark.parametrize("mode", ["histogram", "coverage", "topk", "random"])
def test_native_combo_sampler_shape_and_statistics(mode: str) -> None:
    beliefs = sample_preflop_beliefs(
        256,
        num_players=2,
        profile="actions_12_end",
        mode=mode,
        hand_dim=NUM_HANDS,
        generator=_generator(7),
    )
    stats = belief_row_statistics(beliefs)

    assert beliefs.shape == (256, 2, NUM_HANDS)
    assert stats["max_class"].shape == (512,)
    assert stats["aa_mass"].shape == (512,)
    assert stats["entropy"].shape == (512,)
    assert (stats["aa_mass"] >= 0).all()
    assert (stats["aa_mass"] <= 1).all()


def test_native_combo_statistics_aggregate_aa_combos() -> None:
    beliefs = torch.full((3, NUM_HANDS), 1.0 / NUM_HANDS)
    stats = belief_row_statistics(beliefs)

    torch.testing.assert_close(
        stats["aa_mass"],
        torch.full((3,), 6.0 / NUM_HANDS),
    )


def test_native_combo_histogram_sampler_has_scaled_uniform_entropy_tail() -> None:
    beliefs = sample_preflop_beliefs(
        2048,
        num_players=1,
        profile="actions_12_end",
        mode="histogram",
        hand_dim=NUM_HANDS,
        generator=_generator(41),
    )
    entropy = belief_row_statistics(beliefs)["entropy"]
    uniform_share = float((entropy >= math.log(NUM_HANDS) - 1.0e-4).float().mean())

    assert abs(uniform_share - 0.10) <= 0.04


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_native_combo_histogram_sampler_runs_on_cuda() -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(43)
    beliefs = sample_preflop_beliefs(
        128,
        num_players=2,
        profile="actions_4_7",
        mode="histogram",
        hand_dim=NUM_HANDS,
        device="cuda",
        generator=generator,
    )

    assert beliefs.is_cuda
    assert beliefs.shape == (128, 2, NUM_HANDS)
    assert torch.allclose(
        beliefs.sum(dim=-1),
        torch.ones(128, 2, device=beliefs.device),
        atol=1.0e-5,
    )


@pytest.mark.parametrize("profile_name", list(OBSERVED_CASCADE_PROFILES))
def test_histogram_sampler_matches_observed_max_class_shape(
    profile_name: str,
) -> None:
    profile, stats = _sample_histogram_stats(profile_name)
    observed = _observed_values(profile.max_class_quantiles)
    generated = stats["max_class_q"].cpu()

    assert torch.allclose(generated[:4], observed[:4], atol=0.015)
    assert abs(float(generated[4] - observed[4])) <= 0.04


@pytest.mark.parametrize("profile_name", list(OBSERVED_CASCADE_PROFILES))
def test_histogram_sampler_matches_observed_aa_body(profile_name: str) -> None:
    profile, stats = _sample_histogram_stats(profile_name, seed=17)
    observed = _observed_values(profile.aa_mass_quantiles)
    generated = stats["aa_mass_q"].cpu()

    assert torch.allclose(generated[:3], observed[:3], atol=0.006)
    assert generated[4] >= min(0.95, float(observed[4]) - 0.02)


@pytest.mark.parametrize("profile_name", list(OBSERVED_CASCADE_PROFILES))
def test_histogram_sampler_matches_observed_entropy_quantiles(
    profile_name: str,
) -> None:
    profile, stats = _sample_histogram_stats(profile_name, seed=23)
    observed = _observed_values(profile.entropy_quantiles)
    generated = stats["entropy_q"].cpu()

    assert torch.allclose(generated, observed, atol=0.04)


@pytest.mark.parametrize("profile_name", list(OBSERVED_CASCADE_PROFILES))
def test_histogram_sampler_matches_observed_entropy_cdf(
    profile_name: str,
) -> None:
    profile, stats = _sample_histogram_stats(profile_name, rows=16_000, seed=29)
    entropy = stats["entropy"].cpu()
    uniform_entropy = math.log(PREFLOP_HANDS)

    for expected_quantile, threshold in profile.entropy_quantiles[1:-1]:
        if threshold >= uniform_entropy - 1.0e-5:
            continue
        actual_quantile = float((entropy <= threshold).float().mean())
        tolerance = 0.065 if threshold >= uniform_entropy - 0.02 else 0.04
        assert abs(actual_quantile - expected_quantile) <= tolerance

    expected_uniform_share = _expected_uniform_entropy_share(profile)
    actual_uniform_share = float((entropy >= uniform_entropy - 1.0e-4).float().mean())
    tolerance = max(0.006, expected_uniform_share * 0.25)
    assert abs(actual_uniform_share - expected_uniform_share) <= tolerance


def test_mixed_sampler_covers_broad_and_hard_tail_regions() -> None:
    beliefs = sample_preflop_beliefs(
        4096,
        num_players=2,
        profile="actions_12_end",
        mode="mixed",
        generator=_generator(29),
    )
    stats = belief_row_statistics(beliefs)
    max_class = stats["max_class"]

    assert (max_class < 0.08).float().mean() > 0.05
    assert (max_class > 0.40).float().mean() > 0.08
    assert (max_class > 0.90).any()


def test_mixed_sampler_entropy_histogram_covers_observed_tail_regions() -> None:
    beliefs = sample_preflop_beliefs(
        4096,
        num_players=2,
        profile="actions_12_end",
        mode="mixed",
        generator=_generator(37),
    )
    entropy = belief_row_statistics(beliefs)["entropy"]

    assert (entropy < 4.0).float().mean() > 0.05
    assert ((entropy >= 4.0) & (entropy < 5.0)).float().mean() > 0.10
    assert (entropy >= 5.10).float().mean() > 0.05


def test_belief_row_statistics_quantiles_are_default_shape() -> None:
    beliefs = sample_preflop_beliefs(
        32,
        num_players=2,
        profile="actions_4_7",
        mode="coverage",
        generator=_generator(31),
    )
    stats = belief_row_statistics(beliefs)

    assert stats["max_class_q"].shape == (len(DEFAULT_QUANTILES),)
    assert stats["aa_mass_q"].shape == (len(DEFAULT_QUANTILES),)
    assert stats["entropy_q"].shape == (len(DEFAULT_QUANTILES),)
