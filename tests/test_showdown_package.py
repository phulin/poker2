from __future__ import annotations

import torch
import pytest

from p2.showdown.approximate import (
    tier1_hero_removal,
    tier2_first_order_opp_collision,
    tier3_second_order_opp_collision,
    tier4_third_degree_card_collision,
)
from p2.showdown.compare_multiway_showdown_tiers import prepare_random_showdown
from p2.showdown.exact import (
    exact_nway_ie_fast,
    tri_identity_smoke_check,
    tri_weight,
    tri_weight_direct,
)
from p2.showdown.multiway_showdown_estimators import exact_nway_ie
from p2.showdown.monte_carlo import make_batched_alias_tuple_reject_workspace


def test_showdown_package_exports_tier_calculators_and_alias_workspace() -> None:
    assert callable(tier1_hero_removal)
    assert callable(tier2_first_order_opp_collision)
    assert callable(tier3_second_order_opp_collision)
    assert callable(tier4_third_degree_card_collision)
    assert callable(make_batched_alias_tuple_reject_workspace)


def test_tri_weight_matches_direct_three_card_check() -> None:
    pi = torch.ones(3, 3, dtype=torch.float64) - torch.eye(3, dtype=torch.float64)
    assert tri_weight(pi, pi, pi).item() == 27.0
    assert tri_weight_direct(pi, pi, pi).item() == 27.0
    assert tri_identity_smoke_check() == {
        "direct": 27.0,
        "star_minus_duplicate": 21.0,
        "trace_triangle": 6.0,
        "formula_total": 27.0,
    }


@pytest.mark.slow
def test_axb_fast_matches_oracle_for_four_players() -> None:
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(123)
    prepared = prepare_random_showdown(
        players=4,
        device=device,
        generator=generator,
        concentration=1.0,
    )

    oracle = exact_nway_ie(prepared, pattern_chunk=64)
    fast = exact_nway_ie_fast(prepared, pattern_chunk=64)

    assert torch.equal(fast.equity, oracle.equity)
