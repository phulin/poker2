from __future__ import annotations

import pytest
import torch

from p2.env.card_utils import (
    PREFLOP_HANDS,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
)
from p2.search.fused_cfr_triton import (
    fused_preflop169_parent_sum_opp_rank_stats_,
    fused_preflop169_parent_sum_opp_,
    fused_preflop169_project_rows_,
    fused_preflop169_src_weights_from_unblocked_multiway_,
    fused_preflop169_src_weights_multiway_,
    fused_preflop169_src_weights_rank_stats_multiway_,
    fused_preflop169_src_weights_stats_multiway_,
    preflop169_unblocked_rank_stats_out_,
    preflop169_unblocked_rank_mass_triton_out_,
    preflop169_unblocked_mass_triton_out_,
    triton_is_available,
)


def _cuda_available() -> bool:
    return torch.cuda.is_available() and triton_is_available()


def _projection(device: torch.device) -> torch.Tensor:
    multiplicity = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    compatibility = preflop_class_compatibility_counts_tensor(device=device).to(
        torch.float32
    )
    return (compatibility.T * multiplicity.reciprocal()[:, None]).contiguous()


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_preflop169_parent_sum_rank_stats_matches_unblocked() -> None:
    device = torch.device("cuda")
    parents = 11
    players = 6
    max_children = 4
    child_counts = torch.tensor(
        [1, 4, 2, 3, 1, 2, 4, 3, 2, 1, 4],
        device=device,
        dtype=torch.int32,
    )
    child_offsets = torch.empty_like(child_counts)
    child_offsets[0] = parents
    child_offsets[1:] = parents + child_counts[:-1].cumsum(0)
    children = int(child_counts.sum().item())
    total = parents + children
    policy = torch.rand(total, PREFLOP_HANDS, device=device)
    values = torch.randn(total, players, PREFLOP_HANDS, device=device)
    actor_beliefs = torch.rand(parents, PREFLOP_HANDS, device=device)
    marginal_policy = torch.rand(children, PREFLOP_HANDS, device=device)
    prev_actor = torch.arange(total, device=device, dtype=torch.int32) % players
    has_folded = torch.rand(total, players, device=device) < 0.15
    has_folded[:, 0] = False

    denom_unblocked = torch.empty_like(actor_beliefs)
    numer_unblocked = torch.empty_like(marginal_policy)
    actor_stats = torch.empty(parents, 14, device=device)
    marginal_stats = torch.empty(children, 14, device=device)
    marginal_action = torch.empty(children, device=device)
    preflop169_unblocked_rank_mass_triton_out_(
        actor_beliefs.contiguous(),
        denom_unblocked,
        stats_out=actor_stats,
    )
    preflop169_unblocked_rank_mass_triton_out_(
        marginal_policy.contiguous(),
        numer_unblocked,
        stats_out=marginal_stats,
        row_sum=marginal_action,
    )
    preflop169_unblocked_rank_stats_out_(
        actor_beliefs.contiguous(),
        actor_stats,
    )
    preflop169_unblocked_rank_stats_out_(
        marginal_policy.contiguous(),
        marginal_stats,
    )

    expected = values.clone()
    fused_preflop169_parent_sum_opp_(
        values=expected,
        prev_actor=prev_actor.contiguous(),
        policy=policy.contiguous(),
        marginal_action_policy=marginal_action.contiguous(),
        numer_unblocked=numer_unblocked.contiguous(),
        denom_unblocked=denom_unblocked.contiguous(),
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        parent_base=0,
        child_base=parents,
        max_children=max_children,
        has_folded=has_folded.contiguous(),
    )

    actual = values.clone()
    fused_preflop169_parent_sum_opp_rank_stats_(
        values=actual,
        prev_actor=prev_actor.contiguous(),
        policy=policy.contiguous(),
        actor_beliefs=actor_beliefs.contiguous(),
        marginal_policy=marginal_policy.contiguous(),
        actor_stats=actor_stats,
        marginal_stats=marginal_stats,
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        parent_base=0,
        child_base=parents,
        max_children=max_children,
        has_folded=has_folded.contiguous(),
    )

    torch.testing.assert_close(actual[:parents], expected[:parents], rtol=2e-5, atol=5e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_preflop169_project_rows_matches_mm_and_sum() -> None:
    device = torch.device("cuda")
    rows = 37
    source = torch.rand(rows, PREFLOP_HANDS, device=device)
    projection = _projection(device)
    expected = source @ projection
    expected_sum = source.sum(dim=-1)
    actual = torch.empty_like(expected)
    actual_sum = torch.empty(rows, dtype=source.dtype, device=device)

    fused_preflop169_project_rows_(
        source.contiguous(),
        projection,
        actual,
        row_sum=actual_sum,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(actual_sum, expected_sum, rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_preflop169_stats_unblocked_mass_matches_projection() -> None:
    device = torch.device("cuda")
    rows = 41
    source = torch.rand(rows, PREFLOP_HANDS, device=device)
    projection = _projection(device)
    expected = source @ projection
    actual = torch.empty_like(expected)
    row_sum = torch.empty(rows, dtype=source.dtype, device=device)

    preflop169_unblocked_mass_triton_out_(
        source.contiguous(),
        actual,
        row_sum=row_sum,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(row_sum, source.sum(dim=-1), rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_preflop169_rank_stats_unblocked_mass_matches_projection() -> None:
    device = torch.device("cuda")
    rows = 41
    source = torch.rand(rows, PREFLOP_HANDS, device=device)
    projection = _projection(device)
    expected = source @ projection
    actual = torch.empty_like(expected)
    row_sum = torch.empty(rows, dtype=source.dtype, device=device)

    preflop169_unblocked_rank_mass_triton_out_(
        source.contiguous(),
        actual,
        row_sum=row_sum,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(row_sum, source.sum(dim=-1), rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_preflop169_src_weights_multiway_matches_legacy() -> None:
    device = torch.device("cuda")
    top = 31
    players = 6
    class_mass = torch.rand(top, players, PREFLOP_HANDS, device=device)
    class_mass /= class_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = _projection(device)
    to_act = torch.arange(top, device=device) % players
    has_folded = torch.rand(top, players, device=device) < 0.2
    has_folded[torch.arange(top, device=device), to_act] = False
    allowed = torch.rand(top, PREFLOP_HANDS, device=device) > 0.05

    unblocked = class_mass @ projection
    player_ids = torch.arange(players, device=device)
    other_live = player_ids[None, :, None] != to_act[:, None, None]
    other_live &= ~has_folded[:, :, None]
    expected = torch.where(
        other_live,
        unblocked.clamp_min(1e-12),
        torch.ones_like(unblocked),
    ).prod(dim=1)
    expected *= allowed.to(dtype=expected.dtype)

    actual = torch.empty(top, PREFLOP_HANDS, device=device)
    fused_preflop169_src_weights_multiway_(
        class_mass=class_mass.contiguous(),
        projection=projection,
        to_act=to_act,
        allowed_mask=allowed.contiguous(),
        out=actual,
        has_folded=has_folded.contiguous(),
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_preflop169_src_weights_stats_matches_legacy() -> None:
    device = torch.device("cuda")
    top = 31
    players = 6
    class_mass = torch.rand(top, players, PREFLOP_HANDS, device=device)
    class_mass /= class_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = _projection(device)
    to_act = torch.arange(top, device=device) % players
    has_folded = torch.rand(top, players, device=device) < 0.2
    has_folded[torch.arange(top, device=device), to_act] = False
    allowed = (torch.rand(top, PREFLOP_HANDS, device=device) > 0.05).to(torch.float32)

    unblocked = class_mass @ projection
    player_ids = torch.arange(players, device=device)
    other_live = player_ids[None, :, None] != to_act[:, None, None]
    other_live &= ~has_folded[:, :, None]
    expected = torch.where(
        other_live,
        unblocked.clamp_min(1e-12),
        torch.ones_like(unblocked),
    ).prod(dim=1)
    expected *= allowed

    actual = torch.empty(top, PREFLOP_HANDS, device=device)
    fused_preflop169_src_weights_stats_multiway_(
        class_mass=class_mass.contiguous(),
        to_act=to_act,
        allowed_weight=allowed.contiguous(),
        out=actual,
        has_folded=has_folded.contiguous(),
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_preflop169_src_weights_rank_stats_matches_legacy() -> None:
    device = torch.device("cuda")
    top = 31
    players = 6
    class_mass = torch.rand(top, players, PREFLOP_HANDS, device=device)
    class_mass /= class_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = _projection(device)
    to_act = torch.arange(top, device=device) % players
    has_folded = torch.rand(top, players, device=device) < 0.2
    has_folded[torch.arange(top, device=device), to_act] = False
    allowed = (torch.rand(top, PREFLOP_HANDS, device=device) > 0.05).to(torch.float32)

    unblocked = class_mass @ projection
    player_ids = torch.arange(players, device=device)
    other_live = player_ids[None, :, None] != to_act[:, None, None]
    other_live &= ~has_folded[:, :, None]
    expected = torch.where(
        other_live,
        unblocked.clamp_min(1e-12),
        torch.ones_like(unblocked),
    ).prod(dim=1)
    expected *= allowed

    actual = torch.empty(top, PREFLOP_HANDS, device=device)
    fused_preflop169_src_weights_rank_stats_multiway_(
        class_mass=class_mass.contiguous(),
        to_act=to_act,
        allowed_weight=allowed.contiguous(),
        out=actual,
        has_folded=has_folded.contiguous(),
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_preflop169_src_weights_from_unblocked_matches_legacy() -> None:
    device = torch.device("cuda")
    top = 31
    players = 6
    class_mass = torch.rand(top, players, PREFLOP_HANDS, device=device)
    class_mass /= class_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = _projection(device)
    unblocked = (class_mass @ projection).contiguous()
    to_act = torch.arange(top, device=device) % players
    has_folded = torch.rand(top, players, device=device) < 0.2
    has_folded[torch.arange(top, device=device), to_act] = False
    allowed = (torch.rand(top, PREFLOP_HANDS, device=device) > 0.05).to(torch.float32)

    player_ids = torch.arange(players, device=device)
    other_live = player_ids[None, :, None] != to_act[:, None, None]
    other_live &= ~has_folded[:, :, None]
    expected = torch.where(
        other_live,
        unblocked.clamp_min(1e-12),
        torch.ones_like(unblocked),
    ).prod(dim=1)
    expected *= allowed

    actual = torch.empty(top, PREFLOP_HANDS, device=device)
    fused_preflop169_src_weights_from_unblocked_multiway_(
        unblocked=unblocked,
        to_act=to_act,
        allowed_weight=allowed.contiguous(),
        out=actual,
        has_folded=has_folded.contiguous(),
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)
