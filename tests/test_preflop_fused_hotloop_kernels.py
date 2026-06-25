from __future__ import annotations

import pytest
import torch

from p2.env.card_utils import (
    PREFLOP_HANDS,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
)
from p2.search.fused_cfr_triton import (
    fused_hu_closing_postprocess_writeback_multiway_,
    fused_model_values_postprocess_writeback_multiway_,
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
    select_heads_up_beliefs_triton,
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
def test_select_heads_up_beliefs_triton_matches_torch_gather() -> None:
    device = torch.device("cuda")
    rows = 17
    feature_rows = 31
    players = 6
    beliefs = torch.randn(
        feature_rows,
        players * PREFLOP_HANDS,
        device=device,
        dtype=torch.float32,
    )
    positions = torch.tensor(
        [0, 3, 5, 7, 11, 13, 17, 19, 23, 29, 1, 4, 8, 12, 16, 20, 24],
        device=device,
    )
    live_players = torch.stack(
        (
            torch.arange(rows, device=device) % players,
            (torch.arange(rows, device=device) + 2) % players,
        ),
        dim=1,
    ).contiguous()
    expected = beliefs[positions].view(rows, players, PREFLOP_HANDS).gather(
        1,
        live_players[:, :, None].expand(-1, 2, PREFLOP_HANDS),
    )
    actual = select_heads_up_beliefs_triton(
        beliefs,
        positions,
        live_players,
        num_players=players,
        hand_dim=PREFLOP_HANDS,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_model_values_postprocess_writeback_multiway_matches_torch() -> None:
    device = torch.device("cuda")
    rows = 13
    total = 29
    players = 6
    hand_dim = PREFLOP_HANDS
    node_indices = torch.tensor(
        [0, 3, 5, 7, 11, 13, 17, 19, 23, 27, 1, 9, 21],
        dtype=torch.long,
        device=device,
    )
    hand_values = torch.randn(
        rows,
        players,
        hand_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    last_model_values = torch.randn(
        rows,
        players,
        hand_dim,
        device=device,
    )
    beliefs = torch.rand(rows, players, hand_dim, device=device)
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    has_folded = torch.rand(total, players, device=device) < 0.2
    has_folded[:, 0] = False
    starting_stacks = torch.rand(total, players, device=device) * 200.0 + 800.0
    stacks = starting_stacks + torch.randn(total, players, device=device) * 20.0
    scale = torch.rand(total, device=device) * 100.0 + 200.0
    onon = torch.tensor(1.25, device=device)
    oon = torch.tensor(0.25, device=device)

    folded_mask = has_folded[node_indices]
    stack_value = (
        (stacks[node_indices] - starting_stacks[node_indices])
        / scale[node_indices, None].clamp_min(1.0)
    )
    values = torch.where(
        folded_mask[:, :, None],
        stack_value[:, :, None],
        hand_values.float(),
    )
    live_mask = ~folded_mask
    denom = (beliefs * live_mask[:, :, None]).sum(dim=(1, 2), keepdim=True)
    expected_sum = (values * beliefs).sum(dim=(1, 2), keepdim=True)
    processed = torch.where(
        live_mask[:, :, None],
        values - expected_sum / denom.clamp_min(1e-12),
        values,
    )

    expected_latest = torch.zeros(total, players, hand_dim, device=device)
    expected_latest[node_indices] = processed * onon - last_model_values * oon
    expected_last = processed

    actual_latest = torch.zeros_like(expected_latest)
    actual_last = torch.empty_like(last_model_values)
    fused_model_values_postprocess_writeback_multiway_(
        hand_values=hand_values.contiguous(),
        last_model_values=last_model_values.contiguous(),
        beliefs=beliefs.contiguous(),
        node_indices=node_indices.contiguous(),
        latest_values=actual_latest,
        last_out=actual_last,
        has_folded=has_folded.contiguous(),
        stacks=stacks.contiguous(),
        starting_stacks=starting_stacks.contiguous(),
        scale=scale.contiguous(),
        old_plus_new_over_new=onon,
        old_over_new=oon,
        do_mix=True,
        store_last=True,
    )

    torch.testing.assert_close(
        actual_latest[node_indices],
        expected_latest[node_indices],
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(actual_last, expected_last, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA and Triton")
def test_fused_hu_closing_postprocess_writeback_multiway_matches_torch() -> None:
    device = torch.device("cuda")
    rows = 17
    total = 41
    players = 6
    hand_dim = PREFLOP_HANDS
    node_indices = torch.tensor(
        [0, 2, 4, 5, 8, 10, 13, 16, 19, 22, 25, 27, 30, 33, 36, 38, 40],
        dtype=torch.long,
        device=device,
    )
    hand_values = torch.randn(rows, 2, hand_dim, device=device, dtype=torch.bfloat16)
    last_model_values = torch.randn(rows, players, hand_dim, device=device)
    beliefs = torch.rand(rows, players, hand_dim, device=device)
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    has_folded = torch.ones(total, players, device=device, dtype=torch.bool)
    live_players = torch.empty(rows, 2, device=device, dtype=torch.long)
    for r in range(rows):
        live_count = 2 + (r % 3)
        live = torch.randperm(players, device=device)[:live_count]
        has_folded[node_indices[r], live] = False
        live_players[r] = live[:2]
    starting_stacks = torch.rand(total, players, device=device) * 200.0 + 800.0
    stacks = starting_stacks + torch.randn(total, players, device=device) * 20.0
    scale = torch.rand(total, device=device) * 100.0 + 200.0
    onon = torch.tensor(1.2, device=device)
    oon = torch.tensor(0.2, device=device)

    folded_mask = has_folded[node_indices]
    stack_value = (
        (stacks[node_indices] - starting_stacks[node_indices])
        / scale[node_indices, None].clamp_min(1.0)
    )
    values = stack_value[:, :, None].expand(-1, -1, hand_dim).clone()
    values.scatter_(
        1,
        live_players[:, :, None].expand(-1, 2, hand_dim),
        hand_values.float(),
    )
    values = torch.where(folded_mask[:, :, None], stack_value[:, :, None], values)
    live_mask = ~folded_mask
    denom = (beliefs * live_mask[:, :, None]).sum(dim=(1, 2), keepdim=True)
    expected_sum = (values * beliefs).sum(dim=(1, 2), keepdim=True)
    processed = torch.where(
        live_mask[:, :, None],
        values - expected_sum / denom.clamp_min(1e-12),
        values,
    )
    expected_latest = torch.zeros(total, players, hand_dim, device=device)
    expected_latest[node_indices] = processed * onon - last_model_values * oon
    expected_last = processed

    actual_latest = torch.zeros_like(expected_latest)
    actual_last = torch.empty_like(last_model_values)
    fused_hu_closing_postprocess_writeback_multiway_(
        hand_values=hand_values.contiguous(),
        last_model_values=last_model_values.contiguous(),
        beliefs=beliefs.contiguous(),
        node_indices=node_indices.contiguous(),
        live_players=live_players.contiguous(),
        latest_values=actual_latest,
        last_out=actual_last,
        has_folded=has_folded.contiguous(),
        stacks=stacks.contiguous(),
        starting_stacks=starting_stacks.contiguous(),
        scale=scale.contiguous(),
        old_plus_new_over_new=onon,
        old_over_new=oon,
        do_mix=True,
        store_last=True,
    )

    torch.testing.assert_close(
        actual_latest[node_indices],
        expected_latest[node_indices],
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(actual_last, expected_last, rtol=2e-3, atol=2e-3)


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
