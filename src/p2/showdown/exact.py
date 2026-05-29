from __future__ import annotations

from dataclasses import dataclass, field

import torch

from p2.env.card_utils import NUM_HANDS, board_allowed_hands

from .multiway_showdown_estimators import (
    ExactResult,
    PreparedShowdown,
    brute_force_nway,
    exact_nway_ie,
    exact_nway_ie_by_hand,
    exact_nway_ie_chunked_f32,
    exact_threeway_ie,
)
from .results import (
    PerHandEquityResult,
    aggregate_all_players_from_num_denom,
    safe_divide_by_hand,
)


@dataclass
class _PolyRange:
    mats: list[torch.Tensor | "_MaskedBatchMatrix"]
    scalars: list[torch.Tensor]
    marginals: list[torch.Tensor]


@dataclass
class _MaskedBatchMatrix:
    base: torch.Tensor
    card_mask: torch.Tensor
    blocked0: torch.Tensor
    blocked1: torch.Tensor
    _materialized: torch.Tensor | None = field(default=None, init=False, repr=False)

    def materialize(self) -> torch.Tensor:
        if self._materialized is None:
            base = self.base.unsqueeze(0) if self.base.dim() == 2 else self.base
            self._materialized = (
                base * self.card_mask[:, :, None] * self.card_mask[:, None, :]
            )
        return self._materialized


@dataclass
class _RankClassData:
    rank_ids: torch.Tensor
    card_mask: torch.Tensor
    blocked0: torch.Tensor
    blocked1: torch.Tensor
    less_mats: torch.Tensor
    equal_mats: torch.Tensor


def exact_nway_ie_fast(
    prepared: PreparedShowdown,
    *,
    pattern_chunk: int = 4096,
) -> ExactResult:
    """Run the A+xB exact evaluator when available, else fallback to chunked IE."""
    if prepared.beliefs.shape[1] <= 4:
        return exact_nway_ie_axb(prepared)
    return exact_nway_ie_chunked_f32(prepared, pattern_chunk=pattern_chunk)


def exact_nway_ie_fast_by_hand(
    prepared: PreparedShowdown,
    *,
    pattern_chunk: int = 4096,
) -> PerHandEquityResult:
    if prepared.beliefs.shape[1] <= 4:
        return exact_nway_ie_axb_by_hand(prepared)
    return exact_nway_ie_by_hand(
        prepared,
        pattern_chunk=pattern_chunk,
        dtype=torch.float32,
        chunked_f32=True,
    )


def exact_nway_ie_axb_by_hand(
    prepared: PreparedShowdown,
    *,
    dtype: torch.dtype = torch.float64,
) -> PerHandEquityResult:
    """Exact rank-prefix A+xB showdown evaluator for up to 4 players.

    This path sweeps hand-rank classes. For each hero rank ``R`` it represents
    each opponent range as ``A + xB`` where ``A`` is strictly worse than the
    hero and ``B`` ties the hero, then integrates the exact disjoint-opponent
    cluster polynomial over ``x``.

    The implementation supports up to three opponents, where exact opponent
    card-removal clusters are scalar, pair collision, wedge/Delta, and tri.
    Larger games currently fall back to the copied generic IE prototype.
    """
    players = prepared.beliefs.shape[1]
    if players < 2:
        raise ValueError("exact_nway_ie_axb expects at least two players")
    if players > 4:
        raise ValueError("exact_nway_ie_axb currently supports up to 4 players")

    import time

    start = time.perf_counter()
    device = prepared.beliefs.device
    beliefs = prepared.beliefs.to(dtype)
    batch_size = beliefs.shape[0]
    ranks = prepared.hand_ranks.reshape(batch_size, NUM_HANDS)
    combos = prepared.combos.long()
    allowed = board_allowed_hands(prepared.board)
    active_count = 1081
    active_ids = torch.topk(allowed.to(torch.int8), active_count, dim=1).indices
    active_combos = combos[active_ids]
    active_ranks = ranks.gather(1, active_ids)
    active_beliefs = beliefs.gather(2, active_ids[:, None, :].expand(-1, players, -1))

    board_mask = torch.ones(batch_size, 52, dtype=torch.bool, device=device)
    board_mask.scatter_(1, prepared.board.long(), False)
    card_to_local = board_mask.long().cumsum(dim=1) - 1
    card_to_local.masked_fill_(~board_mask, -1)
    local_c0 = card_to_local.gather(1, active_combos[..., 0])
    local_c1 = card_to_local.gather(1, active_combos[..., 1])
    card_count = 47

    all_mats = _build_player_mats(active_beliefs, local_c0, local_c1, card_count)

    numerator_active = torch.zeros(batch_size, players, active_count, dtype=dtype, device=device)
    denominator_active = torch.zeros_like(numerator_active)
    opponent_lists = [
        [player for player in range(players) if player != hero]
        for hero in range(players)
    ]

    denominator_cache: dict[tuple[int, ...], torch.Tensor] = {}
    for hero, opponents in enumerate(opponent_lists):
        opponent_key = tuple(opponents)
        if opponent_key not in denominator_cache:
            denominator_cache[opponent_key] = _denominator_by_active_hand(
                all_mats,
                opponents,
                local_c0,
                local_c1,
                card_count,
            )
        denominator_active[:, hero] = denominator_cache[opponent_key]

    less_mats = torch.zeros_like(all_mats)
    for rank in torch.unique(active_ranks, sorted=True).detach().cpu().tolist():
        rank_mask = active_ranks == rank
        rank_coords = torch.nonzero(rank_mask, as_tuple=False)
        rank_board_ids = rank_coords[:, 0]
        rank_hand_ids = rank_coords[:, 1]
        equal_mats = _build_player_mats_from_indices(
            active_beliefs,
            rank_board_ids,
            rank_hand_ids,
            local_c0[rank_mask],
            local_c1[rank_mask],
            card_count,
        )
        blocked0 = local_c0[rank_mask]
        blocked1 = local_c1[rank_mask]
        card_mask = torch.ones(
            blocked0.numel(),
            card_count,
            dtype=dtype,
            device=device,
        )
        card_mask.scatter_(1, blocked0[:, None], 0.0)
        card_mask.scatter_(1, blocked1[:, None], 0.0)
        masked_ranges = [
            _poly_range(
                [
                    _masked_batch_matrix(
                        less_mats[rank_board_ids, player],
                        card_mask,
                        blocked0,
                        blocked1,
                    ),
                    _masked_batch_matrix(
                        equal_mats[rank_board_ids, player],
                        card_mask,
                        blocked0,
                        blocked1,
                    ),
                ]
            )
            for player in range(players)
        ]
        for hero, opponents in enumerate(opponent_lists):
            numerators = _integrate_unit_interval(
                _valid_mass_poly([masked_ranges[opp] for opp in opponents])
            )
            numerator_active[rank_board_ids, hero, rank_hand_ids] = numerators
        less_mats = less_mats + equal_mats

    numerator_by_hand = _scatter_active_to_full_batched(numerator_active, active_ids)
    denominator_by_hand = _scatter_active_to_full_batched(denominator_active, active_ids)
    equity_by_hand = safe_divide_by_hand(numerator_by_hand, denominator_by_hand)
    aggregate_equity = aggregate_all_players_from_num_denom(
        beliefs,
        numerator_by_hand,
        denominator_by_hand,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand.to(torch.float32),
        aggregate_equity=aggregate_equity,
        denominator_by_hand=denominator_by_hand,
        numerator_by_hand=numerator_by_hand,
        seconds=time.perf_counter() - start,
    )


def exact_nway_ie_axb(
    prepared: PreparedShowdown,
    *,
    dtype: torch.dtype = torch.float64,
) -> ExactResult:
    result = exact_nway_ie_axb_by_hand(prepared, dtype=dtype)
    return ExactResult(equity=result.aggregate_equity, seconds=result.seconds)


def _build_player_mats(
    active_beliefs: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    card_count: int,
) -> torch.Tensor:
    if active_beliefs.dim() == 2:
        players = active_beliefs.shape[0]
        mats = torch.zeros(
            players,
            card_count,
            card_count,
            dtype=active_beliefs.dtype,
            device=active_beliefs.device,
        )
        player_ids = torch.arange(players, device=active_beliefs.device)[:, None]
        c0 = local_c0[None, :].expand(players, -1)
        c1 = local_c1[None, :].expand(players, -1)
        mats[player_ids, c0, c1] = active_beliefs
        mats[player_ids, c1, c0] = active_beliefs
        return mats

    batch_size, players = active_beliefs.shape[:2]
    mats = torch.zeros(
        batch_size,
        players,
        card_count,
        card_count,
        dtype=active_beliefs.dtype,
        device=active_beliefs.device,
    )
    batch_ids = torch.arange(batch_size, device=active_beliefs.device)[:, None, None]
    player_ids = torch.arange(players, device=active_beliefs.device)[None, :, None]
    c0 = local_c0[:, None, :].expand(-1, players, -1)
    c1 = local_c1[:, None, :].expand(-1, players, -1)
    mats[batch_ids, player_ids, c0, c1] = active_beliefs
    mats[batch_ids, player_ids, c1, c0] = active_beliefs
    return mats


def _build_player_mats_from_indices(
    active_beliefs: torch.Tensor,
    board_ids: torch.Tensor,
    hand_ids: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    card_count: int,
) -> torch.Tensor:
    batch_size, players = active_beliefs.shape[:2]
    mats = torch.zeros(
        batch_size,
        players,
        card_count,
        card_count,
        dtype=active_beliefs.dtype,
        device=active_beliefs.device,
    )
    player_ids = torch.arange(players, device=active_beliefs.device)[None, :]
    values = active_beliefs[board_ids, :, hand_ids]
    mats[board_ids[:, None], player_ids, local_c0[:, None], local_c1[:, None]] = values
    mats[board_ids[:, None], player_ids, local_c1[:, None], local_c0[:, None]] = values
    return mats


def _build_rank_class_data(
    active_beliefs: torch.Tensor,
    active_ranks: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    card_count: int,
) -> list[_RankClassData]:
    rank_classes: list[_RankClassData] = []
    less_mats = torch.zeros(
        active_beliefs.shape[0],
        card_count,
        card_count,
        dtype=active_beliefs.dtype,
        device=active_beliefs.device,
    )
    for rank in torch.unique(active_ranks, sorted=True).detach().cpu().tolist():
        rank_ids = torch.nonzero(active_ranks == rank, as_tuple=False).flatten()
        equal_mats = _build_player_mats(
            active_beliefs.index_select(1, rank_ids),
            local_c0.index_select(0, rank_ids),
            local_c1.index_select(0, rank_ids),
            card_count,
        )
        card_mask = torch.ones(
            rank_ids.numel(),
            card_count,
            dtype=active_beliefs.dtype,
            device=active_beliefs.device,
        )
        card_mask.scatter_(1, local_c0.index_select(0, rank_ids)[:, None], 0.0)
        card_mask.scatter_(1, local_c1.index_select(0, rank_ids)[:, None], 0.0)
        rank_classes.append(
            _RankClassData(
                rank_ids=rank_ids,
                card_mask=card_mask,
                blocked0=local_c0.index_select(0, rank_ids),
                blocked1=local_c1.index_select(0, rank_ids),
                less_mats=less_mats.clone(),
                equal_mats=equal_mats,
            )
        )
        less_mats = less_mats + equal_mats
    return rank_classes


def _denominator_by_active_hand(
    all_mats: torch.Tensor,
    opponents: list[int],
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    card_count: int,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    batch_size, active_count = local_c0.shape
    out = torch.empty(batch_size, active_count, dtype=all_mats.dtype, device=all_mats.device)
    batch_ids = torch.arange(batch_size, device=all_mats.device)[:, None]
    for start in range(0, active_count, chunk_size):
        end = min(start + chunk_size, active_count)
        chunk = end - start
        flat_batch_ids = batch_ids.expand(-1, chunk).reshape(-1)
        blocked0 = local_c0[:, start:end].reshape(-1)
        blocked1 = local_c1[:, start:end].reshape(-1)
        card_mask = torch.ones(
            batch_size * chunk,
            card_count,
            dtype=all_mats.dtype,
            device=all_mats.device,
        )
        card_mask.scatter_(1, blocked0[:, None], 0.0)
        card_mask.scatter_(1, blocked1[:, None], 0.0)
        ranges = [
            _poly_range(
                [
                    _masked_batch_matrix(
                        all_mats[flat_batch_ids, opp],
                        card_mask,
                        blocked0,
                        blocked1,
                    )
                ]
            )
            for opp in opponents
        ]
        out[:, start:end] = _valid_mass_poly(ranges)[0].reshape(batch_size, chunk)
    return out


def _scatter_active_to_full_batched(
    active_values: torch.Tensor,
    active_ids: torch.Tensor,
) -> torch.Tensor:
    full = torch.zeros(
        *active_values.shape[:-1],
        NUM_HANDS,
        dtype=active_values.dtype,
        device=active_values.device,
    )
    index = active_ids[:, None, :].expand(-1, active_values.shape[1], -1)
    full.scatter_(2, index, active_values)
    return full


def _masked_batch_matrix(
    base: torch.Tensor,
    card_mask: torch.Tensor,
    blocked0: torch.Tensor,
    blocked1: torch.Tensor,
) -> _MaskedBatchMatrix:
    return _MaskedBatchMatrix(
        base=base,
        card_mask=card_mask,
        blocked0=blocked0,
        blocked1=blocked1,
    )


def _materialize_mat(mat: torch.Tensor | _MaskedBatchMatrix) -> torch.Tensor:
    if isinstance(mat, _MaskedBatchMatrix):
        return mat.materialize()
    return mat


def _gather_vector(vector: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if vector.dim() == 1:
        return vector.index_select(0, index)
    return vector.gather(1, index[:, None]).squeeze(1)


def _gather_matrix_columns(mat: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if mat.dim() == 2:
        return mat.index_select(1, index).T
    return mat.gather(2, index[:, None, None].expand(-1, mat.shape[1], 1)).squeeze(2)


def _gather_matrix_rows(mat: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if mat.dim() == 2:
        return mat.index_select(0, index)
    return mat.gather(1, index[:, None, None].expand(-1, 1, mat.shape[2])).squeeze(1)


def _gather_matrix_entries(
    mat: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    if mat.dim() == 2:
        return mat[row, col]
    ids = torch.arange(row.numel(), device=mat.device)
    return mat[ids, row, col]


def _gather_matrix_blocks(
    mat: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
) -> torch.Tensor:
    if mat.dim() == 2:
        return mat[row, col]
    ids = torch.arange(row.shape[0], device=mat.device)[:, None, None]
    return mat[ids, row, col]


def _masked_scalar(mat: _MaskedBatchMatrix) -> torch.Tensor:
    base_marginal = mat.base.sum(dim=-1)
    scalar = mat.base.triu(1).sum(dim=(-2, -1))
    blocked0_mass = _gather_vector(base_marginal, mat.blocked0)
    blocked1_mass = _gather_vector(base_marginal, mat.blocked1)
    blocked_edge = _gather_matrix_entries(mat.base, mat.blocked0, mat.blocked1)
    return scalar - blocked0_mass - blocked1_mass + blocked_edge


def _masked_marginal(mat: _MaskedBatchMatrix) -> torch.Tensor:
    base_marginal = mat.base.sum(dim=-1)
    blocked0_rows = _gather_matrix_columns(mat.base, mat.blocked0)
    blocked1_rows = _gather_matrix_columns(mat.base, mat.blocked1)
    marginal = base_marginal.unsqueeze(0) if mat.base.dim() == 2 else base_marginal
    marginal = marginal - blocked0_rows - blocked1_rows
    marginal = marginal.clone()
    marginal.scatter_(1, mat.blocked0[:, None], 0.0)
    marginal.scatter_(1, mat.blocked1[:, None], 0.0)
    return marginal


def _mat_scalar(mat: torch.Tensor | _MaskedBatchMatrix) -> torch.Tensor:
    if isinstance(mat, _MaskedBatchMatrix):
        return _masked_scalar(mat)
    return _sum_upper(mat)


def _mat_marginal(mat: torch.Tensor | _MaskedBatchMatrix) -> torch.Tensor:
    if isinstance(mat, _MaskedBatchMatrix):
        return _masked_marginal(mat)
    return mat.sum(dim=-1)


def _poly_range(mats: list[torch.Tensor | _MaskedBatchMatrix]) -> _PolyRange:
    return _PolyRange(
        mats=mats,
        scalars=[_mat_scalar(mat) for mat in mats],
        marginals=[_mat_marginal(mat) for mat in mats],
    )


def _zero_like(value: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(value)


def _poly_add(
    left: list[torch.Tensor],
    right: list[torch.Tensor],
    *,
    scale: float = 1.0,
) -> list[torch.Tensor]:
    degree = max(len(left), len(right))
    exemplar = left[0] if left else right[0]
    out = [_zero_like(exemplar) for _ in range(degree)]
    for idx, value in enumerate(left):
        out[idx] = out[idx] + value
    for idx, value in enumerate(right):
        out[idx] = out[idx] + float(scale) * value
    return out


def _poly_mul(left: list[torch.Tensor], right: list[torch.Tensor]) -> list[torch.Tensor]:
    out = [_zero_like(left[0]) for _ in range(len(left) + len(right) - 1)]
    for left_idx, left_value in enumerate(left):
        for right_idx, right_value in enumerate(right):
            out[left_idx + right_idx] = out[left_idx + right_idx] + left_value * right_value
    return out


def _pair_collision_poly(left: _PolyRange, right: _PolyRange) -> list[torch.Tensor]:
    out = [_zero_like(left.scalars[0]) for _ in range(len(left.mats) + len(right.mats) - 1)]
    for left_idx, left_mat in enumerate(left.mats):
        for right_idx, right_mat in enumerate(right.mats):
            degree = left_idx + right_idx
            marginal_overlap = (left.marginals[left_idx] * right.marginals[right_idx]).sum(
                dim=-1,
            )
            duplicate = _duplicate_product_upper([left_mat, right_mat])
            out[degree] = out[degree] + marginal_overlap - duplicate
    return out


def _delta_poly(center: _PolyRange, left: _PolyRange, right: _PolyRange) -> list[torch.Tensor]:
    degree_count = len(center.mats) + len(left.mats) + len(right.mats) - 2
    out = [_zero_like(center.scalars[0]) for _ in range(degree_count)]
    for center_idx, center_mat in enumerate(center.mats):
        center_tensor = _materialize_mat(center_mat)
        for left_idx, left_mat in enumerate(left.mats):
            left_tensor = _materialize_mat(left_mat)
            left_o = (
                left.marginals[left_idx].unsqueeze(-1)
                + left.marginals[left_idx].unsqueeze(-2)
                - left_tensor
            )
            for right_idx, right_mat in enumerate(right.mats):
                right_tensor = _materialize_mat(right_mat)
                right_o = (
                    right.marginals[right_idx].unsqueeze(-1)
                    + right.marginals[right_idx].unsqueeze(-2)
                    - right_tensor
                )
                degree = center_idx + left_idx + right_idx
                out[degree] = out[degree] + _sum_upper(center_tensor * left_o * right_o)
    return out


def _tri_poly(first: _PolyRange, second: _PolyRange, third: _PolyRange) -> list[torch.Tensor]:
    degree_count = len(first.mats) + len(second.mats) + len(third.mats) - 2
    out = [_zero_like(first.scalars[0]) for _ in range(degree_count)]
    for first_idx, first_mat in enumerate(first.mats):
        for second_idx, second_mat in enumerate(second.mats):
            for third_idx, third_mat in enumerate(third.mats):
                degree = first_idx + second_idx + third_idx
                star = (
                    first.marginals[first_idx]
                    * second.marginals[second_idx]
                    * third.marginals[third_idx]
                ).sum(dim=-1)
                duplicate = _duplicate_product_upper([first_mat, second_mat, third_mat])
                triangle = _trace_product_masked_or_dense(first_mat, second_mat, third_mat)
                out[degree] = out[degree] + star - duplicate + triangle
    return out


def _valid_mass_poly(ranges: list[_PolyRange]) -> list[torch.Tensor]:
    if len(ranges) == 1:
        return ranges[0].scalars
    if len(ranges) == 2:
        prod = _poly_mul(ranges[0].scalars, ranges[1].scalars)
        return _poly_add(prod, _pair_collision_poly(ranges[0], ranges[1]), scale=-1.0)
    if len(ranges) != 3:
        raise ValueError("_valid_mass_poly supports one to three opponents")

    first, second, third = ranges
    result = _poly_mul(_poly_mul(first.scalars, second.scalars), third.scalars)
    result = _poly_add(
        result,
        _poly_mul(_pair_collision_poly(first, second), third.scalars),
        scale=-1.0,
    )
    result = _poly_add(
        result,
        _poly_mul(_pair_collision_poly(first, third), second.scalars),
        scale=-1.0,
    )
    result = _poly_add(
        result,
        _poly_mul(_pair_collision_poly(second, third), first.scalars),
        scale=-1.0,
    )
    result = _poly_add(result, _delta_poly(first, second, third))
    result = _poly_add(result, _delta_poly(second, first, third))
    result = _poly_add(result, _delta_poly(third, first, second))
    result = _poly_add(result, _tri_poly(first, second, third), scale=-1.0)
    return result


def _integrate_unit_interval(coefficients: list[torch.Tensor]) -> torch.Tensor:
    total = _zero_like(coefficients[0])
    for degree, coefficient in enumerate(coefficients):
        total = total + coefficient / float(degree + 1)
    return total


def _sum_upper(mat: torch.Tensor) -> torch.Tensor:
    return mat.triu(1).sum(dim=(-2, -1))


def _duplicate_product_upper(
    mats: list[torch.Tensor | _MaskedBatchMatrix],
) -> torch.Tensor:
    if not mats or not _same_masked_blockers(mats):
        product = _materialize_mat(mats[0])
        for mat in mats[1:]:
            product = product * _materialize_mat(mat)
        return _sum_upper(product)

    first = mats[0]
    assert isinstance(first, _MaskedBatchMatrix)
    base_product = first.base
    for mat in mats[1:]:
        assert isinstance(mat, _MaskedBatchMatrix)
        base_product = base_product * mat.base
    total = base_product.triu(1).sum(dim=(-2, -1))
    row0 = _gather_matrix_rows(base_product, first.blocked0).sum(dim=-1)
    row1 = _gather_matrix_rows(base_product, first.blocked1).sum(dim=-1)
    edge = _gather_matrix_entries(base_product, first.blocked0, first.blocked1)
    return total - row0 - row1 + edge


def _same_masked_blockers(mats: list[torch.Tensor | _MaskedBatchMatrix]) -> bool:
    if not mats or not all(isinstance(mat, _MaskedBatchMatrix) for mat in mats):
        return False
    first = mats[0]
    assert isinstance(first, _MaskedBatchMatrix)
    return all(
        _same_tensor_view(first.blocked0, mat.blocked0)
        and _same_tensor_view(first.blocked1, mat.blocked1)
        for mat in mats[1:]
        if isinstance(mat, _MaskedBatchMatrix)
    )


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left is right:
        return True
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.storage_offset() == right.storage_offset()
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _trace_product_masked_or_dense(
    first: torch.Tensor | _MaskedBatchMatrix,
    second: torch.Tensor | _MaskedBatchMatrix,
    third: torch.Tensor | _MaskedBatchMatrix,
) -> torch.Tensor:
    mats = [first, second, third]
    if not _same_masked_blockers(mats):
        return _trace_product(
            _materialize_mat(first),
            _materialize_mat(second),
            _materialize_mat(third),
        )

    assert isinstance(first, _MaskedBatchMatrix)
    assert isinstance(second, _MaskedBatchMatrix)
    assert isinstance(third, _MaskedBatchMatrix)
    return _masked_trace_product_low_rank(first, second, third)


def _masked_trace_product_low_rank(
    first: _MaskedBatchMatrix,
    second: _MaskedBatchMatrix,
    third: _MaskedBatchMatrix,
) -> torch.Tensor:
    """Trace of three matrices with the same two card rows/cols deleted.

    The dense value is ``sum_ijk A_ij B_jk C_ki``.  Masking a hero hand deletes
    two card indices from each of ``i``, ``j``, and ``k``.  This computes the
    full trace once, then applies inclusion/exclusion over those two blocked
    indices with batched gathers.
    """
    first_second = first.base @ second.base
    second_third = second.base @ third.base
    third_first = third.base @ first.base

    diag_first_second_third = (first_second * third.base.transpose(-1, -2)).sum(dim=-1)
    diag_second_third_first = (second_third * first.base.transpose(-1, -2)).sum(dim=-1)
    diag_third_first_second = (third_first * second.base.transpose(-1, -2)).sum(dim=-1)
    full = diag_first_second_third.sum(dim=-1)

    blocked = torch.stack((first.blocked0, first.blocked1), dim=-1)
    if diag_first_second_third.dim() == 1:
        single = (
            diag_first_second_third.index_select(0, blocked.reshape(-1))
            .reshape(-1, 2)
            .sum(dim=-1)
            + diag_second_third_first.index_select(0, blocked.reshape(-1))
            .reshape(-1, 2)
            .sum(dim=-1)
            + diag_third_first_second.index_select(0, blocked.reshape(-1))
            .reshape(-1, 2)
            .sum(dim=-1)
        )
    else:
        single = (
            diag_first_second_third.gather(1, blocked).sum(dim=-1)
            + diag_second_third_first.gather(1, blocked).sum(dim=-1)
            + diag_third_first_second.gather(1, blocked).sum(dim=-1)
        )

    row = blocked[:, :, None].expand(-1, 2, 2)
    col = blocked[:, None, :].expand(-1, 2, 2)
    pair = (
        _gather_matrix_blocks(first.base, row, col) * _gather_matrix_blocks(second_third, col, row)
        + _gather_matrix_blocks(third.base, col, row)
        * _gather_matrix_blocks(first_second, row, col)
        + _gather_matrix_blocks(second.base, row, col)
        * _gather_matrix_blocks(third_first, col, row)
    ).sum(dim=(-2, -1))

    first_block = _gather_matrix_blocks(first.base, row, col)
    second_block = _gather_matrix_blocks(second.base, row, col)
    third_block = _gather_matrix_blocks(third.base, row, col)
    triple = torch.einsum("bij,bjk,bki->b", first_block, second_block, third_block)
    return full - single + pair - triple


def _trace_product(
    first: torch.Tensor,
    second: torch.Tensor,
    third: torch.Tensor,
) -> torch.Tensor:
    if first.dim() == 2:
        return torch.trace(first @ second @ third)
    return torch.einsum("...ij,...jk,...ki->...", first, second, third)


def tri_weight(
    pi_j: torch.Tensor,
    pi_k: torch.Tensor,
    pi_l: torch.Tensor,
) -> torch.Tensor:
    """Return the all-three-pairwise-collide weight for three range matrices."""
    m_j = pi_j.sum(dim=-1)
    m_k = pi_k.sum(dim=-1)
    m_l = pi_l.sum(dim=-1)
    star = (m_j * m_k * m_l).sum()
    duplicate = (pi_j * pi_k * pi_l).triu(1).sum()
    triangle = torch.trace(pi_j @ pi_k @ pi_l)
    return star - duplicate + triangle


def tri_weight_direct(
    pi_j: torch.Tensor,
    pi_k: torch.Tensor,
    pi_l: torch.Tensor,
) -> torch.Tensor:
    """Slow checker for ``tri_weight`` on small matrices."""
    card_count = pi_j.shape[0]
    total = pi_j.new_zeros(())
    for a in range(card_count):
        for b in range(a + 1, card_count):
            for c in range(card_count):
                for d in range(c + 1, card_count):
                    if not ({a, b} & {c, d}):
                        continue
                    for e in range(card_count):
                        for f in range(e + 1, card_count):
                            if ({a, b} & {e, f}) and ({c, d} & {e, f}):
                                total = total + pi_j[a, b] * pi_k[c, d] * pi_l[e, f]
    return total


def tri_identity_smoke_check(device: torch.device | None = None) -> dict[str, float]:
    """Return the three-card uniform check from the design note."""
    device = device or torch.device("cpu")
    pi = torch.ones(3, 3, dtype=torch.float64, device=device) - torch.eye(
        3,
        dtype=torch.float64,
        device=device,
    )
    direct = tri_weight_direct(pi, pi, pi)
    formula = tri_weight(pi, pi, pi)
    star_minus_duplicate = (pi.sum(dim=-1) ** 3).sum() - (pi * pi * pi).triu(1).sum()
    triangle = torch.trace(pi @ pi @ pi)
    return {
        "direct": float(direct.item()),
        "star_minus_duplicate": float(star_minus_duplicate.item()),
        "trace_triangle": float(triangle.item()),
        "formula_total": float(formula.item()),
    }


__all__ = [
    "ExactResult",
    "PerHandEquityResult",
    "PreparedShowdown",
    "brute_force_nway",
    "exact_nway_ie",
    "exact_nway_ie_axb_by_hand",
    "exact_nway_ie_by_hand",
    "exact_nway_ie_axb",
    "exact_nway_ie_chunked_f32",
    "exact_nway_ie_fast",
    "exact_nway_ie_fast_by_hand",
    "exact_threeway_ie",
    "tri_identity_smoke_check",
    "tri_weight",
    "tri_weight_direct",
]
