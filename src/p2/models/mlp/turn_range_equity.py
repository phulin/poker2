from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import torch
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA optimization
    triton = None
    tl = None

from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import rank_hands_triton, triton_is_available
from p2.models.mlp.better_features import PlayerContext, ValueScalarContext
from p2.models.mlp.mlp_features import MLPFeatures


RankGroupsFn = Callable[[torch.Tensor], torch.Tensor]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if triton is not None:

    @triton.jit
    def _turn_rank_cumulative_prefix_hu_kernel(
        beliefs_ptr,
        cache_index_ptr,
        sorted_hands_ptr,
        bin_offsets_ptr,
        rank_cumsum_ptr,
        total_rows: tl.constexpr,
        rank_bins: tl.constexpr,
        hand_count: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_BINS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        river = pid % 48
        player = (pid // 48) & 1
        row = pid // 96
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        hand_offs = tl.arange(0, BLOCK_H)
        base_offsets = (cache_row * 48 + river) * (rank_bins + 1)
        legal_total = tl.load(
            bin_offsets_ptr + base_offsets + rank_bins,
            mask=row < total_rows,
            other=0,
        )
        hand_mask = (row < total_rows) & (hand_offs < legal_total)
        hands = tl.load(
            sorted_hands_ptr + (cache_row * 48 + river) * hand_count + hand_offs,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        vals = tl.load(
            beliefs_ptr + (row * 2 + opp_player) * hand_count + hands,
            mask=hand_mask,
            other=0.0,
        ).to(tl.float32)
        prefix = tl.cumsum(vals, 0)
        bin_offs = tl.arange(0, BLOCK_BINS)
        bin_mask = bin_offs < rank_bins
        ends = tl.load(
            bin_offsets_ptr + base_offsets + bin_offs + 1,
            mask=(row < total_rows) & bin_mask,
            other=0,
        )
        end_idx = ends - 1
        end_idx = tl.maximum(end_idx, 0)
        cumulative = tl.gather(
            prefix,
            end_idx,
            0,
        )
        cumulative = tl.where((row < total_rows) & bin_mask & (ends > 0), cumulative, 0.0)
        out = ((row * 2 + player) * 48 + river) * rank_bins + bin_offs
        tl.store(rank_cumsum_ptr + out, cumulative, mask=(row < total_rows) & bin_mask)

    @triton.jit
    def _turn_rank_mass_bins_hu_kernel(
        beliefs_ptr,
        cache_index_ptr,
        sorted_hands_ptr,
        bin_offsets_ptr,
        rank_mass_ptr,
        total_rows: tl.constexpr,
        rank_bins: tl.constexpr,
        hand_count: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bin_id = pid % rank_bins
        river = (pid // rank_bins) % 48
        player = (pid // (rank_bins * 48)) & 1
        row = pid // (rank_bins * 96)
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        base_offset = (cache_row * 48 + river) * (rank_bins + 1)
        start = tl.load(
            bin_offsets_ptr + base_offset + bin_id,
            mask=row < total_rows,
            other=0,
        )
        end = tl.load(
            bin_offsets_ptr + base_offset + bin_id + 1,
            mask=row < total_rows,
            other=0,
        )
        count = end - start
        acc = tl.zeros((), dtype=tl.float32)
        for block_start in tl.static_range(0, hand_count, BLOCK_K):
            offs = block_start + tl.arange(0, BLOCK_K)
            mask = (row < total_rows) & (offs < count)
            hands = tl.load(
                sorted_hands_ptr
                + (cache_row * 48 + river) * hand_count
                + start
                + offs,
                mask=mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                beliefs_ptr + (row * 2 + opp_player) * hand_count + hands,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(vals, axis=0)
        tl.store(
            rank_mass_ptr + ((row * 2 + player) * 48 + river) * rank_bins + bin_id,
            acc,
            mask=row < total_rows,
        )

    @triton.jit
    def _turn_rank_mass_hu_kernel(
        beliefs_ptr,
        cache_index_ptr,
        board_ok_ptr,
        rivers_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_mass_ptr,
        total_rows: tl.constexpr,
        hand_count: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        river = pid_rp % 48
        player = (pid_rp // 48) & 1
        row = pid_rp // 96
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (row < total_rows) & (offs < hand_count)
        card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=-1)
        card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=-2)
        river_card = tl.load(
            rivers_ptr + cache_row * 48 + river,
            mask=row < total_rows,
            other=-3,
        )
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=mask,
            other=0,
        )
        ok = board_ok & (card_a != river_card) & (card_b != river_card)
        opp = tl.load(
            beliefs_ptr + (row * 2 + opp_player) * hand_count + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        ranks = tl.load(
            rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
            mask=mask,
            other=0,
        ).to(tl.int32)
        out = ((row * 2 + player) * 48 + river) * rank_bins + ranks
        tl.atomic_add(rank_mass_ptr + out, opp, sem="relaxed", mask=mask & ok)

    @triton.jit
    def _turn_rank_mass_kernel(
        beliefs_ptr,
        cache_index_ptr,
        board_ok_ptr,
        rivers_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_mass_ptr,
        total_rows: tl.constexpr,
        num_players: tl.constexpr,
        hand_count: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        river = pid_rp % 48
        player = (pid_rp // 48) % num_players
        row = pid_rp // (48 * num_players)
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (row < total_rows) & (offs < hand_count)
        card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=-1)
        card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=-2)
        river_card = tl.load(
            rivers_ptr + cache_row * 48 + river,
            mask=row < total_rows,
            other=-3,
        )
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=mask,
            other=0,
        )
        ok = board_ok & (card_a != river_card) & (card_b != river_card)
        own = tl.load(
            beliefs_ptr + (row * num_players + player) * hand_count + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        total = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for p in tl.static_range(0, num_players):
            total += tl.load(
                beliefs_ptr + (row * num_players + p) * hand_count + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        opp = total - own
        ranks = tl.load(
            rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
            mask=mask,
            other=0,
        ).to(tl.int32)
        out = ((row * num_players + player) * 48 + river) * rank_bins + ranks
        tl.atomic_add(
            rank_mass_ptr + out,
            opp,
            sem="relaxed",
            mask=mask & ok,
        )

    @triton.jit
    def _turn_rank_cumsum_kernel(
        rank_mass_ptr,
        total_groups: tl.constexpr,
        rank_bins: tl.constexpr,
        BLOCK_BINS: tl.constexpr,
    ):
        group = tl.program_id(0)
        offs = tl.arange(0, BLOCK_BINS)
        mask = offs < rank_bins
        ptrs = rank_mass_ptr + group * rank_bins + offs
        mass = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        cumulative = tl.cumsum(mass, 0)
        tl.store(ptrs, cumulative, mask=(group < total_groups) & mask)

    @triton.jit
    def _turn_baseline_from_cumsum_kernel(
        beliefs_ptr,
        context_ptr,
        cache_index_ptr,
        board_ok_ptr,
        hand_runout_ok_ptr,
        rivers_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_cumsum_ptr,
        baseline_ptr,
        total_rows: tl.constexpr,
        num_players: tl.constexpr,
        hand_count: tl.constexpr,
        context_stride: tl.constexpr,
        pot_index: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        has_hand_runout_ok: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_player: tl.constexpr,
        pos_scale: tl.constexpr,
        neg_scale: tl.constexpr,
        intercept: tl.constexpr,
        baseline_scale: tl.constexpr,
        use_pos_neg: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        player = pid_rp % num_players
        row = pid_rp // num_players
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = (row < total_rows) & (offs < hand_count)
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=hand_mask,
            other=0,
        )
        if not has_hand_runout_ok:
            card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=-1)
            card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=-2)
        lower_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        tie_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for river in tl.static_range(0, 48):
            ok = board_ok
            if has_hand_runout_ok:
                ok = tl.load(
                    hand_runout_ok_ptr + (cache_row * 48 + river) * hand_count + offs,
                    mask=hand_mask,
                    other=0,
                )
            else:
                river_card = tl.load(
                    rivers_ptr + cache_row * 48 + river,
                    mask=row < total_rows,
                    other=-3,
                )
                ok = board_ok & (card_a != river_card) & (card_b != river_card)
            ranks = tl.load(
                rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            ).to(tl.int32)
            base = ((row * num_players + player) * 48 + river) * rank_bins
            cum = tl.load(
                rank_cumsum_ptr + base + ranks,
                mask=hand_mask,
                other=0.0,
            ).to(tl.float32)
            prev_rank = ranks - 1
            prev = tl.load(
                rank_cumsum_ptr + base + prev_rank,
                mask=hand_mask & (ranks > 0),
                other=0.0,
            ).to(tl.float32)
            total = tl.load(
                rank_cumsum_ptr + base + (rank_bins - 1),
                mask=row < total_rows,
                other=0.0,
            ).to(tl.float32)
            lower_sum += tl.where(ok, prev, 0.0)
            tie_sum += tl.where(ok, cum - prev, 0.0)
            total_sum += tl.where(ok, total, 0.0)
        safe_total = tl.maximum(total_sum, 1.0e-8)
        equity = (2.0 * lower_sum + tie_sum - total_sum) / safe_total
        equity = tl.where(total_sum > 0.0, equity, 0.0)
        pot = tl.load(
            context_ptr + row * context_stride + pot_index,
            mask=row < total_rows,
            other=0.0,
        ).to(tl.float32)
        sdv = equity * pot
        if use_pos_neg:
            value = (
                tl.maximum(sdv, 0.0) * pos_scale
                + tl.minimum(sdv, 0.0) * neg_scale
                + intercept
            )
        else:
            value = sdv * baseline_scale
        tl.store(
            baseline_ptr + row * out_stride_row + player * out_stride_player + offs,
            value,
            mask=hand_mask,
        )


@dataclass(frozen=True)
class TurnRangeEquityConfig:
    rank_bins: int
    chunk_size: int
    blockers: bool
    baseline_scale: float
    pot_power: float
    pos_scale: float
    neg_scale: float
    intercept: float


@dataclass(frozen=True)
class TurnRangeEquityBoardCache:
    """Board-only turn equity data reusable while CFR beliefs change.

    Large tensors are cache-row aligned, usually one row per unique turn board.
    ``leaf_to_cache`` maps model leaf rows to cache rows when those differ.
    ``rank_groups`` is int16 and already binned to the configured rank-bin
    count.
    """

    board4: torch.Tensor
    rivers: torch.Tensor
    rank_groups: torch.Tensor
    board_ok: torch.Tensor
    hand_runout_ok: torch.Tensor | None
    leaf_to_cache: torch.Tensor | None = None
    sorted_hands: torch.Tensor | None = None
    bin_offsets: torch.Tensor | None = None

    def slice(self, rows: torch.Tensor) -> "TurnRangeEquityBoardCache":
        return TurnRangeEquityBoardCache(
            board4=self.board4,
            rivers=self.rivers,
            rank_groups=self.rank_groups,
            board_ok=self.board_ok,
            hand_runout_ok=self.hand_runout_ok,
            leaf_to_cache=(
                rows
                if self.leaf_to_cache is None
                else self.leaf_to_cache.index_select(0, rows)
            ),
            sorted_hands=self.sorted_hands,
            bin_offsets=self.bin_offsets,
        )


def river_rank_groups(board: torch.Tensor) -> torch.Tensor:
    if board.device.type == "cuda" and triton_is_available():
        try:
            hand_ranks, sorted_indices = rank_hands_triton(board.int())
        except Exception:
            hand_ranks, sorted_indices = rank_hands_torch(board.int())
    else:
        hand_ranks, sorted_indices = rank_hands_torch(board.int())
    sorted_ranks = hand_ranks.gather(1, sorted_indices.long())
    group_start = sorted_ranks[:, 1:] != sorted_ranks[:, :-1]
    group_start = torch.cat(
        (
            torch.ones(
                sorted_ranks.shape[0],
                1,
                device=sorted_ranks.device,
                dtype=torch.bool,
            ),
            group_start,
        ),
        dim=1,
    )
    sorted_groups = group_start.to(dtype=torch.long).cumsum(dim=1) - 1
    rank_groups = torch.empty_like(sorted_groups)
    rank_groups.scatter_(1, sorted_indices.long(), sorted_groups)
    return rank_groups


def turn_runout_boards(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    board4 = board[:, :4].long()
    cards = torch.arange(52, dtype=torch.long, device=board.device)
    cards_expanded = cards.view(1, 52).expand(board4.shape[0], -1)
    river_ok = (cards_expanded[:, :, None] != board4[:, None, :]).all(dim=2)
    removed_before = (board4[:, None, :] < cards_expanded[:, :, None]).sum(dim=2)
    river_slots = (cards_expanded - removed_before).clamp(min=0, max=47)
    river_src = torch.where(
        river_ok,
        cards_expanded + 1,
        torch.zeros_like(cards_expanded),
    )
    rivers = torch.zeros(board4.shape[0], 48, dtype=torch.long, device=board.device)
    rivers.scatter_reduce_(
        1,
        river_slots,
        river_src,
        reduce="amax",
        include_self=True,
    )
    rivers = (rivers - 1).clamp_min(0)
    full = torch.cat(
        (
            board4[:, None, :].expand(-1, 48, -1),
            rivers[:, :, None],
        ),
        dim=2,
    ).reshape(-1, 5)
    return rivers, full


def build_turn_range_equity_board_cache(
    board4: torch.Tensor,
    *,
    rank_bins: int,
    rank_groups_fn: RankGroupsFn = river_rank_groups,
    rank_chunk_size: int = 256,
    include_hand_runout_ok: bool = True,
    dedupe_boards: bool = False,
    balanced_bins: bool = True,
    include_sorted_bins: bool = False,
) -> TurnRangeEquityBoardCache:
    leaf_board = board4[:, :4].long()
    if dedupe_boards and leaf_board.shape[0] > 0:
        board, leaf_to_cache = torch.unique(
            leaf_board,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        leaf_to_cache = leaf_to_cache.to(torch.int32).contiguous()
    else:
        board = leaf_board
        leaf_to_cache = None
    device = board.device
    combos = hand_combos_tensor(device=device)
    card_a = combos[:, 0]
    card_b = combos[:, 1]
    rows = int(board.shape[0])
    rivers = torch.empty(rows, 48, dtype=torch.long, device=device)
    rank_groups = torch.empty(rows, 48, NUM_HANDS, dtype=torch.int16, device=device)
    for start in range(0, rows, int(rank_chunk_size)):
        end = min(rows, start + int(rank_chunk_size))
        chunk_rivers, full_boards = turn_runout_boards(board[start:end])
        rivers[start:end] = chunk_rivers
        chunk_groups = rank_groups_fn(full_boards).clamp_min(0)
        if balanced_bins:
            num_groups = chunk_groups.amax(dim=1, keepdim=True).add_(1).clamp_min_(1)
            chunk_groups = (chunk_groups * rank_bins) // num_groups
        chunk_groups = chunk_groups.clamp(max=rank_bins - 1)
        rank_groups[start:end] = chunk_groups.view(
            end - start,
            48,
            NUM_HANDS,
        ).to(torch.int16)
    board_ok = (
        (card_a[None, :, None] != board[:, None, :])
        & (card_b[None, :, None] != board[:, None, :])
    ).all(dim=2)
    hand_runout_ok = None
    sorted_hands = None
    bin_offsets = None
    river_blocked = None
    if include_hand_runout_ok:
        river_blocked = (card_a[None, None, :] == rivers[:, :, None]) | (
            card_b[None, None, :] == rivers[:, :, None]
        )
        hand_runout_ok = board_ok[:, None, :] & ~river_blocked
    if include_sorted_bins:
        if river_blocked is None:
            river_blocked = (card_a[None, None, :] == rivers[:, :, None]) | (
                card_b[None, None, :] == rivers[:, :, None]
            )
        legal = board_ok[:, None, :] & ~river_blocked
        hand_ids = torch.arange(NUM_HANDS, dtype=torch.int32, device=device)
        bin_counts = torch.zeros(rows, 48, rank_bins, dtype=torch.int32, device=device)
        bin_counts.scatter_add_(
            2,
            rank_groups.long(),
            legal.to(torch.int32),
        )
        bin_offsets = torch.empty(
            rows,
            48,
            rank_bins + 1,
            dtype=torch.int32,
            device=device,
        )
        bin_offsets[:, :, 0] = 0
        bin_offsets[:, :, 1:] = bin_counts.cumsum(dim=2)
        rank_major_key = torch.where(
            legal,
            rank_groups.long() * NUM_HANDS + hand_ids.view(1, 1, NUM_HANDS),
            torch.full(
                (),
                rank_bins * NUM_HANDS + NUM_HANDS,
                dtype=torch.long,
                device=device,
            ),
        )
        sorted_hands = torch.argsort(rank_major_key, dim=2).to(torch.int16)
    return TurnRangeEquityBoardCache(
        board4=board,
        rivers=rivers,
        rank_groups=rank_groups,
        board_ok=board_ok,
        hand_runout_ok=hand_runout_ok,
        leaf_to_cache=leaf_to_cache,
        sorted_hands=sorted_hands,
        bin_offsets=bin_offsets,
    )


def player_spr_context(context: torch.Tensor, num_players: int) -> torch.Tensor:
    base = ValueScalarContext.NUM_SCALAR_CONTEXT.value
    stride = PlayerContext.NUM_PLAYER_CONTEXT.value
    spr_idx = base + torch.arange(
        num_players,
        device=context.device,
        dtype=torch.long,
    ) * stride + PlayerContext.SPR.value
    return context.index_select(1, spr_idx)


def _can_use_triton_turn_baseline(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    config: TurnRangeEquityConfig,
    board_cache: TurnRangeEquityBoardCache | None,
) -> bool:
    return (
        triton is not None
        and player_beliefs.device.type == "cuda"
        and features.context.device.type == "cuda"
        and board_cache is not None
        and board_cache.rank_groups.is_contiguous()
        and board_cache.rivers.is_contiguous()
        and board_cache.board_ok.is_contiguous()
        and (
            board_cache.leaf_to_cache is None
            or board_cache.leaf_to_cache.is_contiguous()
        )
        and (
            board_cache.sorted_hands is None
            or board_cache.sorted_hands.is_contiguous()
        )
        and (
            board_cache.bin_offsets is None
            or board_cache.bin_offsets.is_contiguous()
        )
        and (
            board_cache.hand_runout_ok is None
            or board_cache.hand_runout_ok.is_contiguous()
        )
        and not config.blockers
        and config.pot_power == 1.0
        and config.rank_bins <= 256
    )


def turn_range_equity_baseline(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    *,
    config: TurnRangeEquityConfig,
    dtype: torch.dtype,
    board_cache: TurnRangeEquityBoardCache | None = None,
    rank_groups_fn: RankGroupsFn = river_rank_groups,
) -> torch.Tensor:
    if _can_use_triton_turn_baseline(
        player_beliefs,
        features,
        config,
        board_cache,
    ):
        assert board_cache is not None
        kernel_beliefs = player_beliefs.contiguous()
        kernel_context = features.context.contiguous()
        kernel_board_ok = board_cache.board_ok.contiguous()
        kernel_hand_runout_ok = (
            kernel_board_ok
            if board_cache.hand_runout_ok is None
            else board_cache.hand_runout_ok.contiguous()
        )
        has_hand_runout_ok = board_cache.hand_runout_ok is not None
        kernel_cache_index = (
            kernel_beliefs
            if board_cache.leaf_to_cache is None
            else board_cache.leaf_to_cache.contiguous()
        )
        has_cache_index = board_cache.leaf_to_cache is not None
        batch_size = int(kernel_beliefs.shape[0])
        num_players = int(kernel_beliefs.shape[1])
        rank_bins = int(config.rank_bins)
        baseline = kernel_beliefs.new_empty(
            batch_size,
            num_players,
            NUM_HANDS,
            dtype=dtype,
        )
        rank_mass = kernel_beliefs.new_empty(
            batch_size,
            num_players,
            48,
            rank_bins,
            dtype=torch.float32,
        )
        rank_mass.zero_()
        combos = hand_combos_tensor(device=kernel_beliefs.device)
        card_a = combos[:, 0].contiguous()
        card_b = combos[:, 1].contiguous()
        block_h_mass = _env_int("P2_TURN_EQUITY_BLOCK_H_MASS", 128)
        use_sorted_bins = (
            _env_int("P2_TURN_EQUITY_SORTED_BIN_KERNEL", 0) != 0
            and
            num_players == 2
            and board_cache.sorted_hands is not None
            and board_cache.bin_offsets is not None
        )
        use_prefix_bins = (
            _env_int("P2_TURN_EQUITY_PREFIX_KERNEL", 1) != 0
            and num_players == 2
            and board_cache.sorted_hands is not None
            and board_cache.bin_offsets is not None
        )
        rank_mass_is_cumulative = False
        if use_prefix_bins:
            _turn_rank_cumulative_prefix_hu_kernel[(batch_size * 2 * 48,)](
                kernel_beliefs,
                kernel_cache_index,
                board_cache.sorted_hands,
                board_cache.bin_offsets,
                rank_mass,
                batch_size,
                rank_bins,
                NUM_HANDS,
                has_cache_index,
                BLOCK_H=triton.next_power_of_2(NUM_HANDS),
                BLOCK_BINS=triton.next_power_of_2(rank_bins),
                num_warps=8,
            )
            rank_mass_is_cumulative = True
        elif use_sorted_bins:
            _turn_rank_mass_bins_hu_kernel[(batch_size * 2 * 48 * rank_bins,)](
                kernel_beliefs,
                kernel_cache_index,
                board_cache.sorted_hands,
                board_cache.bin_offsets,
                rank_mass,
                batch_size,
                rank_bins,
                NUM_HANDS,
                has_cache_index,
                BLOCK_K=_env_int("P2_TURN_EQUITY_BIN_BLOCK_K", 128),
                num_warps=4,
            )
        elif num_players == 2:
            _turn_rank_mass_hu_kernel[
                (
                    batch_size * 2 * 48,
                    triton.cdiv(NUM_HANDS, block_h_mass),
                )
            ](
                kernel_beliefs,
                kernel_cache_index,
                kernel_board_ok,
                board_cache.rivers,
                card_a,
                card_b,
                board_cache.rank_groups,
                rank_mass,
                batch_size,
                NUM_HANDS,
                rank_bins,
                has_cache_index,
                BLOCK_H=block_h_mass,
                num_warps=4,
            )
        else:
            _turn_rank_mass_kernel[
                (
                    batch_size * num_players * 48,
                    triton.cdiv(NUM_HANDS, block_h_mass),
                )
            ](
                kernel_beliefs,
                kernel_cache_index,
                kernel_board_ok,
                board_cache.rivers,
                card_a,
                card_b,
                board_cache.rank_groups,
                rank_mass,
                batch_size,
                num_players,
                NUM_HANDS,
                rank_bins,
                has_cache_index,
                BLOCK_H=block_h_mass,
                num_warps=4,
            )
        if not rank_mass_is_cumulative:
            block_bins = triton.next_power_of_2(rank_bins)
            _turn_rank_cumsum_kernel[(batch_size * num_players * 48,)](
                rank_mass,
                batch_size * num_players * 48,
                rank_bins,
                BLOCK_BINS=block_bins,
                num_warps=4,
            )
        block_h_out = _env_int("P2_TURN_EQUITY_BLOCK_H_OUT", 128)
        _turn_baseline_from_cumsum_kernel[
            (
                batch_size * num_players,
                triton.cdiv(NUM_HANDS, block_h_out),
            )
        ](
            kernel_beliefs,
            kernel_context,
            kernel_cache_index,
            kernel_board_ok,
            kernel_hand_runout_ok,
            board_cache.rivers,
            card_a,
            card_b,
            board_cache.rank_groups,
            rank_mass,
            baseline,
            batch_size,
            num_players,
            NUM_HANDS,
            int(kernel_context.stride(0)),
            ValueScalarContext.POT.value,
            rank_bins,
            has_cache_index,
            has_hand_runout_ok,
            int(baseline.stride(0)),
            int(baseline.stride(1)),
            float(config.pos_scale),
            float(config.neg_scale),
            float(config.intercept),
            float(config.baseline_scale),
            bool(config.pos_scale >= 0.0),
            BLOCK_H=block_h_out,
            num_warps=4,
        )
        return baseline

    if board_cache is not None and board_cache.hand_runout_ok is None:
        checks = {
            "triton": triton is not None,
            "beliefs_cuda": player_beliefs.device.type == "cuda",
            "context_cuda": features.context.device.type == "cuda",
            "rank_groups_contiguous": board_cache.rank_groups.is_contiguous(),
            "rivers_contiguous": board_cache.rivers.is_contiguous(),
            "board_ok_contiguous": board_cache.board_ok.is_contiguous(),
            "leaf_to_cache_contiguous": (
                board_cache.leaf_to_cache is None
                or board_cache.leaf_to_cache.is_contiguous()
            ),
            "sorted_hands_contiguous": (
                board_cache.sorted_hands is None
                or board_cache.sorted_hands.is_contiguous()
            ),
            "bin_offsets_contiguous": (
                board_cache.bin_offsets is None
                or board_cache.bin_offsets.is_contiguous()
            ),
            "hand_runout_ok_contiguous": (
                board_cache.hand_runout_ok is None
                or board_cache.hand_runout_ok.is_contiguous()
            ),
            "not_blockers": not config.blockers,
            "pot_power_one": config.pot_power == 1.0,
            "rank_bins": config.rank_bins <= 256,
        }
        failed = ",".join(name for name, ok in checks.items() if not ok)
        raise ValueError(f"Compact turn equity cache requires Triton path: {failed}")

    baseline, _ = turn_range_equity_features(
        player_beliefs,
        features,
        config=config,
        dtype=dtype,
        board_cache=board_cache,
        rank_groups_fn=rank_groups_fn,
    )
    return baseline


def _turn_range_equity_chunk(
    *,
    beliefs: torch.Tensor,
    context: torch.Tensor,
    cache: TurnRangeEquityBoardCache,
    config: TurnRangeEquityConfig,
    num_players: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = int(beliefs.shape[0])
    rank_bins = int(config.rank_bins)
    opponents = beliefs.sum(dim=1, keepdim=True) - beliefs
    hand_runout_ok = cache.hand_runout_ok
    if hand_runout_ok is None:
        raise ValueError("PyTorch turn equity path requires hand_runout_ok in cache")
    if cache.leaf_to_cache is not None:
        cache_rows = cache.leaf_to_cache.to(device=beliefs.device, dtype=torch.long)
        rank_groups = cache.rank_groups.index_select(0, cache_rows).long()
        hand_runout_ok = hand_runout_ok.index_select(0, cache_rows)
    else:
        rank_groups = cache.rank_groups.long()
    rank_idx = rank_groups[:, None, :, :].expand(-1, num_players, -1, -1)
    opp_weights = opponents[:, :, None, :] * hand_runout_ok[:, None, :, :]
    flat_rank_idx = rank_idx.reshape(-1, NUM_HANDS)
    flat_opp_weights = opp_weights.reshape(-1, NUM_HANDS)
    rank_mass = beliefs.new_zeros(flat_opp_weights.shape[0], rank_bins)
    rank_mass.scatter_add_(1, flat_rank_idx, flat_opp_weights)
    cumulative = rank_mass.cumsum(dim=1)
    tie = rank_mass.gather(1, flat_rank_idx)
    lower = cumulative.gather(1, flat_rank_idx) - tie
    per_river_total = rank_mass.sum(dim=1)

    if config.blockers:
        combos = hand_combos_tensor(device=beliefs.device)
        card_a = combos[:, 0]
        card_b = combos[:, 1]
        card_rank_bins = 52 * rank_bins
        card_rank_mass = beliefs.new_zeros(
            flat_opp_weights.shape[0],
            card_rank_bins,
        )
        card_a_idx = card_a.view(1, NUM_HANDS).expand_as(flat_rank_idx)
        card_b_idx = card_b.view(1, NUM_HANDS).expand_as(flat_rank_idx)
        flat_idx_a = card_a_idx * rank_bins + flat_rank_idx
        flat_idx_b = card_b_idx * rank_bins + flat_rank_idx
        card_rank_mass.scatter_add_(1, flat_idx_a, flat_opp_weights)
        card_rank_mass.scatter_add_(1, flat_idx_b, flat_opp_weights)
        card_rank_view = card_rank_mass.view(
            flat_opp_weights.shape[0],
            52,
            rank_bins,
        )
        card_mass = card_rank_view.sum(dim=2)
        card_rank_cumulative = card_rank_view.cumsum(dim=2).reshape(
            flat_opp_weights.shape[0],
            card_rank_bins,
        )
        card_tie_a = card_rank_mass.gather(1, flat_idx_a)
        card_tie_b = card_rank_mass.gather(1, flat_idx_b)
        card_lower_a = card_rank_cumulative.gather(1, flat_idx_a) - card_tie_a
        card_lower_b = card_rank_cumulative.gather(1, flat_idx_b) - card_tie_b
        same_combo_mass = flat_opp_weights
        blocked_tie = card_tie_a + card_tie_b - same_combo_mass
        blocked_lower = card_lower_a + card_lower_b
        blocked_total = (
            card_mass.gather(1, card_a_idx)
            + card_mass.gather(1, card_b_idx)
            - same_combo_mass
        )
        tie = (tie - blocked_tie).clamp_min(0.0)
        lower = (lower - blocked_lower).clamp_min(0.0)
        total = (per_river_total[:, None] - blocked_total).clamp_min(0.0).view(
            chunk,
            num_players,
            48,
            NUM_HANDS,
        )
    else:
        total = per_river_total.view(chunk, num_players, 48, 1)

    hero_ok = hand_runout_ok[:, None, :, :].to(dtype=beliefs.dtype)
    lower_sum = (lower.view(chunk, num_players, 48, NUM_HANDS) * hero_ok).sum(dim=2)
    tie_sum = (tie.view(chunk, num_players, 48, NUM_HANDS) * hero_ok).sum(dim=2)
    total_sum = (total * hero_ok).sum(dim=2)
    safe_total = total_sum.clamp_min(1e-8)
    equity_score = (2.0 * lower_sum + tie_sum - total_sum) / safe_total
    equity_score = torch.where(
        total_sum > 0.0,
        equity_score,
        torch.zeros_like(equity_score),
    )

    pot_scale = context[:, ValueScalarContext.POT.value].float()
    if config.pot_power != 1.0:
        pot_scale = pot_scale.clamp_min(0.0).pow(config.pot_power)
    sdv = equity_score * pot_scale[:, None, None]
    if config.pos_scale >= 0.0:
        value = (
            sdv.clamp_min(0.0) * config.pos_scale
            + sdv.clamp_max(0.0) * config.neg_scale
            + config.intercept
        )
    else:
        value = sdv * config.baseline_scale

    valid_rivers = hand_runout_ok.sum(dim=1).clamp_min(1)
    avg_total_mass = total_sum / valid_rivers[:, None, :].to(dtype=total_sum.dtype)
    spr = player_spr_context(context, num_players).float()
    feature_values = torch.stack(
        (
            sdv,
            beliefs,
            avg_total_mass,
            torch.zeros_like(avg_total_mass),
            pot_scale[:, None, None].expand_as(equity_score),
            spr[:, :, None].expand_as(equity_score),
        ),
        dim=-1,
    )
    return value, feature_values


def turn_range_equity_features(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    *,
    config: TurnRangeEquityConfig,
    dtype: torch.dtype,
    board_cache: TurnRangeEquityBoardCache | None = None,
    rank_groups_fn: RankGroupsFn = river_rank_groups,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = player_beliefs.new_zeros(
        player_beliefs.shape[0],
        player_beliefs.shape[1],
        NUM_HANDS,
        dtype=dtype,
    )
    feature_values = player_beliefs.new_zeros(
        player_beliefs.shape[0],
        player_beliefs.shape[1],
        NUM_HANDS,
        6,
        dtype=dtype,
    )
    rows = torch.arange(player_beliefs.shape[0], device=player_beliefs.device)
    board4_all = features.board[:, :4].long()

    for start in range(0, int(rows.shape[0]), int(config.chunk_size)):
        chunk_rows = rows[start : start + int(config.chunk_size)]
        beliefs = player_beliefs[chunk_rows].float()
        if board_cache is None:
            chunk_cache = build_turn_range_equity_board_cache(
                board4_all[chunk_rows],
                rank_bins=config.rank_bins,
                rank_groups_fn=rank_groups_fn,
            )
        else:
            chunk_cache = board_cache.slice(chunk_rows)
        value, stacked_features = _turn_range_equity_chunk(
            beliefs=beliefs,
            context=features.context[chunk_rows],
            cache=chunk_cache,
            config=config,
            num_players=player_beliefs.shape[1],
        )
        baseline[chunk_rows] = value.to(dtype=dtype)
        feature_values[chunk_rows] = stacked_features.to(dtype=dtype)
    return baseline, feature_values
