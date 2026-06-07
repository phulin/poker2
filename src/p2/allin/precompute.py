from __future__ import annotations

import argparse
import io
import math
import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    board_allowed_hands,
    canonical_full_boards_with_weights,
    combo_to_preflop_class_tensor,
    combo_suit_permutation_tensor,
    hand_combos_tensor,
    preflop_class_multiplicity_tensor,
)
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import triton_is_available
from p2.search.allin_payoff import I16_SCALE, load_preflop_payoff_i16

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


TOTAL_PREFLOP_BOARDS = math.comb(52, 5)
TOTAL_TRIPLE_BOARDS = math.comb(46, 5)
_MAX_CLASS_MULTIPLICITY = 12
_FORMAT = "p2.allin.preflop_allin_169.v1"
_QUANTIZED_FORMAT = "p2.allin.preflop_allin_169.u16.v1"
_U16_SCALE = float(torch.iinfo(torch.uint16).max)
_COMPILED_3P_SHARE0_INNER: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None


@dataclass
class PreflopAllIn169Tensors:
    """Exact class-level preflop all-in showdown tensors."""

    allin_2p_share0: torch.Tensor | None = None
    allin_2p_count: torch.Tensor | None = None
    allin_3p_share0: torch.Tensor | None = None
    allin_3p_count: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def payload(self) -> dict[str, object]:
        out: dict[str, object] = {"format": _FORMAT, "metadata": self.metadata or {}}
        if self.allin_2p_share0 is not None:
            out["allin_2p_share0"] = self.allin_2p_share0
        if self.allin_2p_count is not None:
            out["allin_2p_count"] = self.allin_2p_count
        if self.allin_3p_share0 is not None:
            out["allin_3p_share0"] = self.allin_3p_share0
        if self.allin_3p_count is not None:
            out["allin_3p_count"] = self.allin_3p_count
        return out


def _rank_hands(board: torch.Tensor, *, use_triton: bool) -> torch.Tensor:
    if use_triton:
        from p2.env.rules_triton import rank_hand_scores_triton

        ranks = rank_hand_scores_triton(board.int())
    else:
        ranks, _ = rank_hands_torch(board.int())
    return ranks.to(torch.int32).contiguous()


def _sample_full_boards(
    count: int,
    *,
    seed: int,
    board_chunk: int,
) -> Iterator[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for start in range(0, count, board_chunk):
        cur = min(board_chunk, count - start)
        keys = torch.rand(cur, 52, generator=generator)
        yield keys.topk(5, dim=1).indices.sort(dim=1).values.to(torch.long)


def _exhaustive_full_boards(*, board_chunk: int) -> Iterator[torch.Tensor]:
    buf: list[tuple[int, int, int, int, int]] = []
    for board in combinations(range(52), 5):
        buf.append(board)
        if len(buf) == board_chunk:
            yield torch.tensor(buf, dtype=torch.long)
            buf.clear()
    if buf:
        yield torch.tensor(buf, dtype=torch.long)


def _canonical_full_board_chunks(
    *,
    board_chunk: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    boards, weights = canonical_full_boards_with_weights()
    for start in range(0, boards.shape[0], board_chunk):
        end = min(start + board_chunk, boards.shape[0])
        yield boards[start:end], weights[start:end]


def _weighted_board_chunks(
    *,
    sample_boards: int | None,
    seed: int,
    board_chunk: int,
    canonical_boards: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if board_chunk <= 0:
        raise ValueError("board_chunk must be positive")
    if sample_boards is not None:
        if sample_boards <= 0:
            raise ValueError("sample_boards must be positive")
        for boards in _sample_full_boards(
            sample_boards,
            seed=seed,
            board_chunk=board_chunk,
        ):
            yield boards, torch.ones(boards.shape[0], dtype=torch.float32)
        return
    if canonical_boards:
        yield from _canonical_full_board_chunks(board_chunk=board_chunk)
        return
    for boards in _exhaustive_full_boards(board_chunk=board_chunk):
        yield boards, torch.ones(boards.shape[0], dtype=torch.float32)


def _board_chunks(
    *,
    sample_boards: int | None,
    seed: int,
    board_chunk: int,
) -> Iterator[torch.Tensor]:
    for boards, _weights in _weighted_board_chunks(
        sample_boards=sample_boards,
        seed=seed,
        board_chunk=board_chunk,
        canonical_boards=False,
    ):
        yield boards


def _class_combo_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    class_ids = combo_to_preflop_class_tensor(device=torch.device("cpu"))
    table = torch.full(
        (PREFLOP_HANDS, _MAX_CLASS_MULTIPLICITY),
        -1,
        dtype=torch.long,
    )
    mask = torch.zeros_like(table, dtype=torch.bool)
    for class_id in range(PREFLOP_HANDS):
        combos = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        count = int(combos.numel())
        table[class_id, :count] = combos
        mask[class_id, :count] = True
    return table.to(device), mask.to(device)


def _combo_masks(device: torch.device) -> torch.Tensor:
    combos = hand_combos_tensor(device=device)
    one = torch.ones((), dtype=torch.int64, device=device)
    return (
        torch.bitwise_left_shift(one, combos[:, 0])
        | torch.bitwise_left_shift(one, combos[:, 1])
    ).contiguous()


def _combo_card_masks26(device: torch.device) -> torch.Tensor:
    combos = hand_combos_tensor(device=device)
    one = torch.ones((), dtype=torch.int64, device=device)
    low_shift = combos.clamp_max(25)
    high_shift = (combos - 26).clamp_min(0)
    low = torch.where(
        combos < 26,
        torch.bitwise_left_shift(one, low_shift),
        torch.zeros((), dtype=torch.int64, device=device),
    ).sum(dim=1)
    high = torch.where(
        combos >= 26,
        torch.bitwise_left_shift(one, high_shift),
        torch.zeros((), dtype=torch.int64, device=device),
    ).sum(dim=1)
    return torch.stack((low, high), dim=1).to(torch.int32).contiguous()


def _board_card_masks26(boards: torch.Tensor) -> torch.Tensor:
    boards = boards.long()
    one = torch.ones((), dtype=torch.int64, device=boards.device)
    low_shift = boards.clamp_max(25)
    high_shift = (boards - 26).clamp_min(0)
    low = torch.where(
        boards < 26,
        torch.bitwise_left_shift(one, low_shift),
        torch.zeros((), dtype=torch.int64, device=boards.device),
    ).sum(dim=1)
    high = torch.where(
        boards >= 26,
        torch.bitwise_left_shift(one, high_shift),
        torch.zeros((), dtype=torch.int64, device=boards.device),
    ).sum(dim=1)
    return torch.stack((low, high), dim=1).to(torch.int32).contiguous()


def _tuple_card_masks26(
    tuples: torch.Tensor,
    *,
    combo_card_masks: torch.Tensor,
) -> torch.Tensor:
    masks = combo_card_masks.index_select(0, tuples.reshape(-1)).reshape(-1, 3, 2)
    return torch.bitwise_or(
        torch.bitwise_or(masks[:, 0], masks[:, 1]),
        masks[:, 2],
    ).contiguous()


def _mirror_3p_caller_order_(flat: torch.Tensor) -> None:
    values = flat.reshape(PREFLOP_HANDS, PREFLOP_HANDS, PREFLOP_HANDS)
    lower = torch.tril_indices(
        PREFLOP_HANDS,
        PREFLOP_HANDS,
        offset=-1,
        device=flat.device,
    )
    values[:, lower[0], lower[1]] = values[:, lower[1], lower[0]]


def _canonicalize_tuple_suit_orbits(
    tuples: torch.Tensor,
    *,
    combo_suit_permutation: torch.Tensor,
    combo_class_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuples.numel() == 0:
        return (
            tuples,
            torch.empty(0, dtype=torch.float32, device=tuples.device),
            torch.empty(0, dtype=torch.long, device=tuples.device),
        )

    h0 = combo_suit_permutation[:, tuples[:, 0]].to(torch.int64)
    h1 = combo_suit_permutation[:, tuples[:, 1]].to(torch.int64)
    h2 = combo_suit_permutation[:, tuples[:, 2]].to(torch.int64)
    keys = (h0 * NUM_HANDS + h1) * NUM_HANDS + h2
    canonical_keys = keys.min(dim=0).values
    unique_keys, counts = torch.unique(canonical_keys, sorted=True, return_counts=True)
    c0 = unique_keys // (NUM_HANDS * NUM_HANDS)
    rem = unique_keys - c0 * NUM_HANDS * NUM_HANDS
    c1 = rem // NUM_HANDS
    c2 = rem - c1 * NUM_HANDS
    canonical_tuples = torch.stack((c0, c1, c2), dim=1).to(torch.long).contiguous()
    classes = combo_class_ids.index_select(0, canonical_tuples.reshape(-1)).reshape(
        -1,
        3,
    )
    owners = (classes[:, 0] * PREFLOP_HANDS + classes[:, 1]) * PREFLOP_HANDS + classes[
        :, 2
    ]
    return canonical_tuples, counts.to(torch.float32).contiguous(), owners.contiguous()


def _linear_class_ids(
    players: int,
    *,
    class_ids: Sequence[int] | None,
    device: torch.device,
) -> torch.Tensor:
    if players not in (2, 3):
        raise ValueError("players must be 2 or 3")
    if class_ids is None:
        return torch.arange(PREFLOP_HANDS**players, dtype=torch.long, device=device)

    classes = torch.tensor(tuple(class_ids), dtype=torch.long, device=device)
    if classes.numel() == 0:
        raise ValueError("class_ids cannot be empty")
    if bool(((classes < 0) | (classes >= PREFLOP_HANDS)).any()):
        raise ValueError(f"class ids must be in [0, {PREFLOP_HANDS})")
    grids = torch.cartesian_prod(*([classes] * players))
    if players == 2:
        return grids[:, 0] * PREFLOP_HANDS + grids[:, 1]
    return (grids[:, 0] * PREFLOP_HANDS + grids[:, 1]) * PREFLOP_HANDS + grids[:, 2]


def _linear_3p_share0_class_ids(
    *,
    class_ids: Sequence[int] | None,
    device: torch.device,
    opponent_symmetry: bool,
) -> torch.Tensor:
    if not opponent_symmetry:
        return _linear_class_ids(3, class_ids=class_ids, device=device)
    if class_ids is None:
        classes = torch.arange(PREFLOP_HANDS, dtype=torch.long, device=device)
    else:
        classes = torch.tensor(tuple(class_ids), dtype=torch.long, device=device)
        if classes.numel() == 0:
            raise ValueError("class_ids cannot be empty")
        if bool(((classes < 0) | (classes >= PREFLOP_HANDS)).any()):
            raise ValueError(f"class ids must be in [0, {PREFLOP_HANDS})")
        classes = classes.unique(sorted=True)

    caller_pairs = torch.triu_indices(
        classes.numel(),
        classes.numel(),
        offset=0,
        device=device,
    )
    c1 = classes.index_select(0, caller_pairs[0])
    c2 = classes.index_select(0, caller_pairs[1])
    c0 = classes[:, None].expand(-1, c1.numel()).reshape(-1)
    c1 = c1[None, :].expand(classes.numel(), -1).reshape(-1)
    c2 = c2[None, :].expand(classes.numel(), -1).reshape(-1)
    return (c0 * PREFLOP_HANDS + c1) * PREFLOP_HANDS + c2


def _decode_linear_classes(linear: torch.Tensor, players: int) -> torch.Tensor:
    if players == 2:
        c0 = linear // PREFLOP_HANDS
        c1 = linear - c0 * PREFLOP_HANDS
        return torch.stack((c0, c1), dim=1)
    c0 = linear // (PREFLOP_HANDS * PREFLOP_HANDS)
    rem = linear - c0 * PREFLOP_HANDS * PREFLOP_HANDS
    c1 = rem // PREFLOP_HANDS
    c2 = rem - c1 * PREFLOP_HANDS
    return torch.stack((c0, c1, c2), dim=1)


def allin_2p_169_share0_from_combo_payoff(
    payoff: torch.Tensor,
    *,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse a combo-level preflop payoff table to seat-0 ``[169, 169]`` share."""
    if tuple(payoff.shape) != (NUM_HANDS, NUM_HANDS):
        raise ValueError(
            f"expected payoff shape {(NUM_HANDS, NUM_HANDS)}, got {tuple(payoff.shape)}"
        )

    device = payoff.device
    class_ids = combo_to_preflop_class_tensor(device=device)
    combo_masks = _combo_masks(device)
    compatible = (combo_masks[:, None] & combo_masks[None, :]) == 0
    pair_ids = (class_ids[:, None] * PREFLOP_HANDS + class_ids[None, :]).reshape(-1)
    flat_compatible = compatible.reshape(-1)
    flat_ids = pair_ids[flat_compatible]

    sums = torch.zeros(PREFLOP_HANDS * PREFLOP_HANDS, dtype=torch.float64, device=device)
    counts = torch.zeros_like(sums, dtype=torch.int64)
    values = 0.5 * (payoff.reshape(-1)[flat_compatible].to(torch.float64) / float(scale) + 1.0)
    sums.scatter_add_(0, flat_ids, values)
    counts.scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.int64))
    avg = torch.where(
        counts > 0,
        sums / counts.to(torch.float64).clamp_min(1),
        torch.zeros_like(sums),
    )
    return (
        avg.reshape(PREFLOP_HANDS, PREFLOP_HANDS).to(torch.float32).contiguous(),
        counts.reshape(PREFLOP_HANDS, PREFLOP_HANDS).contiguous(),
    )


def _class_tuple_combos(
    class_tuple: torch.Tensor,
    *,
    class_combo: torch.Tensor,
    class_combo_mask: torch.Tensor,
    combo_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    players = int(class_tuple.shape[1])
    gathered = class_combo.index_select(0, class_tuple.reshape(-1)).reshape(
        class_tuple.shape[0],
        players,
        _MAX_CLASS_MULTIPLICITY,
    )
    gathered_mask = class_combo_mask.index_select(0, class_tuple.reshape(-1)).reshape(
        class_tuple.shape[0],
        players,
        _MAX_CLASS_MULTIPLICITY,
    )
    if players == 2:
        h0 = gathered[:, 0, :, None]
        h1 = gathered[:, 1, None, :]
        m0 = gathered_mask[:, 0, :, None]
        m1 = gathered_mask[:, 1, None, :]
        cm0 = combo_masks.index_select(0, h0.clamp_min(0).reshape(-1)).reshape_as(h0)
        cm1 = combo_masks.index_select(0, h1.clamp_min(0).reshape(-1)).reshape_as(h1)
        valid = m0 & m1 & ((cm0 & cm1) == 0)
        combos = torch.stack(
            (
                h0.expand(-1, -1, _MAX_CLASS_MULTIPLICITY),
                h1.expand(-1, _MAX_CLASS_MULTIPLICITY, -1),
            ),
            dim=-1,
        )
        return combos.reshape(class_tuple.shape[0], -1, players), valid.reshape(
            class_tuple.shape[0], -1
        )

    h0 = gathered[:, 0, :, None, None]
    h1 = gathered[:, 1, None, :, None]
    h2 = gathered[:, 2, None, None, :]
    m0 = gathered_mask[:, 0, :, None, None]
    m1 = gathered_mask[:, 1, None, :, None]
    m2 = gathered_mask[:, 2, None, None, :]
    cm0 = combo_masks.index_select(0, h0.clamp_min(0).reshape(-1)).reshape_as(h0)
    cm1 = combo_masks.index_select(0, h1.clamp_min(0).reshape(-1)).reshape_as(h1)
    cm2 = combo_masks.index_select(0, h2.clamp_min(0).reshape(-1)).reshape_as(h2)
    valid = (
        m0
        & m1
        & m2
        & ((cm0 & cm1) == 0)
        & ((cm0 & cm2) == 0)
        & ((cm1 & cm2) == 0)
    )
    combos = torch.stack(
        (
            h0.expand(-1, -1, _MAX_CLASS_MULTIPLICITY, _MAX_CLASS_MULTIPLICITY),
            h1.expand(-1, _MAX_CLASS_MULTIPLICITY, -1, _MAX_CLASS_MULTIPLICITY),
            h2.expand(-1, _MAX_CLASS_MULTIPLICITY, _MAX_CLASS_MULTIPLICITY, -1),
        ),
        dim=-1,
    )
    return combos.reshape(class_tuple.shape[0], -1, players), valid.reshape(
        class_tuple.shape[0], -1
    )


def _allin_3p_share0_accumulate_inner(
    ranks: torch.Tensor,
    allowed: torch.Tensor,
    tuples: torch.Tensor,
    board_weights: torch.Tensor,
    tuple_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h0 = tuples[:, 0]
    h1 = tuples[:, 1]
    h2 = tuples[:, 2]
    r0 = ranks.index_select(1, h0)
    r1 = ranks.index_select(1, h1)
    r2 = ranks.index_select(1, h2)
    valid = (
        allowed.index_select(1, h0)
        & allowed.index_select(1, h1)
        & allowed.index_select(1, h2)
    )

    # Store shares in sixths: win=6, two-way tie=3, three-way tie=2. Keep this
    # as fp32 on CUDA; int64 scatter/add is much slower and the artifact share
    # is emitted as fp32.
    share6 = torch.zeros(r0.shape, dtype=torch.float32, device=ranks.device)
    share6 = torch.where((r0 > r1) & (r0 > r2), 6.0, share6)
    share6 = torch.where((r0 == r1) & (r0 > r2), 3.0, share6)
    share6 = torch.where((r0 == r2) & (r0 > r1), 3.0, share6)
    share6 = torch.where((r0 == r1) & (r0 == r2), 2.0, share6)
    share6 = torch.where(valid, share6, torch.zeros_like(share6))

    weights = board_weights.to(dtype=torch.float32, device=ranks.device)[:, None]
    tuple_weight = tuple_weights.to(dtype=torch.float32, device=ranks.device)
    share6_by_tuple = (share6 * weights).sum(dim=0) * tuple_weight
    count_by_tuple = (valid.to(torch.float32) * weights).sum(dim=0) * tuple_weight
    return share6_by_tuple, count_by_tuple


def _allin_3p_share0_inner(
    *,
    compile_inner: bool,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    global _COMPILED_3P_SHARE0_INNER
    if not compile_inner:
        return _allin_3p_share0_accumulate_inner
    if _COMPILED_3P_SHARE0_INNER is None:
        _COMPILED_3P_SHARE0_INNER = torch.compile(
            _allin_3p_share0_accumulate_inner,
            dynamic=True,
            fullgraph=False,
        )
    return _COMPILED_3P_SHARE0_INNER


if triton is not None:

    @triton.jit
    def _allin_3p_share0_accumulate_kernel(
        ranks_ptr,
        board_masks_ptr,
        board_weights_ptr,
        tuples_ptr,
        tuple_masks_ptr,
        tuple_weights_ptr,
        owners_ptr,
        mirror_owners_ptr,
        share6_sum_ptr,
        count_ptr,
        tuple_count: tl.constexpr,
        board_count: tl.constexpr,
        H: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BOARD_GROUP: tl.constexpr,
        ACCUM_COUNT: tl.constexpr,
        USE_MIRROR: tl.constexpr,
    ):
        t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        b0 = tl.program_id(1) * BLOCK_B * BOARD_GROUP
        b_inner = tl.arange(0, BLOCK_B)
        t_mask = t < tuple_count

        h0 = tl.load(tuples_ptr + t * 3, mask=t_mask, other=0)
        h1 = tl.load(tuples_ptr + t * 3 + 1, mask=t_mask, other=0)
        h2 = tl.load(tuples_ptr + t * 3 + 2, mask=t_mask, other=0)
        tuple_weight = tl.load(tuple_weights_ptr + t, mask=t_mask, other=0.0).to(
            tl.float32
        )
        tuple_low = tl.load(tuple_masks_ptr + t * 2, mask=t_mask, other=0)
        tuple_high = tl.load(tuple_masks_ptr + t * 2 + 1, mask=t_mask, other=0)
        share6_acc = tl.zeros((BLOCK_T,), tl.float32)
        count_acc = tl.zeros((BLOCK_T,), tl.float32)
        for group in tl.range(0, BOARD_GROUP):
            b = b0 + group * BLOCK_B + b_inner
            b_mask = b < board_count
            board_low = tl.load(board_masks_ptr + b * 2, mask=b_mask, other=-1)
            board_high = tl.load(board_masks_ptr + b * 2 + 1, mask=b_mask, other=-1)
            offsets0 = b[:, None] * H + h0[None, :]
            offsets1 = b[:, None] * H + h1[None, :]
            offsets2 = b[:, None] * H + h2[None, :]
            load_mask = b_mask[:, None] & t_mask[None, :]
            blocks_board = ((board_low[:, None] & tuple_low[None, :]) != 0) | (
                (board_high[:, None] & tuple_high[None, :]) != 0
            )
            valid = load_mask & ~blocks_board

            r0 = tl.load(ranks_ptr + offsets0, mask=valid, other=-1)
            r1 = tl.load(ranks_ptr + offsets1, mask=valid, other=-1)
            r2 = tl.load(ranks_ptr + offsets2, mask=valid, other=-1)

            share6 = tl.zeros((BLOCK_B, BLOCK_T), tl.float32)
            share6 = tl.where((r0 > r1) & (r0 > r2), 6.0, share6)
            share6 = tl.where((r0 == r1) & (r0 > r2), 3.0, share6)
            share6 = tl.where((r0 == r2) & (r0 > r1), 3.0, share6)
            share6 = tl.where((r0 == r1) & (r0 == r2), 2.0, share6)
            share6 = tl.where(valid, share6, 0.0)
            board_weight = tl.load(board_weights_ptr + b, mask=b_mask, other=0.0).to(
                tl.float32
            )
            share6_acc += tl.sum(share6 * board_weight[:, None], axis=0)
            if ACCUM_COUNT:
                count_acc += tl.sum(valid.to(tl.float32) * board_weight[:, None], axis=0)

        share6_by_tuple = share6_acc * tuple_weight
        owner = tl.load(owners_ptr + t, mask=t_mask, other=0)
        tl.atomic_add(share6_sum_ptr + owner, share6_by_tuple, sem="relaxed", mask=t_mask)
        if USE_MIRROR:
            mirror_owner = tl.load(mirror_owners_ptr + t, mask=t_mask, other=-1)
            mirror_mask = t_mask & (mirror_owner >= 0)
            tl.atomic_add(
                share6_sum_ptr + mirror_owner,
                share6_by_tuple,
                sem="relaxed",
                mask=mirror_mask,
            )
        if ACCUM_COUNT:
            count_by_tuple = count_acc * tuple_weight
            tl.atomic_add(count_ptr + owner, count_by_tuple, sem="relaxed", mask=t_mask)
            if USE_MIRROR:
                tl.atomic_add(
                    count_ptr + mirror_owner,
                    count_by_tuple,
                    sem="relaxed",
                    mask=mirror_mask,
                )


def _allin_3p_share0_triton_accumulate_(
    *,
    ranks: torch.Tensor,
    board_masks: torch.Tensor,
    board_weights: torch.Tensor,
    tuples: torch.Tensor,
    tuple_masks: torch.Tensor,
    tuple_weights: torch.Tensor,
    owners: torch.Tensor,
    mirror_owners: torch.Tensor,
    share6_sum: torch.Tensor,
    count: torch.Tensor,
    accumulate_count: bool,
    use_mirror: bool,
    block_t: int,
    block_b: int,
    board_group: int,
) -> None:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if tuples.numel() == 0:
        return
    grid = (
        triton.cdiv(int(tuples.shape[0]), int(block_t)),
        triton.cdiv(int(ranks.shape[0]), int(block_b) * int(board_group)),
    )
    _allin_3p_share0_accumulate_kernel[grid](
        ranks,
        board_masks,
        board_weights,
        tuples,
        tuple_masks,
        tuple_weights,
        owners,
        mirror_owners,
        share6_sum,
        count,
        int(tuples.shape[0]),
        int(ranks.shape[0]),
        NUM_HANDS,
        BLOCK_T=int(block_t),
        BLOCK_B=int(block_b),
        BOARD_GROUP=int(board_group),
        ACCUM_COUNT=bool(accumulate_count),
        USE_MIRROR=bool(use_mirror),
        num_warps=8,
    )


@torch.no_grad()
def precompute_allin_class_shares(
    *,
    players: int,
    device: torch.device | str,
    sample_boards: int | None = None,
    seed: int = 0,
    board_chunk: int = 256,
    class_chunk: int = 8192,
    tuple_chunk: int = 65_536,
    class_ids: Sequence[int] | None = None,
    use_triton: bool | None = None,
    progress: Callable[[dict[str, float]], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Precompute exact ordered 2-player or 3-player class all-in shares.

    Exactness is with respect to the board source: pass ``sample_boards=None``
    for all ``C(52, 5)`` boards, or a positive ``sample_boards`` for a
    deterministic smoke/approximation run.
    """
    if players not in (2, 3):
        raise ValueError("players must be 2 or 3")
    if class_chunk <= 0 or tuple_chunk <= 0:
        raise ValueError("class_chunk and tuple_chunk must be positive")

    device = torch.device(device)
    if use_triton is None:
        use_triton = device.type == "cuda" and triton_is_available()
    if use_triton and device.type != "cuda":
        raise ValueError("Triton ranker requires CUDA")

    linear = _linear_class_ids(players, class_ids=class_ids, device=device)
    total_entries = PREFLOP_HANDS**players
    share_sum = torch.zeros(players, total_entries, dtype=torch.float64, device=device)
    count = torch.zeros(total_entries, dtype=torch.int64, device=device)
    class_combo, class_combo_mask = _class_combo_table(device)
    combo_masks = _combo_masks(device)
    started = time.perf_counter()
    board_total = TOTAL_PREFLOP_BOARDS if sample_boards is None else int(sample_boards)
    done_boards = 0

    for boards_cpu in _board_chunks(
        sample_boards=sample_boards,
        seed=seed,
        board_chunk=board_chunk,
    ):
        boards = boards_cpu.to(device=device, non_blocking=True)
        ranks = _rank_hands(boards, use_triton=bool(use_triton))
        allowed = board_allowed_hands(boards)
        done_boards += int(boards.shape[0])

        for start in range(0, linear.numel(), class_chunk):
            ids = linear[start : start + class_chunk]
            class_tuple = _decode_linear_classes(ids, players)
            tuple_grid, tuple_valid = _class_tuple_combos(
                class_tuple,
                class_combo=class_combo,
                class_combo_mask=class_combo_mask,
                combo_masks=combo_masks,
            )
            flat_valid = tuple_valid.reshape(-1)
            if not bool(flat_valid.any()):
                continue
            flat_tuple = tuple_grid.reshape(-1, players)[flat_valid].contiguous()
            owners = ids.repeat_interleave(tuple_valid.shape[1])[flat_valid]

            for tuple_start in range(0, flat_tuple.shape[0], tuple_chunk):
                tuples = flat_tuple[tuple_start : tuple_start + tuple_chunk]
                owner = owners[tuple_start : tuple_start + tuple_chunk]
                flat_hands = tuples.reshape(-1)
                tuple_ranks = ranks.index_select(1, flat_hands).reshape(
                    boards.shape[0],
                    tuples.shape[0],
                    players,
                )
                tuple_allowed = allowed.index_select(1, flat_hands).reshape(
                    boards.shape[0],
                    tuples.shape[0],
                    players,
                )
                board_valid = tuple_allowed.all(dim=2)
                max_rank = tuple_ranks.max(dim=2, keepdim=True).values
                winners = tuple_ranks == max_rank
                winner_count = winners.sum(dim=2, keepdim=True).clamp_min(1)
                shares = winners.to(torch.float64) / winner_count.to(torch.float64)
                shares = torch.where(board_valid[:, :, None], shares, 0.0)
                share_by_tuple = shares.sum(dim=0).T.contiguous()
                count_by_tuple = board_valid.sum(dim=0, dtype=torch.int64)
                for player in range(players):
                    share_sum[player].scatter_add_(0, owner, share_by_tuple[player])
                count.scatter_add_(0, owner, count_by_tuple)

        if progress is not None:
            elapsed = time.perf_counter() - started
            progress(
                {
                    "players": float(players),
                    "boards_done": float(done_boards),
                    "boards_total": float(board_total),
                    "seconds": float(elapsed),
                    "boards_per_second": float(done_boards / max(elapsed, 1.0e-9)),
                }
            )

    avg = torch.where(
        count[None, :] > 0,
        share_sum / count[None, :].to(torch.float64).clamp_min(1),
        torch.zeros_like(share_sum),
    )
    shape = (players, *([PREFLOP_HANDS] * players))
    metadata: dict[str, object] = {
        "players": players,
        "num_boards": board_total,
        "exact": sample_boards is None,
        "sample_seed": int(seed),
        "board_chunk": int(board_chunk),
        "class_chunk": int(class_chunk),
        "tuple_chunk": int(tuple_chunk),
        "class_ids": None if class_ids is None else [int(x) for x in class_ids],
        "device": str(device),
        "ranker": "triton" if use_triton else "torch",
    }
    return (
        avg.reshape(shape).to(torch.float32).contiguous(),
        count.reshape(*([PREFLOP_HANDS] * players)).contiguous(),
        metadata,
    )


@torch.no_grad()
def precompute_allin_3p_share0(
    *,
    device: torch.device | str,
    sample_boards: int | None = None,
    seed: int = 0,
    board_chunk: int = 256,
    class_chunk: int = 8192,
    tuple_chunk: int = 65_536,
    class_ids: Sequence[int] | None = None,
    use_triton: bool | None = None,
    use_accumulation_kernel: bool | None = None,
    compile_inner: bool | None = None,
    triton_block_t: int = 64,
    triton_block_b: int = 16,
    triton_board_group: int = 24,
    canonical_boards: bool | None = None,
    tuple_orbits: bool | None = None,
    opponent_symmetry: bool = True,
    progress: Callable[[dict[str, float]], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Precompute canonical 3-player seat-0 shares with weighted accumulation."""
    if class_chunk <= 0 or tuple_chunk <= 0:
        raise ValueError("class_chunk and tuple_chunk must be positive")

    device = torch.device(device)
    if use_triton is None:
        use_triton = device.type == "cuda" and triton_is_available()
    if use_triton and device.type != "cuda":
        raise ValueError("Triton ranker requires CUDA")
    if use_accumulation_kernel is None:
        use_accumulation_kernel = device.type == "cuda" and triton is not None
    if use_accumulation_kernel and (device.type != "cuda" or triton is None):
        raise ValueError("Triton accumulation kernel requires CUDA and Triton")
    if compile_inner is None:
        compile_inner = device.type == "cuda" and not use_accumulation_kernel
    if triton_block_t <= 0 or triton_block_b <= 0:
        raise ValueError("triton_block_t and triton_block_b must be positive")
    if triton_board_group <= 0:
        raise ValueError("triton_board_group must be positive")
    if canonical_boards is None:
        canonical_boards = sample_boards is None
    if tuple_orbits is None:
        tuple_orbits = sample_boards is None and not canonical_boards
    if tuple_orbits and sample_boards is not None:
        raise ValueError("tuple_orbits is only exact for exhaustive board runs")
    if tuple_orbits and canonical_boards:
        raise ValueError("tuple_orbits cannot be combined with canonical_boards")

    linear = _linear_3p_share0_class_ids(
        class_ids=class_ids,
        device=device,
        opponent_symmetry=opponent_symmetry,
    )
    total_entries = PREFLOP_HANDS**3
    share6_sum = torch.zeros(total_entries, dtype=torch.float32, device=device)
    use_analytic_count = sample_boards is None
    count_exact = (
        torch.zeros(total_entries, dtype=torch.int64, device=device)
        if use_analytic_count
        else None
    )
    count_accum = torch.zeros(
        1 if use_analytic_count else total_entries,
        dtype=torch.float32,
        device=device,
    )
    class_combo, class_combo_mask = _class_combo_table(device)
    combo_masks = _combo_masks(device)
    combo_card_masks = _combo_card_masks26(device)
    combo_suit_permutation = (
        combo_suit_permutation_tensor(device=device) if tuple_orbits else None
    )
    combo_class_ids = combo_to_preflop_class_tensor(device=device) if tuple_orbits else None
    inner = _allin_3p_share0_inner(compile_inner=compile_inner)
    started = time.perf_counter()
    board_total = TOTAL_PREFLOP_BOARDS if sample_boards is None else int(sample_boards)
    board_representatives_total = (
        canonical_full_boards_with_weights()[0].shape[0]
        if sample_boards is None and canonical_boards
        else board_total
    )
    done_boards = 0.0
    done_board_representatives = 0

    if use_analytic_count:
        if count_exact is None:
            raise RuntimeError("count_exact missing for analytic count mode")
        for start in range(0, linear.numel(), class_chunk):
            ids = linear[start : start + class_chunk]
            class_tuple = _decode_linear_classes(ids, 3)
            _tuple_grid, tuple_valid = _class_tuple_combos(
                class_tuple,
                class_combo=class_combo,
                class_combo_mask=class_combo_mask,
                combo_masks=combo_masks,
            )
            compatible_triples = tuple_valid.sum(dim=1, dtype=torch.int64)
            count_exact.index_copy_(0, ids, compatible_triples * TOTAL_TRIPLE_BOARDS)

    def accumulate_board_batch(
        *,
        ranks: torch.Tensor,
        allowed: torch.Tensor | None,
        board_masks: torch.Tensor | None,
        board_weights: torch.Tensor,
        progress_by_class: bool,
    ) -> None:
        for start in range(0, linear.numel(), class_chunk):
            ids = linear[start : start + class_chunk]
            class_tuple = _decode_linear_classes(ids, 3)
            tuple_grid, tuple_valid = _class_tuple_combos(
                class_tuple,
                class_combo=class_combo,
                class_combo_mask=class_combo_mask,
                combo_masks=combo_masks,
            )
            flat_valid = tuple_valid.reshape(-1)
            if not bool(flat_valid.any()):
                continue
            flat_tuple = tuple_grid.reshape(-1, 3)[flat_valid].contiguous()
            owners = ids.repeat_interleave(tuple_valid.shape[1])[flat_valid].contiguous()
            mirror_owners = torch.empty(0, dtype=torch.long, device=device)
            if tuple_orbits:
                if combo_suit_permutation is None:
                    raise RuntimeError("combo_suit_permutation missing")
                if combo_class_ids is None:
                    raise RuntimeError("combo_class_ids missing")
                flat_tuple, tuple_weights, owners = _canonicalize_tuple_suit_orbits(
                    flat_tuple,
                    combo_suit_permutation=combo_suit_permutation,
                    combo_class_ids=combo_class_ids,
                )
            else:
                tuple_weights = torch.ones(
                    flat_tuple.shape[0],
                    dtype=torch.float32,
                    device=device,
                )

            for tuple_start in range(0, flat_tuple.shape[0], tuple_chunk):
                tuples = flat_tuple[tuple_start : tuple_start + tuple_chunk]
                tuple_masks = _tuple_card_masks26(
                    tuples,
                    combo_card_masks=combo_card_masks,
                )
                tuple_weight = tuple_weights[tuple_start : tuple_start + tuple_chunk]
                owner = owners[tuple_start : tuple_start + tuple_chunk]
                if use_accumulation_kernel:
                    if board_masks is None:
                        raise RuntimeError("board_masks missing for accumulation kernel")
                    _allin_3p_share0_triton_accumulate_(
                        ranks=ranks,
                        board_masks=board_masks,
                        board_weights=board_weights,
                        tuples=tuples,
                        tuple_masks=tuple_masks,
                        tuple_weights=tuple_weight,
                        owners=owner,
                        mirror_owners=mirror_owners,
                        share6_sum=share6_sum,
                        count=count_accum,
                        accumulate_count=not use_analytic_count,
                        use_mirror=False,
                        block_t=triton_block_t,
                        block_b=triton_block_b,
                        board_group=triton_board_group,
                    )
                else:
                    if allowed is None:
                        raise RuntimeError("allowed missing for torch accumulation")
                    chunk_share6, chunk_count = inner(
                        ranks,
                        allowed,
                        tuples,
                        board_weights,
                        tuple_weight,
                    )
                    share6_sum.scatter_add_(0, owner, chunk_share6)
                    if not use_analytic_count:
                        count_accum.scatter_add_(0, owner, chunk_count)

            if progress is not None and progress_by_class:
                elapsed = time.perf_counter() - started
                progress(
                    {
                        "players": 3.0,
                        "boards_done": done_boards,
                        "boards_total": float(board_total),
                        "board_representatives_done": float(done_board_representatives),
                        "board_representatives_total": float(
                            board_representatives_total
                        ),
                        "class_entries_done": float(start + ids.numel()),
                        "class_entries_total": float(linear.numel()),
                        "seconds": float(elapsed),
                        "boards_per_second": float(
                            done_boards / max(elapsed, 1.0e-9)
                        ),
                    }
                )

    resident_board_tensors = (
        sample_boards is None
        and bool(canonical_boards)
        and bool(use_accumulation_kernel)
        and device.type == "cuda"
    )

    if resident_board_tensors:
        boards_cpu, weights_cpu = canonical_full_boards_with_weights()
        boards = boards_cpu.to(device=device, non_blocking=True)
        board_weights = weights_cpu.to(device=device, non_blocking=True).contiguous()
        ranks = _rank_hands(boards, use_triton=bool(use_triton))
        allowed = None if use_accumulation_kernel else board_allowed_hands(boards)
        if allowed is not None:
            allowed = allowed.contiguous()
        board_masks = _board_card_masks26(boards) if use_accumulation_kernel else None
        done_boards = float(weights_cpu.sum().item())
        done_board_representatives = int(boards.shape[0])
        accumulate_board_batch(
            ranks=ranks,
            allowed=allowed,
            board_masks=board_masks,
            board_weights=board_weights,
            progress_by_class=True,
        )
    else:
        for boards_cpu, weights_cpu in _weighted_board_chunks(
            sample_boards=sample_boards,
            seed=seed,
            board_chunk=board_chunk,
            canonical_boards=bool(canonical_boards),
        ):
            boards = boards_cpu.to(device=device, non_blocking=True)
            board_weights = weights_cpu.to(device=device, non_blocking=True).contiguous()
            ranks = _rank_hands(boards, use_triton=bool(use_triton))
            allowed = None if use_accumulation_kernel else board_allowed_hands(boards)
            if allowed is not None:
                allowed = allowed.contiguous()
            board_masks = _board_card_masks26(boards) if use_accumulation_kernel else None
            done_boards += float(weights_cpu.sum().item())
            done_board_representatives += int(boards.shape[0])
            accumulate_board_batch(
                ranks=ranks,
                allowed=allowed,
                board_masks=board_masks,
                board_weights=board_weights,
                progress_by_class=False,
            )

            if progress is not None:
                elapsed = time.perf_counter() - started
                progress(
                    {
                        "players": 3.0,
                        "boards_done": done_boards,
                        "boards_total": float(board_total),
                        "board_representatives_done": float(done_board_representatives),
                        "board_representatives_total": float(
                            board_representatives_total
                        ),
                        "seconds": float(elapsed),
                        "boards_per_second": float(
                            done_boards / max(elapsed, 1.0e-9)
                        ),
                    }
                )

    if opponent_symmetry:
        _mirror_3p_caller_order_(share6_sum)

    count = (
        count_exact
        if use_analytic_count and count_exact is not None
        else count_accum.round().to(torch.int64)
    )
    if opponent_symmetry:
        _mirror_3p_caller_order_(count)
    share0 = torch.where(
        count > 0,
        share6_sum / (count.to(torch.float32) * 6.0).clamp_min(1.0),
        torch.zeros_like(share6_sum),
    )
    metadata: dict[str, object] = {
        "players": 3,
        "num_boards": board_total,
        "num_board_representatives": int(board_representatives_total),
        "exact": sample_boards is None,
        "sample_seed": int(seed),
        "board_chunk": int(board_chunk),
        "class_chunk": int(class_chunk),
        "tuple_chunk": int(tuple_chunk),
        "class_ids": None if class_ids is None else [int(x) for x in class_ids],
        "device": str(device),
        "ranker": "triton" if use_triton else "torch",
        "accumulation_kernel": "triton" if use_accumulation_kernel else "torch",
        "compile_inner": bool(compile_inner),
        "triton_block_t": int(triton_block_t),
        "triton_block_b": int(triton_block_b),
        "triton_board_group": int(triton_board_group),
        "canonical_boards": bool(canonical_boards),
        "tuple_orbits": bool(tuple_orbits),
        "opponent_symmetry": bool(opponent_symmetry),
        "board_tensor_mode": "resident_exact_canonical"
        if resident_board_tensors
        else "streamed",
        "share_accumulator": "fp32 weighted sixths",
        "count_mode": "analytic_compatible_triples_x_C46_5"
        if use_analytic_count
        else "streamed_board_valid_fp32",
    }
    return (
        share0.reshape(PREFLOP_HANDS, PREFLOP_HANDS, PREFLOP_HANDS).contiguous(),
        count.reshape(PREFLOP_HANDS, PREFLOP_HANDS, PREFLOP_HANDS).contiguous(),
        metadata,
    )


@torch.no_grad()
def precompute_allin_169_tensors(
    *,
    output: str | Path | None = None,
    device: torch.device | str = "cuda",
    players: Sequence[int] = (2, 3),
    preflop_payoff_path: str | Path | None = None,
    sample_boards: int | None = None,
    seed: int = 0,
    board_chunk: int = 256,
    class_chunk: int = 8192,
    tuple_chunk: int = 65_536,
    class_ids: Sequence[int] | None = None,
    use_triton: bool | None = None,
    use_accumulation_kernel: bool | None = None,
    compile_inner: bool | None = None,
    triton_block_t: int = 64,
    triton_block_b: int = 16,
    triton_board_group: int = 24,
    canonical_boards: bool | None = None,
    tuple_orbits: bool | None = None,
    opponent_symmetry: bool = True,
    progress: Callable[[dict[str, float]], None] | None = None,
) -> PreflopAllIn169Tensors:
    """Build and optionally save preflop all-in 169-class tensors."""
    device = torch.device(device)
    wanted = tuple(int(p) for p in players)
    if any(p not in (2, 3) for p in wanted):
        raise ValueError("players entries must be 2 or 3")

    result = PreflopAllIn169Tensors(metadata={"format": _FORMAT})
    timings: dict[str, float] = {}

    if 2 in wanted and preflop_payoff_path is not None and class_ids is None:
        start = time.perf_counter()
        table = load_preflop_payoff_i16(preflop_payoff_path, device)
        share0, counts = allin_2p_169_share0_from_combo_payoff(table, scale=I16_SCALE)
        result.allin_2p_share0 = share0.cpu()
        result.allin_2p_count = counts.cpu()
        timings["allin_2p_seconds"] = time.perf_counter() - start
    elif 2 in wanted:
        start = time.perf_counter()
        shares, counts, meta = precompute_allin_class_shares(
            players=2,
            device=device,
            sample_boards=sample_boards,
            seed=seed,
            board_chunk=board_chunk,
            class_chunk=class_chunk,
            tuple_chunk=tuple_chunk,
            class_ids=class_ids,
            use_triton=use_triton,
            progress=progress,
        )
        share0 = torch.where(
            counts > 0,
            shares[0],
            torch.zeros_like(shares[0]),
        )
        result.allin_2p_share0 = share0.cpu()
        result.allin_2p_count = counts.cpu()
        timings["allin_2p_seconds"] = time.perf_counter() - start
        result.metadata = {**(result.metadata or {}), "allin_2p_stream": meta}

    if 3 in wanted:
        start = time.perf_counter()
        share0, counts, meta = precompute_allin_3p_share0(
            device=device,
            sample_boards=sample_boards,
            seed=seed,
            board_chunk=board_chunk,
            class_chunk=class_chunk,
            tuple_chunk=tuple_chunk,
            class_ids=class_ids,
            use_triton=use_triton,
            use_accumulation_kernel=use_accumulation_kernel,
            compile_inner=compile_inner,
            triton_block_t=triton_block_t,
            triton_block_b=triton_block_b,
            triton_board_group=triton_board_group,
            canonical_boards=canonical_boards,
            tuple_orbits=tuple_orbits,
            opponent_symmetry=opponent_symmetry,
            progress=progress,
        )
        result.allin_3p_share0 = share0.cpu()
        result.allin_3p_count = counts.cpu()
        timings["allin_3p_seconds"] = time.perf_counter() - start
        result.metadata = {**(result.metadata or {}), "allin_3p_stream": meta}

    result.metadata = {
        **(result.metadata or {}),
        "created_unix_time": time.time(),
        "players": list(wanted),
        "preflop_payoff_path": None
        if preflop_payoff_path is None
        else str(preflop_payoff_path),
        "sample_boards": sample_boards,
        "exact": sample_boards is None,
        "class_order": "p2.env.card_utils.combo_to_preflop_class_tensor",
        "class_multiplicity": preflop_class_multiplicity_tensor().to(torch.int16),
        "timings": timings,
    }
    if output is not None:
        save_allin_169_tensors(result, output)
    return result


def save_allin_169_tensors(result: PreflopAllIn169Tensors, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = result.payload()
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if output.suffix == ".zst":
        import zstandard as zstd

        buffer = io.BytesIO()
        torch.save(payload, buffer)
        tmp.write_bytes(zstd.ZstdCompressor(level=10).compress(buffer.getvalue()))
        tmp.replace(output)
        return
    torch.save(payload, tmp)
    tmp.replace(output)


def _quantize_share_u16(share: torch.Tensor) -> torch.Tensor:
    return (
        share.to(torch.float32)
        .clamp(0.0, 1.0)
        .mul(_U16_SCALE)
        .round()
        .to(torch.uint16)
        .contiguous()
    )


def quantized_allin_169_payload(result: PreflopAllIn169Tensors) -> dict[str, object]:
    metadata = {
        **(result.metadata or {}),
        "format": _QUANTIZED_FORMAT,
        "share_quantization": {
            "dtype": "uint16",
            "scale": int(_U16_SCALE),
            "quantization": "round(clamp(share, 0, 1) * 65535).to(uint16)",
            "dequantization": "share_u16.float() / 65535",
        },
    }
    payload: dict[str, object] = {"format": _QUANTIZED_FORMAT, "metadata": metadata}
    if result.allin_2p_share0 is not None:
        payload["allin_2p_share0_u16"] = _quantize_share_u16(result.allin_2p_share0)
    if result.allin_3p_share0 is not None:
        payload["allin_3p_share0_u16"] = _quantize_share_u16(result.allin_3p_share0)
    return payload


def save_quantized_allin_169_tensors(
    result: PreflopAllIn169Tensors,
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = quantized_allin_169_payload(result)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if output.suffix == ".zst":
        import zstandard as zstd

        buffer = io.BytesIO()
        torch.save(payload, buffer)
        tmp.write_bytes(zstd.ZstdCompressor(level=10).compress(buffer.getvalue()))
        tmp.replace(output)
        return
    torch.save(payload, tmp)
    tmp.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quantized-shares",
        action="store_true",
        help="Save share tensors as uint16 with implicit division by 65535.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--players", type=int, nargs="+", default=[2, 3], choices=[2, 3])
    parser.add_argument("--preflop-payoff-path", type=Path, default=None)
    parser.add_argument("--sample-boards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--board-chunk", type=int, default=256)
    parser.add_argument("--class-chunk", type=int, default=8192)
    parser.add_argument("--tuple-chunk", type=int, default=65_536)
    parser.add_argument("--class-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--accumulation-kernel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force Triton accumulation kernel on/off; default auto-enables on CUDA.",
    )
    parser.add_argument(
        "--compile-inner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force torch.compile on/off for the 3p CUDA inner loop; default auto-enables on CUDA.",
    )
    parser.add_argument("--triton-block-t", type=int, default=64)
    parser.add_argument("--triton-block-b", type=int, default=16)
    parser.add_argument("--triton-board-group", type=int, default=24)
    parser.add_argument(
        "--canonical-boards",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use suit-canonical full-board representatives with orbit weights for "
            "exact 3p; default auto-enables for exact runs."
        ),
    )
    parser.add_argument(
        "--tuple-orbits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use suit-canonical concrete hole-tuple representatives with orbit "
            "weights for exact 3p; default auto-enables for exact runs."
        ),
    )
    parser.add_argument(
        "--opponent-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror 3p seat-0 results across caller order; default enabled.",
    )
    parser.add_argument("--no-triton", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_boards is None and args.device == "cpu":
        print(
            "Warning: exhaustive CPU generation is expected to be very slow; "
            "use CUDA or --sample-boards for a smoke run.",
            flush=True,
        )

    def progress(info: dict[str, float]) -> None:
        reps = ""
        if info.get("board_representatives_total", info["boards_total"]) != info[
            "boards_total"
        ]:
            reps = (
                f" reps={int(info['board_representatives_done']):,}/"
                f"{int(info['board_representatives_total']):,}"
            )
        print(
            f"p{int(info['players'])} boards={int(info['boards_done']):,}/"
            f"{int(info['boards_total']):,}{reps} "
            f"rate={info['boards_per_second']:,.1f}/s "
            f"elapsed={info['seconds'] / 60.0:.1f}m",
            flush=True,
        )

    result = precompute_allin_169_tensors(
        output=None if args.quantized_shares else args.output,
        device=args.device,
        players=args.players,
        preflop_payoff_path=args.preflop_payoff_path,
        sample_boards=args.sample_boards,
        seed=args.seed,
        board_chunk=args.board_chunk,
        class_chunk=args.class_chunk,
        tuple_chunk=args.tuple_chunk,
        class_ids=args.class_ids,
        use_triton=False if args.no_triton else None,
        use_accumulation_kernel=args.accumulation_kernel,
        compile_inner=args.compile_inner,
        triton_block_t=args.triton_block_t,
        triton_block_b=args.triton_block_b,
        triton_board_group=args.triton_board_group,
        canonical_boards=args.canonical_boards,
        tuple_orbits=args.tuple_orbits,
        opponent_symmetry=args.opponent_symmetry,
        progress=progress,
    )
    if args.quantized_shares:
        save_quantized_allin_169_tensors(result, args.output)
    payload = result.payload()
    print(f"wrote {args.output}", flush=True)
    printed_payload = quantized_allin_169_payload(result) if args.quantized_shares else payload
    for key, value in printed_payload.items():
        if torch.is_tensor(value):
            mb = value.numel() * value.element_size() / 1024 / 1024
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype} {mb:.2f} MiB")


if __name__ == "__main__":
    main()
