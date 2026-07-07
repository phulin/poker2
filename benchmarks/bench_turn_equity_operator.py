#!/usr/bin/env python3
"""Benchmark a precomputed turn hand-pair operator baseline.

The experiment compares the current turn-equity baseline against a root-grouped
matrix formulation:

    numerator[root]   = pair_payoff[root] @ opponent_beliefs[root]
    denominator[root] = pair_count[root] @ opponent_beliefs[root]

The pair operators are precomputed per unique turn root and reused for all
leaves under that root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA benchmark path
    triton = None
    tl = None

from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.models.mlp.better_features import (
    PlayerContext,
    ValueScalarContext,
    context_length,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.turn_range_equity import (
    TurnRangeEquityConfig,
    build_turn_range_equity_board_cache,
    turn_range_equity_baseline,
)


if triton is not None:

    @triton.jit
    def _turn_operator_denominator_hu_kernel(
        beliefs_ptr,
        grouped_leaf_indices_ptr,
        legal_hands_ptr,
        card_a_ptr,
        card_b_ptr,
        den_ptr,
        total_programs: tl.constexpr,
        leaves_per_root: tl.constexpr,
        hand_count: tl.constexpr,
        BLOCK_ALL_HANDS: tl.constexpr,
        BLOCK_OUT_HANDS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        player = pid & 1
        leaf_local = (pid // 2) % leaves_per_root
        root = pid // (2 * leaves_per_root)
        opp_player = 1 - player
        leaf_pos = root * leaves_per_root + leaf_local
        leaf = tl.load(
            grouped_leaf_indices_ptr + leaf_pos, mask=pid < total_programs, other=0
        )

        all_offs = tl.arange(0, BLOCK_ALL_HANDS)
        all_mask = all_offs < 1326
        vals = tl.load(
            beliefs_ptr + (leaf * 2 + opp_player) * 1326 + all_offs,
            mask=(pid < total_programs) & all_mask,
            other=0.0,
        ).to(tl.float32)
        opp_card_a = tl.load(card_a_ptr + all_offs, mask=all_mask, other=-1)
        opp_card_b = tl.load(card_b_ptr + all_offs, mask=all_mask, other=-2)
        total = tl.sum(vals, axis=0)

        out_offs = tl.arange(0, BLOCK_OUT_HANDS)
        out_mask = out_offs < hand_count
        hero_hands = tl.load(
            legal_hands_ptr + root * hand_count + out_offs,
            mask=(pid < total_programs) & out_mask,
            other=0,
        ).to(tl.int32)
        hero_card_a = tl.load(card_a_ptr + hero_hands, mask=out_mask, other=-3)
        hero_card_b = tl.load(card_b_ptr + hero_hands, mask=out_mask, other=-4)
        den = tl.full((BLOCK_OUT_HANDS,), 44.0, dtype=tl.float32) * total
        for card in tl.static_range(0, 52):
            card_mass = tl.sum(
                tl.where((opp_card_a == card) | (opp_card_b == card), vals, 0.0),
                axis=0,
            )
            den += tl.where(
                (hero_card_a == card) | (hero_card_b == card), card_mass, 0.0
            )
        tl.store(
            den_ptr
            + ((root * leaves_per_root + leaf_local) * 2 + player) * hand_count
            + out_offs,
            den,
            mask=(pid < total_programs) & out_mask,
        )

    @triton.jit
    def _turn_operator_formula_postprocess_hu_kernel(
        beliefs_ptr,
        grouped_leaf_indices_ptr,
        legal_hands_ptr,
        card_a_ptr,
        card_b_ptr,
        num0_ptr,
        num1_ptr,
        pot_ptr,
        compact_values_ptr,
        total_programs: tl.constexpr,
        leaves_per_root: tl.constexpr,
        hand_count: tl.constexpr,
        BLOCK_ALL_HANDS: tl.constexpr,
        BLOCK_OUT_HANDS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        leaf_local = pid % leaves_per_root
        root = pid // leaves_per_root
        leaf_pos = root * leaves_per_root + leaf_local
        leaf = tl.load(
            grouped_leaf_indices_ptr + leaf_pos, mask=pid < total_programs, other=0
        )

        all_offs = tl.arange(0, BLOCK_ALL_HANDS)
        all_mask = all_offs < 1326
        vals0 = tl.load(
            beliefs_ptr + (leaf * 2 + 1) * 1326 + all_offs,
            mask=(pid < total_programs) & all_mask,
            other=0.0,
        ).to(tl.float32)
        vals1 = tl.load(
            beliefs_ptr + (leaf * 2) * 1326 + all_offs,
            mask=(pid < total_programs) & all_mask,
            other=0.0,
        ).to(tl.float32)
        opp_card_a = tl.load(card_a_ptr + all_offs, mask=all_mask, other=-1)
        opp_card_b = tl.load(card_b_ptr + all_offs, mask=all_mask, other=-2)
        total0 = tl.sum(vals0, axis=0)
        total1 = tl.sum(vals1, axis=0)

        out_offs = tl.arange(0, BLOCK_OUT_HANDS)
        out_mask = out_offs < hand_count
        hero_hands = tl.load(
            legal_hands_ptr + root * hand_count + out_offs,
            mask=(pid < total_programs) & out_mask,
            other=0,
        ).to(tl.int32)
        hero_card_a = tl.load(card_a_ptr + hero_hands, mask=out_mask, other=-3)
        hero_card_b = tl.load(card_b_ptr + hero_hands, mask=out_mask, other=-4)
        den0 = tl.full((BLOCK_OUT_HANDS,), 44.0, dtype=tl.float32) * total0
        den1 = tl.full((BLOCK_OUT_HANDS,), 44.0, dtype=tl.float32) * total1
        for card in tl.static_range(0, 52):
            card_match = (opp_card_a == card) | (opp_card_b == card)
            card_mass0 = tl.sum(tl.where(card_match, vals0, 0.0), axis=0)
            card_mass1 = tl.sum(tl.where(card_match, vals1, 0.0), axis=0)
            hero_match = (hero_card_a == card) | (hero_card_b == card)
            den0 += tl.where(hero_match, card_mass0, 0.0)
            den1 += tl.where(hero_match, card_mass1, 0.0)

        pot = tl.load(pot_ptr + leaf_pos, mask=pid < total_programs, other=0.0).to(
            tl.float32
        )
        num0 = tl.load(
            num0_ptr + (root * hand_count + out_offs) * leaves_per_root + leaf_local,
            mask=(pid < total_programs) & out_mask,
            other=0.0,
        ).to(tl.float32)
        num1 = tl.load(
            num1_ptr + (root * hand_count + out_offs) * leaves_per_root + leaf_local,
            mask=(pid < total_programs) & out_mask,
            other=0.0,
        ).to(tl.float32)
        val0 = (num0 / tl.maximum(den0, 1.0e-8)) * pot
        val1 = (num1 / tl.maximum(den1, 1.0e-8)) * pot
        out_base = ((root * leaves_per_root + leaf_local) * 2) * hand_count + out_offs
        tl.store(
            compact_values_ptr + out_base, val0, mask=(pid < total_programs) & out_mask
        )
        tl.store(
            compact_values_ptr + out_base + hand_count,
            val1,
            mask=(pid < total_programs) & out_mask,
        )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "turn_equity_operator_microbench.json"
DEFAULT_CUDA_DRIVER_DIR = Path("/usr/lib/x86_64-linux-gnu")


def _ensure_cuda_driver_path() -> None:
    libcuda = DEFAULT_CUDA_DRIVER_DIR / "libcuda.so.1"
    if not libcuda.exists():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(":") if part]
    driver_dir = str(DEFAULT_CUDA_DRIVER_DIR)
    if driver_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([driver_dir, *parts])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(samples: list[float], *, prefix: str) -> dict[str, float]:
    if not samples:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_min_ms": 0.0,
            f"{prefix}_max_ms": 0.0,
            f"{prefix}_std_ms": 0.0,
            f"{prefix}_cv": 0.0,
        }
    sorted_samples = sorted(samples)
    mean = sum(samples) / len(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / len(samples)
    std = variance**0.5
    return {
        f"{prefix}_mean_ms": mean,
        f"{prefix}_p50_ms": sorted_samples[len(sorted_samples) // 2],
        f"{prefix}_p90_ms": sorted_samples[
            min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.9))
        ],
        f"{prefix}_min_ms": min(samples),
        f"{prefix}_max_ms": max(samples),
        f"{prefix}_std_ms": std,
        f"{prefix}_cv": std / max(mean, 1e-9),
    }


def _time_call(
    device: torch.device,
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    include_samples: bool,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type != "cuda":
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append(1e3 * (time.perf_counter() - start))
        out = _summarize(samples, prefix="wall")
        if include_samples:
            out["wall_samples_ms"] = samples
        return out

    cuda_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(iters):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        cuda_samples.append(float(start_ev.elapsed_time(end_ev)))
        wall_samples.append(1e3 * (time.perf_counter() - start_wall))
    out = _summarize(cuda_samples, prefix="cuda")
    out.update(_summarize(wall_samples, prefix="wall"))
    if include_samples:
        out["cuda_samples_ms"] = cuda_samples
        out["wall_samples_ms"] = wall_samples
    return out


def _random_turn_roots(
    *,
    roots: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.rand(roots, 52, device=device).argsort(dim=1)[:, :4].long()


def _features_for_roots(
    *,
    root_boards: torch.Tensor,
    leaves_per_root: int,
    pot: float,
) -> tuple[MLPFeatures, torch.Tensor]:
    device = root_boards.device
    roots = int(root_boards.shape[0])
    leaf_board4 = root_boards.repeat_interleave(leaves_per_root, dim=0)
    leaves = int(leaf_board4.shape[0])
    board = torch.full((leaves, 5), -1, dtype=torch.long, device=device)
    board[:, :4] = leaf_board4

    combos = hand_combos_tensor(device=device)
    card_a = combos[:, 0]
    card_b = combos[:, 1]
    board_ok = (
        (card_a[None, :] != leaf_board4[:, :, None])
        & (card_b[None, :] != leaf_board4[:, :, None])
    ).all(dim=1)
    beliefs = torch.rand(leaves, 2, NUM_HANDS, dtype=torch.float32, device=device)
    beliefs = beliefs * board_ok[:, None, :].to(dtype=beliefs.dtype)
    beliefs = beliefs / beliefs.sum(dim=2, keepdim=True).clamp_min(1e-8)

    context = torch.zeros(
        leaves,
        context_length(2),
        dtype=torch.float32,
        device=device,
    )
    context[:, ValueScalarContext.POT.value] = pot
    spr_start = ValueScalarContext.NUM_SCALAR_CONTEXT.value
    context[:, spr_start + PlayerContext.SPR.value] = 1.0
    context[
        :,
        spr_start + PlayerContext.NUM_PLAYER_CONTEXT.value + PlayerContext.SPR.value,
    ] = 1.0

    features = MLPFeatures(
        context=context,
        street=torch.full((leaves,), 2, dtype=torch.long, device=device),
        to_act=torch.zeros(leaves, dtype=torch.long, device=device),
        board=board,
        beliefs=beliefs.reshape(leaves, -1),
    )
    return features, beliefs


def _legal_hands_from_board_ok(board_ok: torch.Tensor) -> torch.Tensor:
    roots = int(board_ok.shape[0])
    legal_count = int(board_ok.sum(dim=1).min().item())
    hand_ids = torch.arange(NUM_HANDS, dtype=torch.int32, device=board_ok.device)
    key = torch.where(
        board_ok,
        hand_ids.view(1, NUM_HANDS),
        torch.full(
            (roots, NUM_HANDS), NUM_HANDS, dtype=torch.int32, device=board_ok.device
        ),
    )
    return torch.argsort(key, dim=1)[:, :legal_count].to(torch.int64).contiguous()


def _build_pair_operators(
    *,
    rank_groups: torch.Tensor,
    hand_runout_ok: torch.Tensor,
    legal_hands: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    roots = int(rank_groups.shape[0])
    hand_count = int(legal_hands.shape[1])
    pair_payoff = torch.empty(
        roots,
        hand_count,
        hand_count,
        dtype=dtype,
        device=rank_groups.device,
    )
    pair_count = torch.empty_like(pair_payoff)
    gather_idx = legal_hands[:, None, :].expand(-1, 48, -1)
    compact_ranks = rank_groups.long().gather(2, gather_idx)
    compact_ok = hand_runout_ok.gather(2, gather_idx)

    for root in range(roots):
        payoff = torch.zeros(
            hand_count,
            hand_count,
            dtype=torch.float32,
            device=rank_groups.device,
        )
        count = torch.zeros_like(payoff)
        for river in range(48):
            ok = compact_ok[root, river]
            ranks = compact_ranks[root, river].to(torch.int16)
            pair_ok = ok[:, None] & ok[None, :]
            diff = ranks[:, None] - ranks[None, :]
            payoff += torch.sign(diff).to(torch.float32) * pair_ok.to(torch.float32)
            count += pair_ok.to(torch.float32)
        pair_payoff[root] = payoff.to(dtype)
        pair_count[root] = count.to(dtype)
    return pair_payoff.contiguous(), pair_count.contiguous()


def _build_card_incidence(
    *,
    legal_hands: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    roots = int(legal_hands.shape[0])
    hand_count = int(legal_hands.shape[1])
    combos = hand_combos_tensor(device=legal_hands.device)
    legal_hand_cards = combos[legal_hands.long()].contiguous()
    card_incidence = torch.zeros(
        roots,
        hand_count,
        52,
        dtype=dtype,
        device=legal_hands.device,
    )
    card_incidence.scatter_(
        2,
        legal_hand_cards,
        torch.ones_like(legal_hand_cards, dtype=dtype),
    )
    return card_incidence.contiguous(), legal_hand_cards


def _operator_baseline(
    *,
    beliefs: torch.Tensor,
    context: torch.Tensor,
    legal_hands: torch.Tensor,
    pair_payoff: torch.Tensor,
    pair_count: torch.Tensor,
    grouped_leaf_indices: torch.Tensor,
    leaves_per_root: int,
    matmul_dtype: torch.dtype,
) -> torch.Tensor:
    roots = int(legal_hands.shape[0])
    hand_count = int(legal_hands.shape[1])
    grouped_beliefs = beliefs.index_select(
        0,
        grouped_leaf_indices.long(),
    ).view(roots, leaves_per_root, 2, NUM_HANDS)
    compact_idx = legal_hands[:, None, None, :].expand(
        roots,
        leaves_per_root,
        2,
        hand_count,
    )
    compact_beliefs = torch.gather(grouped_beliefs, 3, compact_idx)
    opp_for_p0 = compact_beliefs[:, :, 1, :].transpose(1, 2).to(matmul_dtype)
    opp_for_p1 = compact_beliefs[:, :, 0, :].transpose(1, 2).to(matmul_dtype)

    num0 = torch.bmm(pair_payoff, opp_for_p0)
    den0 = torch.bmm(pair_count, opp_for_p0)
    num1 = torch.bmm(pair_payoff, opp_for_p1)
    den1 = torch.bmm(pair_count, opp_for_p1)
    val0 = (num0.float() / den0.float().clamp_min(1e-8)).transpose(1, 2)
    val1 = (num1.float() / den1.float().clamp_min(1e-8)).transpose(1, 2)
    pot = context.index_select(0, grouped_leaf_indices.long())[
        :,
        ValueScalarContext.POT.value,
    ].view(roots, leaves_per_root)
    compact_values = torch.stack((val0, val1), dim=2) * pot[:, :, None, None]

    dense = beliefs.new_zeros(roots, leaves_per_root, 2, NUM_HANDS)
    dense.scatter_(
        3,
        compact_idx,
        compact_values.to(dtype=dense.dtype),
    )
    grouped_dense = dense.view(roots * leaves_per_root, 2, NUM_HANDS)
    out = beliefs.new_empty(beliefs.shape)
    out.index_copy_(0, grouped_leaf_indices.long(), grouped_dense)
    return out


def _prepare_operator_inputs(
    *,
    beliefs: torch.Tensor,
    context: torch.Tensor,
    legal_hands: torch.Tensor,
    grouped_leaf_indices: torch.Tensor,
    leaves_per_root: int,
    matmul_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    roots = int(legal_hands.shape[0])
    hand_count = int(legal_hands.shape[1])
    grouped_beliefs = beliefs.index_select(
        0,
        grouped_leaf_indices.long(),
    ).view(roots, leaves_per_root, 2, NUM_HANDS)
    compact_idx = legal_hands[:, None, None, :].expand(
        roots,
        leaves_per_root,
        2,
        hand_count,
    )
    compact_beliefs = torch.gather(grouped_beliefs, 3, compact_idx)
    opp_for_p0 = compact_beliefs[:, :, 1, :].transpose(1, 2).to(matmul_dtype)
    opp_for_p1 = compact_beliefs[:, :, 0, :].transpose(1, 2).to(matmul_dtype)
    pot = context.index_select(0, grouped_leaf_indices.long())[
        :,
        ValueScalarContext.POT.value,
    ].view(roots, leaves_per_root)
    return compact_idx, opp_for_p0, opp_for_p1, pot


def _operator_gemm(
    *,
    pair_payoff: torch.Tensor,
    pair_count: torch.Tensor,
    opp_for_p0: torch.Tensor,
    opp_for_p1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num0 = torch.bmm(pair_payoff, opp_for_p0)
    den0 = torch.bmm(pair_count, opp_for_p0)
    num1 = torch.bmm(pair_payoff, opp_for_p1)
    den1 = torch.bmm(pair_count, opp_for_p1)
    return num0, den0, num1, den1


def _denominator_formula_cardmatmul(
    *,
    opp_for_p0: torch.Tensor,
    opp_for_p1: torch.Tensor,
    card_incidence: torch.Tensor,
    legal_hand_cards: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    card_mass0 = torch.bmm(opp_for_p0.transpose(1, 2), card_incidence)
    card_mass1 = torch.bmm(opp_for_p1.transpose(1, 2), card_incidence)
    total0 = opp_for_p0.sum(dim=1).float()
    total1 = opp_for_p1.sum(dim=1).float()
    gather_idx = legal_hand_cards[:, None, :, :].expand(
        -1,
        int(opp_for_p0.shape[2]),
        -1,
        -1,
    )
    den0 = 44.0 * total0[:, :, None]
    den0 = den0 + card_mass0.float().gather(2, gather_idx[..., 0])
    den0 = den0 + card_mass0.float().gather(2, gather_idx[..., 1])
    den1 = 44.0 * total1[:, :, None]
    den1 = den1 + card_mass1.float().gather(2, gather_idx[..., 0])
    den1 = den1 + card_mass1.float().gather(2, gather_idx[..., 1])
    return den0.transpose(1, 2).contiguous(), den1.transpose(1, 2).contiguous()


def _operator_postprocess(
    *,
    num0: torch.Tensor,
    den0: torch.Tensor,
    num1: torch.Tensor,
    den1: torch.Tensor,
    pot: torch.Tensor,
) -> torch.Tensor:
    val0 = (num0.float() / den0.float().clamp_min(1e-8)).transpose(1, 2)
    val1 = (num1.float() / den1.float().clamp_min(1e-8)).transpose(1, 2)
    return torch.stack((val0, val1), dim=2) * pot[:, :, None, None]


def _denominator_formula_triton(
    *,
    beliefs: torch.Tensor,
    legal_hands: torch.Tensor,
    grouped_leaf_indices: torch.Tensor,
    leaves_per_root: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for formula denominator benchmark")
    roots = int(legal_hands.shape[0])
    hand_count = int(legal_hands.shape[1])
    den = beliefs.new_empty(roots, leaves_per_root, 2, hand_count)
    combos = hand_combos_tensor(device=beliefs.device)
    card_a = combos[:, 0].contiguous()
    card_b = combos[:, 1].contiguous()
    total_programs = roots * leaves_per_root * 2
    _turn_operator_denominator_hu_kernel[(total_programs,)](
        beliefs,
        grouped_leaf_indices,
        legal_hands,
        card_a,
        card_b,
        den,
        total_programs,
        leaves_per_root,
        hand_count,
        BLOCK_ALL_HANDS=triton.next_power_of_2(NUM_HANDS),
        BLOCK_OUT_HANDS=triton.next_power_of_2(hand_count),
        num_warps=8,
    )
    return den


def _operator_postprocess_formula_den(
    *,
    num0: torch.Tensor,
    num1: torch.Tensor,
    den: torch.Tensor,
    pot: torch.Tensor,
) -> torch.Tensor:
    den0 = den[:, :, 0, :]
    den1 = den[:, :, 1, :]
    val0 = num0.float().transpose(1, 2) / den0.float().clamp_min(1e-8)
    val1 = num1.float().transpose(1, 2) / den1.float().clamp_min(1e-8)
    return torch.stack((val0, val1), dim=2) * pot[:, :, None, None]


def _operator_formula_postprocess_triton(
    *,
    beliefs: torch.Tensor,
    legal_hands: torch.Tensor,
    grouped_leaf_indices: torch.Tensor,
    num0: torch.Tensor,
    num1: torch.Tensor,
    pot: torch.Tensor,
    leaves_per_root: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for formula postprocess benchmark")
    roots = int(legal_hands.shape[0])
    hand_count = int(legal_hands.shape[1])
    compact_values = beliefs.new_empty(roots, leaves_per_root, 2, hand_count)
    combos = hand_combos_tensor(device=beliefs.device)
    card_a = combos[:, 0].contiguous()
    card_b = combos[:, 1].contiguous()
    total_programs = roots * leaves_per_root
    _turn_operator_formula_postprocess_hu_kernel[(total_programs,)](
        beliefs,
        grouped_leaf_indices,
        legal_hands,
        card_a,
        card_b,
        num0.contiguous(),
        num1.contiguous(),
        pot.contiguous(),
        compact_values,
        total_programs,
        leaves_per_root,
        hand_count,
        BLOCK_ALL_HANDS=triton.next_power_of_2(NUM_HANDS),
        BLOCK_OUT_HANDS=triton.next_power_of_2(hand_count),
        num_warps=8,
    )
    return compact_values


def _operator_dense_scatter(
    *,
    compact_values: torch.Tensor,
    compact_idx: torch.Tensor,
    beliefs: torch.Tensor,
    grouped_leaf_indices: torch.Tensor,
) -> torch.Tensor:
    roots, leaves_per_root = int(compact_values.shape[0]), int(compact_values.shape[1])
    dense = beliefs.new_zeros(roots, leaves_per_root, 2, NUM_HANDS)
    dense.scatter_(
        3,
        compact_idx,
        compact_values.to(dtype=dense.dtype),
    )
    grouped_dense = dense.view(roots * leaves_per_root, 2, NUM_HANDS)
    out = beliefs.new_empty(beliefs.shape)
    out.index_copy_(0, grouped_leaf_indices.long(), grouped_dense)
    return out


def _format_row(row: dict[str, Any]) -> str:
    ms_key = "cuda_mean_ms" if "cuda_mean_ms" in row else "wall_mean_ms"
    std_key = "cuda_std_ms" if "cuda_std_ms" in row else "wall_std_ms"
    cv_key = "cuda_cv" if "cuda_cv" in row else "wall_cv"
    parts = [str(row["kind"])]
    for key in ("roots", "leaves_per_root", "leaves", "operator_dtype"):
        if key in row:
            parts.append(f"{key}={row[key]}")
    parts.append(f"{ms_key}={float(row[ms_key]):.3f}")
    if std_key in row:
        parts.append(f"{std_key}={float(row[std_key]):.3f}")
    if cv_key in row:
        parts.append(f"{cv_key}={float(row[cv_key]):.3f}")
    return " ".join(parts)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=int, default=512)
    parser.add_argument("--leaves-per-root", type=int, default=20)
    parser.add_argument("--rank-bins", type=int, default=144)
    parser.add_argument("--pot", type=float, default=100.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--component-timings", action="store_true")
    parser.add_argument("--compile-operator", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--operator-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    _ensure_cuda_driver_path()
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    operator_dtype = dtype_by_name[args.operator_dtype]

    root_boards = _random_turn_roots(roots=args.roots, device=device)
    features, beliefs = _features_for_roots(
        root_boards=root_boards,
        leaves_per_root=args.leaves_per_root,
        pot=args.pot,
    )
    config = TurnRangeEquityConfig(
        rank_bins=args.rank_bins,
        chunk_size=64,
        blockers=False,
        baseline_scale=1.0,
        pot_power=1.0,
        pos_scale=-1.0,
        neg_scale=1.0,
        intercept=0.0,
    )
    leaf_cache = build_turn_range_equity_board_cache(
        features.board[:, :4],
        rank_bins=args.rank_bins,
        include_hand_runout_ok=True,
        include_sorted_bins=True,
        dedupe_boards=True,
    )
    if leaf_cache.grouped_leaf_indices is None or leaf_cache.root_leaf_offsets is None:
        raise RuntimeError("expected grouped leaf cache metadata")
    root_counts = leaf_cache.root_leaf_offsets[1:] - leaf_cache.root_leaf_offsets[:-1]
    if not bool((root_counts == args.leaves_per_root).all().item()):
        raise RuntimeError(
            "benchmark expects uniform root reuse; try a smaller root count or "
            "different seed to avoid duplicate sampled roots"
        )
    legal_hands = _legal_hands_from_board_ok(leaf_cache.board_ok)
    build_start = time.perf_counter()
    pair_payoff, pair_count = _build_pair_operators(
        rank_groups=leaf_cache.rank_groups,
        hand_runout_ok=leaf_cache.hand_runout_ok,
        legal_hands=legal_hands,
        dtype=operator_dtype,
    )
    card_incidence, legal_hand_cards = _build_card_incidence(
        legal_hands=legal_hands,
        dtype=operator_dtype,
    )
    _sync(device)
    build_ms = 1e3 * (time.perf_counter() - build_start)

    def current_baseline() -> torch.Tensor:
        return turn_range_equity_baseline(
            beliefs,
            features,
            config=config,
            dtype=torch.float32,
            board_cache=leaf_cache,
        )

    def operator_baseline() -> torch.Tensor:
        return _operator_baseline(
            beliefs=beliefs,
            context=features.context,
            legal_hands=legal_hands,
            pair_payoff=pair_payoff,
            pair_count=pair_count,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
            leaves_per_root=args.leaves_per_root,
            matmul_dtype=operator_dtype,
        )

    def operator_formula_den_baseline() -> torch.Tensor:
        compact_idx, opp_for_p0, opp_for_p1, pot = _prepare_operator_inputs(
            beliefs=beliefs,
            context=features.context,
            legal_hands=legal_hands,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
            leaves_per_root=args.leaves_per_root,
            matmul_dtype=operator_dtype,
        )
        num0 = torch.bmm(pair_payoff, opp_for_p0)
        num1 = torch.bmm(pair_payoff, opp_for_p1)
        den = _denominator_formula_triton(
            beliefs=beliefs,
            legal_hands=legal_hands,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
            leaves_per_root=args.leaves_per_root,
        )
        compact_values = _operator_postprocess_formula_den(
            num0=num0,
            num1=num1,
            den=den,
            pot=pot,
        )
        return _operator_dense_scatter(
            compact_values=compact_values,
            compact_idx=compact_idx,
            beliefs=beliefs,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
        )

    def operator_cardmatmul_den_baseline() -> torch.Tensor:
        compact_idx, opp_for_p0, opp_for_p1, pot = _prepare_operator_inputs(
            beliefs=beliefs,
            context=features.context,
            legal_hands=legal_hands,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
            leaves_per_root=args.leaves_per_root,
            matmul_dtype=operator_dtype,
        )
        num0 = torch.bmm(pair_payoff, opp_for_p0)
        num1 = torch.bmm(pair_payoff, opp_for_p1)
        den0, den1 = _denominator_formula_cardmatmul(
            opp_for_p0=opp_for_p0,
            opp_for_p1=opp_for_p1,
            card_incidence=card_incidence,
            legal_hand_cards=legal_hand_cards,
        )
        compact_values = _operator_postprocess(
            num0=num0,
            den0=den0,
            num1=num1,
            den1=den1,
            pot=pot,
        )
        return _operator_dense_scatter(
            compact_values=compact_values,
            compact_idx=compact_idx,
            beliefs=beliefs,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
        )

    def operator_fused_formula_den_baseline() -> torch.Tensor:
        compact_idx, opp_for_p0, opp_for_p1, pot = _prepare_operator_inputs(
            beliefs=beliefs,
            context=features.context,
            legal_hands=legal_hands,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
            leaves_per_root=args.leaves_per_root,
            matmul_dtype=operator_dtype,
        )
        num0 = torch.bmm(pair_payoff, opp_for_p0)
        num1 = torch.bmm(pair_payoff, opp_for_p1)
        compact_values = _operator_formula_postprocess_triton(
            beliefs=beliefs,
            legal_hands=legal_hands,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
            num0=num0,
            num1=num1,
            pot=pot,
            leaves_per_root=args.leaves_per_root,
        )
        return _operator_dense_scatter(
            compact_values=compact_values,
            compact_idx=compact_idx,
            beliefs=beliefs,
            grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
        )

    compiled_operator_baseline: Callable[[], torch.Tensor] | None = None
    compiled_formula_den_baseline: Callable[[], torch.Tensor] | None = None
    compiled_cardmatmul_den_baseline: Callable[[], torch.Tensor] | None = None
    if args.compile_operator:
        compiled_operator_baseline = torch.compile(
            operator_baseline,
            mode=args.compile_mode,
            fullgraph=False,
        )
        compiled_formula_den_baseline = torch.compile(
            operator_formula_den_baseline,
            mode=args.compile_mode,
            fullgraph=False,
        )
        compiled_cardmatmul_den_baseline = torch.compile(
            operator_cardmatmul_den_baseline,
            mode=args.compile_mode,
            fullgraph=False,
        )

    with torch.no_grad():
        ref = current_baseline()
        got = operator_baseline()
        got_formula_den = operator_formula_den_baseline()
        got_cardmatmul_den = operator_cardmatmul_den_baseline()
        got_fused_formula_den = operator_fused_formula_den_baseline()
        got_compiled = (
            compiled_operator_baseline()
            if compiled_operator_baseline is not None
            else got
        )
        _sync(device)
    diff = (ref - got).abs()
    formula_den_diff = (ref - got_formula_den).abs()
    cardmatmul_den_diff = (ref - got_cardmatmul_den).abs()
    fused_formula_den_diff = (ref - got_fused_formula_den).abs()
    compiled_diff = (ref - got_compiled).abs()
    rows: list[dict[str, Any]] = [
        {
            "kind": "operator_precompute",
            "roots": args.roots,
            "leaves_per_root": args.leaves_per_root,
            "leaves": args.roots * args.leaves_per_root,
            "operator_dtype": args.operator_dtype,
            "wall_mean_ms": build_ms,
            "pair_payoff_bytes": int(pair_payoff.numel() * pair_payoff.element_size()),
            "pair_count_bytes": int(pair_count.numel() * pair_count.element_size()),
            "card_incidence_bytes": int(
                card_incidence.numel() * card_incidence.element_size()
            ),
        },
        {
            "kind": "correctness",
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "formula_den_max_abs": float(formula_den_diff.max().item()),
            "formula_den_mean_abs": float(formula_den_diff.mean().item()),
            "cardmatmul_den_max_abs": float(cardmatmul_den_diff.max().item()),
            "cardmatmul_den_mean_abs": float(cardmatmul_den_diff.mean().item()),
            "fused_formula_den_max_abs": float(fused_formula_den_diff.max().item()),
            "fused_formula_den_mean_abs": float(fused_formula_den_diff.mean().item()),
            "compiled_max_abs": float(compiled_diff.max().item()),
            "compiled_mean_abs": float(compiled_diff.mean().item()),
        },
    ]
    print(_format_row(rows[0]), flush=True)
    print(
        f"correctness max_abs={rows[1]['max_abs']:.6g} "
        f"mean_abs={rows[1]['mean_abs']:.6g} "
        f"formula_den_max_abs={rows[1]['formula_den_max_abs']:.6g} "
        f"formula_den_mean_abs={rows[1]['formula_den_mean_abs']:.6g} "
        f"cardmatmul_den_max_abs={rows[1]['cardmatmul_den_max_abs']:.6g} "
        f"cardmatmul_den_mean_abs={rows[1]['cardmatmul_den_mean_abs']:.6g} "
        f"fused_formula_den_max_abs={rows[1]['fused_formula_den_max_abs']:.6g} "
        f"fused_formula_den_mean_abs={rows[1]['fused_formula_den_mean_abs']:.6g} "
        f"compiled_max_abs={rows[1]['compiled_max_abs']:.6g} "
        f"compiled_mean_abs={rows[1]['compiled_mean_abs']:.6g}",
        flush=True,
    )

    benchmark_fns: list[tuple[str, Callable[[], Any]]] = [
        ("current_prefix_baseline", current_baseline),
        ("pair_operator_baseline", operator_baseline),
        ("pair_operator_formula_den_baseline", operator_formula_den_baseline),
        ("pair_operator_cardmatmul_den_baseline", operator_cardmatmul_den_baseline),
        (
            "pair_operator_fused_formula_den_baseline",
            operator_fused_formula_den_baseline,
        ),
    ]
    if compiled_operator_baseline is not None:
        benchmark_fns.append(
            ("pair_operator_baseline_compiled", compiled_operator_baseline)
        )
    if compiled_formula_den_baseline is not None:
        benchmark_fns.append(
            (
                "pair_operator_formula_den_baseline_compiled",
                compiled_formula_den_baseline,
            )
        )
    if compiled_cardmatmul_den_baseline is not None:
        benchmark_fns.append(
            (
                "pair_operator_cardmatmul_den_baseline_compiled",
                compiled_cardmatmul_den_baseline,
            )
        )

    for kind, fn in benchmark_fns:
        row: dict[str, Any] = {
            "kind": kind,
            "roots": args.roots,
            "leaves_per_root": args.leaves_per_root,
            "leaves": args.roots * args.leaves_per_root,
            "operator_dtype": args.operator_dtype,
        }
        row.update(
            _time_call(
                device,
                fn,
                warmup=args.warmup,
                iters=args.iters,
                include_samples=args.include_samples,
            )
        )
        rows.append(row)
        print(_format_row(row), flush=True)

    if args.component_timings:
        with torch.no_grad():
            compact_idx, opp_for_p0, opp_for_p1, pot = _prepare_operator_inputs(
                beliefs=beliefs,
                context=features.context,
                legal_hands=legal_hands,
                grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                leaves_per_root=args.leaves_per_root,
                matmul_dtype=operator_dtype,
            )
            num0, den0, num1, den1 = _operator_gemm(
                pair_payoff=pair_payoff,
                pair_count=pair_count,
                opp_for_p0=opp_for_p0,
                opp_for_p1=opp_for_p1,
            )
            cardmatmul_den0, cardmatmul_den1 = _denominator_formula_cardmatmul(
                opp_for_p0=opp_for_p0,
                opp_for_p1=opp_for_p1,
                card_incidence=card_incidence,
                legal_hand_cards=legal_hand_cards,
            )
            formula_den = _denominator_formula_triton(
                beliefs=beliefs,
                legal_hands=legal_hands,
                grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                leaves_per_root=args.leaves_per_root,
            )
            compact_values = _operator_postprocess(
                num0=num0,
                den0=den0,
                num1=num1,
                den1=den1,
                pot=pot,
            )
            cardmatmul_compact_values = _operator_postprocess(
                num0=num0,
                den0=cardmatmul_den0,
                num1=num1,
                den1=cardmatmul_den1,
                pot=pot,
            )
            formula_compact_values = _operator_postprocess_formula_den(
                num0=num0,
                num1=num1,
                den=formula_den,
                pot=pot,
            )
            fused_formula_compact_values = _operator_formula_postprocess_triton(
                beliefs=beliefs,
                legal_hands=legal_hands,
                grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                num0=num0,
                num1=num1,
                pot=pot,
                leaves_per_root=args.leaves_per_root,
            )
            _sync(device)

        component_fns: tuple[tuple[str, Callable[[], Any]], ...] = (
            (
                "operator_component_prepare_inputs",
                lambda: _prepare_operator_inputs(
                    beliefs=beliefs,
                    context=features.context,
                    legal_hands=legal_hands,
                    grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                    leaves_per_root=args.leaves_per_root,
                    matmul_dtype=operator_dtype,
                ),
            ),
            (
                "operator_component_gemm_all",
                lambda: _operator_gemm(
                    pair_payoff=pair_payoff,
                    pair_count=pair_count,
                    opp_for_p0=opp_for_p0,
                    opp_for_p1=opp_for_p1,
                ),
            ),
            (
                "operator_component_gemm_payoff",
                lambda: (
                    torch.bmm(pair_payoff, opp_for_p0),
                    torch.bmm(pair_payoff, opp_for_p1),
                ),
            ),
            (
                "operator_component_gemm_count",
                lambda: (
                    torch.bmm(pair_count, opp_for_p0),
                    torch.bmm(pair_count, opp_for_p1),
                ),
            ),
            (
                "operator_component_cardmatmul_den",
                lambda: _denominator_formula_cardmatmul(
                    opp_for_p0=opp_for_p0,
                    opp_for_p1=opp_for_p1,
                    card_incidence=card_incidence,
                    legal_hand_cards=legal_hand_cards,
                ),
            ),
            (
                "operator_component_formula_den",
                lambda: _denominator_formula_triton(
                    beliefs=beliefs,
                    legal_hands=legal_hands,
                    grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                    leaves_per_root=args.leaves_per_root,
                ),
            ),
            (
                "operator_component_postprocess",
                lambda: _operator_postprocess(
                    num0=num0,
                    den0=den0,
                    num1=num1,
                    den1=den1,
                    pot=pot,
                ),
            ),
            (
                "operator_component_postprocess_cardmatmul_den",
                lambda: _operator_postprocess(
                    num0=num0,
                    den0=cardmatmul_den0,
                    num1=num1,
                    den1=cardmatmul_den1,
                    pot=pot,
                ),
            ),
            (
                "operator_component_postprocess_formula_den",
                lambda: _operator_postprocess_formula_den(
                    num0=num0,
                    num1=num1,
                    den=formula_den,
                    pot=pot,
                ),
            ),
            (
                "operator_component_fused_formula_postprocess",
                lambda: _operator_formula_postprocess_triton(
                    beliefs=beliefs,
                    legal_hands=legal_hands,
                    grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                    num0=num0,
                    num1=num1,
                    pot=pot,
                    leaves_per_root=args.leaves_per_root,
                ),
            ),
            (
                "operator_component_dense_scatter",
                lambda: _operator_dense_scatter(
                    compact_values=compact_values,
                    compact_idx=compact_idx,
                    beliefs=beliefs,
                    grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                ),
            ),
            (
                "operator_component_dense_scatter_cardmatmul_den",
                lambda: _operator_dense_scatter(
                    compact_values=cardmatmul_compact_values,
                    compact_idx=compact_idx,
                    beliefs=beliefs,
                    grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                ),
            ),
            (
                "operator_component_dense_scatter_formula_den",
                lambda: _operator_dense_scatter(
                    compact_values=formula_compact_values,
                    compact_idx=compact_idx,
                    beliefs=beliefs,
                    grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                ),
            ),
            (
                "operator_component_dense_scatter_fused_formula_den",
                lambda: _operator_dense_scatter(
                    compact_values=fused_formula_compact_values,
                    compact_idx=compact_idx,
                    beliefs=beliefs,
                    grouped_leaf_indices=leaf_cache.grouped_leaf_indices,
                ),
            ),
        )
        for kind, fn in component_fns:
            row = {
                "kind": kind,
                "roots": args.roots,
                "leaves_per_root": args.leaves_per_root,
                "leaves": args.roots * args.leaves_per_root,
                "operator_dtype": args.operator_dtype,
            }
            row.update(
                _time_call(
                    device,
                    fn,
                    warmup=args.warmup,
                    iters=args.iters,
                    include_samples=args.include_samples,
                )
            )
            rows.append(row)
            print(_format_row(row), flush=True)

    current_ms = next(
        float(row.get("cuda_mean_ms", row.get("wall_mean_ms", 0.0)))
        for row in rows
        if row["kind"] == "current_prefix_baseline"
    )
    operator_ms = next(
        float(row.get("cuda_mean_ms", row.get("wall_mean_ms", 0.0)))
        for row in rows
        if row["kind"] == "pair_operator_baseline"
    )
    summary = {
        "current_mean_ms": current_ms,
        "operator_mean_ms": operator_ms,
        "speedup": current_ms / max(operator_ms, 1e-9),
        "operator_precompute_ms": build_ms,
        "max_abs": rows[1]["max_abs"],
        "mean_abs": rows[1]["mean_abs"],
    }
    payload = {
        "argv": argv,
        "device": str(device),
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
