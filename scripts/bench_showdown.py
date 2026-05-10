#!/usr/bin/env python3
"""Microbenchmark for _showdown_value_both with multiple Triton variants.

Builds a real FusedSparseCFREvaluator + a representative subgame so the
hand_rank_data, showdown_indices, showdown_potential, env.scale, etc. are
all populated correctly, then times the existing PyTorch implementation
against successive Triton kernel variants. Each variant is verified for
correctness vs the PyTorch baseline before being timed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import hydra
import torch
import triton
import triton.language as tl
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


REPEATS = 30
WARMUP = 5
MC = 64  # padded slot count per (env, card) — actual non-board cards have 51


# =============================================================================
# Pre-compute (one per subgame): card_positions per (env, card).
# =============================================================================


def _precompute_card_positions(
    hands_c1c2_sorted: torch.Tensor, num_cards: int = 52, max_per_card: int = MC
) -> torch.Tensor:
    """For each (env m, card c), find sorted j-positions where c appears
    in hands_c1c2_sorted. Returns [M, 52, max_per_card] padded with H."""
    M, H, _ = hands_c1c2_sorted.shape
    device = hands_c1c2_sorted.device
    cards = torch.arange(num_cards, device=device).view(1, 1, num_cards)
    c1 = hands_c1c2_sorted[..., 0].unsqueeze(-1)
    c2 = hands_c1c2_sorted[..., 1].unsqueeze(-1)
    incidence = (c1 == cards) | (c2 == cards)
    j_idx = torch.arange(H, device=device).view(1, H, 1).expand(M, H, num_cards)
    masked_j = torch.where(incidence, j_idx, H)
    masked_j_t = masked_j.transpose(1, 2).contiguous()
    sorted_j, _ = masked_j_t.sort(dim=-1)
    return sorted_j[..., :max_per_card].contiguous()


# =============================================================================
# v1: per-hero Triton kernel — baseline.
# =============================================================================


@triton.jit
def _hero_ev_kernel_v1(
    b_opp_sorted_ptr,        # [M, H]
    P_padded_ptr,            # [M, H+1]
    card_positions_ptr,      # [M, 52, MC] int32, padded with H
    card_cumsum_ptr,         # [M, 52, MC] float32
    L_idx_ptr,               # [M, H]
    R_idx_ptr,               # [M, H]
    c1_sorted_ptr,           # [M, H]
    c2_sorted_ptr,           # [M, H]
    ev_out_ptr,              # [M, H] sorted-order
    H,
    NUM_CARDS: tl.constexpr,
    MC_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < H

    L = tl.load(L_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    R = tl.load(R_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    c1 = tl.load(c1_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    c2 = tl.load(c2_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    b_at_k = tl.load(b_opp_sorted_ptr + m * H + k_offs, mask=mask_k, other=0.0)

    P_L = tl.load(P_padded_ptr + m * (H + 1) + L, mask=mask_k, other=0.0)
    P_R = tl.load(P_padded_ptr + m * (H + 1) + R, mask=mask_k, other=0.0)

    slot_off = tl.arange(0, MC_K)
    pos_c1 = tl.load(
        card_positions_ptr + m * NUM_CARDS * MC_K + c1[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=H,
    )
    pos_c2 = tl.load(
        card_positions_ptr + m * NUM_CARDS * MC_K + c2[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=H,
    )
    cum_c1 = tl.load(
        card_cumsum_ptr + m * NUM_CARDS * MC_K + c1[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=0.0,
    )
    cum_c2 = tl.load(
        card_cumsum_ptr + m * NUM_CARDS * MC_K + c2[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=0.0,
    )

    H_vec = tl.zeros((BLOCK_K,), dtype=tl.int32) + H
    slot_L_c1 = tl.sum((pos_c1 < L[:, None]).to(tl.int32), axis=1)
    slot_L_c2 = tl.sum((pos_c2 < L[:, None]).to(tl.int32), axis=1)
    slot_R_c1 = tl.sum((pos_c1 < R[:, None]).to(tl.int32), axis=1)
    slot_R_c2 = tl.sum((pos_c2 < R[:, None]).to(tl.int32), axis=1)
    slot_last_c1 = tl.sum((pos_c1 < H_vec[:, None]).to(tl.int32), axis=1)
    slot_last_c2 = tl.sum((pos_c2 < H_vec[:, None]).to(tl.int32), axis=1)

    def gather(cum_block, slot, slot_off=slot_off):
        slot_idx = tl.maximum(slot - 1, 0)
        return tl.where(
            slot > 0,
            tl.sum(cum_block * (slot_off[None, :] == slot_idx[:, None]).to(tl.float32), axis=1),
            0.0,
        )

    Pcards_L_c1 = gather(cum_c1, slot_L_c1)
    Pcards_L_c2 = gather(cum_c2, slot_L_c2)
    Pcards_R_c1 = gather(cum_c1, slot_R_c1)
    Pcards_R_c2 = gather(cum_c2, slot_R_c2)
    Pcards_last_c1 = gather(cum_c1, slot_last_c1)
    Pcards_last_c2 = gather(cum_c2, slot_last_c2)

    win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
    seg_sum = P_R - P_L
    seg_c1 = Pcards_R_c1 - Pcards_L_c1
    seg_c2 = Pcards_R_c2 - Pcards_L_c2
    tie_mass = seg_sum - seg_c1 - seg_c2 + b_at_k

    denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
    valid = denom > 1e-8
    denom_safe = tl.where(valid, denom, 1.0)
    win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
    tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
    loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
    EV = win_prob - loss_prob
    tl.store(ev_out_ptr + m * H + k_offs, EV, mask=mask_k)


def _common_setup(beliefs, hero, hrd, card_positions):
    """Build per-call inputs shared by all variants (per-hero)."""
    villain = 1 - hero
    M, _, _ = beliefs.shape
    device = beliefs.device
    dtype = torch.float32
    H_int = hrd.sorted_indices.shape[1]

    b_opp_sorted = beliefs[:, villain, :].to(dtype).gather(1, hrd.sorted_indices).contiguous()
    b_padded = torch.cat(
        [b_opp_sorted, torch.zeros(M, 1, device=device, dtype=dtype)], dim=1
    ).contiguous()

    pos_flat = card_positions.clamp(min=0, max=H_int).view(M, -1)
    b_at_pos = b_padded.gather(1, pos_flat).view(M, card_positions.shape[1], MC)
    card_cumsum = b_at_pos.cumsum(dim=-1).contiguous()

    P = b_opp_sorted.cumsum(dim=1)
    P_padded = torch.cat([torch.zeros(M, 1, device=device, dtype=dtype), P], dim=1).contiguous()
    return b_opp_sorted, P_padded, card_cumsum


def _final_postprocess(ev_sorted, hrd, hero, showdown_potential, env_scale):
    dtype = ev_sorted.dtype
    ev = torch.gather(ev_sorted, 1, hrd.inv_sorted)
    ev = ev * hrd.hand_ok_mask.to(dtype)
    return ev * showdown_potential[:, hero, None] / env_scale[:, None]


def run_v1(beliefs, hero, hrd, card_positions, showdown_potential, env_scale, block_k=64):
    b_opp_sorted, P_padded, card_cumsum = _common_setup(beliefs, hero, hrd, card_positions)
    M, _, NUM_HANDS = beliefs.shape
    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    L_idx = hrd.L_idx.contiguous()
    R_idx = hrd.R_idx.contiguous()
    ev_sorted = torch.empty(M, NUM_HANDS, device=beliefs.device, dtype=torch.float32)
    grid = (M, triton.cdiv(NUM_HANDS, block_k))
    _hero_ev_kernel_v1[grid](
        b_opp_sorted, P_padded, card_positions.to(torch.int32), card_cumsum,
        L_idx, R_idx, c1, c2, ev_sorted,
        NUM_HANDS, NUM_CARDS=card_positions.shape[1], MC_K=MC, BLOCK_K=block_k, num_warps=4,
    )
    return _final_postprocess(ev_sorted, hrd, hero, showdown_potential, env_scale)


def run_v1_both(beliefs, hrd, card_positions, showdown_potential, env_scale, block_k=64):
    """Run v1 twice (one per hero), to compare against PyTorch _showdown_value_both."""
    M, _, NUM_HANDS = beliefs.shape
    out = torch.empty(M, 2, NUM_HANDS, device=beliefs.device, dtype=torch.float32)
    out[:, 0] = run_v1(beliefs, 0, hrd, card_positions, showdown_potential, env_scale, block_k)
    out[:, 1] = run_v1(beliefs, 1, hrd, card_positions, showdown_potential, env_scale, block_k)
    return out


# =============================================================================
# v2: fused-both-heroes kernel. Loads pos/c1/c2/L/R/slot_off once per program;
# per-hero work is the cumsum lookups.
# =============================================================================


@triton.jit
def _hero_ev_kernel_v2(
    b_opp_sorted_ptr,        # [M, 2, H]
    P_padded_ptr,            # [M, 2, H+1]
    card_positions_ptr,      # [M, 52, MC] int32 — shared across heroes
    card_cumsum_ptr,         # [M, 2, 52, MC] float32
    L_idx_ptr,               # [M, H]
    R_idx_ptr,               # [M, H]
    c1_sorted_ptr,           # [M, H]
    c2_sorted_ptr,           # [M, H]
    ev_out_ptr,              # [M, 2, H] sorted-order
    H,
    NUM_CARDS: tl.constexpr,
    MC_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < H

    L = tl.load(L_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    R = tl.load(R_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    c1 = tl.load(c1_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    c2 = tl.load(c2_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)

    slot_off = tl.arange(0, MC_K)
    pos_c1 = tl.load(
        card_positions_ptr + m * NUM_CARDS * MC_K + c1[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=H,
    )
    pos_c2 = tl.load(
        card_positions_ptr + m * NUM_CARDS * MC_K + c2[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=H,
    )

    H_vec = tl.zeros((BLOCK_K,), dtype=tl.int32) + H
    slot_L_c1 = tl.sum((pos_c1 < L[:, None]).to(tl.int32), axis=1)
    slot_L_c2 = tl.sum((pos_c2 < L[:, None]).to(tl.int32), axis=1)
    slot_R_c1 = tl.sum((pos_c1 < R[:, None]).to(tl.int32), axis=1)
    slot_R_c2 = tl.sum((pos_c2 < R[:, None]).to(tl.int32), axis=1)
    slot_last_c1 = tl.sum((pos_c1 < H_vec[:, None]).to(tl.int32), axis=1)
    slot_last_c2 = tl.sum((pos_c2 < H_vec[:, None]).to(tl.int32), axis=1)

    slot_L_c1_idx = tl.maximum(slot_L_c1 - 1, 0)
    slot_L_c2_idx = tl.maximum(slot_L_c2 - 1, 0)
    slot_R_c1_idx = tl.maximum(slot_R_c1 - 1, 0)
    slot_R_c2_idx = tl.maximum(slot_R_c2 - 1, 0)
    slot_last_c1_idx = tl.maximum(slot_last_c1 - 1, 0)
    slot_last_c2_idx = tl.maximum(slot_last_c2 - 1, 0)

    has_L_c1 = slot_L_c1 > 0
    has_L_c2 = slot_L_c2 > 0
    has_R_c1 = slot_R_c1 > 0
    has_R_c2 = slot_R_c2 > 0
    has_last_c1 = slot_last_c1 > 0
    has_last_c2 = slot_last_c2 > 0

    # Loop over heroes — Triton unrolls.
    for hero in tl.static_range(2):
        b_at_k = tl.load(
            b_opp_sorted_ptr + m * 2 * H + hero * H + k_offs, mask=mask_k, other=0.0
        )
        P_L = tl.load(
            P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L, mask=mask_k, other=0.0
        )
        P_R = tl.load(
            P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R, mask=mask_k, other=0.0
        )
        cum_c1 = tl.load(
            card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            + c1[:, None] * MC_K + slot_off[None, :],
            mask=mask_k[:, None], other=0.0,
        )
        cum_c2 = tl.load(
            card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            + c2[:, None] * MC_K + slot_off[None, :],
            mask=mask_k[:, None], other=0.0,
        )

        Pcards_L_c1 = tl.where(
            has_L_c1,
            tl.sum(cum_c1 * (slot_off[None, :] == slot_L_c1_idx[:, None]).to(tl.float32), axis=1),
            0.0,
        )
        Pcards_L_c2 = tl.where(
            has_L_c2,
            tl.sum(cum_c2 * (slot_off[None, :] == slot_L_c2_idx[:, None]).to(tl.float32), axis=1),
            0.0,
        )
        Pcards_R_c1 = tl.where(
            has_R_c1,
            tl.sum(cum_c1 * (slot_off[None, :] == slot_R_c1_idx[:, None]).to(tl.float32), axis=1),
            0.0,
        )
        Pcards_R_c2 = tl.where(
            has_R_c2,
            tl.sum(cum_c2 * (slot_off[None, :] == slot_R_c2_idx[:, None]).to(tl.float32), axis=1),
            0.0,
        )
        Pcards_last_c1 = tl.where(
            has_last_c1,
            tl.sum(cum_c1 * (slot_off[None, :] == slot_last_c1_idx[:, None]).to(tl.float32), axis=1),
            0.0,
        )
        Pcards_last_c2 = tl.where(
            has_last_c2,
            tl.sum(cum_c2 * (slot_off[None, :] == slot_last_c2_idx[:, None]).to(tl.float32), axis=1),
            0.0,
        )

        win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
        seg_c1 = Pcards_R_c1 - Pcards_L_c1
        seg_c2 = Pcards_R_c2 - Pcards_L_c2
        tie_mass = (P_R - P_L) - seg_c1 - seg_c2 + b_at_k

        denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
        valid = denom > 1e-8
        denom_safe = tl.where(valid, denom, 1.0)
        win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
        tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
        loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
        EV = win_prob - loss_prob

        tl.store(ev_out_ptr + m * 2 * H + hero * H + k_offs, EV, mask=mask_k)


# =============================================================================
# v3: v2 + tl.gather instead of masked-sum reduction for cumsum lookup.
# =============================================================================


@triton.jit
def _hero_ev_kernel_v3(
    b_opp_sorted_ptr,        # [M, 2, H]
    P_padded_ptr,            # [M, 2, H+1]
    card_positions_ptr,      # [M, 52, MC] int32
    card_cumsum_ptr,         # [M, 2, 52, MC] float32
    L_idx_ptr,               # [M, H]
    R_idx_ptr,               # [M, H]
    c1_sorted_ptr,           # [M, H]
    c2_sorted_ptr,           # [M, H]
    ev_out_ptr,              # [M, 2, H] sorted
    H,
    NUM_CARDS: tl.constexpr,
    MC_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < H

    L = tl.load(L_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    R = tl.load(R_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    c1 = tl.load(c1_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    c2 = tl.load(c2_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)

    slot_off = tl.arange(0, MC_K)
    pos_c1 = tl.load(
        card_positions_ptr + m * NUM_CARDS * MC_K + c1[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=H,
    )
    pos_c2 = tl.load(
        card_positions_ptr + m * NUM_CARDS * MC_K + c2[:, None] * MC_K + slot_off[None, :],
        mask=mask_k[:, None], other=H,
    )

    H_vec = tl.zeros((BLOCK_K,), dtype=tl.int32) + H
    slot_L_c1 = tl.sum((pos_c1 < L[:, None]).to(tl.int32), axis=1)
    slot_L_c2 = tl.sum((pos_c2 < L[:, None]).to(tl.int32), axis=1)
    slot_R_c1 = tl.sum((pos_c1 < R[:, None]).to(tl.int32), axis=1)
    slot_R_c2 = tl.sum((pos_c2 < R[:, None]).to(tl.int32), axis=1)
    slot_last_c1 = tl.sum((pos_c1 < H_vec[:, None]).to(tl.int32), axis=1)
    slot_last_c2 = tl.sum((pos_c2 < H_vec[:, None]).to(tl.int32), axis=1)

    # tl.gather: pick along axis=1 with per-row slot index. Need shape match.
    has_L_c1 = slot_L_c1 > 0
    has_L_c2 = slot_L_c2 > 0
    has_R_c1 = slot_R_c1 > 0
    has_R_c2 = slot_R_c2 > 0
    has_last_c1 = slot_last_c1 > 0
    has_last_c2 = slot_last_c2 > 0
    slot_L_c1_idx = tl.maximum(slot_L_c1 - 1, 0)[:, None]
    slot_L_c2_idx = tl.maximum(slot_L_c2 - 1, 0)[:, None]
    slot_R_c1_idx = tl.maximum(slot_R_c1 - 1, 0)[:, None]
    slot_R_c2_idx = tl.maximum(slot_R_c2 - 1, 0)[:, None]
    slot_last_c1_idx = tl.maximum(slot_last_c1 - 1, 0)[:, None]
    slot_last_c2_idx = tl.maximum(slot_last_c2 - 1, 0)[:, None]

    for hero in tl.static_range(2):
        b_at_k = tl.load(
            b_opp_sorted_ptr + m * 2 * H + hero * H + k_offs, mask=mask_k, other=0.0
        )
        P_L = tl.load(
            P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L, mask=mask_k, other=0.0
        )
        P_R = tl.load(
            P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R, mask=mask_k, other=0.0
        )
        cum_c1 = tl.load(
            card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            + c1[:, None] * MC_K + slot_off[None, :],
            mask=mask_k[:, None], other=0.0,
        )
        cum_c2 = tl.load(
            card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            + c2[:, None] * MC_K + slot_off[None, :],
            mask=mask_k[:, None], other=0.0,
        )

        # Use tl.gather to pull one element per row.
        Pcards_L_c1 = tl.where(has_L_c1, tl.gather(cum_c1, slot_L_c1_idx, axis=1).reshape((BLOCK_K,)), 0.0)
        Pcards_L_c2 = tl.where(has_L_c2, tl.gather(cum_c2, slot_L_c2_idx, axis=1).reshape((BLOCK_K,)), 0.0)
        Pcards_R_c1 = tl.where(has_R_c1, tl.gather(cum_c1, slot_R_c1_idx, axis=1).reshape((BLOCK_K,)), 0.0)
        Pcards_R_c2 = tl.where(has_R_c2, tl.gather(cum_c2, slot_R_c2_idx, axis=1).reshape((BLOCK_K,)), 0.0)
        Pcards_last_c1 = tl.where(has_last_c1, tl.gather(cum_c1, slot_last_c1_idx, axis=1).reshape((BLOCK_K,)), 0.0)
        Pcards_last_c2 = tl.where(has_last_c2, tl.gather(cum_c2, slot_last_c2_idx, axis=1).reshape((BLOCK_K,)), 0.0)

        win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
        seg_c1 = Pcards_R_c1 - Pcards_L_c1
        seg_c2 = Pcards_R_c2 - Pcards_L_c2
        tie_mass = (P_R - P_L) - seg_c1 - seg_c2 + b_at_k

        denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
        valid = denom > 1e-8
        denom_safe = tl.where(valid, denom, 1.0)
        win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
        tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
        loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
        EV = win_prob - loss_prob
        tl.store(ev_out_ptr + m * 2 * H + hero * H + k_offs, EV, mask=mask_k)


# =============================================================================
# v4: precomputed slot indices per (m, k, lookup_type). Per subgame: precompute
# slot_L_c1, slot_L_c2, slot_R_c1, slot_R_c2, slot_last_c1, slot_last_c2 as
# [M, H] int32 each. Per call: kernel does scalar cumsum lookups instead of
# (BLOCK_K, MC) loads + reductions — eliminates the 64× bandwidth waste.
# =============================================================================


def _precompute_lookup_slots(
    card_positions: torch.Tensor,    # [M, 52, MC] padded with H
    hrd,
) -> tuple[torch.Tensor, ...]:
    """Slot index per (m, k, lookup type). Returns 6 [M, H] int32 tensors:
    slot_L_c1, slot_L_c2, slot_R_c1, slot_R_c2, slot_last_c1, slot_last_c2."""
    M, _, mc = card_positions.shape
    H = card_positions.shape[1] * mc  # placeholder
    H = hrd.L_idx.shape[1]
    L = hrd.L_idx                                      # [M, H]
    R = hrd.R_idx
    c1 = hrd.hands_c1c2_sorted[..., 0]                 # [M, H]
    c2 = hrd.hands_c1c2_sorted[..., 1]
    # Gather per-(m, k) the position list for c1/c2: shape [M, H, MC]
    pos_for_c1 = card_positions.gather(
        1, c1.unsqueeze(-1).expand(-1, -1, mc)
    )                                                   # [M, H, MC]
    pos_for_c2 = card_positions.gather(
        1, c2.unsqueeze(-1).expand(-1, -1, mc)
    )
    H_t = torch.full_like(L, H)
    slot_L_c1 = (pos_for_c1 < L.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
    slot_L_c2 = (pos_for_c2 < L.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
    slot_R_c1 = (pos_for_c1 < R.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
    slot_R_c2 = (pos_for_c2 < R.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
    slot_last_c1 = (pos_for_c1 < H_t.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
    slot_last_c2 = (pos_for_c2 < H_t.unsqueeze(-1)).sum(dim=-1).to(torch.int32)
    return slot_L_c1, slot_L_c2, slot_R_c1, slot_R_c2, slot_last_c1, slot_last_c2


# =============================================================================
# v7: v4 + write at inv_sorted position (skip the post-process gather).
# =============================================================================


@triton.jit
def _hero_ev_kernel_v7(
    b_opp_sorted_ptr,        # [M, 2, H]
    P_padded_ptr,            # [M, 2, H+1]
    card_cumsum_ptr,         # [M, 2, 52, MC]
    L_idx_ptr,               # [M, H]
    R_idx_ptr,               # [M, H]
    c1_sorted_ptr,
    c2_sorted_ptr,
    inv_sorted_ptr,          # [M, H] — write target index for each k
    slot_L_c1_ptr,
    slot_L_c2_ptr,
    slot_R_c1_ptr,
    slot_R_c2_ptr,
    slot_last_c1_ptr,
    slot_last_c2_ptr,
    ev_out_ptr,              # [M, 2, H] in unsorted order
    H,
    NUM_CARDS: tl.constexpr,
    MC_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < H

    L = tl.load(L_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    R = tl.load(R_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    c1 = tl.load(c1_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    c2 = tl.load(c2_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    inv_k = tl.load(inv_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL1 = tl.load(slot_L_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL2 = tl.load(slot_L_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR1 = tl.load(slot_R_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR2 = tl.load(slot_R_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast1 = tl.load(slot_last_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast2 = tl.load(slot_last_c2_ptr + m * H + k_offs, mask=mask_k, other=0)

    has_L1 = sL1 > 0
    has_L2 = sL2 > 0
    has_R1 = sR1 > 0
    has_R2 = sR2 > 0
    has_Last1 = sLast1 > 0
    has_Last2 = sLast2 > 0
    iL1 = tl.maximum(sL1 - 1, 0)
    iL2 = tl.maximum(sL2 - 1, 0)
    iR1 = tl.maximum(sR1 - 1, 0)
    iR2 = tl.maximum(sR2 - 1, 0)
    iLast1 = tl.maximum(sLast1 - 1, 0)
    iLast2 = tl.maximum(sLast2 - 1, 0)

    for hero in tl.static_range(2):
        b_at_k = tl.load(b_opp_sorted_ptr + m * 2 * H + hero * H + k_offs, mask=mask_k, other=0.0)
        P_L = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L, mask=mask_k, other=0.0)
        P_R = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R, mask=mask_k, other=0.0)

        cum_base = card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
        Pcards_L_c1 = tl.where(has_L1, tl.load(cum_base + c1 * MC_K + iL1, mask=mask_k, other=0.0), 0.0)
        Pcards_L_c2 = tl.where(has_L2, tl.load(cum_base + c2 * MC_K + iL2, mask=mask_k, other=0.0), 0.0)
        Pcards_R_c1 = tl.where(has_R1, tl.load(cum_base + c1 * MC_K + iR1, mask=mask_k, other=0.0), 0.0)
        Pcards_R_c2 = tl.where(has_R2, tl.load(cum_base + c2 * MC_K + iR2, mask=mask_k, other=0.0), 0.0)
        Pcards_last_c1 = tl.where(has_Last1, tl.load(cum_base + c1 * MC_K + iLast1, mask=mask_k, other=0.0), 0.0)
        Pcards_last_c2 = tl.where(has_Last2, tl.load(cum_base + c2 * MC_K + iLast2, mask=mask_k, other=0.0), 0.0)

        win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
        seg_c1 = Pcards_R_c1 - Pcards_L_c1
        seg_c2 = Pcards_R_c2 - Pcards_L_c2
        tie_mass = (P_R - P_L) - seg_c1 - seg_c2 + b_at_k

        denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
        valid = denom > 1e-8
        denom_safe = tl.where(valid, denom, 1.0)
        win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
        tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
        loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
        EV = win_prob - loss_prob
        # Write at the original (unsorted) position.
        tl.store(ev_out_ptr + m * 2 * H + hero * H + inv_k, EV, mask=mask_k)


def run_v7_both(beliefs, hrd, card_positions, slot_tensors, showdown_potential, env_scale, block_k=64):
    M, _, NUM_HANDS = beliefs.shape
    device = beliefs.device
    dtype = torch.float32

    b_opp_both, P_both, cum_both = _setup_both(beliefs, hrd, card_positions)
    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    L_idx = hrd.L_idx.contiguous()
    R_idx = hrd.R_idx.contiguous()
    inv_sorted = hrd.inv_sorted.contiguous()
    sL1, sL2, sR1, sR2, sLast1, sLast2 = slot_tensors

    ev_unsorted = torch.zeros(M, 2, NUM_HANDS, device=device, dtype=dtype)
    grid = (M, triton.cdiv(NUM_HANDS, block_k))
    _hero_ev_kernel_v7[grid](
        b_opp_both, P_both, cum_both,
        L_idx, R_idx, c1, c2, inv_sorted,
        sL1, sL2, sR1, sR2, sLast1, sLast2,
        ev_unsorted,
        NUM_HANDS, NUM_CARDS=card_positions.shape[1], MC_K=MC, BLOCK_K=block_k, num_warps=4,
    )
    # Skip the gather; just apply hand_ok / potential / scale.
    ev = ev_unsorted * hrd.hand_ok_mask.to(dtype).unsqueeze(1)
    return ev * showdown_potential.unsqueeze(-1) / env_scale[:, None, None]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": bk}, num_warps=nw)
        for bk in (32, 64, 128, 256)
        for nw in (2, 4, 8)
    ],
    key=["H"],
)
@triton.jit
def _hero_ev_kernel_v5(
    b_opp_sorted_ptr,        # [M, 2, H]
    P_padded_ptr,            # [M, 2, H+1]
    card_cumsum_ptr,         # [M, 2, 52, MC]
    L_idx_ptr,               # [M, H]
    R_idx_ptr,               # [M, H]
    c1_sorted_ptr,           # [M, H]
    c2_sorted_ptr,           # [M, H]
    slot_L_c1_ptr,           # [M, H]
    slot_L_c2_ptr,
    slot_R_c1_ptr,
    slot_R_c2_ptr,
    slot_last_c1_ptr,
    slot_last_c2_ptr,
    ev_out_ptr,              # [M, 2, H] sorted
    H,
    NUM_CARDS: tl.constexpr,
    MC_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < H

    L = tl.load(L_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    R = tl.load(R_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    c1 = tl.load(c1_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    c2 = tl.load(c2_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL1 = tl.load(slot_L_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL2 = tl.load(slot_L_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR1 = tl.load(slot_R_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR2 = tl.load(slot_R_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast1 = tl.load(slot_last_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast2 = tl.load(slot_last_c2_ptr + m * H + k_offs, mask=mask_k, other=0)

    has_L1 = sL1 > 0
    has_L2 = sL2 > 0
    has_R1 = sR1 > 0
    has_R2 = sR2 > 0
    has_Last1 = sLast1 > 0
    has_Last2 = sLast2 > 0
    iL1 = tl.maximum(sL1 - 1, 0)
    iL2 = tl.maximum(sL2 - 1, 0)
    iR1 = tl.maximum(sR1 - 1, 0)
    iR2 = tl.maximum(sR2 - 1, 0)
    iLast1 = tl.maximum(sLast1 - 1, 0)
    iLast2 = tl.maximum(sLast2 - 1, 0)

    for hero in tl.static_range(2):
        b_at_k = tl.load(
            b_opp_sorted_ptr + m * 2 * H + hero * H + k_offs, mask=mask_k, other=0.0
        )
        P_L = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L, mask=mask_k, other=0.0)
        P_R = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R, mask=mask_k, other=0.0)

        cum_base = card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
        Pcards_L_c1 = tl.where(has_L1, tl.load(cum_base + c1 * MC_K + iL1, mask=mask_k, other=0.0), 0.0)
        Pcards_L_c2 = tl.where(has_L2, tl.load(cum_base + c2 * MC_K + iL2, mask=mask_k, other=0.0), 0.0)
        Pcards_R_c1 = tl.where(has_R1, tl.load(cum_base + c1 * MC_K + iR1, mask=mask_k, other=0.0), 0.0)
        Pcards_R_c2 = tl.where(has_R2, tl.load(cum_base + c2 * MC_K + iR2, mask=mask_k, other=0.0), 0.0)
        Pcards_last_c1 = tl.where(has_Last1, tl.load(cum_base + c1 * MC_K + iLast1, mask=mask_k, other=0.0), 0.0)
        Pcards_last_c2 = tl.where(has_Last2, tl.load(cum_base + c2 * MC_K + iLast2, mask=mask_k, other=0.0), 0.0)

        win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
        seg_c1 = Pcards_R_c1 - Pcards_L_c1
        seg_c2 = Pcards_R_c2 - Pcards_L_c2
        tie_mass = (P_R - P_L) - seg_c1 - seg_c2 + b_at_k

        denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
        valid = denom > 1e-8
        denom_safe = tl.where(valid, denom, 1.0)
        win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
        tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
        loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
        EV = win_prob - loss_prob
        tl.store(ev_out_ptr + m * 2 * H + hero * H + k_offs, EV, mask=mask_k)


def run_v5_both(beliefs, hrd, card_positions, slot_tensors, showdown_potential, env_scale):
    M, _, NUM_HANDS = beliefs.shape
    device = beliefs.device
    dtype = torch.float32

    b_opp_both, P_both, cum_both = _setup_both(beliefs, hrd, card_positions)
    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    L_idx = hrd.L_idx.contiguous()
    R_idx = hrd.R_idx.contiguous()
    sL1, sL2, sR1, sR2, sLast1, sLast2 = slot_tensors

    ev_sorted = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
    grid = lambda meta: (M, triton.cdiv(NUM_HANDS, meta["BLOCK_K"]))
    _hero_ev_kernel_v5[grid](
        b_opp_both, P_both, cum_both,
        L_idx, R_idx, c1, c2,
        sL1, sL2, sR1, sR2, sLast1, sLast2,
        ev_sorted,
        NUM_HANDS, NUM_CARDS=card_positions.shape[1], MC_K=MC,
    )
    inv_sorted = hrd.inv_sorted
    ev = ev_sorted.gather(2, inv_sorted.unsqueeze(1).expand(-1, 2, -1))
    ev = ev * hrd.hand_ok_mask.to(dtype).unsqueeze(1)
    return ev * showdown_potential.unsqueeze(-1) / env_scale[:, None, None]


# =============================================================================
# v8: v4 with card_cumsum stored as bf16 (halve memory bandwidth on the
# largest tensor in the kernel).
# =============================================================================


@triton.jit
def _hero_ev_kernel_v8(
    b_opp_sorted_ptr,        # [M, 2, H] fp32
    P_padded_ptr,            # [M, 2, H+1] fp32
    card_cumsum_ptr,         # [M, 2, 52, MC] bf16
    L_idx_ptr, R_idx_ptr,
    c1_sorted_ptr, c2_sorted_ptr,
    slot_L_c1_ptr, slot_L_c2_ptr, slot_R_c1_ptr, slot_R_c2_ptr,
    slot_last_c1_ptr, slot_last_c2_ptr,
    ev_out_ptr,              # [M, 2, H] fp32 sorted
    H,
    NUM_CARDS: tl.constexpr,
    MC_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < H

    L = tl.load(L_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    R = tl.load(R_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    c1 = tl.load(c1_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    c2 = tl.load(c2_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL1 = tl.load(slot_L_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL2 = tl.load(slot_L_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR1 = tl.load(slot_R_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR2 = tl.load(slot_R_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast1 = tl.load(slot_last_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast2 = tl.load(slot_last_c2_ptr + m * H + k_offs, mask=mask_k, other=0)

    has_L1 = sL1 > 0
    has_L2 = sL2 > 0
    has_R1 = sR1 > 0
    has_R2 = sR2 > 0
    has_Last1 = sLast1 > 0
    has_Last2 = sLast2 > 0
    iL1 = tl.maximum(sL1 - 1, 0)
    iL2 = tl.maximum(sL2 - 1, 0)
    iR1 = tl.maximum(sR1 - 1, 0)
    iR2 = tl.maximum(sR2 - 1, 0)
    iLast1 = tl.maximum(sLast1 - 1, 0)
    iLast2 = tl.maximum(sLast2 - 1, 0)

    for hero in tl.static_range(2):
        b_at_k = tl.load(b_opp_sorted_ptr + m * 2 * H + hero * H + k_offs, mask=mask_k, other=0.0)
        P_L = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L, mask=mask_k, other=0.0)
        P_R = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R, mask=mask_k, other=0.0)

        cum_base = card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
        Pcards_L_c1 = tl.where(has_L1, tl.load(cum_base + c1 * MC_K + iL1, mask=mask_k, other=0.0).to(tl.float32), 0.0)
        Pcards_L_c2 = tl.where(has_L2, tl.load(cum_base + c2 * MC_K + iL2, mask=mask_k, other=0.0).to(tl.float32), 0.0)
        Pcards_R_c1 = tl.where(has_R1, tl.load(cum_base + c1 * MC_K + iR1, mask=mask_k, other=0.0).to(tl.float32), 0.0)
        Pcards_R_c2 = tl.where(has_R2, tl.load(cum_base + c2 * MC_K + iR2, mask=mask_k, other=0.0).to(tl.float32), 0.0)
        Pcards_last_c1 = tl.where(has_Last1, tl.load(cum_base + c1 * MC_K + iLast1, mask=mask_k, other=0.0).to(tl.float32), 0.0)
        Pcards_last_c2 = tl.where(has_Last2, tl.load(cum_base + c2 * MC_K + iLast2, mask=mask_k, other=0.0).to(tl.float32), 0.0)

        win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
        seg_c1 = Pcards_R_c1 - Pcards_L_c1
        seg_c2 = Pcards_R_c2 - Pcards_L_c2
        tie_mass = (P_R - P_L) - seg_c1 - seg_c2 + b_at_k

        denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
        valid = denom > 1e-8
        denom_safe = tl.where(valid, denom, 1.0)
        win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
        tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
        loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
        EV = win_prob - loss_prob
        tl.store(ev_out_ptr + m * 2 * H + hero * H + k_offs, EV, mask=mask_k)


def run_v8_both(beliefs, hrd, card_positions, slot_tensors, showdown_potential, env_scale, block_k=64):
    M, _, NUM_HANDS = beliefs.shape
    device = beliefs.device
    dtype = torch.float32

    b_opp_both, P_both, cum_both = _setup_both(beliefs, hrd, card_positions)
    cum_both_bf16 = cum_both.to(torch.bfloat16).contiguous()
    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    L_idx = hrd.L_idx.contiguous()
    R_idx = hrd.R_idx.contiguous()
    sL1, sL2, sR1, sR2, sLast1, sLast2 = slot_tensors

    ev_sorted = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
    grid = (M, triton.cdiv(NUM_HANDS, block_k))
    _hero_ev_kernel_v8[grid](
        b_opp_both, P_both, cum_both_bf16,
        L_idx, R_idx, c1, c2,
        sL1, sL2, sR1, sR2, sLast1, sLast2,
        ev_sorted,
        NUM_HANDS, NUM_CARDS=card_positions.shape[1], MC_K=MC, BLOCK_K=block_k, num_warps=4,
    )
    inv_sorted = hrd.inv_sorted
    ev = ev_sorted.gather(2, inv_sorted.unsqueeze(1).expand(-1, 2, -1))
    ev = ev * hrd.hand_ok_mask.to(dtype).unsqueeze(1)
    return ev * showdown_potential.unsqueeze(-1) / env_scale[:, None, None]


@triton.jit
def _hero_ev_kernel_v4(
    b_opp_sorted_ptr,        # [M, 2, H]
    P_padded_ptr,            # [M, 2, H+1]
    card_cumsum_ptr,         # [M, 2, 52, MC]
    L_idx_ptr,               # [M, H]
    R_idx_ptr,               # [M, H]
    c1_sorted_ptr,           # [M, H]
    c2_sorted_ptr,           # [M, H]
    slot_L_c1_ptr,           # [M, H]
    slot_L_c2_ptr,
    slot_R_c1_ptr,
    slot_R_c2_ptr,
    slot_last_c1_ptr,
    slot_last_c2_ptr,
    ev_out_ptr,              # [M, 2, H] sorted
    H,
    NUM_CARDS: tl.constexpr,
    MC_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0)
    k_block = tl.program_id(1)
    k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offs < H

    L = tl.load(L_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    R = tl.load(R_idx_ptr + m * H + k_offs, mask=mask_k, other=0)
    c1 = tl.load(c1_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    c2 = tl.load(c2_sorted_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL1 = tl.load(slot_L_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sL2 = tl.load(slot_L_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR1 = tl.load(slot_R_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sR2 = tl.load(slot_R_c2_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast1 = tl.load(slot_last_c1_ptr + m * H + k_offs, mask=mask_k, other=0)
    sLast2 = tl.load(slot_last_c2_ptr + m * H + k_offs, mask=mask_k, other=0)

    # cum address base for this (m, hero, c) — built per hero per c lookup
    has_L1 = sL1 > 0
    has_L2 = sL2 > 0
    has_R1 = sR1 > 0
    has_R2 = sR2 > 0
    has_Last1 = sLast1 > 0
    has_Last2 = sLast2 > 0
    iL1 = tl.maximum(sL1 - 1, 0)
    iL2 = tl.maximum(sL2 - 1, 0)
    iR1 = tl.maximum(sR1 - 1, 0)
    iR2 = tl.maximum(sR2 - 1, 0)
    iLast1 = tl.maximum(sLast1 - 1, 0)
    iLast2 = tl.maximum(sLast2 - 1, 0)

    for hero in tl.static_range(2):
        b_at_k = tl.load(
            b_opp_sorted_ptr + m * 2 * H + hero * H + k_offs, mask=mask_k, other=0.0
        )
        P_L = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L, mask=mask_k, other=0.0)
        P_R = tl.load(P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R, mask=mask_k, other=0.0)

        cum_base = card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
        Pcards_L_c1 = tl.where(has_L1, tl.load(cum_base + c1 * MC_K + iL1, mask=mask_k, other=0.0), 0.0)
        Pcards_L_c2 = tl.where(has_L2, tl.load(cum_base + c2 * MC_K + iL2, mask=mask_k, other=0.0), 0.0)
        Pcards_R_c1 = tl.where(has_R1, tl.load(cum_base + c1 * MC_K + iR1, mask=mask_k, other=0.0), 0.0)
        Pcards_R_c2 = tl.where(has_R2, tl.load(cum_base + c2 * MC_K + iR2, mask=mask_k, other=0.0), 0.0)
        Pcards_last_c1 = tl.where(has_Last1, tl.load(cum_base + c1 * MC_K + iLast1, mask=mask_k, other=0.0), 0.0)
        Pcards_last_c2 = tl.where(has_Last2, tl.load(cum_base + c2 * MC_K + iLast2, mask=mask_k, other=0.0), 0.0)

        win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
        seg_c1 = Pcards_R_c1 - Pcards_L_c1
        seg_c2 = Pcards_R_c2 - Pcards_L_c2
        tie_mass = (P_R - P_L) - seg_c1 - seg_c2 + b_at_k

        denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
        valid = denom > 1e-8
        denom_safe = tl.where(valid, denom, 1.0)
        win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
        tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
        loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
        EV = win_prob - loss_prob
        tl.store(ev_out_ptr + m * 2 * H + hero * H + k_offs, EV, mask=mask_k)


def run_v4_both(beliefs, hrd, card_positions, slot_tensors, showdown_potential, env_scale, block_k=64):
    M, _, NUM_HANDS = beliefs.shape
    device = beliefs.device
    dtype = torch.float32

    b_opp_both, P_both, cum_both = _setup_both(beliefs, hrd, card_positions)
    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    L_idx = hrd.L_idx.contiguous()
    R_idx = hrd.R_idx.contiguous()
    sL1, sL2, sR1, sR2, sLast1, sLast2 = slot_tensors

    ev_sorted = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
    grid = (M, triton.cdiv(NUM_HANDS, block_k))
    _hero_ev_kernel_v4[grid](
        b_opp_both, P_both, cum_both,
        L_idx, R_idx, c1, c2,
        sL1, sL2, sR1, sR2, sLast1, sLast2,
        ev_sorted,
        NUM_HANDS, NUM_CARDS=card_positions.shape[1], MC_K=MC, BLOCK_K=block_k, num_warps=4,
    )
    inv_sorted = hrd.inv_sorted
    ev = ev_sorted.gather(2, inv_sorted.unsqueeze(1).expand(-1, 2, -1))
    ev = ev * hrd.hand_ok_mask.to(dtype).unsqueeze(1)
    return ev * showdown_potential.unsqueeze(-1) / env_scale[:, None, None]


def _setup_both(beliefs, hrd, card_positions):
    """Shared setup for both-hero variants."""
    M, _, NUM_HANDS = beliefs.shape
    device = beliefs.device
    dtype = torch.float32
    H_int = hrd.sorted_indices.shape[1]

    b_opp_h0 = beliefs[:, 1, :].gather(1, hrd.sorted_indices)
    b_opp_h1 = beliefs[:, 0, :].gather(1, hrd.sorted_indices)
    b_opp_sorted_both = torch.stack([b_opp_h0, b_opp_h1], dim=1).contiguous()

    zeros = torch.zeros(M, 1, device=device, dtype=dtype)
    P_h0 = b_opp_h0.cumsum(dim=1)
    P_h1 = b_opp_h1.cumsum(dim=1)
    P_padded_both = torch.stack(
        [torch.cat([zeros, P_h0], dim=1), torch.cat([zeros, P_h1], dim=1)], dim=1
    ).contiguous()

    pos_flat = card_positions.clamp(min=0, max=H_int).view(M, -1)
    b_h0_padded = torch.cat([b_opp_h0, zeros], dim=1)
    b_h1_padded = torch.cat([b_opp_h1, zeros], dim=1)
    b_at_pos_h0 = b_h0_padded.gather(1, pos_flat).view(M, 52, MC)
    b_at_pos_h1 = b_h1_padded.gather(1, pos_flat).view(M, 52, MC)
    card_cumsum_both = torch.stack(
        [b_at_pos_h0.cumsum(dim=-1), b_at_pos_h1.cumsum(dim=-1)], dim=1
    ).contiguous()

    return b_opp_sorted_both, P_padded_both, card_cumsum_both


def run_v3_both(beliefs, hrd, card_positions, showdown_potential, env_scale, block_k=64):
    M, _, NUM_HANDS = beliefs.shape
    device = beliefs.device
    dtype = torch.float32

    b_opp_both, P_both, cum_both = _setup_both(beliefs, hrd, card_positions)
    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    L_idx = hrd.L_idx.contiguous()
    R_idx = hrd.R_idx.contiguous()

    ev_sorted = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
    grid = (M, triton.cdiv(NUM_HANDS, block_k))
    _hero_ev_kernel_v3[grid](
        b_opp_both, P_both, card_positions.to(torch.int32),
        cum_both, L_idx, R_idx, c1, c2, ev_sorted,
        NUM_HANDS, NUM_CARDS=card_positions.shape[1], MC_K=MC, BLOCK_K=block_k, num_warps=4,
    )
    inv_sorted = hrd.inv_sorted
    ev = ev_sorted.gather(2, inv_sorted.unsqueeze(1).expand(-1, 2, -1))
    ev = ev * hrd.hand_ok_mask.to(dtype).unsqueeze(1)
    return ev * showdown_potential.unsqueeze(-1) / env_scale[:, None, None]


def run_v2_both(beliefs, hrd, card_positions, showdown_potential, env_scale, block_k=64):
    M, _, NUM_HANDS = beliefs.shape
    device = beliefs.device
    dtype = torch.float32
    H_int = hrd.sorted_indices.shape[1]

    # Per-hero b_opp_sorted [M, 2, H], stacked
    b_opp_h0 = beliefs[:, 1, :].gather(1, hrd.sorted_indices)
    b_opp_h1 = beliefs[:, 0, :].gather(1, hrd.sorted_indices)
    b_opp_sorted_both = torch.stack([b_opp_h0, b_opp_h1], dim=1).contiguous()

    # Per-hero P [M, 2, H+1]
    P_h0 = b_opp_h0.cumsum(dim=1)
    P_h1 = b_opp_h1.cumsum(dim=1)
    zeros = torch.zeros(M, 1, device=device, dtype=dtype)
    P_padded_both = torch.stack(
        [torch.cat([zeros, P_h0], dim=1), torch.cat([zeros, P_h1], dim=1)], dim=1
    ).contiguous()

    # Per-hero card_cumsum [M, 2, 52, MC]
    pos_flat = card_positions.clamp(min=0, max=H_int).view(M, -1)
    b_h0_padded = torch.cat([b_opp_h0, zeros], dim=1)
    b_h1_padded = torch.cat([b_opp_h1, zeros], dim=1)
    b_at_pos_h0 = b_h0_padded.gather(1, pos_flat).view(M, 52, MC)
    b_at_pos_h1 = b_h1_padded.gather(1, pos_flat).view(M, 52, MC)
    card_cumsum_both = torch.stack(
        [b_at_pos_h0.cumsum(dim=-1), b_at_pos_h1.cumsum(dim=-1)], dim=1
    ).contiguous()

    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    L_idx = hrd.L_idx.contiguous()
    R_idx = hrd.R_idx.contiguous()

    ev_sorted = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
    grid = (M, triton.cdiv(NUM_HANDS, block_k))
    _hero_ev_kernel_v2[grid](
        b_opp_sorted_both, P_padded_both, card_positions.to(torch.int32),
        card_cumsum_both, L_idx, R_idx, c1, c2, ev_sorted,
        NUM_HANDS, NUM_CARDS=card_positions.shape[1], MC_K=MC, BLOCK_K=block_k, num_warps=4,
    )
    # Postprocess: gather permutation, hand mask, potential/scale.
    inv_sorted = hrd.inv_sorted
    ev = ev_sorted.gather(2, inv_sorted.unsqueeze(1).expand(-1, 2, -1))
    ev = ev * hrd.hand_ok_mask.to(dtype).unsqueeze(1)
    ev = ev * showdown_potential.unsqueeze(-1) / env_scale[:, None, None]
    return ev


# =============================================================================
# Bench harness.
# =============================================================================


@dataclass
class Variant:
    name: str
    fn_both: Callable  # takes (beliefs, hrd, card_positions, potential, scale) -> [M, 2, H]
    enabled: bool = True


def _bench(fn, *, label: str, m: int) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        fn()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / REPEATS
    print(f"  {label:<48} M={m:>5}  {elapsed_us:>9.1f} us / call")
    return elapsed_us


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    cfg = Config.from_dict_config(dict_config)
    cfg.use_wandb = False
    cfg.num_steps = max(cfg.num_steps, 5)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    print("Building trainer + running CFR until subgame has river leaves...")
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    eval_ = trainer.cfr_evaluator
    for step in range(20):
        trainer._update_model(step)
        if eval_.showdown_indices.numel() > 0:
            print(f"  step {step}: {eval_.showdown_indices.numel()} showdown rows")
            break
    else:
        raise SystemExit("never got showdown rows")

    m = eval_.showdown_indices.numel()
    NUM_HANDS = eval_.beliefs.shape[-1]
    beliefs = torch.rand(m, 2, NUM_HANDS, device=device, dtype=torch.float32)
    beliefs /= beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    hrd = eval_.hand_rank_data
    showdown_potential = eval_.showdown_potential
    env_scale = eval_.env.scale[eval_.showdown_indices]

    print("Precomputing card_positions...")
    t0 = time.perf_counter()
    card_positions = _precompute_card_positions(hrd.hands_c1c2_sorted)
    torch.cuda.synchronize()
    print(f"  {(time.perf_counter() - t0) * 1e3:.1f} ms, shape {tuple(card_positions.shape)}")

    print("Precomputing lookup slots...")
    t0 = time.perf_counter()
    slot_tensors = _precompute_lookup_slots(card_positions, hrd)
    torch.cuda.synchronize()
    print(f"  {(time.perf_counter() - t0) * 1e3:.1f} ms")

    # PyTorch baseline reference (both heroes).
    print("\nReference: PyTorch _showdown_value_both")
    ev_ref = eval_._showdown_value_both(beliefs)

    variants = [
        Variant("v2: fused both-heroes (masked-sum)",
                lambda b: run_v2_both(b, hrd, card_positions, showdown_potential, env_scale)),
        Variant("v3: v2 + tl.gather lookup",
                lambda b: run_v3_both(b, hrd, card_positions, showdown_potential, env_scale)),
        Variant("v4: precomputed slots + scalar cum loads",
                lambda b: run_v4_both(b, hrd, card_positions, slot_tensors, showdown_potential, env_scale)),
        Variant("v5: v4 + autotune block_k/num_warps",
                lambda b: run_v5_both(b, hrd, card_positions, slot_tensors, showdown_potential, env_scale)),
        Variant("v7: v4 + write at inv_sorted (skip post-gather)",
                lambda b: run_v7_both(b, hrd, card_positions, slot_tensors, showdown_potential, env_scale)),
        Variant("v8: v4 + bf16 card_cumsum",
                lambda b: run_v8_both(b, hrd, card_positions, slot_tensors, showdown_potential, env_scale)),
    ]

    print("\nVerifying correctness:")
    for v in variants:
        if not v.enabled:
            continue
        try:
            out = v.fn_both(beliefs)
            diff = (out - ev_ref).abs().max().item()
            ok = "✓" if diff < 1e-3 else "✗ MISMATCH"
            print(f"  {ok}  {v.name}: max abs diff = {diff:.2e}")
            if diff >= 1e-3:
                v.enabled = False
        except Exception as e:
            print(f"  ✗ ERROR  {v.name}: {type(e).__name__}: {e}")
            v.enabled = False

    print("\nBenchmarking:")
    timings = {"PyTorch _showdown_value_both (compiled)": 0.0}
    timings["PyTorch _showdown_value_both (compiled)"] = _bench(
        lambda: eval_._showdown_value_both(beliefs),
        label="PyTorch _showdown_value_both (compiled)", m=m,
    )
    for v in variants:
        if not v.enabled:
            continue
        timings[v.name] = _bench(lambda v=v: v.fn_both(beliefs), label=v.name, m=m)

    # Summary table
    print("\n" + "=" * 78)
    print(f"  Summary at M={m}")
    print("=" * 78)
    base = timings["PyTorch _showdown_value_both (compiled)"]
    print(f"  {'variant':<48} {'us/call':>10}  {'vs base':>8}")
    print("  " + "-" * 76)
    for name, t in timings.items():
        rel = base / t if t > 0 else 0
        print(f"  {name:<48} {t:>10.1f}  {rel:>7.2f}x")


if __name__ == "__main__":
    main()
