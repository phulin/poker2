"""Benchmark exact 3-player native-169 all-in share kernels.

Compares the current PyTorch implementation, which materializes
``[B, 169 * 169]`` opponent outer products, against a fused Triton kernel that
streams opponent-class products inside the kernel.

Example:

    uv run python benchmarks/bench_preflop_allin_3p_kernel.py --batch-sizes 64 256 1024
"""

from __future__ import annotations

import argparse

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - benchmark-only optional dependency
    triton = None
    tl = None

from p2.allin.oracle import PreflopAllIn169Oracle, _preflop_3p_compatibility_counts
from p2.env.card_utils import (
    PREFLOP_HANDS,
    combo_to_onehot_tensor,
    combo_to_preflop_class_tensor,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
)

AB = PREFLOP_HANDS * PREFLOP_HANDS


if triton is not None:

    @triton.jit
    def _share3_fused_kernel(
        opp0_ptr,
        opp1_ptr,
        weighted_flat_ptr,
        compat_flat_ptr,
        out_ptr,
        rows: tl.constexpr,
        stride_row: tl.constexpr,
        hand_count: tl.constexpr,
        ab_count: tl.constexpr,
        block_ab: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        h_start = tl.program_id(1) * block_h
        h = h_start + tl.arange(0, block_h)
        h_mask = h < hand_count

        acc_num = tl.zeros((1, block_h), dtype=tl.float32)
        acc_den = tl.zeros((1, block_h), dtype=tl.float32)

        for ab_start in range(0, ab_count, block_ab):
            ab = ab_start + tl.arange(0, block_ab)
            ab_mask = ab < ab_count
            a = ab // hand_count
            b = ab - a * hand_count

            o0 = tl.load(opp0_ptr + row * hand_count + a, mask=ab_mask, other=0.0)
            o1 = tl.load(opp1_ptr + row * hand_count + b, mask=ab_mask, other=0.0)
            weight = (o0 * o1).to(tl.float32)
            offsets = ab[:, None] * hand_count + h[None, :]
            table_mask = ab_mask[:, None] & h_mask[None, :]
            weighted = tl.load(weighted_flat_ptr + offsets, mask=table_mask, other=0.0)
            compat = tl.load(compat_flat_ptr + offsets, mask=table_mask, other=0.0)
            acc_num += tl.dot(weight[None, :], weighted)
            acc_den += tl.dot(weight[None, :], compat)

        value = acc_num / tl.maximum(acc_den, 1.0e-8)
        tl.store(
            out_ptr + row * stride_row + h[None, :],
            value,
            mask=(row < rows) & h_mask[None, :],
        )


    @triton.jit
    def _share3_fused_blockm_kernel(
        opp0_ptr,
        opp1_ptr,
        weighted_flat_ptr,
        compat_flat_ptr,
        out_ptr,
        rows: tl.constexpr,
        stride_row: tl.constexpr,
        hand_count: tl.constexpr,
        ab_count: tl.constexpr,
        block_m: tl.constexpr,
        block_ab: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_m
        h_start = tl.program_id(1) * block_h
        row = row_start + tl.arange(0, block_m)
        h = h_start + tl.arange(0, block_h)
        row_mask = row < rows
        h_mask = h < hand_count

        acc_num = tl.zeros((block_m, block_h), dtype=tl.float32)
        acc_den = tl.zeros((block_m, block_h), dtype=tl.float32)

        for ab_start in range(0, ab_count, block_ab):
            ab = ab_start + tl.arange(0, block_ab)
            ab_mask = ab < ab_count
            a = ab // hand_count
            b = ab - a * hand_count

            o0 = tl.load(
                opp0_ptr + row[:, None] * hand_count + a[None, :],
                mask=row_mask[:, None] & ab_mask[None, :],
                other=0.0,
            )
            o1 = tl.load(
                opp1_ptr + row[:, None] * hand_count + b[None, :],
                mask=row_mask[:, None] & ab_mask[None, :],
                other=0.0,
            )
            weight = (o0 * o1).to(tl.float32)
            offsets = ab[:, None] * hand_count + h[None, :]
            table_mask = ab_mask[:, None] & h_mask[None, :]
            weighted = tl.load(weighted_flat_ptr + offsets, mask=table_mask, other=0.0)
            compat = tl.load(compat_flat_ptr + offsets, mask=table_mask, other=0.0)
            acc_num += tl.dot(weight, weighted)
            acc_den += tl.dot(weight, compat)

        value = acc_num / tl.maximum(acc_den, 1.0e-8)
        tl.store(
            out_ptr + row[:, None] * stride_row + h[None, :],
            value,
            mask=row_mask[:, None] & h_mask[None, :],
        )


    @triton.jit
    def _packed_outer_kernel(
        opp0_ptr,
        opp1_ptr,
        tri_i_ptr,
        tri_j_ptr,
        out_ptr,
        rows: tl.constexpr,
        packed_count: tl.constexpr,
        hand_count: tl.constexpr,
        stride_out: tl.constexpr,
        block_m: tl.constexpr,
        block_p: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_m
        col_start = tl.program_id(1) * block_p
        row = row_start + tl.arange(0, block_m)
        col = col_start + tl.arange(0, block_p)
        row_mask = row < rows
        col_mask = col < packed_count
        i = tl.load(tri_i_ptr + col, mask=col_mask, other=0)
        j = tl.load(tri_j_ptr + col, mask=col_mask, other=0)
        p0_i = tl.load(
            opp0_ptr + row[:, None] * hand_count + i[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        p1_j = tl.load(
            opp1_ptr + row[:, None] * hand_count + j[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        p0_j = tl.load(
            opp0_ptr + row[:, None] * hand_count + j[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        p1_i = tl.load(
            opp1_ptr + row[:, None] * hand_count + i[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        offdiag = i != j
        value = p0_i * p1_j + tl.where(offdiag[None, :], p0_j * p1_i, 0.0)
        tl.store(
            out_ptr + row[:, None] * stride_out + col[None, :],
            value,
            mask=row_mask[:, None] & col_mask[None, :],
        )


    @triton.jit
    def _share3_fused_packed_num_kernel(
        opp0_ptr,
        opp1_ptr,
        tri_i_ptr,
        tri_j_ptr,
        weighted_packed_ptr,
        numer_ptr,
        rows: tl.constexpr,
        packed_count: tl.constexpr,
        hand_count: tl.constexpr,
        stride_numer: tl.constexpr,
        block_m: tl.constexpr,
        block_p: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_m
        h_start = tl.program_id(1) * block_h
        row = row_start + tl.arange(0, block_m)
        h = h_start + tl.arange(0, block_h)
        row_mask = row < rows
        h_mask = h < hand_count

        acc = tl.zeros((block_m, block_h), dtype=tl.float32)
        for p_start in range(0, packed_count, block_p):
            p = p_start + tl.arange(0, block_p)
            p_mask = p < packed_count
            i = tl.load(tri_i_ptr + p, mask=p_mask, other=0)
            j = tl.load(tri_j_ptr + p, mask=p_mask, other=0)
            p0_i = tl.load(
                opp0_ptr + row[:, None] * hand_count + i[None, :],
                mask=row_mask[:, None] & p_mask[None, :],
                other=0.0,
            )
            p1_j = tl.load(
                opp1_ptr + row[:, None] * hand_count + j[None, :],
                mask=row_mask[:, None] & p_mask[None, :],
                other=0.0,
            )
            p0_j = tl.load(
                opp0_ptr + row[:, None] * hand_count + j[None, :],
                mask=row_mask[:, None] & p_mask[None, :],
                other=0.0,
            )
            p1_i = tl.load(
                opp1_ptr + row[:, None] * hand_count + i[None, :],
                mask=row_mask[:, None] & p_mask[None, :],
                other=0.0,
            )
            offdiag = i != j
            lhs = (p0_i * p1_j + tl.where(offdiag[None, :], p0_j * p1_i, 0.0)).to(
                tl.float32
            )
            rhs = tl.load(
                weighted_packed_ptr + p[:, None] * hand_count + h[None, :],
                mask=p_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(lhs, rhs, input_precision="ieee")

        tl.store(
            numer_ptr + row[:, None] * stride_numer + h[None, :],
            acc,
            mask=row_mask[:, None] & h_mask[None, :],
        )


    @triton.jit
    def _denom_card_formula_kernel(
        q0_ptr,
        q1_ptr,
        base0_ptr,
        base1_ptr,
        total0_ptr,
        total1_ptr,
        hero0_ptr,
        hero1_ptr,
        pair0_ptr,
        pair1_ptr,
        out_ptr,
        rows: tl.constexpr,
        hand_count: tl.constexpr,
        card_count: tl.constexpr,
        stride_out: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_m
        h_start = tl.program_id(1) * block_h
        row = row_start + tl.arange(0, block_m)
        h = h_start + tl.arange(0, block_h)
        row_mask = row < rows
        h_mask = h < hand_count

        h0 = tl.load(hero0_ptr + h, mask=h_mask, other=0)
        h1 = tl.load(hero1_ptr + h, mask=h_mask, other=0)
        total0 = tl.load(total0_ptr + row, mask=row_mask, other=0.0)
        total1 = tl.load(total1_ptr + row, mask=row_mask, other=0.0)
        base0_h0 = tl.load(
            base0_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base0_h1 = tl.load(
            base0_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base1_h0 = tl.load(
            base1_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base1_h1 = tl.load(
            base1_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        q0_h = tl.load(
            q0_ptr + row[:, None] * hand_count + h[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        q1_h = tl.load(
            q1_ptr + row[:, None] * hand_count + h[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        allowed0 = total0[:, None] - base0_h0 - base0_h1 + q0_h
        allowed1 = total1[:, None] - base1_h0 - base1_h1 + q1_h

        collision = tl.zeros((block_m, block_h), dtype=tl.float32)
        for c in range(0, card_count):
            card_is_live = (c != h0) & (c != h1)
            cls0 = tl.load(pair0_ptr + h * card_count + c, mask=h_mask, other=0)
            cls1 = tl.load(pair1_ptr + h * card_count + c, mask=h_mask, other=0)
            b0 = tl.load(base0_ptr + row * card_count + c, mask=row_mask, other=0.0)
            b1 = tl.load(base1_ptr + row * card_count + c, mask=row_mask, other=0.0)
            sub00 = tl.load(
                q0_ptr + row[:, None] * hand_count + cls0[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub01 = tl.load(
                q0_ptr + row[:, None] * hand_count + cls1[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub10 = tl.load(
                q1_ptr + row[:, None] * hand_count + cls0[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub11 = tl.load(
                q1_ptr + row[:, None] * hand_count + cls1[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            c0 = tl.where(card_is_live[None, :], b0[:, None] - sub00 - sub01, 0.0)
            c1 = tl.where(card_is_live[None, :], b1[:, None] - sub10 - sub11, 0.0)
            collision += c0 * c1

        # compat2[h, a] equals the number of combos in class a disjoint from
        # hero h. For same-combo correction with q0*q1, only classes compatible
        # with h contribute by that count. This scalar correction is easier to
        # keep in PyTorch for now, so this kernel stores allowed/collision only
        # plus q0_h*q1_h placeholder is not sufficient for all classes.
        denom = allowed0 * allowed1 - collision
        tl.store(
            out_ptr + row[:, None] * stride_out + h[None, :],
            denom,
            mask=row_mask[:, None] & h_mask[None, :],
        )

    @triton.jit
    def _denom_card_formula_same_kernel(
        q0_ptr,
        q1_ptr,
        base0_ptr,
        base1_ptr,
        total0_ptr,
        total1_ptr,
        same_base_ptr,
        same_total_ptr,
        hero0_ptr,
        hero1_ptr,
        pair0_ptr,
        pair1_ptr,
        out_ptr,
        rows: tl.constexpr,
        hand_count: tl.constexpr,
        card_count: tl.constexpr,
        stride_out: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row_start = tl.program_id(0) * block_m
        h_start = tl.program_id(1) * block_h
        row = row_start + tl.arange(0, block_m)
        h = h_start + tl.arange(0, block_h)
        row_mask = row < rows
        h_mask = h < hand_count

        h0 = tl.load(hero0_ptr + h, mask=h_mask, other=0)
        h1 = tl.load(hero1_ptr + h, mask=h_mask, other=0)
        total0 = tl.load(total0_ptr + row, mask=row_mask, other=0.0)
        total1 = tl.load(total1_ptr + row, mask=row_mask, other=0.0)
        same_total = tl.load(same_total_ptr + row, mask=row_mask, other=0.0)
        base0_h0 = tl.load(
            base0_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base0_h1 = tl.load(
            base0_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base1_h0 = tl.load(
            base1_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        base1_h1 = tl.load(
            base1_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        same_h0 = tl.load(
            same_base_ptr + row[:, None] * card_count + h0[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        same_h1 = tl.load(
            same_base_ptr + row[:, None] * card_count + h1[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        q0_h = tl.load(
            q0_ptr + row[:, None] * hand_count + h[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        q1_h = tl.load(
            q1_ptr + row[:, None] * hand_count + h[None, :],
            mask=row_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        allowed0 = total0[:, None] - base0_h0 - base0_h1 + q0_h
        allowed1 = total1[:, None] - base1_h0 - base1_h1 + q1_h
        same_allowed = same_total[:, None] - same_h0 - same_h1 + q0_h * q1_h

        collision = tl.zeros((block_m, block_h), dtype=tl.float32)
        for card in range(0, card_count):
            card_is_live = (card != h0) & (card != h1)
            cls0 = tl.load(pair0_ptr + h * card_count + card, mask=h_mask, other=0)
            cls1 = tl.load(pair1_ptr + h * card_count + card, mask=h_mask, other=0)
            b0 = tl.load(base0_ptr + row * card_count + card, mask=row_mask, other=0.0)
            b1 = tl.load(base1_ptr + row * card_count + card, mask=row_mask, other=0.0)
            sub00 = tl.load(
                q0_ptr + row[:, None] * hand_count + cls0[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub01 = tl.load(
                q0_ptr + row[:, None] * hand_count + cls1[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub10 = tl.load(
                q1_ptr + row[:, None] * hand_count + cls0[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            sub11 = tl.load(
                q1_ptr + row[:, None] * hand_count + cls1[None, :],
                mask=row_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            c0 = tl.where(card_is_live[None, :], b0[:, None] - sub00 - sub01, 0.0)
            c1 = tl.where(card_is_live[None, :], b1[:, None] - sub10 - sub11, 0.0)
            collision += c0 * c1

        denom = tl.maximum(allowed0 * allowed1 - collision + same_allowed, 1.0e-8)
        tl.store(
            out_ptr + row[:, None] * stride_out + h[None, :],
            denom,
            mask=row_mask[:, None] & h_mask[None, :],
        )


def _sync() -> None:
    torch.cuda.synchronize()


def _time_ms(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / repeats


def _random_ranges(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1234 + batch_size)
    opp0 = torch.rand(batch_size, PREFLOP_HANDS, device=device, generator=generator)
    opp1 = torch.rand(batch_size, PREFLOP_HANDS, device=device, generator=generator)
    opp0 = opp0 / opp0.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    opp1 = opp1 / opp1.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return opp0.contiguous(), opp1.contiguous()


@torch.no_grad()
def _current_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_flat: torch.Tensor,
    compat_flat: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        outer = (opp0[start:end, :, None] * opp1[start:end, None, :]).flatten(1)
        numer = outer @ weighted_flat
        denom = (outer @ compat_flat).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


@torch.no_grad()
def _combined_gemm_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    combined_flat: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        outer = (opp0[start:end, :, None] * opp1[start:end, None, :]).flatten(1)
        combined = outer @ combined_flat
        numer, denom = combined.split(PREFLOP_HANDS, dim=-1)
        out[start:end] = numer / denom.clamp_min(1.0e-8)
    return out


@torch.no_grad()
def _einsum_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted: torch.Tensor,
    compat3: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        o0 = opp0[start:end]
        o1 = opp1[start:end]
        numer = torch.einsum("ra,rb,hab->rh", o0, o1, weighted)
        denom = torch.einsum("ra,rb,hab->rh", o0, o1, compat3).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


@torch.no_grad()
def _left_contract_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_a_hb: torch.Tensor,
    compat_a_hb: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        numer_tmp = (opp0[start:end] @ weighted_a_hb).view(-1, PREFLOP_HANDS, PREFLOP_HANDS)
        denom_tmp = (opp0[start:end] @ compat_a_hb).view(-1, PREFLOP_HANDS, PREFLOP_HANDS)
        o1 = opp1[start:end, None, :]
        numer = (numer_tmp * o1).sum(dim=-1)
        denom = (denom_tmp * o1).sum(dim=-1).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


@torch.no_grad()
def _right_contract_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_b_ha: torch.Tensor,
    compat_b_ha: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        numer_tmp = (opp1[start:end] @ weighted_b_ha).view(-1, PREFLOP_HANDS, PREFLOP_HANDS)
        denom_tmp = (opp1[start:end] @ compat_b_ha).view(-1, PREFLOP_HANDS, PREFLOP_HANDS)
        o0 = opp0[start:end, None, :]
        numer = (numer_tmp * o0).sum(dim=-1)
        denom = (denom_tmp * o0).sum(dim=-1).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


@torch.no_grad()
def _card_count_denom_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_flat: torch.Tensor,
    compat2: torch.Tensor,
    card_counts_a_hc: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        o0 = opp0[start:end]
        o1 = opp1[start:end]
        outer = (o0[:, :, None] * o1[:, None, :]).flatten(1)
        numer = outer @ weighted_flat

        o0_allowed = o0 @ compat2.T
        o1_allowed = o1 @ compat2.T
        o0_cards = (o0 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        o1_cards = (o1 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        card_collisions = (o0_cards * o1_cards).sum(dim=-1)
        same_combo_correction = (o0 * o1) @ compat2.T
        denom = (
            o0_allowed * o1_allowed - card_collisions + same_combo_correction
        ).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


@torch.no_grad()
def _packed_sym_card_denom_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_diag: torch.Tensor,
    weighted_off: torch.Tensor,
    off_i: torch.Tensor,
    off_j: torch.Tensor,
    compat2: torch.Tensor,
    card_counts_a_hc: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        o0 = opp0[start:end]
        o1 = opp1[start:end]
        off_outer = o0[:, off_i] * o1[:, off_j] + o0[:, off_j] * o1[:, off_i]
        numer = off_outer @ weighted_off + (o0 * o1) @ weighted_diag

        o0_allowed = o0 @ compat2.T
        o1_allowed = o1 @ compat2.T
        o0_cards = (o0 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        o1_cards = (o1 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        card_collisions = (o0_cards * o1_cards).sum(dim=-1)
        same_combo_correction = (o0 * o1) @ compat2.T
        denom = (
            o0_allowed * o1_allowed - card_collisions + same_combo_correction
        ).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


@torch.no_grad()
def _triton_packed_sym_card_denom_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_packed: torch.Tensor,
    tri_i: torch.Tensor,
    tri_j: torch.Tensor,
    compat2: torch.Tensor,
    card_counts_a_hc: torch.Tensor,
    *,
    chunk_size: int,
    block_m: int,
    block_p: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    packed_count = tri_i.shape[0]
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    packed = torch.empty(chunk_size, packed_count, dtype=torch.float32, device=opp0.device)
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        rows = end - start
        packed_chunk = packed[:rows]
        grid = (triton.cdiv(rows, block_m), triton.cdiv(packed_count, block_p))
        _packed_outer_kernel[grid](
            opp0[start:end],
            opp1[start:end],
            tri_i,
            tri_j,
            packed_chunk,
            rows,
            packed_count,
            PREFLOP_HANDS,
            packed_chunk.stride(0),
            block_m,
            block_p,
            num_warps=8,
        )
        numer = packed_chunk @ weighted_packed

        o0 = opp0[start:end]
        o1 = opp1[start:end]
        o0_allowed = o0 @ compat2.T
        o1_allowed = o1 @ compat2.T
        o0_cards = (o0 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        o1_cards = (o1 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        card_collisions = (o0_cards * o1_cards).sum(dim=-1)
        same_combo_correction = (o0 * o1) @ compat2.T
        denom = (
            o0_allowed * o1_allowed - card_collisions + same_combo_correction
        ).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


@torch.no_grad()
def _fused_packed_num_card_denom_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_packed: torch.Tensor,
    tri_i: torch.Tensor,
    tri_j: torch.Tensor,
    compat2: torch.Tensor,
    card_counts_a_hc: torch.Tensor,
    *,
    chunk_size: int,
    block_m: int,
    block_p: int,
    block_h: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    packed_count = tri_i.shape[0]
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    numer_buf = torch.empty(
        min(chunk_size, opp0.shape[0]),
        PREFLOP_HANDS,
        dtype=torch.float32,
        device=opp0.device,
    )
    for start in range(0, opp0.shape[0], chunk_size):
        end = min(start + chunk_size, opp0.shape[0])
        rows = end - start
        numer = numer_buf[:rows]
        grid = (triton.cdiv(rows, block_m), triton.cdiv(PREFLOP_HANDS, block_h))
        _share3_fused_packed_num_kernel[grid](
            opp0[start:end],
            opp1[start:end],
            tri_i,
            tri_j,
            weighted_packed,
            numer,
            rows,
            packed_count,
            PREFLOP_HANDS,
            numer.stride(0),
            block_m,
            block_p,
            block_h,
            num_warps=4,
        )
        o0 = opp0[start:end]
        o1 = opp1[start:end]
        o0_allowed = o0 @ compat2.T
        o1_allowed = o1 @ compat2.T
        o0_cards = (o0 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        o1_cards = (o1 @ card_counts_a_hc).view(-1, PREFLOP_HANDS, 52)
        card_collisions = (o0_cards * o1_cards).sum(dim=-1)
        same_combo_correction = (o0 * o1) @ compat2.T
        denom = (
            o0_allowed * o1_allowed - card_collisions + same_combo_correction
        ).clamp_min(1.0e-8)
        out[start:end] = numer / denom
    return out


def _preflop_3p_card_counts(device: torch.device) -> torch.Tensor:
    class_ids = combo_to_preflop_class_tensor(device=torch.device("cpu"))
    combo_cards = combo_to_onehot_tensor(device=torch.device("cpu")).to(torch.float32)
    combo_compatible = (combo_cards @ combo_cards.T) < 0.5
    class_onehot = torch.zeros(
        combo_cards.shape[0],
        PREFLOP_HANDS,
        dtype=torch.float32,
    )
    class_onehot.scatter_(1, class_ids[:, None], 1.0)
    representative = torch.empty(PREFLOP_HANDS, dtype=torch.int64)
    for class_id in range(PREFLOP_HANDS):
        representative[class_id] = torch.nonzero(
            class_ids == class_id,
            as_tuple=False,
        )[0, 0]

    counts = torch.empty(PREFLOP_HANDS, PREFLOP_HANDS, 52, dtype=torch.float32)
    for hero_class in range(PREFLOP_HANDS):
        allowed = combo_compatible[representative[hero_class]].to(torch.float32)
        counts[hero_class] = class_onehot.T @ (combo_cards * allowed[:, None])
    return counts.to(device=device, non_blocking=True).contiguous()


@torch.no_grad()
def _fused_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_flat: torch.Tensor,
    compat_flat: torch.Tensor,
    *,
    block_ab: int,
    block_h: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    grid = (opp0.shape[0], triton.cdiv(PREFLOP_HANDS, block_h))
    _share3_fused_kernel[grid](
        opp0,
        opp1,
        weighted_flat,
        compat_flat,
        out,
        opp0.shape[0],
        out.stride(0),
        PREFLOP_HANDS,
        AB,
        block_ab,
        block_h,
        num_warps=4,
    )
    return out


@torch.no_grad()
def _fused_blockm_share3(
    opp0: torch.Tensor,
    opp1: torch.Tensor,
    weighted_flat: torch.Tensor,
    compat_flat: torch.Tensor,
    *,
    block_m: int,
    block_ab: int,
    block_h: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    out = torch.empty(opp0.shape[0], PREFLOP_HANDS, dtype=torch.float32, device=opp0.device)
    grid = (triton.cdiv(opp0.shape[0], block_m), triton.cdiv(PREFLOP_HANDS, block_h))
    _share3_fused_blockm_kernel[grid](
        opp0,
        opp1,
        weighted_flat,
        compat_flat,
        out,
        opp0.shape[0],
        out.stride(0),
        PREFLOP_HANDS,
        AB,
        block_m,
        block_ab,
        block_h,
        num_warps=4,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-path",
        default="outputs/precompute/preflop_allin_169_u16.pt.zst",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--block-ab", type=int, default=128)
    parser.add_argument("--block-h", type=int, default=32)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-p", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    device = torch.device("cuda")
    oracle = PreflopAllIn169Oracle(device=device, exact_cache_path=args.cache_path)
    oracle._ensure_exact()
    assert oracle._share3 is not None

    compat2 = preflop_class_compatibility_counts_tensor(device=device).to(torch.float32)
    compat3 = _preflop_3p_compatibility_counts(device)
    card_counts = _preflop_3p_card_counts(device)
    multiplicity = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    weighted = (compat3 * oracle._share3).contiguous()
    weighted_flat = weighted.reshape(PREFLOP_HANDS, AB).T.contiguous()
    compat_flat = compat3.reshape(PREFLOP_HANDS, AB).T.contiguous()
    combined_flat = torch.cat([weighted_flat, compat_flat], dim=-1).contiguous()
    weighted_a_hb = weighted.permute(1, 0, 2).reshape(PREFLOP_HANDS, AB).contiguous()
    compat_a_hb = compat3.permute(1, 0, 2).reshape(PREFLOP_HANDS, AB).contiguous()
    weighted_b_ha = weighted.permute(2, 0, 1).reshape(PREFLOP_HANDS, AB).contiguous()
    compat_b_ha = compat3.permute(2, 0, 1).reshape(PREFLOP_HANDS, AB).contiguous()
    card_counts_a_hc = (
        card_counts.permute(1, 0, 2).reshape(PREFLOP_HANDS, PREFLOP_HANDS * 52).contiguous()
    )
    tri = torch.triu_indices(PREFLOP_HANDS, PREFLOP_HANDS, offset=1, device=device)
    off_i = tri[0].contiguous()
    off_j = tri[1].contiguous()
    diag = torch.arange(PREFLOP_HANDS, device=device)
    weighted_off = weighted[:, off_i, off_j].T.contiguous()
    weighted_diag = weighted[:, diag, diag].T.contiguous()
    tri_all = torch.triu_indices(PREFLOP_HANDS, PREFLOP_HANDS, device=device)
    weighted_packed = weighted[:, tri_all[0], tri_all[1]].T.contiguous()
    tri_i = tri_all[0].to(torch.int32).contiguous()
    tri_j = tri_all[1].to(torch.int32).contiguous()
    symmetry_err = float((weighted - weighted.transpose(1, 2)).abs().max().item())

    print(
        "batch,current_ms,combined_ms,einsum_ms,left_ms,right_ms,"
        "card_denom_ms,packed_sym_ms,triton_packed_ms,fused_ms,fused_blockm_ms,"
        "best_speedup,max_abs,max_rel,"
        f"chunk={args.chunk_size},block_m={args.block_m},"
        f"block_ab={args.block_ab},block_h={args.block_h},block_p={args.block_p},"
        f"symmetry_err={symmetry_err:.3e}"
    )
    for batch_size in args.batch_sizes:
        opp0, opp1 = _random_ranges(batch_size, device)
        opp0_per_combo = (opp0 / multiplicity).contiguous()
        opp1_per_combo = (opp1 / multiplicity).contiguous()

        current = _current_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_flat,
            compat_flat,
            chunk_size=args.chunk_size,
        )
        combined = _combined_gemm_share3(
            opp0_per_combo,
            opp1_per_combo,
            combined_flat,
            chunk_size=args.chunk_size,
        )
        fused = _fused_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_flat,
            compat_flat,
            block_ab=args.block_ab,
            block_h=args.block_h,
        )
        fused_blockm = _fused_blockm_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_flat,
            compat_flat,
            block_m=args.block_m,
            block_ab=args.block_ab,
            block_h=args.block_h,
        )
        left = _left_contract_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_a_hb,
            compat_a_hb,
            chunk_size=args.chunk_size,
        )
        right = _right_contract_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_b_ha,
            compat_b_ha,
            chunk_size=args.chunk_size,
        )
        card_denom = _card_count_denom_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_flat,
            compat2,
            card_counts_a_hc,
            chunk_size=args.chunk_size,
        )
        packed_sym = _packed_sym_card_denom_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_diag,
            weighted_off,
            off_i,
            off_j,
            compat2,
            card_counts_a_hc,
            chunk_size=args.chunk_size,
        )
        triton_packed = _triton_packed_sym_card_denom_share3(
            opp0_per_combo,
            opp1_per_combo,
            weighted_packed,
            tri_i,
            tri_j,
            compat2,
            card_counts_a_hc,
            chunk_size=args.chunk_size,
            block_m=args.block_m,
            block_p=args.block_p,
        )
        _sync()
        abs_err = torch.maximum(
            torch.maximum(
                torch.maximum(
                    torch.maximum((current - fused).abs(), (current - left).abs()),
                    (current - right).abs(),
                ),
                torch.maximum((current - combined).abs(), (current - fused_blockm).abs()),
            ),
            torch.maximum(
                torch.maximum((current - card_denom).abs(), (current - packed_sym).abs()),
                (current - triton_packed).abs(),
            ),
        )
        max_abs = float(abs_err.max().item())
        max_rel = float((abs_err / current.abs().clamp_min(1e-6)).max().item())

        current_ms = _time_ms(
            lambda: _current_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_flat,
                compat_flat,
                chunk_size=args.chunk_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        combined_ms = _time_ms(
            lambda: _combined_gemm_share3(
                opp0_per_combo,
                opp1_per_combo,
                combined_flat,
                chunk_size=args.chunk_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        einsum_ms = _time_ms(
            lambda: _einsum_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted,
                compat3,
                chunk_size=args.chunk_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        left_ms = _time_ms(
            lambda: _left_contract_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_a_hb,
                compat_a_hb,
                chunk_size=args.chunk_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        right_ms = _time_ms(
            lambda: _right_contract_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_b_ha,
                compat_b_ha,
                chunk_size=args.chunk_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        card_denom_ms = _time_ms(
            lambda: _card_count_denom_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_flat,
                compat2,
                card_counts_a_hc,
                chunk_size=args.chunk_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        packed_sym_ms = _time_ms(
            lambda: _packed_sym_card_denom_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_diag,
                weighted_off,
                off_i,
                off_j,
                compat2,
                card_counts_a_hc,
                chunk_size=args.chunk_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        triton_packed_ms = _time_ms(
            lambda: _triton_packed_sym_card_denom_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_packed,
                tri_i,
                tri_j,
                compat2,
                card_counts_a_hc,
                chunk_size=args.chunk_size,
                block_m=args.block_m,
                block_p=args.block_p,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        fused_ms = _time_ms(
            lambda: _fused_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_flat,
                compat_flat,
                block_ab=args.block_ab,
                block_h=args.block_h,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        fused_blockm_ms = _time_ms(
            lambda: _fused_blockm_share3(
                opp0_per_combo,
                opp1_per_combo,
                weighted_flat,
                compat_flat,
                block_m=args.block_m,
                block_ab=args.block_ab,
                block_h=args.block_h,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        best_ms = min(
            combined_ms,
            einsum_ms,
            left_ms,
            right_ms,
            card_denom_ms,
            packed_sym_ms,
            triton_packed_ms,
            fused_ms,
            fused_blockm_ms,
        )
        print(
            f"{batch_size},{current_ms:.4f},{combined_ms:.4f},{einsum_ms:.4f},"
            f"{left_ms:.4f},{right_ms:.4f},{card_denom_ms:.4f},"
            f"{packed_sym_ms:.4f},{triton_packed_ms:.4f},"
            f"{fused_ms:.4f},{fused_blockm_ms:.4f},"
            f"{current_ms / best_ms:.3f},{max_abs:.3e},{max_rel:.3e}"
        )


if __name__ == "__main__":
    main()
