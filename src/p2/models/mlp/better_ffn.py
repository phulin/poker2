from __future__ import annotations

import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA optimization
    triton = None
    tl = None

from p2.core.structured_config import NonlinearityType, StreetValueHeads
from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.env.card_utils import PREFLOP_HANDS
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import rank_hands_triton, triton_is_available
from p2.models.activation_utils import get_activation, SwiGLU
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp.better_feature_encoder import (
    BetterFeatureEncoder,
    BetterPolicyFeatureEncoder,
    BetterPreflopPolicyFeatureEncoder,
    BetterPreflopValueFeatureEncoder,
    BetterStreetValueFeatureEncoder,
)
from p2.models.mlp.better_features import (
    ChancePhase,
    context_schemas,
    PlayerContext,
    ValueScalarContext,
    context_length,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.turn_range_equity import (
    TurnRangeEquityBoardCache,
    TurnRangeEquityConfig,
    apply_turn_pair_operator_baseline_value,
    turn_range_equity_baseline,
    turn_range_equity_features,
    turn_runout_boards as turn_equity_runout_boards,
)
from p2.models.model_output import ModelOutput
from p2.utils.profiling import profile


HAND_STATIC_FEATURE_DIM = 8
HAND_DYNAMIC_FEATURE_DIM = 15
_PREFLOP_NEXT_NORM_MAX_BATCH = 16_384


def _preflop_eval_cache_enabled() -> bool:
    return os.environ.get("P2_PREFLOP_EVAL_CACHE", "1").lower() not in {
        "0",
        "false",
        "off",
    }


def _preflop_compiled_ffn_boundary_enabled() -> bool:
    explicit = os.environ.get("P2_PREFLOP_COMPILED_FFN_BOUNDARY")
    if explicit is not None:
        return explicit.lower() in {
            "1",
            "true",
            "on",
        }
    return os.environ.get(
        "P2_DISABLE_PREFLOP_COMPILED_FFN_BOUNDARY",
        "0",
    ).lower() not in {
        "1",
        "true",
        "on",
    }


if triton is not None:

    @triton.jit
    def _preflop_token_mixer_leaky_relu_kernel(
        y_ptr,
        w_in_ptr,
        w_out_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)

        y0 = tl.load(y_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0)
        y1 = tl.load(y_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0)
        y2 = tl.load(y_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0)
        y3 = tl.load(y_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0)
        y4 = tl.load(y_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0)
        y5 = tl.load(y_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0)
        y6 = tl.load(y_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0)

        out0 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out1 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out2 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out3 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out4 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out5 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out6 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)

        for h in tl.static_range(0, 28):
            hidden = (
                y0 * tl.load(w_in_ptr + h * 7 + 0)
                + y1 * tl.load(w_in_ptr + h * 7 + 1)
                + y2 * tl.load(w_in_ptr + h * 7 + 2)
                + y3 * tl.load(w_in_ptr + h * 7 + 3)
                + y4 * tl.load(w_in_ptr + h * 7 + 4)
                + y5 * tl.load(w_in_ptr + h * 7 + 5)
                + y6 * tl.load(w_in_ptr + h * 7 + 6)
            )
            hidden = tl.where(hidden >= 0.0, hidden, hidden * 0.01)
            out0 += hidden * tl.load(w_out_ptr + 0 * 28 + h)
            out1 += hidden * tl.load(w_out_ptr + 1 * 28 + h)
            out2 += hidden * tl.load(w_out_ptr + 2 * 28 + h)
            out3 += hidden * tl.load(w_out_ptr + 3 * 28 + h)
            out4 += hidden * tl.load(w_out_ptr + 4 * 28 + h)
            out5 += hidden * tl.load(w_out_ptr + 5 * 28 + h)
            out6 += hidden * tl.load(w_out_ptr + 6 * 28 + h)

        tl.store(out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], out0, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], out1, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], out2, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], out3, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], out4, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], out5, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], out6, mask=mask)

    @triton.jit
    def _river_card_rank_prefix_kernel(
        card_rank_mass_ptr,
        card_prefix_ptr,
        card_total_ptr,
        batch_size,
        RANK_BINS: tl.constexpr,
        NUM_PLAYERS: tl.constexpr,
        NUM_CARDS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        CARD_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        player = tl.program_id(1)
        card_block = tl.program_id(2)
        rank_off = tl.arange(0, BLOCK_R)[:, None]
        card_off = tl.arange(0, CARD_BLOCK)[None, :]
        cards = card_block * CARD_BLOCK + card_off
        mask = (row < batch_size) & (rank_off < RANK_BINS) & (cards < NUM_CARDS)
        base = (
            (row * NUM_PLAYERS + player) * NUM_CARDS * RANK_BINS
            + cards * RANK_BINS
            + rank_off
        )
        mass = tl.load(card_rank_mass_ptr + base, mask=mask, other=0.0).to(tl.float32)
        prefix = tl.cumsum(mass, axis=0)
        tl.store(card_prefix_ptr + base, prefix, mask=mask)
        total = tl.sum(mass, axis=0)
        total_base = (row * NUM_PLAYERS + player) * NUM_CARDS + cards
        tl.store(card_total_ptr + total_base, total, mask=(row < batch_size) & (cards < NUM_CARDS))

    @triton.jit
    def _preflop_gate_residual_combine_kernel(
        x_ptr,
        mixed_ptr,
        gate_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mixed = tl.load(mixed_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        out = x + mixed * tl.sigmoid(gate) * scale
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _preflop_token_mixer_gate_residual_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        w_in_ptr,
        w_out_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)

        y0 = tl.load(y_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0)
        y1 = tl.load(y_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0)
        y2 = tl.load(y_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0)
        y3 = tl.load(y_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0)
        y4 = tl.load(y_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0)
        y5 = tl.load(y_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0)
        y6 = tl.load(y_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0)

        out0 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out1 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out2 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out3 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out4 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out5 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out6 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)

        for h in tl.static_range(0, 28):
            hidden = (
                y0 * tl.load(w_in_ptr + h * 7 + 0)
                + y1 * tl.load(w_in_ptr + h * 7 + 1)
                + y2 * tl.load(w_in_ptr + h * 7 + 2)
                + y3 * tl.load(w_in_ptr + h * 7 + 3)
                + y4 * tl.load(w_in_ptr + h * 7 + 4)
                + y5 * tl.load(w_in_ptr + h * 7 + 5)
                + y6 * tl.load(w_in_ptr + h * 7 + 6)
            )
            hidden = tl.where(hidden >= 0.0, hidden, hidden * 0.01)
            out0 += hidden * tl.load(w_out_ptr + 0 * 28 + h)
            out1 += hidden * tl.load(w_out_ptr + 1 * 28 + h)
            out2 += hidden * tl.load(w_out_ptr + 2 * 28 + h)
            out3 += hidden * tl.load(w_out_ptr + 3 * 28 + h)
            out4 += hidden * tl.load(w_out_ptr + 4 * 28 + h)
            out5 += hidden * tl.load(w_out_ptr + 5 * 28 + h)
            out6 += hidden * tl.load(w_out_ptr + 6 * 28 + h)

        x0 = tl.load(x_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x4 = tl.load(x_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x5 = tl.load(x_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x6 = tl.load(x_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)

        g0 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g1 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g3 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g4 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g5 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g6 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)

        out0 = x0 + out0 * tl.sigmoid(g0) * scale
        out1 = x1 + out1 * tl.sigmoid(g1) * scale
        out2 = x2 + out2 * tl.sigmoid(g2) * scale
        out3 = x3 + out3 * tl.sigmoid(g3) * scale
        out4 = x4 + out4 * tl.sigmoid(g4) * scale
        out5 = x5 + out5 * tl.sigmoid(g5) * scale
        out6 = x6 + out6 * tl.sigmoid(g6) * scale

        tl.store(out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], out0, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], out1, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], out2, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], out3, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], out4, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], out5, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], out6, mask=mask)

    @triton.jit
    def _preflop_token_mixer_gate_residual_persistent_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        w_in_ptr,
        w_out_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_PROGRAMS: tl.constexpr,
    ):
        start_pid = tl.program_id(axis=0)
        num_pid_b = tl.cdiv(batch_size, BLOCK_B)
        num_pid_d = tl.cdiv(dim, BLOCK_D)
        num_tiles = num_pid_b * num_pid_d

        for tile_id in tl.range(start_pid, num_tiles, NUM_PROGRAMS, flatten=True):
            pid_b = tile_id // num_pid_d
            pid_d = tile_id - pid_b * num_pid_d
            offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
            offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
            mask = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)

            y0 = tl.load(y_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0)
            y1 = tl.load(y_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0)
            y2 = tl.load(y_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0)
            y3 = tl.load(y_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0)
            y4 = tl.load(y_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0)
            y5 = tl.load(y_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0)
            y6 = tl.load(y_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0)

            out0 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
            out1 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
            out2 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
            out3 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
            out4 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
            out5 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
            out6 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)

            for h in tl.static_range(0, 28):
                hidden = (
                    y0 * tl.load(w_in_ptr + h * 7 + 0)
                    + y1 * tl.load(w_in_ptr + h * 7 + 1)
                    + y2 * tl.load(w_in_ptr + h * 7 + 2)
                    + y3 * tl.load(w_in_ptr + h * 7 + 3)
                    + y4 * tl.load(w_in_ptr + h * 7 + 4)
                    + y5 * tl.load(w_in_ptr + h * 7 + 5)
                    + y6 * tl.load(w_in_ptr + h * 7 + 6)
                )
                hidden = tl.where(hidden >= 0.0, hidden, hidden * 0.01)
                out0 += hidden * tl.load(w_out_ptr + 0 * 28 + h)
                out1 += hidden * tl.load(w_out_ptr + 1 * 28 + h)
                out2 += hidden * tl.load(w_out_ptr + 2 * 28 + h)
                out3 += hidden * tl.load(w_out_ptr + 3 * 28 + h)
                out4 += hidden * tl.load(w_out_ptr + 4 * 28 + h)
                out5 += hidden * tl.load(w_out_ptr + 5 * 28 + h)
                out6 += hidden * tl.load(w_out_ptr + 6 * 28 + h)

            x0 = tl.load(x_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(x_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            x3 = tl.load(x_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            x4 = tl.load(x_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            x5 = tl.load(x_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            x6 = tl.load(x_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)

            g0 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            g1 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            g2 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            g3 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            g4 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            g5 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
            g6 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)

            out0 = x0 + out0 * tl.sigmoid(g0) * scale
            out1 = x1 + out1 * tl.sigmoid(g1) * scale
            out2 = x2 + out2 * tl.sigmoid(g2) * scale
            out3 = x3 + out3 * tl.sigmoid(g3) * scale
            out4 = x4 + out4 * tl.sigmoid(g4) * scale
            out5 = x5 + out5 * tl.sigmoid(g5) * scale
            out6 = x6 + out6 * tl.sigmoid(g6) * scale

            tl.store(out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], out0, mask=mask)
            tl.store(out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], out1, mask=mask)
            tl.store(out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], out2, mask=mask)
            tl.store(out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], out3, mask=mask)
            tl.store(out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], out4, mask=mask)
            tl.store(out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], out5, mask=mask)
            tl.store(out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], out6, mask=mask)

    @triton.jit
    def _preflop_token_mixer_gate_residual_next_norm_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        w_in_ptr,
        w_out_ptr,
        norm_weight_ptr,
        out_ptr,
        normed_out_ptr,
        batch_size: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)

        y0 = tl.load(y_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0)
        y1 = tl.load(y_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0)
        y2 = tl.load(y_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0)
        y3 = tl.load(y_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0)
        y4 = tl.load(y_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0)
        y5 = tl.load(y_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0)
        y6 = tl.load(y_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0)

        out0 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out1 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out2 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out3 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out4 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out5 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        out6 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)

        for h in tl.static_range(0, 28):
            hidden = (
                y0 * tl.load(w_in_ptr + h * 7 + 0)
                + y1 * tl.load(w_in_ptr + h * 7 + 1)
                + y2 * tl.load(w_in_ptr + h * 7 + 2)
                + y3 * tl.load(w_in_ptr + h * 7 + 3)
                + y4 * tl.load(w_in_ptr + h * 7 + 4)
                + y5 * tl.load(w_in_ptr + h * 7 + 5)
                + y6 * tl.load(w_in_ptr + h * 7 + 6)
            )
            hidden = tl.where(hidden >= 0.0, hidden, hidden * 0.01)
            out0 += hidden * tl.load(w_out_ptr + 0 * 28 + h)
            out1 += hidden * tl.load(w_out_ptr + 1 * 28 + h)
            out2 += hidden * tl.load(w_out_ptr + 2 * 28 + h)
            out3 += hidden * tl.load(w_out_ptr + 3 * 28 + h)
            out4 += hidden * tl.load(w_out_ptr + 4 * 28 + h)
            out5 += hidden * tl.load(w_out_ptr + 5 * 28 + h)
            out6 += hidden * tl.load(w_out_ptr + 6 * 28 + h)

        x0 = tl.load(x_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x4 = tl.load(x_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x5 = tl.load(x_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x6 = tl.load(x_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)

        g0 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g1 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g2 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g3 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g4 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g5 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        g6 = tl.load(gate_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)

        out0 = x0 + out0 * tl.sigmoid(g0) * scale
        out1 = x1 + out1 * tl.sigmoid(g1) * scale
        out2 = x2 + out2 * tl.sigmoid(g2) * scale
        out3 = x3 + out3 * tl.sigmoid(g3) * scale
        out4 = x4 + out4 * tl.sigmoid(g4) * scale
        out5 = x5 + out5 * tl.sigmoid(g5) * scale
        out6 = x6 + out6 * tl.sigmoid(g6) * scale

        norm_weight = tl.load(norm_weight_ptr + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        ss0 = tl.sum(tl.where(mask, out0 * out0, 0.0), axis=1)
        ss1 = tl.sum(tl.where(mask, out1 * out1, 0.0), axis=1)
        ss2 = tl.sum(tl.where(mask, out2 * out2, 0.0), axis=1)
        ss3 = tl.sum(tl.where(mask, out3 * out3, 0.0), axis=1)
        ss4 = tl.sum(tl.where(mask, out4 * out4, 0.0), axis=1)
        ss5 = tl.sum(tl.where(mask, out5 * out5, 0.0), axis=1)
        ss6 = tl.sum(tl.where(mask, out6 * out6, 0.0), axis=1)

        norm0 = out0 * tl.rsqrt(ss0[:, None] / dim + eps) * norm_weight[None, :]
        norm1 = out1 * tl.rsqrt(ss1[:, None] / dim + eps) * norm_weight[None, :]
        norm2 = out2 * tl.rsqrt(ss2[:, None] / dim + eps) * norm_weight[None, :]
        norm3 = out3 * tl.rsqrt(ss3[:, None] / dim + eps) * norm_weight[None, :]
        norm4 = out4 * tl.rsqrt(ss4[:, None] / dim + eps) * norm_weight[None, :]
        norm5 = out5 * tl.rsqrt(ss5[:, None] / dim + eps) * norm_weight[None, :]
        norm6 = out6 * tl.rsqrt(ss6[:, None] / dim + eps) * norm_weight[None, :]

        tl.store(out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], out0, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], out1, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], out2, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], out3, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], out4, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], out5, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], out6, mask=mask)

        tl.store(normed_out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], norm0, mask=mask)
        tl.store(normed_out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], norm1, mask=mask)
        tl.store(normed_out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], norm2, mask=mask)
        tl.store(normed_out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], norm3, mask=mask)
        tl.store(normed_out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], norm4, mask=mask)
        tl.store(normed_out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], norm5, mask=mask)
        tl.store(normed_out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], norm6, mask=mask)

    @triton.jit
    def _preflop_ffn_residual_next_token_norm_kernel(
        residual_ptr,
        ffn_out_ptr,
        norm_weight_ptr,
        out_ptr,
        normed_out_ptr,
        batch_size: tl.constexpr,
        token_count: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)
        base = (offs_b[:, None] * token_count + pid_t) * dim + offs_d[None, :]

        residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
        ffn_out = tl.load(ffn_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        out = residual + ffn_out * scale
        ss = tl.sum(tl.where(mask, out * out, 0.0), axis=1)
        norm_weight = tl.load(
            norm_weight_ptr + offs_d,
            mask=offs_d < dim,
            other=0.0,
        ).to(tl.float32)
        normed = out * tl.rsqrt(ss[:, None] / dim + eps) * norm_weight[None, :]

        tl.store(out_ptr + base, out, mask=mask)
        tl.store(normed_out_ptr + base, normed, mask=mask)

    @triton.jit
    def _preflop_token_mixer_norm_gate_residual_kernel(
        x_ptr,
        norm_weight_ptr,
        gate_weight_ptr,
        gate_bias_ptr,
        w_in_ptr,
        w_out_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GATE_DOT_BF16: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)

        ss0 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ss1 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ss2 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ss3 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ss4 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ss5 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        ss6 = tl.zeros((BLOCK_B,), dtype=tl.float32)
        for k0 in tl.range(0, dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = (offs_b[:, None] < batch_size) & (offs_k[None, :] < dim)
            x0k = tl.load(x_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_k[None, :], mask=k_mask, other=0.0).to(tl.float32)
            x1k = tl.load(x_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_k[None, :], mask=k_mask, other=0.0).to(tl.float32)
            x2k = tl.load(x_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_k[None, :], mask=k_mask, other=0.0).to(tl.float32)
            x3k = tl.load(x_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_k[None, :], mask=k_mask, other=0.0).to(tl.float32)
            x4k = tl.load(x_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_k[None, :], mask=k_mask, other=0.0).to(tl.float32)
            x5k = tl.load(x_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_k[None, :], mask=k_mask, other=0.0).to(tl.float32)
            x6k = tl.load(x_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_k[None, :], mask=k_mask, other=0.0).to(tl.float32)
            ss0 += tl.sum(x0k * x0k, axis=1)
            ss1 += tl.sum(x1k * x1k, axis=1)
            ss2 += tl.sum(x2k * x2k, axis=1)
            ss3 += tl.sum(x3k * x3k, axis=1)
            ss4 += tl.sum(x4k * x4k, axis=1)
            ss5 += tl.sum(x5k * x5k, axis=1)
            ss6 += tl.sum(x6k * x6k, axis=1)

        inv0 = tl.rsqrt(ss0 / dim + eps)
        inv1 = tl.rsqrt(ss1 / dim + eps)
        inv2 = tl.rsqrt(ss2 / dim + eps)
        inv3 = tl.rsqrt(ss3 / dim + eps)
        inv4 = tl.rsqrt(ss4 / dim + eps)
        inv5 = tl.rsqrt(ss5 / dim + eps)
        inv6 = tl.rsqrt(ss6 / dim + eps)

        norm_d = tl.load(norm_weight_ptr + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        x0 = tl.load(x_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x4 = tl.load(x_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x5 = tl.load(x_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        x6 = tl.load(x_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)

        y0 = x0 * inv0[:, None] * norm_d[None, :]
        y1 = x1 * inv1[:, None] * norm_d[None, :]
        y2 = x2 * inv2[:, None] * norm_d[None, :]
        y3 = x3 * inv3[:, None] * norm_d[None, :]
        y4 = x4 * inv4[:, None] * norm_d[None, :]
        y5 = x5 * inv5[:, None] * norm_d[None, :]
        y6 = x6 * inv6[:, None] * norm_d[None, :]

        mix0 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        mix1 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        mix2 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        mix3 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        mix4 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        mix5 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
        mix6 = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)

        for h in tl.static_range(0, 28):
            hidden = (
                y0 * tl.load(w_in_ptr + h * 7 + 0)
                + y1 * tl.load(w_in_ptr + h * 7 + 1)
                + y2 * tl.load(w_in_ptr + h * 7 + 2)
                + y3 * tl.load(w_in_ptr + h * 7 + 3)
                + y4 * tl.load(w_in_ptr + h * 7 + 4)
                + y5 * tl.load(w_in_ptr + h * 7 + 5)
                + y6 * tl.load(w_in_ptr + h * 7 + 6)
            )
            hidden = tl.where(hidden >= 0.0, hidden, hidden * 0.01)
            mix0 += hidden * tl.load(w_out_ptr + 0 * 28 + h)
            mix1 += hidden * tl.load(w_out_ptr + 1 * 28 + h)
            mix2 += hidden * tl.load(w_out_ptr + 2 * 28 + h)
            mix3 += hidden * tl.load(w_out_ptr + 3 * 28 + h)
            mix4 += hidden * tl.load(w_out_ptr + 4 * 28 + h)
            mix5 += hidden * tl.load(w_out_ptr + 5 * 28 + h)
            mix6 += hidden * tl.load(w_out_ptr + 6 * 28 + h)

        bias = tl.load(gate_bias_ptr + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        gate0 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        gate1 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        gate2 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        gate3 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        gate4 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        gate5 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        gate6 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        for k0 in tl.range(0, dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            k_mask = (offs_b[:, None] < batch_size) & (offs_k[None, :] < dim)
            norm_k = tl.load(norm_weight_ptr + offs_k, mask=offs_k < dim, other=0.0).to(tl.float32)
            w_gate = tl.load(
                gate_weight_ptr + offs_k[:, None] + offs_d[None, :] * dim,
                mask=(offs_k[:, None] < dim) & (offs_d[None, :] < dim),
                other=0.0,
            )
            if not GATE_DOT_BF16:
                w_gate = w_gate.to(tl.float32)
            x0k = tl.load(x_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_k[None, :], mask=k_mask, other=0.0)
            x1k = tl.load(x_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_k[None, :], mask=k_mask, other=0.0)
            x2k = tl.load(x_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_k[None, :], mask=k_mask, other=0.0)
            x3k = tl.load(x_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_k[None, :], mask=k_mask, other=0.0)
            x4k = tl.load(x_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_k[None, :], mask=k_mask, other=0.0)
            x5k = tl.load(x_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_k[None, :], mask=k_mask, other=0.0)
            x6k = tl.load(x_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_k[None, :], mask=k_mask, other=0.0)
            y0k = x0k.to(tl.float32) * inv0[:, None] * norm_k[None, :]
            y1k = x1k.to(tl.float32) * inv1[:, None] * norm_k[None, :]
            y2k = x2k.to(tl.float32) * inv2[:, None] * norm_k[None, :]
            y3k = x3k.to(tl.float32) * inv3[:, None] * norm_k[None, :]
            y4k = x4k.to(tl.float32) * inv4[:, None] * norm_k[None, :]
            y5k = x5k.to(tl.float32) * inv5[:, None] * norm_k[None, :]
            y6k = x6k.to(tl.float32) * inv6[:, None] * norm_k[None, :]
            if GATE_DOT_BF16:
                y0k = y0k.to(tl.bfloat16)
                y1k = y1k.to(tl.bfloat16)
                y2k = y2k.to(tl.bfloat16)
                y3k = y3k.to(tl.bfloat16)
                y4k = y4k.to(tl.bfloat16)
                y5k = y5k.to(tl.bfloat16)
                y6k = y6k.to(tl.bfloat16)
            gate0 += tl.dot(y0k, w_gate)
            gate1 += tl.dot(y1k, w_gate)
            gate2 += tl.dot(y2k, w_gate)
            gate3 += tl.dot(y3k, w_gate)
            gate4 += tl.dot(y4k, w_gate)
            gate5 += tl.dot(y5k, w_gate)
            gate6 += tl.dot(y6k, w_gate)

        out0 = x0 + mix0 * tl.sigmoid(gate0) * scale
        out1 = x1 + mix1 * tl.sigmoid(gate1) * scale
        out2 = x2 + mix2 * tl.sigmoid(gate2) * scale
        out3 = x3 + mix3 * tl.sigmoid(gate3) * scale
        out4 = x4 + mix4 * tl.sigmoid(gate4) * scale
        out5 = x5 + mix5 * tl.sigmoid(gate5) * scale
        out6 = x6 + mix6 * tl.sigmoid(gate6) * scale

        tl.store(out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], out0, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], out1, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], out2, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], out3, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], out4, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], out5, mask=mask)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], out6, mask=mask)


def _preflop_token_mixer_leaky_relu_triton(
    y: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not y.is_contiguous():
        y = y.contiguous()
    out = torch.empty_like(y)
    batch_size, token_count, dim = y.shape
    if token_count != 7 or w_in.shape != (28, 7) or w_out.shape != (7, 28):
        raise ValueError("specialized preflop token mixer requires 7 -> 28 -> 7")
    block_b = 8
    block_d = 32
    grid = (triton.cdiv(batch_size, block_b), triton.cdiv(dim, block_d))
    _preflop_token_mixer_leaky_relu_kernel[grid](
        y,
        w_in,
        w_out,
        out,
        batch_size,
        dim,
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def _river_card_rank_prefix_triton(
    card_rank_mass: torch.Tensor,
    *,
    rank_bins: int,
    num_players: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not card_rank_mass.is_cuda:
        raise ValueError("river card-rank prefix Triton path requires CUDA")
    batch_size = card_rank_mass.shape[0]
    prefix = torch.empty_like(card_rank_mass, dtype=torch.float32)
    card_total = torch.empty(
        batch_size,
        num_players,
        52,
        device=card_rank_mass.device,
        dtype=torch.float32,
    )
    block_r = 1 << (int(rank_bins) - 1).bit_length()
    card_block = 16
    _river_card_rank_prefix_kernel[
        (batch_size, num_players, triton.cdiv(52, card_block))
    ](
        card_rank_mass,
        prefix,
        card_total,
        batch_size,
        RANK_BINS=int(rank_bins),
        NUM_PLAYERS=int(num_players),
        NUM_CARDS=52,
        BLOCK_R=block_r,
        CARD_BLOCK=card_block,
        num_warps=8,
    )
    return prefix, card_total


def _preflop_gate_residual_combine_triton(
    x: torch.Tensor,
    mixed: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not x.is_contiguous():
        x = x.contiguous()
    if not mixed.is_contiguous():
        mixed = mixed.contiguous()
    if not gate.is_contiguous():
        gate = gate.contiguous()
    if mixed.shape != x.shape or gate.shape != x.shape:
        raise ValueError("x, mixed, and gate must have matching shapes")
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_n = 256
    grid = (triton.cdiv(n_elements, block_n),)
    _preflop_gate_residual_combine_kernel[grid](
        x,
        mixed,
        gate,
        out,
        n_elements,
        1.0 / math.sqrt(2.0),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out


def _preflop_token_mixer_gate_residual_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    *,
    block_b: int = 8,
    block_d: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    if not gate.is_contiguous():
        gate = gate.contiguous()
    out = torch.empty_like(x)
    batch_size, token_count, dim = x.shape
    if (
        token_count != 7
        or y.shape != x.shape
        or gate.shape != x.shape
        or w_in.shape != (28, 7)
        or w_out.shape != (7, 28)
    ):
        raise ValueError("specialized preflop token mixer gate requires 7 -> 28 -> 7")
    if block_b <= 0 or block_d <= 0:
        raise ValueError("block_b and block_d must be positive")
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")
    grid = (triton.cdiv(batch_size, block_b), triton.cdiv(dim, block_d))
    _preflop_token_mixer_gate_residual_kernel[grid](
        x,
        y,
        gate,
        w_in,
        w_out,
        out,
        batch_size,
        dim,
        1.0 / math.sqrt(2.0),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out


def _preflop_token_mixer_gate_residual_persistent_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    *,
    programs_per_sm: int = 8,
    block_b: int = 8,
    block_d: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    if not gate.is_contiguous():
        gate = gate.contiguous()
    out = torch.empty_like(x)
    batch_size, token_count, dim = x.shape
    if (
        token_count != 7
        or y.shape != x.shape
        or gate.shape != x.shape
        or w_in.shape != (28, 7)
        or w_out.shape != (7, 28)
    ):
        raise ValueError(
            "specialized persistent preflop token mixer gate requires 7 -> 28 -> 7"
        )
    if programs_per_sm <= 0:
        raise ValueError("programs_per_sm must be positive")
    if block_b <= 0 or block_d <= 0:
        raise ValueError("block_b and block_d must be positive")
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")
    num_tiles = triton.cdiv(batch_size, block_b) * triton.cdiv(dim, block_d)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    num_programs = min(num_sms * int(programs_per_sm), num_tiles)
    grid = (num_programs,)
    _preflop_token_mixer_gate_residual_persistent_kernel[grid](
        x,
        y,
        gate,
        w_in,
        w_out,
        out,
        batch_size,
        dim,
        1.0 / math.sqrt(2.0),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        NUM_PROGRAMS=num_programs,
        num_warps=num_warps,
    )
    return out


def _preflop_token_mixer_gate_residual_next_norm_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    block_b: int = 1,
    block_d: int = 256,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    if not gate.is_contiguous():
        gate = gate.contiguous()
    out = torch.empty_like(x)
    normed_out = torch.empty_like(x)
    batch_size, token_count, dim = x.shape
    if (
        token_count != 7
        or y.shape != x.shape
        or gate.shape != x.shape
        or w_in.shape != (28, 7)
        or w_out.shape != (7, 28)
        or norm_weight.shape != (dim,)
    ):
        raise ValueError(
            "specialized preflop token mixer next-norm path requires "
            "7 -> 28 -> 7 token weights and a matching norm weight"
        )
    if dim > block_d:
        raise ValueError("block_d must cover the full hidden dimension")
    if block_b <= 0 or block_d <= 0:
        raise ValueError("block_b and block_d must be positive")
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")
    grid = (triton.cdiv(batch_size, block_b),)
    _preflop_token_mixer_gate_residual_next_norm_kernel[grid](
        x,
        y,
        gate,
        w_in,
        w_out,
        norm_weight,
        out,
        normed_out,
        batch_size,
        dim,
        eps,
        1.0 / math.sqrt(2.0),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out, normed_out


def _preflop_ffn_residual_next_token_norm_triton(
    residual: torch.Tensor,
    ffn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float = 1e-5,
    block_b: int = 2,
    block_d: int = 256,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not ffn_out.is_contiguous():
        ffn_out = ffn_out.contiguous()
    if residual.shape != ffn_out.shape:
        raise ValueError("residual and ffn_out must have matching shapes")
    out = torch.empty_like(residual)
    normed_out = torch.empty_like(residual)
    batch_size, token_count, dim = residual.shape
    if norm_weight.shape != (dim,):
        raise ValueError("norm_weight must match the hidden dimension")
    if dim > block_d:
        raise ValueError("block_d must cover the full hidden dimension")
    if block_b <= 0 or block_d <= 0:
        raise ValueError("block_b and block_d must be positive")
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")
    grid = (triton.cdiv(batch_size, block_b), token_count)
    _preflop_ffn_residual_next_token_norm_kernel[grid](
        residual,
        ffn_out,
        norm_weight,
        out,
        normed_out,
        batch_size,
        token_count,
        dim,
        eps,
        1.0 / math.sqrt(2.0),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out, normed_out


def _preflop_token_mixer_norm_gate_residual_triton(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, token_count, dim = x.shape
    if (
        token_count != 7
        or norm_weight.shape != (dim,)
        or gate_weight.shape != (dim, dim)
        or gate_bias.shape != (dim,)
        or w_in.shape != (28, 7)
        or w_out.shape != (7, 28)
    ):
        raise ValueError(
            "specialized preflop token mixer megakernel requires "
            "7 tokens, 7 -> 28 -> 7 token weights, and dim -> dim gate"
        )
    block_b = 8
    block_d = 32
    block_k = 32
    grid = (triton.cdiv(batch_size, block_b), triton.cdiv(dim, block_d))
    _preflop_token_mixer_norm_gate_residual_kernel[grid](
        x,
        norm_weight,
        gate_weight,
        gate_bias,
        w_in,
        w_out,
        out,
        batch_size,
        dim,
        eps,
        1.0 / math.sqrt(2.0),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        GATE_DOT_BF16=False,
        num_warps=4,
    )
    return out


def _preflop_token_mixer_norm_gate_residual_bf16_gate_triton(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    batch_size, token_count, dim = x.shape
    if (
        token_count != 7
        or norm_weight.shape != (dim,)
        or gate_weight.shape != (dim, dim)
        or gate_bias.shape != (dim,)
        or w_in.shape != (28, 7)
        or w_out.shape != (7, 28)
    ):
        raise ValueError(
            "specialized preflop token mixer bf16-gate megakernel requires "
            "7 tokens, 7 -> 28 -> 7 token weights, and dim -> dim gate"
        )
    block_b = 8
    block_d = 32
    block_k = 32
    grid = (triton.cdiv(batch_size, block_b), triton.cdiv(dim, block_d))
    _preflop_token_mixer_norm_gate_residual_kernel[grid](
        x,
        norm_weight,
        gate_weight,
        gate_bias,
        w_in,
        w_out,
        out,
        batch_size,
        dim,
        eps,
        1.0 / math.sqrt(2.0),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        GATE_DOT_BF16=True,
        num_warps=4,
    )
    return out


def _validate_internal_zero_sum(num_players: int, enforce_zero_sum: bool) -> None:
    if int(num_players) != 2 and bool(enforce_zero_sum):
        raise ValueError(
            "Internal zero-sum projection is heads-up only; multiway value "
            "models must use fold-aware external value postprocessing."
        )


def _is_cuda_graph_capturing(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda" or not torch.cuda.is_available():
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


class ResidualBlock(nn.Module):
    def __init__(self, inner: nn.Module, alpha: float) -> None:
        super().__init__()
        self.inner = inner
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.inner(x) + x


def ffn_block(
    in_dim: int,
    hidden_dim: int,
    out_dim: int | None = None,
    nonlinearity: NonlinearityType = NonlinearityType.gelu,
) -> nn.Module:
    if out_dim is None:
        out_dim = in_dim
    if nonlinearity == NonlinearityType.swiglu:
        return nn.Sequential(
            OrderedDict(
                [
                    ("norm", nn.RMSNorm(in_dim, eps=1e-5)),
                    ("swiglu", SwiGLU(in_dim, hidden_dim, out_dim)),
                ]
            )
        )
    else:
        return nn.Sequential(
            OrderedDict(
                [
                    ("norm", nn.RMSNorm(in_dim, eps=1e-5)),
                    ("linear_in", nn.Linear(in_dim, hidden_dim, bias=False)),
                    ("activation", get_activation(nonlinearity)),
                    ("linear_out", nn.Linear(hidden_dim, out_dim)),
                ]
            )
        )


def output_projection(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        OrderedDict(
            [
                ("norm", nn.RMSNorm(in_dim, eps=1e-5)),
                ("linear_out", nn.Linear(in_dim, out_dim)),
            ]
        )
    )


class CardTokenValueHead(nn.Module):
    """Blocker-aware value head that aggregates opponent range card tokens."""

    def __init__(
        self,
        hidden_dim: int,
        token_dim: int,
        num_players: int,
        nonlinearity: NonlinearityType,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        self.token_dim = int(token_dim)
        self.num_players = int(num_players)
        self.hand_value_proj = output_projection(hidden_dim, self.token_dim)
        self.trunk_proj = output_projection(hidden_dim, self.token_dim)
        self.card_ffn = ResidualBlock(
            ffn_block(
                self.token_dim,
                self.token_dim * 2,
                self.token_dim,
                nonlinearity,
            ),
            1.0 / math.sqrt(2.0),
        )
        self.per_hand_value_head = output_projection(self.token_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        player_beliefs: torch.Tensor,
        hand_emb: torch.Tensor,
        hand_card_a: torch.Tensor,
        hand_card_b: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        num_players = player_beliefs.shape[1]
        if num_players != self.num_players:
            raise ValueError(
                f"expected {self.num_players} players, got {num_players}"
            )

        total_belief = player_beliefs.sum(dim=1, keepdim=True)
        opp_belief = total_belief - player_beliefs
        hand_token = self.hand_value_proj(hand_emb)
        if hand_token.dim() == 2:
            hand_token = hand_token[None, None]
        elif hand_token.dim() == 3:
            hand_token = hand_token[:, None]
        else:
            raise ValueError("hand_emb must have shape [N, H] or [B, N, H]")
        weighted = opp_belief[..., None].to(dtype=hand_token.dtype) * hand_token

        card_feat = x.new_zeros(
            batch_size,
            num_players,
            52,
            self.token_dim,
            dtype=weighted.dtype,
        )
        index_a = hand_card_a.view(1, 1, -1, 1).expand_as(weighted)
        index_b = hand_card_b.view(1, 1, -1, 1).expand_as(weighted)
        card_feat.scatter_add_(2, index_a, weighted)
        card_feat.scatter_add_(2, index_b, weighted)
        card_feat = self.card_ffn(card_feat)

        total = card_feat.sum(dim=2)
        per_hand = (
            total[:, :, None]
            - card_feat[:, :, hand_card_a]
            - card_feat[:, :, hand_card_b]
        )
        trunk_token = self.trunk_proj(x)[:, None, None].expand_as(per_hand)
        value_input = torch.cat((trunk_token, per_hand), dim=-1)
        return self.per_hand_value_head(value_input).squeeze(-1)

    def scale_output(self, scale: float) -> None:
        self.per_hand_value_head.get_submodule("linear_out").weight.data.mul_(
            scale
        )


class StrengthBucketEncoder(nn.Module):
    """Board and bet-context conditioned soft response buckets over hands."""

    def __init__(
        self,
        hand_dim: int,
        board_dim: int,
        bet_dim: int,
        bucket_count: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        self.bucket_count = int(bucket_count)
        self.hand_proj = nn.Linear(hand_dim, hidden_dim, bias=False)
        self.board_proj = nn.Linear(board_dim, hidden_dim, bias=False)
        self.bet_proj = nn.Linear(bet_dim, hidden_dim, bias=True)
        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, bucket_count)
        self.tau = nn.Parameter(torch.ones(()))

    def _bucket_weights(
        self,
        hand_emb: torch.Tensor,
        board_ctx: torch.Tensor,
        bet_ctx: torch.Tensor,
    ) -> torch.Tensor:
        hand_hidden = self.hand_proj(hand_emb)
        if hand_hidden.dim() == 2:
            hand_hidden = hand_hidden[None]
        elif hand_hidden.dim() != 3:
            raise ValueError("hand_emb must have shape [N, H] or [B, N, H]")
        hidden = (
            hand_hidden
            + self.board_proj(board_ctx).to(dtype=hand_hidden.dtype)[:, None, :]
            + self.bet_proj(bet_ctx).to(dtype=hand_hidden.dtype)[:, None, :]
        )
        logits = self.out(self.norm(self.activation(hidden)))
        tau = self.tau.abs().clamp_min(0.1).to(dtype=logits.dtype)
        return torch.softmax(logits / tau, dim=-1)

    def forward(
        self,
        hand_emb: torch.Tensor,
        board_ctx: torch.Tensor,
        bet_ctx: torch.Tensor,
        player_beliefs: torch.Tensor,
        hand_card_a: torch.Tensor,
        hand_card_b: torch.Tensor,
        use_blockers: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bucket_weights = self._bucket_weights(hand_emb, board_ctx, bet_ctx)
        batch_size, num_players, num_hands = player_beliefs.shape
        if bucket_weights.shape[0] == 1 and batch_size != 1:
            bucket_weights = bucket_weights.expand(batch_size, -1, -1)
        if bucket_weights.shape[:2] != (batch_size, num_hands):
            raise ValueError("bucket weights must have shape [B, N, K]")

        opp_beliefs = player_beliefs.sum(dim=1, keepdim=True) - player_beliefs
        weighted = opp_beliefs[..., None].to(dtype=bucket_weights.dtype) * (
            bucket_weights[:, None, :, :]
        )
        total = weighted.sum(dim=2)
        if not use_blockers:
            compat_bucket = total[:, :, None, :].expand(
                -1,
                -1,
                num_hands,
                -1,
            )
            return compat_bucket, bucket_weights

        card_bucket = weighted.new_zeros(
            batch_size,
            num_players,
            52,
            self.bucket_count,
        )
        index_a = hand_card_a.view(1, 1, -1, 1).expand_as(weighted)
        index_b = hand_card_b.view(1, 1, -1, 1).expand_as(weighted)
        card_bucket.scatter_add_(2, index_a, weighted)
        card_bucket.scatter_add_(2, index_b, weighted)

        compat_bucket = (
            total[:, :, None, :]
            - card_bucket[:, :, hand_card_a, :]
            - card_bucket[:, :, hand_card_b, :]
            + weighted
        )
        return compat_bucket, bucket_weights


class ValueStratificationHead(nn.Module):
    """Small per-player residual head for hand-level bucket features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_players: int,
        nonlinearity: NonlinearityType,
        state_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_players = int(num_players)
        self.state_dim = int(state_dim)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = get_activation(nonlinearity)
        if self.state_dim > 0:
            self.film = nn.Linear(self.state_dim, 2 * hidden_dim)
        self.out = nn.Linear(hidden_dim, num_players)

    def forward(
        self,
        features: torch.Tensor,
        player_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.activation(self.hidden(features))
        if self.state_dim > 0:
            if player_state is None:
                raise ValueError("player_state is required for stratification FiLM")
            film = self.film(player_state.to(dtype=hidden.dtype)).view(
                features.shape[0],
                self.num_players,
                1,
                2,
                hidden.shape[-1],
            )
            scale = film[:, :, :, 0, :]
            shift = film[:, :, :, 1, :]
            hidden = hidden * (1.0 + scale.tanh()) + shift
        all_players = self.out(hidden)
        index = torch.arange(self.num_players, device=features.device).view(
            1,
            self.num_players,
            1,
            1,
        )
        index = index.expand(
            features.shape[0],
            self.num_players,
            features.shape[2],
            1,
        )
        return all_players.gather(-1, index).squeeze(-1)

    def scale_output(self, scale: float) -> None:
        self.out.weight.data.mul_(scale)


class LatentBucketValueResidual(nn.Module):
    """Perceiver-style board-conditioned latent buckets for value residuals."""

    def __init__(
        self,
        hidden_dim: int,
        bucket_count: int,
        bucket_dim: int,
        num_players: int,
        nonlinearity: NonlinearityType,
    ) -> None:
        super().__init__()
        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if bucket_dim <= 0:
            raise ValueError("bucket_dim must be positive")
        self.bucket_count = int(bucket_count)
        self.bucket_dim = int(bucket_dim)
        self.num_players = int(num_players)
        self.hand_key = nn.Linear(hidden_dim, bucket_dim, bias=False)
        self.hand_value = nn.Linear(hidden_dim, bucket_dim, bias=False)
        self.board_query = nn.Linear(hidden_dim, bucket_count * bucket_dim)
        self.state_query = nn.Linear(hidden_dim, num_players * bucket_count * bucket_dim)
        self.bucket_query = nn.Parameter(torch.empty(bucket_count, bucket_dim))
        nn.init.zeros_(self.bucket_query)
        self.bucket_norm = nn.LayerNorm(bucket_dim)
        self.bucket_value = nn.Sequential(
            nn.Linear(bucket_dim, bucket_dim),
            get_activation(nonlinearity),
            nn.Linear(bucket_dim, 1),
        )

    def forward(
        self,
        hand_emb: torch.Tensor,
        board_ctx: torch.Tensor,
        state: torch.Tensor,
        player_beliefs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_players, num_hands = player_beliefs.shape
        if num_players != self.num_players:
            raise ValueError(
                f"expected {self.num_players} players, got {num_players}"
            )
        if hand_emb.dim() == 2:
            hand_emb = hand_emb[None].expand(batch_size, -1, -1)
        elif hand_emb.dim() != 3:
            raise ValueError("hand_emb must have shape [N, H] or [B, N, H]")
        if hand_emb.shape[:2] != (batch_size, num_hands):
            raise ValueError("hand_emb must have shape [B, N, D]")

        hand_key = self.hand_key(hand_emb)
        hand_value = self.hand_value(hand_emb)
        board_query = self.board_query(board_ctx).view(
            batch_size,
            1,
            self.bucket_count,
            self.bucket_dim,
        )
        state_query = self.state_query(state).view(
            batch_size,
            num_players,
            self.bucket_count,
            self.bucket_dim,
        )
        query = (
            self.bucket_query.to(dtype=board_query.dtype).view(
                1,
                1,
                self.bucket_count,
                self.bucket_dim,
            )
            + board_query
            + state_query
        )
        logits = torch.einsum("bpkd,bhd->bpkh", query, hand_key) / math.sqrt(
            float(self.bucket_dim)
        )

        opp_beliefs = player_beliefs.sum(dim=1, keepdim=True) - player_beliefs
        attn_logits = logits + opp_beliefs[:, :, None, :].to(
            dtype=logits.dtype
        ).clamp_min(1e-8).log()
        attend = torch.softmax(attn_logits, dim=-1)
        bucket_state = torch.einsum("bpkh,bhd->bpkd", attend, hand_value)
        bucket_state = self.bucket_norm(bucket_state + query)
        bucket_values = self.bucket_value(bucket_state).squeeze(-1)

        hand_bucket = torch.softmax(logits.transpose(2, 3), dim=-1)
        hand_residual = torch.einsum("bphk,bpk->bph", hand_bucket, bucket_values)
        bucket_weights = hand_bucket.mean(dim=1)
        return hand_residual, bucket_weights

    def scale_output(self, scale: float) -> None:
        self.bucket_value[-1].weight.data.mul_(scale)


class LowRankValueHead(nn.Module):
    """Value tower with a factorized final per-hand output projection."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_value_layers: int,
        num_hidden_layers: int,
        num_players: int,
        rank: int,
        nonlinearity: NonlinearityType,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        alpha = 1 / math.sqrt(num_hidden_layers + num_value_layers)
        self.tower = nn.Sequential(
            *[
                ResidualBlock(
                    ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity),
                    alpha,
                )
                for _ in range(num_value_layers)
            ]
        )
        self.left = output_projection(hidden_dim, rank)
        self.right = nn.Linear(rank, num_players * NUM_HANDS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(self.tower(x)))

    def scale_output(self, scale: float) -> None:
        self.right.weight.data.mul_(scale)


class HandBasisValueHead(nn.Module):
    """Hand-aware low-rank value head using a shared learned hand basis."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_value_layers: int,
        num_hidden_layers: int,
        num_players: int,
        rank: int,
        nonlinearity: NonlinearityType,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.num_players = int(num_players)
        self.rank = int(rank)
        alpha = 1 / math.sqrt(num_hidden_layers + num_value_layers)
        self.tower = nn.Sequential(
            *[
                ResidualBlock(
                    ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity),
                    alpha,
                )
                for _ in range(num_value_layers)
            ]
        )
        self.state_proj = output_projection(hidden_dim, self.num_players * self.rank)
        self.hand_basis_proj = output_projection(hidden_dim, self.rank)
        self.state_bias = output_projection(hidden_dim, self.num_players)
        self.hand_bias = output_projection(hidden_dim, self.num_players)

    def forward(self, x: torch.Tensor, hand_emb: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            player_state = (
                x[:, 1:] if x.shape[1] == self.num_players + 1 else x
            )
            if player_state.shape[1] != self.num_players:
                raise ValueError("token value input must have one token per player")
            batch_size = player_state.shape[0]
            state = self.tower(player_state.reshape(-1, player_state.shape[-1]))
            coeff = self.state_proj(state).view(
                batch_size,
                self.num_players,
                self.num_players,
                self.rank,
            )
            player_idx = torch.arange(self.num_players, device=x.device)
            coeff = coeff[:, player_idx, player_idx, :]
            state_bias = self.state_bias(state).view(
                batch_size,
                self.num_players,
                self.num_players,
            )[:, player_idx, player_idx]
        else:
            state = self.tower(x)
            coeff = self.state_proj(state).view(
                x.shape[0],
                self.num_players,
                self.rank,
            )
            state_bias = self.state_bias(state)

        basis = self.hand_basis_proj(hand_emb)
        hand_bias = self.hand_bias(hand_emb)
        if basis.dim() == 2:
            values = torch.einsum("bpr,nr->bpn", coeff, basis)
            values = values + hand_bias.transpose(0, 1)[None]
        else:
            values = torch.einsum("bpr,bnr->bpn", coeff, basis)
            values = values + hand_bias.permute(0, 2, 1)
        return values + state_bias[:, :, None]

    def scale_output(self, scale: float) -> None:
        self.state_proj.get_submodule("linear_out").weight.data.mul_(scale)
        self.state_bias.get_submodule("linear_out").weight.data.mul_(scale)
        self.hand_bias.get_submodule("linear_out").weight.data.mul_(scale)


class _RiverCanonicalMixerBlock(nn.Module):
    """One MLP-mixer block over canonical strength tokens (token-mix + channel-MLP)."""

    def __init__(self, dim: int, num_bins: int, nonlinearity: NonlinearityType) -> None:
        super().__init__()
        self.token_norm = nn.RMSNorm(dim, eps=1e-5)
        self.token_mix = nn.Linear(num_bins, num_bins)
        self.channel_norm = nn.RMSNorm(dim, eps=1e-5)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            get_activation(nonlinearity),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [N, K, dim]
        mixed = self.token_norm(tokens).transpose(1, 2)  # [N, dim, K]
        mixed = self.token_mix(mixed).transpose(1, 2)  # [N, K, dim]
        tokens = tokens + mixed
        tokens = tokens + self.channel_mlp(self.channel_norm(tokens))
        return tokens


class RiverCanonicalValueHead(nn.Module):
    """Canonical-strength mixer producing per-player nodal river values.

    Consumes per-bin token features (both players' mass, mean-coordinate, equity
    and blocked-mass row) plus broadcast globals, and emits ``num_players``
    nodal values at each quantile bin midpoint. Suit isomorphism is exact by
    construction: no 52-card identity ever enters the token features.
    """

    def __init__(
        self,
        num_bins: int,
        dim: int,
        num_layers: int,
        num_players: int,
        num_globals: int,
        nonlinearity: NonlinearityType,
        per_player_extra: int = 0,
    ) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.num_players = int(num_players)
        # Per-bin token: both players' [mass, mean_u, equity, (extra), blocked-mass
        # row (K)]. ``per_player_extra`` covers optional scalars such as the
        # per-bin analytic-baseline value.
        per_player = 3 + int(per_player_extra) + self.num_bins
        token_dim = 2 * per_player
        self.input_proj = nn.Linear(token_dim + int(num_globals), dim)
        self.blocks = nn.ModuleList(
            [
                _RiverCanonicalMixerBlock(dim, self.num_bins, nonlinearity)
                for _ in range(int(num_layers))
            ]
        )
        self.out_norm = nn.RMSNorm(dim, eps=1e-5)
        self.output_proj = nn.Linear(dim, self.num_players)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self, token_features: torch.Tensor, globals_features: torch.Tensor
    ) -> torch.Tensor:
        # token_features: [N, K, token_dim]; globals_features: [N, num_globals]
        broadcast = globals_features[:, None, :].expand(-1, self.num_bins, -1)
        tokens = self.input_proj(torch.cat((token_features, broadcast), dim=-1))
        for block in self.blocks:
            tokens = block(tokens)
        nodal = self.output_proj(self.out_norm(tokens))  # [N, K, num_players]
        return nodal.transpose(1, 2)  # [N, num_players, K]


class BetterFFN(BaseMLPModel):
    """Better PBS feed-forward poker model."""

    hand_dim = NUM_HANDS

    def __init__(
        self,
        num_actions: int,
        hidden_dim: int = 1024,
        range_hidden_dim: int = 256,
        ffn_dim: int = 1024,
        num_hidden_layers: int = 3,
        num_policy_layers: int = 3,
        num_value_layers: int = 7,
        num_players: int = 2,
        shared_trunk: bool = True,
        enforce_zero_sum: bool = True,
        board_interaction_dim: int = 0,
        board_interaction_skip_out: bool = False,
        board_interaction_gated: bool = False,
        policy_rank: int = 64,
        policy_hand_bias_rank: int = 32,
        value_per_hand_residual: bool = False,
        board_conditioned_hand_embedding_dim: int = 0,
        cross_range_rank: int = 0,
        card_token_value_head_dim: int = 0,
        context_range_stats: bool = False,
        postflop_multi_token_trunk: bool = False,
        belief_second_moment: bool = False,
        value_strength_bucket_count: int = 0,
        value_strength_bucket_film: bool = False,
        value_strength_bucket_relative: bool = False,
        value_strength_bucket_board_only: bool = False,
        value_strength_bucket_blockers: bool = True,
        value_strength_bucket_coarse_residual: bool = False,
        value_bucket_coarse_dim: int = 16,
        value_latent_bucket_count: int = 0,
        value_latent_bucket_dim: int = 32,
        value_exact_river_features: bool = False,
        value_showdown_baseline: bool = False,
        value_river_range_equity_baseline: bool = False,
        value_river_range_equity_baseline_scale: float = 0.65,
        value_river_range_equity_pot_power: float = 1.0,
        value_river_range_equity_pos_scale: float = -1.0,
        value_river_range_equity_neg_scale: float = -1.0,
        value_river_range_equity_intercept: float = 0.0,
        value_river_range_equity_blockers: bool = False,
        value_river_range_equity_rank_bins: int = 144,
        value_river_range_equity_feature_head: bool = False,
        value_river_range_equity_trunk_context: bool = False,
        value_river_range_equity_film_rank: int = 0,
        value_river_range_equity_film_hidden_dim: int = 16,
        value_turn_range_equity_baseline: bool = False,
        value_turn_range_equity_baseline_scale: float = 0.65,
        value_turn_range_equity_pot_power: float = 1.0,
        value_turn_range_equity_pos_scale: float = -1.0,
        value_turn_range_equity_neg_scale: float = -1.0,
        value_turn_range_equity_intercept: float = 0.0,
        value_turn_range_equity_blockers: bool = False,
        value_turn_range_equity_rank_bins: int = 144,
        value_turn_range_equity_feature_head: bool = False,
        value_turn_range_equity_decomposition_features: bool = False,
        value_turn_range_equity_runout_std_feature: bool = False,
        value_turn_range_equity_blocker_interactions: bool = False,
        value_turn_range_equity_feature_hidden_dim: int = 16,
        value_turn_range_equity_board_film: bool = False,
        value_turn_range_equity_hand_board_film: bool = False,
        value_turn_range_equity_chunk_size: int = 64,
        value_river_canonical_head: bool = False,
        value_river_canonical_bins: int = 32,
        value_river_canonical_dim: int = 64,
        value_river_canonical_layers: int = 2,
        value_river_canonical_blocker_rows: bool = True,
        value_river_canonical_baseline_input: bool = False,
        value_river_canonical_init_scale: float = 0.0,
        value_river_canonical_only: bool = False,
        value_river_showdown_range_encoder: bool = False,
        value_river_showdown_perhand_head: bool = False,
        value_river_showdown_perhand_dim: int = 0,
        value_output_init_scale: float = 0.0,
        value_action_summary_head: bool = False,
        value_head_rank: int = 0,
        value_hand_basis_rank: int = 0,
        belief_low_rank_dim: int = 0,
        belief_low_rank_board_conditioned: bool = False,
        belief_skip_matching_encoder: bool = False,
        belief_linear_encoder: bool = False,
        belief_board_film: bool = False,
        belief_board_bilinear_rank: int = 0,
        belief_board_mass_features: bool = False,
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
    ) -> None:
        super().__init__()
        _validate_internal_zero_sum(num_players, enforce_zero_sum)
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_value_layers = num_value_layers
        self.num_players = num_players
        self.shared_trunk = shared_trunk
        self.enforce_zero_sum = enforce_zero_sum
        self.board_interaction_dim = board_interaction_dim
        self.board_interaction_skip_out = bool(board_interaction_skip_out)
        self.board_interaction_gated = bool(board_interaction_gated)
        self.policy_rank = policy_rank
        self.policy_hand_bias_rank = policy_hand_bias_rank
        self.value_per_hand_residual = bool(value_per_hand_residual)
        self.board_conditioned_hand_embedding_dim = int(
            board_conditioned_hand_embedding_dim
        )
        self.cross_range_rank = int(cross_range_rank)
        self.card_token_value_head_dim = int(card_token_value_head_dim)
        self.context_range_stats = bool(context_range_stats)
        self.postflop_multi_token_trunk = bool(postflop_multi_token_trunk)
        self.belief_second_moment = bool(belief_second_moment)
        self.value_strength_bucket_count = int(value_strength_bucket_count)
        self.value_strength_bucket_film = bool(value_strength_bucket_film)
        self.value_strength_bucket_relative = bool(value_strength_bucket_relative)
        self.value_strength_bucket_board_only = bool(value_strength_bucket_board_only)
        self.value_strength_bucket_blockers = bool(value_strength_bucket_blockers)
        self.value_strength_bucket_coarse_residual = bool(
            value_strength_bucket_coarse_residual
        )
        self.value_bucket_coarse_dim = int(value_bucket_coarse_dim)
        self.value_latent_bucket_count = int(value_latent_bucket_count)
        self.value_latent_bucket_dim = int(value_latent_bucket_dim)
        self.value_exact_river_features = bool(value_exact_river_features)
        self.value_showdown_baseline = bool(value_showdown_baseline)
        self.value_river_range_equity_baseline = bool(
            value_river_range_equity_baseline
        )
        self.value_river_range_equity_baseline_scale = float(
            value_river_range_equity_baseline_scale
        )
        self.value_river_range_equity_pot_power = float(
            value_river_range_equity_pot_power
        )
        self.value_river_range_equity_pos_scale = float(
            value_river_range_equity_pos_scale
        )
        self.value_river_range_equity_neg_scale = float(
            value_river_range_equity_neg_scale
        )
        self.value_river_range_equity_intercept = float(
            value_river_range_equity_intercept
        )
        self.value_river_range_equity_blockers = bool(
            value_river_range_equity_blockers
        )
        self.value_river_range_equity_rank_bins = int(
            value_river_range_equity_rank_bins
        )
        self.value_river_range_equity_feature_head = bool(
            value_river_range_equity_feature_head
        )
        self.value_river_range_equity_trunk_context = bool(
            value_river_range_equity_trunk_context
        )
        self.value_river_range_equity_film_rank = int(
            value_river_range_equity_film_rank
        )
        self.value_river_range_equity_film_hidden_dim = int(
            value_river_range_equity_film_hidden_dim
        )
        self.value_turn_range_equity_baseline = bool(
            value_turn_range_equity_baseline
        )
        self.value_turn_range_equity_baseline_scale = float(
            value_turn_range_equity_baseline_scale
        )
        self.value_turn_range_equity_pot_power = float(
            value_turn_range_equity_pot_power
        )
        self.value_turn_range_equity_pos_scale = float(
            value_turn_range_equity_pos_scale
        )
        self.value_turn_range_equity_neg_scale = float(
            value_turn_range_equity_neg_scale
        )
        self.value_turn_range_equity_intercept = float(
            value_turn_range_equity_intercept
        )
        self.value_turn_range_equity_blockers = bool(
            value_turn_range_equity_blockers
        )
        self.value_turn_range_equity_rank_bins = int(
            value_turn_range_equity_rank_bins
        )
        self.value_turn_range_equity_feature_head = bool(
            value_turn_range_equity_feature_head
        )
        self.value_turn_range_equity_decomposition_features = bool(
            value_turn_range_equity_decomposition_features
        )
        self.value_turn_range_equity_runout_std_feature = bool(
            value_turn_range_equity_runout_std_feature
        )
        self.value_turn_range_equity_blocker_interactions = bool(
            value_turn_range_equity_blocker_interactions
        )
        self.value_turn_range_equity_feature_hidden_dim = int(
            value_turn_range_equity_feature_hidden_dim
        )
        self.value_turn_range_equity_board_film = bool(
            value_turn_range_equity_board_film
        )
        self.value_turn_range_equity_hand_board_film = bool(
            value_turn_range_equity_hand_board_film
        )
        self.value_turn_range_equity_chunk_size = int(
            value_turn_range_equity_chunk_size
        )
        self.value_river_canonical_head = bool(value_river_canonical_head)
        self.value_river_canonical_bins = int(value_river_canonical_bins)
        self.value_river_canonical_dim = int(value_river_canonical_dim)
        self.value_river_canonical_layers = int(value_river_canonical_layers)
        self.value_river_canonical_blocker_rows = bool(
            value_river_canonical_blocker_rows
        )
        self.value_river_canonical_baseline_input = bool(
            value_river_canonical_baseline_input
        )
        self.value_river_canonical_init_scale = float(
            value_river_canonical_init_scale
        )
        self.value_river_canonical_only = bool(value_river_canonical_only)
        self.value_river_showdown_range_encoder = bool(
            value_river_showdown_range_encoder
        )
        self.value_river_showdown_perhand_head = bool(
            value_river_showdown_perhand_head
        )
        self.value_river_showdown_perhand_dim = (
            int(value_river_showdown_perhand_dim)
            if int(value_river_showdown_perhand_dim) > 0
            else int(hidden_dim)
        )
        self.value_output_init_scale = float(value_output_init_scale)
        self.value_action_summary_head = bool(value_action_summary_head)
        self.value_head_rank = int(value_head_rank)
        self.value_hand_basis_rank = int(value_hand_basis_rank)
        self.belief_low_rank_dim = int(belief_low_rank_dim)
        self.belief_low_rank_board_conditioned = bool(
            belief_low_rank_board_conditioned
        )
        self.belief_skip_matching_encoder = bool(belief_skip_matching_encoder)
        self.belief_linear_encoder = bool(belief_linear_encoder)
        self.belief_board_film = bool(belief_board_film)
        self.belief_board_bilinear_rank = int(belief_board_bilinear_rank)
        self.belief_board_mass_features = bool(belief_board_mass_features)
        self.nonlinearity = nonlinearity

        if range_hidden_dim < 0:
            raise ValueError("range_hidden_dim must be non-negative")
        if board_interaction_dim < 0:
            raise ValueError("board_interaction_dim must be non-negative")
        if (
            self.board_interaction_skip_out
            and num_players * board_interaction_dim != hidden_dim
        ):
            raise ValueError(
                "board_interaction_skip_out requires "
                "num_players * board_interaction_dim == hidden_dim"
            )
        if self.board_conditioned_hand_embedding_dim < 0:
            raise ValueError("board_conditioned_hand_embedding_dim must be non-negative")
        if self.cross_range_rank < 0:
            raise ValueError("cross_range_rank must be non-negative")
        if self.card_token_value_head_dim < 0:
            raise ValueError("card_token_value_head_dim must be non-negative")
        if self.value_strength_bucket_count < 0:
            raise ValueError("value_strength_bucket_count must be non-negative")
        if self.value_bucket_coarse_dim <= 0:
            raise ValueError("value_bucket_coarse_dim must be positive")
        if self.value_latent_bucket_count < 0:
            raise ValueError("value_latent_bucket_count must be non-negative")
        if self.value_latent_bucket_dim <= 0:
            raise ValueError("value_latent_bucket_dim must be positive")
        if self.value_river_range_equity_baseline_scale < 0.0:
            raise ValueError(
                "value_river_range_equity_baseline_scale must be non-negative"
            )
        if self.value_river_range_equity_pot_power < 0.0:
            raise ValueError("value_river_range_equity_pot_power must be non-negative")
        if (self.value_river_range_equity_pos_scale >= 0.0) != (
            self.value_river_range_equity_neg_scale >= 0.0
        ):
            raise ValueError(
                "value_river_range_equity_pos_scale and "
                "value_river_range_equity_neg_scale must both be negative "
                "or both be non-negative"
            )
        if self.value_river_range_equity_rank_bins <= 0:
            raise ValueError("value_river_range_equity_rank_bins must be positive")
        if self.value_river_range_equity_rank_bins > NUM_HANDS:
            raise ValueError(
                "value_river_range_equity_rank_bins must be <= NUM_HANDS"
            )
        if (
            self.value_river_range_equity_feature_head
            or self.value_river_range_equity_trunk_context
            or self.value_river_range_equity_film_rank > 0
        ) and not self.value_river_range_equity_baseline:
            raise ValueError(
                "river range equity feature/context/FiLM heads require "
                "value_river_range_equity_baseline=True"
            )
        if self.value_river_range_equity_film_rank < 0:
            raise ValueError("value_river_range_equity_film_rank must be non-negative")
        if self.value_river_range_equity_film_hidden_dim <= 0:
            raise ValueError(
                "value_river_range_equity_film_hidden_dim must be positive"
            )
        if self.value_turn_range_equity_baseline_scale < 0.0:
            raise ValueError(
                "value_turn_range_equity_baseline_scale must be non-negative"
            )
        if self.value_turn_range_equity_pot_power < 0.0:
            raise ValueError("value_turn_range_equity_pot_power must be non-negative")
        if (self.value_turn_range_equity_pos_scale >= 0.0) != (
            self.value_turn_range_equity_neg_scale >= 0.0
        ):
            raise ValueError(
                "value_turn_range_equity_pos_scale and "
                "value_turn_range_equity_neg_scale must both be negative "
                "or both be non-negative"
            )
        if self.value_turn_range_equity_rank_bins <= 0:
            raise ValueError("value_turn_range_equity_rank_bins must be positive")
        if self.value_turn_range_equity_rank_bins > NUM_HANDS:
            raise ValueError("value_turn_range_equity_rank_bins must be <= NUM_HANDS")
        if self.value_turn_range_equity_feature_hidden_dim < 2:
            raise ValueError("turn equity feature hidden dim must be at least 2")
        if (
            self.value_turn_range_equity_feature_head
            and not self.value_turn_range_equity_baseline
        ):
            raise ValueError(
                "turn range equity feature head requires "
                "value_turn_range_equity_baseline=True"
            )
        if (
            self.value_turn_range_equity_board_film
            or self.value_turn_range_equity_hand_board_film
            or self.value_turn_range_equity_decomposition_features
            or self.value_turn_range_equity_runout_std_feature
            or self.value_turn_range_equity_blocker_interactions
        ) and not self.value_turn_range_equity_feature_head:
            raise ValueError("turn equity FiLM requires the turn equity feature head")
        if (
            self.value_turn_range_equity_board_film
            and self.value_turn_range_equity_hand_board_film
        ):
            raise ValueError("turn equity board FiLM modes are mutually exclusive")
        if self.value_turn_range_equity_chunk_size <= 0:
            raise ValueError("value_turn_range_equity_chunk_size must be positive")
        if self.value_river_canonical_head:
            if self.value_river_canonical_bins <= 1:
                raise ValueError("value_river_canonical_bins must be > 1")
            if self.value_river_canonical_dim <= 0:
                raise ValueError("value_river_canonical_dim must be positive")
            if self.value_river_canonical_layers <= 0:
                raise ValueError("value_river_canonical_layers must be positive")
        if self.value_river_canonical_only and not self.value_river_canonical_head:
            raise ValueError(
                "value_river_canonical_only requires value_river_canonical_head"
            )
        if self.value_head_rank < 0:
            raise ValueError("value_head_rank must be non-negative")
        if self.value_output_init_scale < 0.0:
            raise ValueError("value_output_init_scale must be non-negative")
        if self.value_river_canonical_init_scale < 0.0:
            raise ValueError(
                "value_river_canonical_init_scale must be non-negative"
            )
        if self.value_hand_basis_rank < 0:
            raise ValueError("value_hand_basis_rank must be non-negative")
        if self.belief_low_rank_dim < 0:
            raise ValueError("belief_low_rank_dim must be non-negative")
        if self.belief_board_bilinear_rank < 0:
            raise ValueError("belief_board_bilinear_rank must be non-negative")
        if self.belief_low_rank_board_conditioned and self.belief_low_rank_dim <= 0:
            raise ValueError(
                "belief_low_rank_board_conditioned requires belief_low_rank_dim > 0"
            )
        if self.value_head_rank > 0 and self.value_hand_basis_rank > 0:
            raise ValueError("value_head_rank and value_hand_basis_rank are exclusive")
        if self.belief_low_rank_dim > 0 and self.postflop_multi_token_trunk:
            raise ValueError(
                "belief_low_rank_dim is not compatible with postflop_multi_token_trunk"
            )
        if self.cross_range_rank > 0 and num_players != 2:
            raise ValueError("cross_range_rank is currently heads-up only")
        if self.context_range_stats and num_players != 2:
            raise ValueError("context_range_stats is currently heads-up only")
        if self.postflop_multi_token_trunk and num_players < 2:
            raise ValueError("postflop_multi_token_trunk requires at least two players")
        if policy_rank <= 0:
            raise ValueError("policy_rank must be positive")
        if policy_hand_bias_rank <= 0:
            raise ValueError("policy_hand_bias_rank must be positive")

        self.street_embedding = nn.Embedding(5, hidden_dim)
        self.rank_embedding = nn.Embedding(13 + 1, hidden_dim, padding_idx=13)
        self.suit_embedding = nn.Embedding(4 + 1, hidden_dim, padding_idx=4)
        self.card_embedding = nn.Embedding(52, hidden_dim)
        # Hand-aware belief encoder: project per-player belief vectors through a
        # hand embedding, then fuse across players. range_hidden_dim=0 keeps
        # this path and uses ffn_dim for the belief FFN width instead of
        # num_players * range_hidden_dim.
        combos = hand_combos_tensor()  # [NUM_HANDS, 2]
        self.register_buffer("hand_combos", combos, persistent=False)
        self.register_buffer(
            "hand_card_a",
            combos[:, 0].long().contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "hand_card_b",
            combos[:, 1].long().contiguous(),
            persistent=False,
        )
        card_ids = torch.arange(52, dtype=torch.long)
        self.register_buffer("card_ids", card_ids, persistent=False)
        self.register_buffer("card_ranks", card_ids % 13, persistent=False)
        self.register_buffer("card_suits", card_ids // 13, persistent=False)
        self.register_buffer(
            "card_rank_one_hot",
            torch.nn.functional.one_hot(card_ids % 13, 13).to(torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "card_suit_one_hot",
            torch.nn.functional.one_hot(card_ids // 13, 4).to(torch.float32),
            persistent=False,
        )
        self.register_buffer("hand_ranks", combos % 13, persistent=False)
        self.register_buffer("hand_suits", combos // 13, persistent=False)
        hand_static_features = self._build_hand_static_features(
            self.hand_ranks, self.hand_suits
        )
        self.register_buffer(
            "hand_static_features", hand_static_features, persistent=False
        )
        hand_rank_pair_idx = self._unordered_pair_index(
            self.hand_ranks[:, 0], self.hand_ranks[:, 1], 13
        )
        self.register_buffer(
            "hand_rank_pair_one_hot",
            torch.nn.functional.one_hot(hand_rank_pair_idx, 91).to(torch.float32),
            persistent=False,
        )
        hand_suit_pair_idx = self._unordered_pair_index(
            self.hand_suits[:, 0], self.hand_suits[:, 1], 4
        )
        self.register_buffer(
            "hand_suit_pair_one_hot",
            torch.nn.functional.one_hot(hand_suit_pair_idx, 10).to(torch.float32),
            persistent=False,
        )
        if range_hidden_dim == 0 and ffn_dim % num_players != 0:
            raise ValueError(
                "ffn_dim must be divisible by num_players when range_hidden_dim is 0"
            )
        effective_range_hidden_dim = (
            ffn_dim // num_players if range_hidden_dim == 0 else range_hidden_dim
        )
        self.belief_feature_dim = (
            self.belief_low_rank_dim if self.belief_low_rank_dim > 0 else hidden_dim
        )
        belief_moment_count = 2 if self.belief_second_moment else 1
        belief_in_dim = num_players * self.belief_feature_dim * belief_moment_count
        belief_hidden_dim = num_players * effective_range_hidden_dim
        self.hand_feature_proj = nn.Linear(
            HAND_STATIC_FEATURE_DIM, hidden_dim, bias=False
        )
        if self.belief_low_rank_dim > 0:
            self.belief_hand_low_proj = nn.Linear(
                hidden_dim,
                self.belief_low_rank_dim,
                bias=False,
            )
            if self.belief_low_rank_board_conditioned:
                self.belief_low_board_proj = nn.Linear(
                    hidden_dim,
                    self.belief_low_rank_dim,
                    bias=False,
                )
                self.belief_low_card_id_embedding = nn.Embedding(
                    52, self.belief_low_rank_dim
                )
                self.belief_low_card_rank_embedding = nn.Embedding(
                    13, self.belief_low_rank_dim
                )
                self.belief_low_card_suit_embedding = nn.Embedding(
                    4, self.belief_low_rank_dim
                )
                self.belief_low_card_offset = nn.Sequential(
                    nn.Linear(4 * self.belief_low_rank_dim, self.belief_low_rank_dim),
                    get_activation(nonlinearity),
                    nn.Linear(self.belief_low_rank_dim, self.belief_low_rank_dim),
                )
        if self.board_conditioned_hand_embedding_dim > 0:
            self.card_board_proj = nn.Linear(
                hidden_dim,
                52 * self.board_conditioned_hand_embedding_dim,
                bias=False,
            )
            self.card_offset_up = nn.Linear(
                self.board_conditioned_hand_embedding_dim,
                hidden_dim,
                bias=False,
            )
        if self.belief_skip_matching_encoder:
            if self.belief_linear_encoder:
                raise ValueError(
                    "belief_linear_encoder is not compatible with "
                    "belief_skip_matching_encoder"
                )
            if belief_in_dim != hidden_dim or belief_hidden_dim != hidden_dim:
                raise ValueError(
                    "belief_skip_matching_encoder requires belief input, hidden, "
                    "and output dimensions to match"
                )
            self.belief_proj = nn.RMSNorm(hidden_dim, eps=1e-5)
        elif self.belief_linear_encoder:
            self.belief_proj = nn.Linear(belief_in_dim, hidden_dim)
        else:
            self.belief_proj = ffn_block(
                belief_in_dim, belief_hidden_dim, hidden_dim, nonlinearity
            )
        if self.belief_board_film:
            self.belief_board_film_proj = nn.Linear(
                hidden_dim,
                2 * num_players * self.belief_feature_dim,
            )
        if self.belief_board_bilinear_rank > 0:
            self.belief_board_bilinear_left = nn.Linear(
                num_players * self.belief_feature_dim,
                self.belief_board_bilinear_rank,
                bias=False,
            )
            self.belief_board_bilinear_right = nn.Linear(
                hidden_dim,
                self.belief_board_bilinear_rank,
                bias=False,
            )
            self.belief_board_bilinear_out = nn.Linear(
                self.belief_board_bilinear_rank,
                hidden_dim,
                bias=False,
            )
        if self.belief_board_mass_features:
            self.belief_board_mass_proj = nn.Linear(
                num_players * (5 + 13 + 4),
                hidden_dim,
                bias=False,
            )
        if self.cross_range_rank > 0:
            self.cross_left = nn.Linear(
                self.belief_feature_dim,
                self.cross_range_rank,
                bias=False,
            )
            self.cross_right = nn.Linear(
                self.belief_feature_dim,
                self.cross_range_rank,
                bias=False,
            )
            self.cross_proj = nn.Linear(self.cross_range_rank, hidden_dim, bias=False)
        context_in_dim = context_length(num_players) + (
            5 if self.context_range_stats else 0
        )
        self.context_in_dim = int(context_in_dim)
        self.context_encoder = ffn_block(
            context_in_dim, hidden_dim, hidden_dim, nonlinearity
        )
        if self.value_strength_bucket_count > 0:
            strength_bet_dim = 32
            strength_hidden_dim = 64
            strat_input_dim = 2 * self.value_strength_bucket_count
            if self.value_strength_bucket_relative:
                strat_input_dim += 2 * self.value_strength_bucket_count
            self.strength_bet_ctx_proj = nn.Linear(5, strength_bet_dim)
            self.strength_bucket_enc = StrengthBucketEncoder(
                hidden_dim,
                hidden_dim,
                strength_bet_dim,
                self.value_strength_bucket_count,
                strength_hidden_dim,
            )
            bucket_head_name = (
                "value_bucket_coarse_residual_head"
                if self.value_strength_bucket_coarse_residual
                else "value_strat_head"
            )
            setattr(
                self,
                bucket_head_name,
                ValueStratificationHead(
                strat_input_dim,
                self.value_bucket_coarse_dim,
                num_players,
                nonlinearity,
                state_dim=hidden_dim if self.value_strength_bucket_film else 0,
                ),
            )
        if self.value_latent_bucket_count > 0:
            self.value_latent_bucket_residual = LatentBucketValueResidual(
                hidden_dim,
                self.value_latent_bucket_count,
                self.value_latent_bucket_dim,
                num_players,
                nonlinearity,
            )
        if self.value_exact_river_features:
            self.value_exact_feature_head = nn.Sequential(
                nn.Linear(4, 16),
                get_activation(nonlinearity),
                nn.Linear(16, 1),
            )
        if self.value_river_range_equity_feature_head:
            self.value_river_equity_feature_head = nn.Sequential(
                nn.Linear(6, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            self._init_river_equity_feature_head()
        if self.value_turn_range_equity_feature_head:
            turn_equity_feature_dim = 6
            if self.value_turn_range_equity_decomposition_features:
                turn_equity_feature_dim += 2
            if self.value_turn_range_equity_runout_std_feature:
                turn_equity_feature_dim += 1
            if self.value_turn_range_equity_blocker_interactions:
                turn_equity_feature_dim += 3
            turn_equity_hidden_dim = self.value_turn_range_equity_feature_hidden_dim
            self.value_turn_equity_feature_head = nn.Sequential(
                nn.Linear(turn_equity_feature_dim, turn_equity_hidden_dim),
                nn.ReLU(),
                nn.Linear(turn_equity_hidden_dim, 1),
            )
            self._init_turn_equity_feature_head()
        if self.value_turn_range_equity_board_film:
            self.value_turn_equity_board_film_proj = nn.Linear(
                hidden_dim, 2 * self.value_turn_range_equity_feature_hidden_dim
            )
            nn.init.zeros_(self.value_turn_equity_board_film_proj.weight)
            nn.init.zeros_(self.value_turn_equity_board_film_proj.bias)
        if self.value_turn_range_equity_hand_board_film:
            self.value_turn_equity_hand_film_proj = nn.Linear(
                hidden_dim,
                2 * self.value_turn_range_equity_feature_hidden_dim,
                bias=False,
            )
            self.value_turn_equity_hand_board_film_proj = nn.Linear(
                hidden_dim, 2 * self.value_turn_range_equity_feature_hidden_dim
            )
            nn.init.zeros_(self.value_turn_equity_hand_film_proj.weight)
            nn.init.zeros_(self.value_turn_equity_hand_board_film_proj.weight)
            nn.init.zeros_(self.value_turn_equity_hand_board_film_proj.bias)
        if self.value_river_range_equity_trunk_context:
            self.value_river_equity_context_proj = nn.Linear(
                2 * self.value_river_range_equity_rank_bins,
                hidden_dim,
                bias=False,
            )
            nn.init.zeros_(self.value_river_equity_context_proj.weight)
        if self.value_river_range_equity_film_rank > 0:
            rank = self.value_river_range_equity_film_rank
            film_hidden = self.value_river_range_equity_film_hidden_dim
            self.value_river_equity_film_state = nn.Linear(
                hidden_dim,
                rank,
                bias=False,
            )
            self.value_river_equity_film = nn.Sequential(
                nn.Linear(6, film_hidden),
                nn.ReLU(),
                nn.Linear(film_hidden, 2 * rank),
            )
            self.value_river_equity_film_out = nn.Linear(rank, 1, bias=False)
            nn.init.zeros_(self.value_river_equity_film_out.weight)
        if self.value_river_canonical_head:
            # Globals: pot + per-player SPR.
            num_globals = 1 + self.num_players
            self.value_river_canonical = RiverCanonicalValueHead(
                num_bins=self.value_river_canonical_bins,
                dim=self.value_river_canonical_dim,
                num_layers=self.value_river_canonical_layers,
                num_players=self.num_players,
                num_globals=num_globals,
                nonlinearity=nonlinearity,
                per_player_extra=(
                    1 if self.value_river_canonical_baseline_input else 0
                ),
            )
        if self.value_river_showdown_range_encoder:
            # Pool each of the 4 per-hand channels (belief, blocker-corrected
            # win / tie / loss mass) against the hand embedding, then project the
            # concatenation into the trunk. Zero-init so it starts as a no-op.
            self.showdown_range_proj = nn.Linear(
                num_players * 4 * hidden_dim, hidden_dim, bias=False
            )
        if self.value_river_showdown_perhand_head:
            # Dense showdown trunk encoder: for each player, map the full
            # blocker-corrected showdown vector over all hands
            # ([belief, win, tie, loss] concatenated -> 4*NUM_HANDS) straight to
            # hidden_dim with a dense projection -- no pooling through the hand
            # embedding, so there is no low-rank "range bottleneck". Then fuse the
            # two players ([dim, P] -> hidden_dim) and add into the trunk. The
            # fuse output is zero-initialised so it starts as a no-op.
            sd_dim = self.value_river_showdown_perhand_dim
            self.showdown_perhand_in = nn.Linear(4 * NUM_HANDS, sd_dim)
            self.showdown_perhand_fuse = nn.Linear(num_players * sd_dim, hidden_dim)
            self.showdown_perhand_act = get_activation(nonlinearity)
        if self.value_action_summary_head:
            self.value_action_summary = output_projection(hidden_dim, num_players)
        if board_interaction_dim > 0:
            self.rank_pair_low_embedding = nn.Embedding(91, board_interaction_dim)
            self.board_rank_low = nn.Linear(13, board_interaction_dim, bias=False)
            self.suit_pair_low_embedding = nn.Embedding(10, board_interaction_dim)
            self.board_suit_low = nn.Linear(4, board_interaction_dim, bias=False)
            if self.board_interaction_gated:
                self.board_interaction_gate = nn.Parameter(torch.zeros(()))
            if self.board_interaction_skip_out:
                self.board_interaction_norm = nn.RMSNorm(hidden_dim, eps=1e-5)
            else:
                self.rank_board_interaction_out = nn.Linear(
                    num_players * board_interaction_dim, hidden_dim, bias=False
                )
                self.suit_board_interaction_out = nn.Linear(
                    num_players * board_interaction_dim, hidden_dim, bias=False
                )

        # Build trunk
        # Default alpha is always based on hidden + value layers
        alpha = 1 / math.sqrt(num_hidden_layers + num_value_layers)
        if self.postflop_multi_token_trunk:
            self.trunk = nn.ModuleList(
                [
                    _PreflopGatedTokenMixerBlock(
                        hidden_dim,
                        token_count=num_players + 1,
                        ffn_dim=ffn_dim,
                        nonlinearity=nonlinearity,
                    )
                    for _ in range(num_hidden_layers)
                ]
            )
        else:
            layers = [
                ResidualBlock(
                    ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), alpha
                )
                for _ in range(num_hidden_layers)
            ]
            self.trunk = nn.Sequential(*layers)

        # Heads
        # If shared_trunk is False, use separate alpha for policy_head based on num_policy_layers
        policy_alpha = alpha if shared_trunk else 1 / math.sqrt(num_policy_layers)

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), policy_alpha
            )
            for _ in range(num_policy_layers)
        ]
        self.policy_tower = nn.Sequential(*layers)
        self.policy_hand_proj = output_projection(hidden_dim, self.policy_rank)
        self.policy_action_head = output_projection(
            hidden_dim, num_actions * self.policy_rank
        )
        self.policy_hand_gate = output_projection(hidden_dim, self.policy_rank)
        self.policy_dynamic_coeff = output_projection(
            hidden_dim, num_actions * HAND_DYNAMIC_FEATURE_DIM
        )
        self.policy_action_bias = output_projection(hidden_dim, num_actions)
        self.policy_hand_bias = output_projection(
            hidden_dim, self.policy_hand_bias_rank
        )
        self.policy_hand_bias_action = output_projection(
            hidden_dim, num_actions * self.policy_hand_bias_rank
        )
        self.policy_hand_norm = nn.RMSNorm(hidden_dim, eps=1e-5)

        self.hand_value_head = self._make_value_head()
        if self.value_per_hand_residual:
            self.value_residual = nn.Sequential(
                nn.Linear(3, 8),
                get_activation(nonlinearity),
                nn.Linear(8, 1),
            )

    def _make_value_head(self) -> nn.Module:
        if self.card_token_value_head_dim > 0:
            return CardTokenValueHead(
                self.hidden_dim,
                self.card_token_value_head_dim,
                self.num_players,
                self.nonlinearity,
            )
        if self.value_head_rank > 0:
            return LowRankValueHead(
                self.hidden_dim,
                self.ffn_dim,
                self.num_value_layers,
                self.num_hidden_layers,
                self.num_players,
                self.value_head_rank,
                self.nonlinearity,
            )
        if self.value_hand_basis_rank > 0:
            return HandBasisValueHead(
                self.hidden_dim,
                self.ffn_dim,
                self.num_value_layers,
                self.num_hidden_layers,
                self.num_players,
                self.value_hand_basis_rank,
                self.nonlinearity,
            )
        alpha = 1 / math.sqrt(self.num_hidden_layers + self.num_value_layers)
        layers = [
            ResidualBlock(
                ffn_block(
                    self.hidden_dim,
                    self.ffn_dim,
                    nonlinearity=self.nonlinearity,
                ),
                alpha,
            )
            for _ in range(self.num_value_layers)
        ]
        layers.append(output_projection(self.hidden_dim, self.num_players * NUM_HANDS))
        return nn.Sequential(*layers)

    def _init_river_equity_feature_head(self) -> None:
        if not hasattr(self, "value_river_equity_feature_head"):
            return
        first = self.value_river_equity_feature_head[0]
        last = self.value_river_equity_feature_head[-1]
        if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
            return
        nn.init.zeros_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        first.weight.data[0, 0] = 1.0
        first.weight.data[1, 0] = -1.0
        if self.value_river_range_equity_pos_scale >= 0.0:
            pos_scale = self.value_river_range_equity_pos_scale
            neg_scale = self.value_river_range_equity_neg_scale
            intercept = self.value_river_range_equity_intercept
        else:
            pos_scale = self.value_river_range_equity_baseline_scale
            neg_scale = self.value_river_range_equity_baseline_scale
            intercept = 0.0
        last.weight.data[0, 0] = pos_scale
        last.weight.data[0, 1] = -neg_scale
        last.bias.data.fill_(intercept)

    def _init_turn_equity_feature_head(self) -> None:
        if not hasattr(self, "value_turn_equity_feature_head"):
            return
        first = self.value_turn_equity_feature_head[0]
        last = self.value_turn_equity_feature_head[-1]
        if not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear):
            return
        nn.init.zeros_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        first.weight.data[0, 0] = 1.0
        first.weight.data[1, 0] = -1.0
        if self.value_turn_range_equity_pos_scale >= 0.0:
            pos_scale = self.value_turn_range_equity_pos_scale
            neg_scale = self.value_turn_range_equity_neg_scale
            intercept = self.value_turn_range_equity_intercept
        else:
            pos_scale = self.value_turn_range_equity_baseline_scale
            neg_scale = self.value_turn_range_equity_baseline_scale
            intercept = 0.0
        last.weight.data[0, 0] = pos_scale
        last.weight.data[0, 1] = -neg_scale
        last.bias.data.fill_(intercept)

    @staticmethod
    def _unordered_pair_index(
        first: torch.Tensor, second: torch.Tensor, num_items: int
    ) -> torch.Tensor:
        lo = torch.minimum(first, second)
        hi = torch.maximum(first, second)
        return lo * num_items - (lo * (lo - 1)) // 2 + (hi - lo)

    @staticmethod
    def _index_counts(
        indices: torch.Tensor,
        valid: torch.Tensor,
        num_items: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        safe_indices = torch.where(valid, indices, torch.zeros_like(indices))
        counts = torch.zeros(
            indices.shape[0],
            num_items,
            device=indices.device,
            dtype=dtype,
        )
        return counts.scatter_add(1, safe_indices, valid.to(dtype))

    @staticmethod
    def _build_hand_static_features(
        hand_ranks: torch.Tensor, hand_suits: torch.Tensor
    ) -> torch.Tensor:
        rank_a = hand_ranks[:, 0].to(torch.float32)
        rank_b = hand_ranks[:, 1].to(torch.float32)
        suit_a = hand_suits[:, 0]
        suit_b = hand_suits[:, 1]
        hi = torch.maximum(rank_a, rank_b)
        lo = torch.minimum(rank_a, rank_b)
        gap = (hi - lo).clamp(min=0.0)
        return torch.stack(
            [
                (rank_a == rank_b).to(torch.float32),
                (suit_a == suit_b).to(torch.float32),
                gap / 12.0,
                hi / 12.0,
                lo / 12.0,
                (hi == 12).to(torch.float32),
                (lo >= 8).to(torch.float32),
                (gap <= 1).to(torch.float32),
            ],
            dim=-1,
        )

    def _board_context(self, board: torch.Tensor) -> torch.Tensor:
        ranks = torch.where(board >= 0, board % 13, torch.full_like(board, 13))
        suits = torch.where(board >= 0, board // 13, torch.full_like(board, 4))
        return (self.rank_embedding(ranks) + self.suit_embedding(suits)).sum(dim=1)

    def _hand_embedding(self, board_context: torch.Tensor | None = None) -> torch.Tensor:
        """Per-hand exact-card embedding — shape [NUM_HANDS, hidden_dim]."""
        card_emb = self.card_embedding(self.hand_combos)
        static = self.hand_static_features.to(dtype=card_emb.dtype)
        hand_emb = card_emb.sum(dim=1) + self.hand_feature_proj(static)
        if self.board_conditioned_hand_embedding_dim <= 0 or board_context is None:
            return hand_emb
        card_offsets = self.card_board_proj(board_context).view(
            board_context.shape[0],
            52,
            self.board_conditioned_hand_embedding_dim,
        )
        hand_offset = card_offsets[:, self.hand_card_a] + card_offsets[
            :, self.hand_card_b
        ]
        return hand_emb[None] + self.card_offset_up(hand_offset)

    def _belief_moments(
        self,
        player_beliefs: torch.Tensor,
        hand_emb: torch.Tensor,
        board_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        belief_hand_emb = (
            self.belief_hand_low_proj(hand_emb)
            if self.belief_low_rank_dim > 0
            else hand_emb
        )
        if (
            self.belief_low_rank_dim > 0
            and self.belief_low_rank_board_conditioned
            and board_context is not None
        ):
            board_token = self.belief_low_board_proj(board_context)
            dtype = board_token.dtype
            card_input = torch.cat(
                (
                    board_token[:, None, :].expand(-1, 52, -1),
                    self.belief_low_card_id_embedding(self.card_ids)
                    .to(dtype=dtype)[None]
                    .expand(board_context.shape[0], -1, -1),
                    self.belief_low_card_rank_embedding(self.card_ranks)
                    .to(dtype=dtype)[None]
                    .expand(board_context.shape[0], -1, -1),
                    self.belief_low_card_suit_embedding(self.card_suits)
                    .to(dtype=dtype)[None]
                    .expand(board_context.shape[0], -1, -1),
                ),
                dim=-1,
            )
            card_offsets = self.belief_low_card_offset(card_input)
            hand_offsets = card_offsets[:, self.hand_card_a] + card_offsets[
                :, self.hand_card_b
            ]
            belief_hand_emb = (
                belief_hand_emb[None] + hand_offsets
                if belief_hand_emb.dim() == 2
                else belief_hand_emb + hand_offsets
            )
        if belief_hand_emb.dim() == 2:
            mu = player_beliefs @ belief_hand_emb
            if not self.belief_second_moment:
                return mu, None
            mu2 = player_beliefs @ belief_hand_emb.square()
        else:
            mu = torch.einsum("bpn,bnh->bph", player_beliefs, belief_hand_emb)
            if not self.belief_second_moment:
                return mu, None
            mu2 = torch.einsum(
                "bpn,bnh->bph",
                player_beliefs,
                belief_hand_emb.square(),
            )
        return mu, mu2 - mu.square()

    def _apply_belief_board_film(
        self,
        per_player_belief: torch.Tensor,
        board_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.belief_board_film or board_context is None:
            return per_player_belief
        film = self.belief_board_film_proj(board_context).view(
            -1,
            self.num_players,
            2,
            self.belief_feature_dim,
        )
        gate = 0.1 * film[:, :, 0].tanh()
        shift = 0.1 * film[:, :, 1]
        return per_player_belief * (1.0 + gate) + shift

    def _belief_board_bilinear(
        self,
        per_player_belief: torch.Tensor,
        board_context: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.belief_board_bilinear_rank <= 0 or board_context is None:
            return None
        belief_term = self.belief_board_bilinear_left(per_player_belief.flatten(1))
        board_term = self.belief_board_bilinear_right(board_context)
        return self.belief_board_bilinear_out(belief_term * board_term)

    def _belief_board_mass_features(
        self,
        player_beliefs: torch.Tensor,
        board: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.belief_board_mass_features:
            return None
        card_mass = self._card_mass(player_beliefs)
        valid = board >= 0
        board_safe = torch.where(valid, board, torch.zeros_like(board))
        gather_idx = board_safe[:, None, :].expand(
            -1,
            self.num_players,
            -1,
        )
        board_card_mass = card_mass.gather(2, gather_idx) * valid[:, None, :].to(
            dtype=card_mass.dtype
        )
        rank_mass = card_mass @ self.card_rank_one_hot.to(dtype=card_mass.dtype)
        suit_mass = card_mass @ self.card_suit_one_hot.to(dtype=card_mass.dtype)
        mass_features = torch.cat(
            (board_card_mass, rank_mass, suit_mass),
            dim=-1,
        ).flatten(1)
        return self.belief_board_mass_proj(mass_features)

    def _belief_projection_input(
        self,
        per_player_belief: torch.Tensor,
        per_player_variance: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.belief_second_moment:
            return per_player_belief.flatten(1)
        if per_player_variance is None:
            raise RuntimeError("belief_second_moment requires variance features")
        return torch.cat((per_player_belief, per_player_variance), dim=-1).flatten(1)

    def _postflop_trunk_output(
        self,
        game_token: torch.Tensor,
        player_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.postflop_multi_token_trunk:
            tokens = torch.cat((game_token[:, None, :], player_tokens), dim=1)
            return _run_preflop_gated_token_mixer_blocks(self.trunk, tokens)
        return self.trunk(game_token)

    def _policy_input_from_base(
        self,
        flat_features: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if not self.shared_trunk:
            return flat_features.detach()
        if x.dim() == 3:
            return x[:, 0]
        return x

    def _card_mass_and_unblocked_mass(
        self, belief: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        card_mass = self._card_mass(belief)
        return card_mass, self._unblocked_mass_from_card_mass(belief, card_mass)

    def _card_mass(self, belief: torch.Tensor) -> torch.Tensor:
        belief_batched = belief.view(-1, NUM_HANDS).float()
        card_mass = torch.zeros(
            belief_batched.shape[0],
            52,
            dtype=belief_batched.dtype,
            device=belief_batched.device,
        )
        card_a_idx = self.hand_card_a[None, :].expand(belief_batched.shape[0], -1)
        card_b_idx = self.hand_card_b[None, :].expand(belief_batched.shape[0], -1)
        card_mass.scatter_add_(1, card_a_idx, belief_batched)
        card_mass.scatter_add_(1, card_b_idx, belief_batched)
        return card_mass.view(*belief.shape[:-1], 52)

    def _unblocked_mass_from_card_mass(
        self, belief: torch.Tensor, card_mass: torch.Tensor
    ) -> torch.Tensor:
        belief = belief.float()
        card_mass = card_mass.float()
        total = belief.sum(dim=-1, keepdim=True)
        unblocked = (
            total
            - card_mass[..., self.hand_card_a]
            - card_mass[..., self.hand_card_b]
            + belief
        )
        return unblocked.clamp_min(0.0)

    def _calculate_unblocked_mass(self, target: torch.Tensor) -> torch.Tensor:
        _, unblocked = self._card_mass_and_unblocked_mass(target)
        return unblocked

    def _board_stats(
        self, board: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = board >= 0
        ranks = torch.where(valid, board % 13, torch.zeros_like(board))
        suits = torch.where(valid, board // 13, torch.zeros_like(board))
        rank_counts = self._index_counts(ranks, valid, 13, dtype)
        suit_counts = self._index_counts(suits, valid, 4, dtype)

        board_safe = torch.where(valid, board, torch.full_like(board, 52))
        board_onehot = torch.zeros(
            board.shape[0], 53, dtype=torch.bool, device=board.device
        )
        board_onehot.scatter_(1, board_safe, valid)
        board_onehot = board_onehot[:, :52]
        return rank_counts, suit_counts, board_onehot

    def _board_hand_feature_dot(
        self,
        board: torch.Tensor,
        coeff: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dtype = coeff.dtype
        if board_stats is None:
            rank_counts, suit_counts, board_onehot = self._board_stats(board, dtype)
        else:
            rank_counts, suit_counts, board_onehot = board_stats
            rank_counts = rank_counts.to(dtype=dtype)
            suit_counts = suit_counts.to(dtype=dtype)

        rank_a = rank_counts[:, self.hand_ranks[:, 0]]
        rank_b = rank_counts[:, self.hand_ranks[:, 1]]
        suit_a = suit_counts[:, self.hand_suits[:, 0]]
        suit_b = suit_counts[:, self.hand_suits[:, 1]]
        blocked = (
            board_onehot[:, self.hand_card_a] | board_onehot[:, self.hand_card_b]
        ).to(dtype)

        if coeff.dim() == 3:
            rank_a = rank_a[:, None]
            rank_b = rank_b[:, None]
            suit_a = suit_a[:, None]
            suit_b = suit_b[:, None]
            blocked = blocked[:, None]
            coeff = coeff[:, :, :, None]
        else:
            coeff = coeff[:, :, None]

        rank_sum = rank_a + rank_b
        suit_sum = suit_a + suit_b
        out = (rank_a / 4.0) * coeff[..., 0, :]
        out = out + (rank_b / 4.0) * coeff[..., 1, :]
        out = out + (rank_sum / 4.0) * coeff[..., 2, :]
        out = out + ((rank_a * rank_b) / 16.0) * coeff[..., 3, :]
        out = out + (suit_a / 5.0) * coeff[..., 4, :]
        out = out + (suit_b / 5.0) * coeff[..., 5, :]
        out = out + (suit_sum / 5.0) * coeff[..., 6, :]
        out = out + blocked * coeff[..., 7, :]
        return out

    def _board_hand_features_from_stats(
        self,
        board: torch.Tensor,
        dtype: torch.dtype,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if board_stats is None:
            rank_counts, suit_counts, board_onehot = self._board_stats(board, dtype)
        else:
            rank_counts, suit_counts, board_onehot = board_stats
            rank_counts = rank_counts.to(dtype=dtype)
            suit_counts = suit_counts.to(dtype=dtype)

        rank_a = rank_counts[:, self.hand_ranks[:, 0]]
        rank_b = rank_counts[:, self.hand_ranks[:, 1]]
        suit_a = suit_counts[:, self.hand_suits[:, 0]]
        suit_b = suit_counts[:, self.hand_suits[:, 1]]
        blocked = (
            board_onehot[:, self.hand_card_a] | board_onehot[:, self.hand_card_b]
        ).to(dtype)
        rank_sum = rank_a + rank_b
        suit_sum = suit_a + suit_b
        return torch.stack(
            [
                rank_a / 4.0,
                rank_b / 4.0,
                rank_sum / 4.0,
                (rank_a * rank_b) / 16.0,
                suit_a / 5.0,
                suit_b / 5.0,
                suit_sum / 5.0,
                blocked,
            ],
            dim=-1,
        )

    def _dynamic_hand_features_from_stats(
        self,
        own_belief: torch.Tensor,
        own_card_mass: torch.Tensor,
        opp_card_mass: torch.Tensor,
        opp_unblocked: torch.Tensor,
        board: torch.Tensor,
        dtype: torch.dtype,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        own_belief = own_belief.to(dtype=dtype)
        opp_unblocked = opp_unblocked.to(dtype=dtype)
        own_card_mass = own_card_mass.to(dtype=dtype)
        opp_card_mass = opp_card_mass.to(dtype=dtype)
        own_card_a = own_card_mass[..., self.hand_card_a]
        own_card_b = own_card_mass[..., self.hand_card_b]
        opp_card_a = opp_card_mass[..., self.hand_card_a]
        opp_card_b = opp_card_mass[..., self.hand_card_b]
        board_features = self._board_hand_features_from_stats(board, dtype, board_stats)
        if own_belief.dim() == 3:
            board_features = board_features[:, None].expand(
                -1, own_belief.shape[1], -1, -1
            )
        range_features = torch.stack(
            [
                own_belief,
                own_belief.clamp_min(1e-8).log(),
                opp_unblocked,
                own_card_a,
                own_card_b,
                opp_card_a,
                opp_card_b,
            ],
            dim=-1,
        )
        return torch.cat([range_features, board_features], dim=-1)

    def _dynamic_hand_feature_dot_from_stats(
        self,
        own_belief: torch.Tensor,
        own_card_mass: torch.Tensor,
        opp_card_mass: torch.Tensor,
        opp_unblocked: torch.Tensor,
        board: torch.Tensor,
        coeff: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dtype = coeff.dtype
        own_belief = own_belief.to(dtype=dtype)
        opp_unblocked = opp_unblocked.to(dtype=dtype)
        own_card_mass = own_card_mass.to(dtype=dtype)
        opp_card_mass = opp_card_mass.to(dtype=dtype)
        own_card_a = own_card_mass[..., self.hand_card_a]
        own_card_b = own_card_mass[..., self.hand_card_b]
        opp_card_a = opp_card_mass[..., self.hand_card_a]
        opp_card_b = opp_card_mass[..., self.hand_card_b]
        if coeff.dim() == own_belief.dim() + 1:
            own_belief = own_belief[:, None, :]
            opp_unblocked = opp_unblocked[:, None, :]
            own_card_a = own_card_a[:, None, :]
            own_card_b = own_card_b[:, None, :]
            opp_card_a = opp_card_a[:, None, :]
            opp_card_b = opp_card_b[:, None, :]

        out = own_belief * coeff[..., 0, None]
        out = out + own_belief.clamp_min(1e-8).log() * coeff[..., 1, None]
        out = out + opp_unblocked * coeff[..., 2, None]
        out = out + own_card_a * coeff[..., 3, None]
        out = out + own_card_b * coeff[..., 4, None]
        out = out + opp_card_a * coeff[..., 5, None]
        out = out + opp_card_b * coeff[..., 6, None]

        board_coeff = coeff[..., 7:HAND_DYNAMIC_FEATURE_DIM]
        return out + self._board_hand_feature_dot(board, board_coeff, board_stats)

    def _policy_dynamic_logits(
        self,
        player_beliefs: torch.Tensor,
        actor: torch.Tensor,
        board: torch.Tensor,
        coeff: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        actor_belief = player_beliefs.gather(
            1,
            actor[:, None, None].expand(-1, 1, NUM_HANDS),
        ).squeeze(1)

        card_mass = self._card_mass(player_beliefs)
        actor_card_mass = card_mass.gather(
            1,
            actor[:, None, None].expand(-1, 1, 52),
        ).squeeze(1)
        player_ids = torch.arange(self.num_players, device=player_beliefs.device)
        non_actor = player_ids[None, :] != actor[:, None]
        opp_card_mass = torch.where(
            non_actor[:, :, None],
            card_mass,
            torch.zeros_like(card_mass),
        ).sum(dim=1)
        opp_belief = torch.where(
            non_actor[:, :, None],
            player_beliefs,
            torch.zeros_like(player_beliefs),
        ).sum(dim=1)
        opp_unblocked = self._unblocked_mass_from_card_mass(opp_belief, opp_card_mass)
        dynamic_logits = self._dynamic_hand_feature_dot_from_stats(
            actor_belief,
            actor_card_mass,
            opp_card_mass,
            opp_unblocked,
            board,
            coeff,
            board_stats,
        )
        return dynamic_logits.transpose(1, 2)

    def _policy_dynamic_features(
        self,
        player_beliefs: torch.Tensor,
        actor: torch.Tensor,
        board: torch.Tensor,
        dtype: torch.dtype,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        actor_belief = player_beliefs.gather(
            1,
            actor[:, None, None].expand(-1, 1, NUM_HANDS),
        ).squeeze(1)

        card_mass = self._card_mass(player_beliefs)
        actor_card_mass = card_mass.gather(
            1,
            actor[:, None, None].expand(-1, 1, 52),
        ).squeeze(1)
        player_ids = torch.arange(self.num_players, device=player_beliefs.device)
        non_actor = player_ids[None, :] != actor[:, None]
        opp_card_mass = torch.where(
            non_actor[:, :, None],
            card_mass,
            torch.zeros_like(card_mass),
        ).sum(dim=1)
        opp_belief = torch.where(
            non_actor[:, :, None],
            player_beliefs,
            torch.zeros_like(player_beliefs),
        ).sum(dim=1)
        opp_unblocked = self._unblocked_mass_from_card_mass(opp_belief, opp_card_mass)
        return self._dynamic_hand_features_from_stats(
            actor_belief,
            actor_card_mass,
            opp_card_mass,
            opp_unblocked,
            board,
            dtype,
            board_stats,
        )

    def _hand_value_logits(
        self,
        value_input: torch.Tensor,
    ) -> torch.Tensor:
        return self._hand_value_logits_from_head(value_input, self.hand_value_head)

    def _hand_value_logits_from_head(
        self, value_input: torch.Tensor, head: nn.Module
    ) -> torch.Tensor:
        if value_input.dim() == 3:
            player_state = (
                value_input[:, 1:]
                if value_input.shape[1] == self.num_players + 1
                else value_input
            )
            if player_state.shape[1] != self.num_players:
                raise ValueError(
                    "token value input must have one token per player, optionally "
                    "preceded by one game token"
                )
            batch_size = player_state.shape[0]
            token_values = head(player_state.reshape(-1, player_state.shape[-1])).view(
                batch_size,
                self.num_players,
                self.num_players,
                NUM_HANDS,
            )
            player_idx = torch.arange(self.num_players, device=value_input.device)
            return token_values[:, player_idx, player_idx, :]
        return head(value_input).view(-1, self.num_players, NUM_HANDS)

    def _hand_value_logits_and_state_from_head(
        self, value_input: torch.Tensor, head: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(head, nn.Sequential) or len(head) == 0:
            return self._hand_value_logits_from_head(value_input, head), None
        body = head[:-1]
        output = head[-1]
        if value_input.dim() == 3:
            player_state = (
                value_input[:, 1:]
                if value_input.shape[1] == self.num_players + 1
                else value_input
            )
            if player_state.shape[1] != self.num_players:
                raise ValueError(
                    "token value input must have one token per player, optionally "
                    "preceded by one game token"
                )
            batch_size = player_state.shape[0]
            flat_state = player_state.reshape(-1, player_state.shape[-1])
            hidden = body(flat_state) if len(body) > 0 else flat_state
            token_values = output(hidden).view(
                batch_size,
                self.num_players,
                self.num_players,
                NUM_HANDS,
            )
            player_idx = torch.arange(self.num_players, device=value_input.device)
            return (
                token_values[:, player_idx, player_idx, :],
                hidden.view(batch_size, self.num_players, -1),
            )
        hidden = body(value_input) if len(body) > 0 else value_input
        hand_values = output(hidden).view(-1, self.num_players, NUM_HANDS)
        return hand_values, hidden[:, None, :].expand(-1, self.num_players, -1)

    def _policy_logits(
        self,
        policy_input: torch.Tensor,
        player_beliefs: torch.Tensor,
        actor: torch.Tensor,
        board: torch.Tensor,
        hand_emb: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        policy_state = self.policy_tower(policy_input)
        action_emb = self.policy_action_head(policy_state).view(
            -1, self.num_actions, self.policy_rank
        )
        hand_gate = 1.0 + self.policy_hand_gate(policy_state).tanh()
        action_emb = action_emb * hand_gate[:, None, :]
        hand_vec = self.policy_hand_proj(hand_emb)
        if hand_vec.dim() == 2:
            logits = torch.einsum("hr,bar->bha", hand_vec, action_emb)
        else:
            logits = torch.einsum("bhr,bar->bha", hand_vec, action_emb)
        logits = logits / math.sqrt(self.policy_rank)
        hand_bias = self.policy_hand_bias(hand_emb)
        hand_bias_action = self.policy_hand_bias_action(policy_state).view(
            -1, self.num_actions, self.policy_hand_bias_rank
        )
        if hand_bias.dim() == 2:
            logits = logits + torch.einsum("hk,bak->bha", hand_bias, hand_bias_action)
        else:
            logits = logits + torch.einsum(
                "bhk,bak->bha", hand_bias, hand_bias_action
            )

        dynamic_coeff = self.policy_dynamic_coeff(policy_state).view(
            -1, self.num_actions, HAND_DYNAMIC_FEATURE_DIM
        )
        if torch.is_grad_enabled():
            dynamic_features = self._policy_dynamic_features(
                player_beliefs, actor, board, policy_state.dtype, board_stats
            )
            logits = logits + torch.einsum(
                "bhf,baf->bha", dynamic_features, dynamic_coeff
            )
        else:
            logits = logits + self._policy_dynamic_logits(
                player_beliefs, actor, board, dynamic_coeff, board_stats
            )
        logits = logits + self.policy_action_bias(policy_state)[:, None, :]
        return logits

    def _belief_board_interaction(
        self,
        player_beliefs: torch.Tensor,
        board_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor | None:
        if self.board_interaction_dim <= 0:
            return None
        board_rank_counts, board_suit_counts, _ = board_stats

        rank_pair_mass = player_beliefs @ self.hand_rank_pair_one_hot.to(
            dtype=player_beliefs.dtype
        )
        rank_pair_low = rank_pair_mass @ self.rank_pair_low_embedding.weight
        board_rank_low = self.board_rank_low(board_rank_counts)
        rank_features = (rank_pair_low * board_rank_low[:, None, :]).flatten(1)

        suit_pair_mass = player_beliefs @ self.hand_suit_pair_one_hot.to(
            dtype=player_beliefs.dtype
        )
        suit_pair_low = suit_pair_mass @ self.suit_pair_low_embedding.weight
        board_suit_low = self.board_suit_low(board_suit_counts)
        suit_features = (suit_pair_low * board_suit_low[:, None, :]).flatten(1)

        if self.board_interaction_skip_out:
            out = 0.1 * self.board_interaction_norm(rank_features + suit_features)
            if self.board_interaction_gated:
                out = out * self.board_interaction_gate.tanh()
            return out

        out = self.rank_board_interaction_out(
            rank_features
        ) + self.suit_board_interaction_out(suit_features)
        if self.board_interaction_gated:
            out = out * self.board_interaction_gate.tanh()
        return out

    def _cross_range_interaction(
        self, per_player_belief: torch.Tensor
    ) -> torch.Tensor | None:
        if self.cross_range_rank <= 0:
            return None
        p0 = per_player_belief[:, 0]
        p1 = per_player_belief[:, 1]
        cross = self.cross_left(p0) * self.cross_right(p1)
        return self.cross_proj(cross)

    def _range_stats(self, player_beliefs: torch.Tensor) -> torch.Tensor:
        b0 = player_beliefs[:, 0]
        b1 = player_beliefs[:, 1]
        b0_norm = b0 / b0.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        b1_norm = b1 / b1.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.stack(
            [
                -(b0_norm * b0_norm.clamp_min(1e-8).log()).sum(dim=-1),
                -(b1_norm * b1_norm.clamp_min(1e-8).log()).sum(dim=-1),
                (b0_norm * b1_norm).sum(dim=-1),
                b0_norm.square().sum(dim=-1).rsqrt(),
                b1_norm.square().sum(dim=-1).rsqrt(),
            ],
            dim=-1,
        )

    def _context_with_range_stats(
        self, context: torch.Tensor, player_beliefs: torch.Tensor
    ) -> torch.Tensor:
        base_context_dim = self.context_in_dim - 5
        if context.shape[-1] > base_context_dim:
            context = context[..., :base_context_dim]
        elif context.shape[-1] < base_context_dim:
            pad = context.new_zeros(
                *context.shape[:-1], base_context_dim - context.shape[-1]
            )
            context = torch.cat((context, pad), dim=-1)
        stats = self._range_stats(player_beliefs).to(dtype=context.dtype)
        return torch.cat((context, stats), dim=-1)

    def _range_context_delta(
        self, context: torch.Tensor, player_beliefs: torch.Tensor
    ) -> torch.Tensor | None:
        if not self.context_range_stats:
            return None
        full_context = self._context_with_range_stats(context, player_beliefs)
        zero_stats_context = torch.cat(
            (
                full_context[..., :-5],
                full_context.new_zeros(*full_context.shape[:-1], 5),
            ),
            dim=-1,
        )
        return self.context_encoder(full_context) - self.context_encoder(
            zero_stats_context
        )

    def _strength_bet_context(self, context: torch.Tensor) -> torch.Tensor:
        bet_scalars = torch.stack(
            [
                context[:, ValueScalarContext.POT.value],
                context[:, ValueScalarContext.MIN_RAISE.value],
                context[:, ValueScalarContext.LOG_POT_BB.value],
                context[:, ValueScalarContext.LOG_STACK_DEPTH_BB.value],
                context[:, ValueScalarContext.MAX_COMMITTED.value],
            ],
            dim=-1,
        )
        return self.strength_bet_ctx_proj(bet_scalars)

    def _river_rank_percentile(self, board: torch.Tensor) -> torch.Tensor:
        if board.device.type == "cuda" and triton_is_available():
            try:
                hand_ranks, sorted_indices = rank_hands_triton(board.int())
            except Exception:
                hand_ranks, sorted_indices = rank_hands_torch(board.int())
        else:
            hand_ranks, sorted_indices = rank_hands_torch(board.int())
        del hand_ranks
        positions = torch.arange(NUM_HANDS, device=board.device, dtype=torch.float32)
        positions = positions.view(1, NUM_HANDS).expand(board.shape[0], -1)
        rank_percentile = torch.empty_like(positions)
        rank_percentile.scatter_(1, sorted_indices.long(), positions)
        return rank_percentile / float(NUM_HANDS - 1)

    def _river_rank_groups(self, board: torch.Tensor) -> torch.Tensor:
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

    def _player_spr_context(self, context: torch.Tensor) -> torch.Tensor:
        base = ValueScalarContext.NUM_SCALAR_CONTEXT.value
        stride = PlayerContext.NUM_PLAYER_CONTEXT.value
        spr_idx = base + torch.arange(
            self.num_players,
            device=context.device,
            dtype=torch.long,
        ) * stride + PlayerContext.SPR.value
        return context.index_select(1, spr_idx)

    def _river_range_equity_context_delta(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.value_river_range_equity_trunk_context:
            return None
        delta = player_beliefs.new_zeros(
            player_beliefs.shape[0],
            self.hidden_dim,
            dtype=dtype,
        )
        river_mask = (features.street == 3) & (features.board >= 0).all(dim=1)
        if not river_mask.any():
            return delta
        rows = torch.where(river_mask)[0]
        rank_bins = self.value_river_range_equity_rank_bins
        rank_groups = self._river_rank_groups(features.board[rows]).clamp(
            min=0,
            max=rank_bins - 1,
        )
        beliefs = player_beliefs[rows].float()
        rank_idx = rank_groups[:, None, :].expand(-1, self.num_players, -1)
        rank_mass = beliefs.new_zeros(
            beliefs.shape[0],
            self.num_players,
            rank_bins,
        )
        rank_mass.scatter_add_(2, rank_idx, beliefs)
        context_features = rank_mass.flatten(1).to(dtype=dtype)
        delta[rows] = self.value_river_equity_context_proj(context_features).to(
            dtype=dtype
        )
        return delta

    def _river_showdown_masses(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ):
        """Blocker-corrected opponent showdown decomposition for each hero hand
        on river rows. Returns row-indexed tensors ``(rows, beliefs, lower_mass,
        tie_mass, total_mass, blocked_top_decile)`` where ``lower_mass`` is the
        opponent belief mass the hero beats (win), ``tie_mass`` ties, and
        ``total_mass - lower_mass - tie_mass`` losses; all corrected for card
        removal when ``value_river_range_equity_blockers`` is set. Returns
        ``None`` when the batch has no river rows. Shared by the analytic
        equity baseline and the showdown-mass range encoder so the blocker math
        lives in one place."""
        river_mask = (features.street == 3) & (features.board >= 0).all(dim=1)
        if not river_mask.any():
            return None
        rows = torch.where(river_mask)[0]
        rank_bins = self.value_river_range_equity_rank_bins
        rank_groups = self._river_rank_groups(features.board[rows]).clamp(
            min=0,
            max=rank_bins - 1,
        )
        beliefs = player_beliefs[rows].float()
        opponent_beliefs = beliefs.sum(dim=1, keepdim=True) - beliefs
        rank_idx = rank_groups[:, None, :].expand(-1, self.num_players, -1)
        rank_mass = beliefs.new_zeros(
            beliefs.shape[0],
            self.num_players,
            rank_bins,
        )
        rank_mass.scatter_add_(2, rank_idx, opponent_beliefs)
        cumulative = rank_mass.cumsum(dim=2)
        tie_mass = rank_mass.gather(2, rank_idx)
        lower_mass = cumulative.gather(2, rank_idx) - tie_mass
        total_mass = rank_mass.sum(dim=2, keepdim=True).clamp_min(1e-8)
        blocked_top_decile = beliefs.new_zeros(
            beliefs.shape[0],
            self.num_players,
            NUM_HANDS,
        )
        if self.value_river_range_equity_blockers:
            card_a = self.hand_card_a.to(device=beliefs.device)
            card_b = self.hand_card_b.to(device=beliefs.device)
            card_a_idx = card_a.view(1, 1, NUM_HANDS).expand_as(rank_idx)
            card_b_idx = card_b.view(1, 1, NUM_HANDS).expand_as(rank_idx)
            card_rank_bins = 52 * rank_bins
            card_rank_mass = beliefs.new_zeros(
                beliefs.shape[0],
                self.num_players,
                card_rank_bins,
            )
            flat_idx_a = card_a_idx * rank_bins + rank_idx
            flat_idx_b = card_b_idx * rank_bins + rank_idx
            card_rank_mass.scatter_add_(2, flat_idx_a, opponent_beliefs)
            card_rank_mass.scatter_add_(2, flat_idx_b, opponent_beliefs)
            if (
                triton is not None
                and card_rank_mass.is_cuda
                and rank_bins <= 256
            ):
                card_rank_cumulative, card_mass = _river_card_rank_prefix_triton(
                    card_rank_mass,
                    rank_bins=rank_bins,
                    num_players=self.num_players,
                )
                same_combo_mass = opponent_beliefs
                card_tie_a = card_rank_mass.gather(2, flat_idx_a)
                card_tie_b = card_rank_mass.gather(2, flat_idx_b)
                card_lower_a = card_rank_cumulative.gather(2, flat_idx_a) - card_tie_a
                card_lower_b = card_rank_cumulative.gather(2, flat_idx_b) - card_tie_b
                blocked_tie = card_tie_a + card_tie_b - same_combo_mass
                blocked_lower = card_lower_a + card_lower_b
                blocked_total = (
                    card_mass.gather(2, card_a_idx)
                    + card_mass.gather(2, card_b_idx)
                    - same_combo_mass
                )
                top_start = rank_bins - max(1, math.ceil(rank_bins / 10))
                card_rank_view = card_rank_mass.view(
                    beliefs.shape[0],
                    self.num_players,
                    52,
                    rank_bins,
                )
                card_top_mass = card_rank_view[..., top_start:].sum(dim=3)
                same_combo_top = same_combo_mass * (rank_idx >= top_start).to(
                    dtype=same_combo_mass.dtype
                )
                blocked_top_decile = (
                    card_top_mass.gather(2, card_a_idx)
                    + card_top_mass.gather(2, card_b_idx)
                    - same_combo_top
                ).clamp_min(0.0)
            else:
                card_rank_view = card_rank_mass.view(
                    beliefs.shape[0],
                    self.num_players,
                    52,
                    rank_bins,
                )
                card_mass = card_rank_view.sum(dim=3)
                card_rank_cumulative = card_rank_view.cumsum(dim=3).reshape(
                    beliefs.shape[0],
                    self.num_players,
                    card_rank_bins,
                )
                card_tie_a = card_rank_mass.gather(2, flat_idx_a)
                card_tie_b = card_rank_mass.gather(2, flat_idx_b)
                card_lower_a = card_rank_cumulative.gather(2, flat_idx_a) - card_tie_a
                card_lower_b = card_rank_cumulative.gather(2, flat_idx_b) - card_tie_b
                same_combo_mass = opponent_beliefs
                blocked_tie = card_tie_a + card_tie_b - same_combo_mass
                blocked_lower = card_lower_a + card_lower_b
                blocked_total = (
                    card_mass.gather(2, card_a_idx)
                    + card_mass.gather(2, card_b_idx)
                    - same_combo_mass
                )
                top_start = rank_bins - max(1, math.ceil(rank_bins / 10))
                card_top_mass = card_rank_view[..., top_start:].sum(dim=3)
                same_combo_top = same_combo_mass * (rank_idx >= top_start).to(
                    dtype=same_combo_mass.dtype
                )
                blocked_top_decile = (
                    card_top_mass.gather(2, card_a_idx)
                    + card_top_mass.gather(2, card_b_idx)
                    - same_combo_top
                ).clamp_min(0.0)
            tie_mass = (tie_mass - blocked_tie).clamp_min(0.0)
            lower_mass = (lower_mass - blocked_lower).clamp_min(0.0)
            total_mass = (total_mass - blocked_total).clamp_min(1e-8)
        return (
            rows,
            beliefs,
            lower_mass,
            tie_mass,
            total_mass,
            blocked_top_decile,
        )

    def _river_showdown_range_features(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        hand_emb: torch.Tensor,
    ) -> torch.Tensor | None:
        """Wide range-encoder input: for each player/hand, pool [belief, win,
        tie, loss] mass against the hand embedding and project into the trunk.
        Gives a single ordinary value head explicit, per-hand, blocker-corrected
        showdown information as input (rather than an additive equity baseline).
        River rows only; zero on other rows."""
        if not self.value_river_showdown_range_encoder:
            return None
        masses = self._river_showdown_masses(player_beliefs, features)
        if masses is None:
            return None
        rows, beliefs_rows, lower_mass, tie_mass, total_mass, _ = masses
        loss_mass = (total_mass - lower_mass - tie_mass).clamp_min(0.0)
        n = player_beliefs.shape[0]

        def _full(rows_value: torch.Tensor) -> torch.Tensor:
            full = player_beliefs.new_zeros(
                n, self.num_players, NUM_HANDS, dtype=torch.float32
            )
            full[rows] = rows_value.to(dtype=torch.float32)
            return full

        # Broadcast loss to [n_rows, P, H] before scattering (total_mass may be
        # [n_rows, P, 1] when blockers are disabled).
        loss_mass = loss_mass.expand(-1, self.num_players, NUM_HANDS)
        channels = (
            _full(beliefs_rows),
            _full(lower_mass),
            _full(tie_mass),
            _full(loss_mass),
        )
        he = hand_emb
        he_dtype = he.dtype

        def _pool(channel: torch.Tensor) -> torch.Tensor:
            c = channel.to(dtype=he_dtype)
            if he.dim() == 2:
                return c @ he  # [N, P, hidden]
            return torch.einsum("bph,bhd->bpd", c, he)

        pooled = torch.stack([_pool(c) for c in channels], dim=2)  # [N,P,4,hd]
        flat = pooled.reshape(n, -1)  # [N, P * 4 * hidden]
        return self.showdown_range_proj(flat)

    def _river_showdown_dense_features(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
    ) -> torch.Tensor | None:
        """Dense showdown trunk feature. For each player, the full
        blocker-corrected showdown vector over all hands (``[belief, win, tie,
        loss]`` -> ``4 * NUM_HANDS``) is projected straight to ``dim`` with a
        dense linear (contracting the hand axis, so no low-rank range
        bottleneck), the two players are fused ``[dim, P] -> hidden_dim``, and the
        result is added into the trunk. River rows only; zero on other rows so it
        starts (and stays off-river) as a no-op."""
        if not self.value_river_showdown_perhand_head:
            return None
        full = player_beliefs.new_zeros(
            player_beliefs.shape[0],
            self.hidden_dim,
            dtype=self.showdown_perhand_in.weight.dtype,
        )
        masses = self._river_showdown_masses(player_beliefs, features)
        if masses is None:
            return full
        rows, beliefs, lower_mass, tie_mass, total_mass, _ = masses
        players = self.num_players
        n_rows = rows.shape[0]
        wdtype = self.showdown_perhand_in.weight.dtype

        # Broadcast to [n_rows, P, H] (total_mass may be [n_rows, P, 1] when
        # blockers are disabled) and assemble the per-player showdown channels.
        lower = lower_mass.expand(-1, players, NUM_HANDS)
        tie = tie_mass.expand(-1, players, NUM_HANDS)
        total = total_mass.expand(-1, players, NUM_HANDS)
        loss = (total - lower - tie).clamp_min(0.0)
        channels = torch.stack((beliefs, lower, tie, loss), dim=2)  # [Nr, P, 4, H]
        x = channels.reshape(n_rows, players, -1)  # [Nr, P, 4*H]

        # [Nr, P, 4H] -> [Nr, P, dim] (dense over hands), then fuse players.
        per_player = self.showdown_perhand_act(
            self.showdown_perhand_in(x.to(dtype=wdtype))
        )  # [Nr, P, dim]
        fused = self.showdown_perhand_fuse(
            per_player.reshape(n_rows, -1)
        )  # [Nr, hidden_dim]
        full[rows] = fused.to(dtype=wdtype)
        return full

    def _river_range_equity_features(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        baseline = player_beliefs.new_zeros(
            player_beliefs.shape[0],
            self.num_players,
            NUM_HANDS,
            dtype=dtype,
        )
        feature_values = player_beliefs.new_zeros(
            player_beliefs.shape[0],
            self.num_players,
            NUM_HANDS,
            6,
            dtype=dtype,
        )
        masses = self._river_showdown_masses(player_beliefs, features)
        if masses is None:
            return baseline, feature_values
        rows, beliefs, lower_mass, tie_mass, total_mass, blocked_top_decile = (
            masses
        )
        equity_score = (2.0 * lower_mass + tie_mass - total_mass) / total_mass
        pot_scale = features.context[rows, ValueScalarContext.POT.value].float()
        if self.value_river_range_equity_pot_power != 1.0:
            pot_scale = pot_scale.clamp_min(0.0).pow(
                self.value_river_range_equity_pot_power
            )
        sdv = equity_score * pot_scale[:, None, None]
        if self.value_river_range_equity_pos_scale >= 0.0:
            value = (
                sdv.clamp_min(0.0) * self.value_river_range_equity_pos_scale
                + sdv.clamp_max(0.0) * self.value_river_range_equity_neg_scale
                + self.value_river_range_equity_intercept
            )
        else:
            value = sdv * self.value_river_range_equity_baseline_scale
        baseline[rows] = value.to(dtype=dtype)
        spr = self._player_spr_context(features.context[rows]).float()
        feature_values[rows] = torch.stack(
            (
                sdv,
                beliefs,
                total_mass.expand_as(equity_score),
                blocked_top_decile,
                pot_scale[:, None, None].expand_as(equity_score),
                spr[:, :, None].expand_as(equity_score),
            ),
            dim=-1,
        ).to(dtype=dtype)
        return baseline, feature_values

    def _turn_runout_boards(
        self, board: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return turn_equity_runout_boards(board)

    def _turn_range_equity_config(self) -> TurnRangeEquityConfig:
        return TurnRangeEquityConfig(
            rank_bins=self.value_turn_range_equity_rank_bins,
            chunk_size=self.value_turn_range_equity_chunk_size,
            blockers=self.value_turn_range_equity_blockers,
            baseline_scale=self.value_turn_range_equity_baseline_scale,
            pot_power=self.value_turn_range_equity_pot_power,
            pos_scale=self.value_turn_range_equity_pos_scale,
            neg_scale=self.value_turn_range_equity_neg_scale,
            intercept=self.value_turn_range_equity_intercept,
            runout_std=self.value_turn_range_equity_runout_std_feature,
            decomposition=self.value_turn_range_equity_decomposition_features,
        )

    def _turn_range_equity_features(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
        board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return turn_range_equity_features(
            player_beliefs,
            features,
            config=self._turn_range_equity_config(),
            dtype=dtype,
            board_cache=board_cache,
            rank_groups_fn=self._river_rank_groups,
        )

    def _shared_river_range_equity(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Compute the analytic river-equity features once when more than one
        consumer needs them, so the expensive blocker/equity kernel is not run
        twice per forward. Returns a float32 ``(baseline, feature_values)`` tuple
        (consumers cast to their own dtype) or ``None`` when sharing does not
        apply."""
        if (
            self.value_river_range_equity_baseline
            and self.value_river_canonical_head
            and self.value_river_canonical_baseline_input
        ):
            return self._river_range_equity_features(
                player_beliefs, features, torch.float32
            )
        return None

    def _river_range_equity_baseline(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
        equity: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if equity is not None:
            baseline, _ = equity
            return baseline.to(dtype=dtype)
        baseline, _ = self._river_range_equity_features(
            player_beliefs,
            features,
            dtype,
        )
        return baseline

    def _turn_range_equity_value(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
        board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> torch.Tensor:
        if not self.value_turn_range_equity_feature_head:
            return turn_range_equity_baseline(
                player_beliefs,
                features,
                config=self._turn_range_equity_config(),
                dtype=dtype,
                board_cache=board_cache,
                rank_groups_fn=self._river_rank_groups,
            )
        baseline, feature_values = self._turn_range_equity_features(
            player_beliefs,
            features,
            dtype,
            board_cache=board_cache,
        )
        del baseline
        selected_features = [feature_values[..., :6]]
        if self.value_turn_range_equity_decomposition_features:
            selected_features.append(feature_values[..., 6:8])
        if self.value_turn_range_equity_runout_std_feature:
            selected_features.append(feature_values[..., 8:9])
        if self.value_turn_range_equity_blocker_interactions:
            selected_features.append(feature_values[..., 9:12])
        feature_values = torch.cat(selected_features, dim=-1)
        hidden = self.value_turn_equity_feature_head[1](
            self.value_turn_equity_feature_head[0](feature_values)
        )
        if self.value_turn_range_equity_board_film:
            film = self.value_turn_equity_board_film_proj(
                self._board_context(features.board)
            )[:, None, None, :]
            gamma, beta = film.chunk(2, dim=-1)
            hidden = hidden * (1.0 + gamma) + beta
        elif self.value_turn_range_equity_hand_board_film:
            board_film = self.value_turn_equity_hand_board_film_proj(
                self._board_context(features.board)
            )[:, None, :]
            hand_film = self.value_turn_equity_hand_film_proj(
                self._hand_embedding()
            )[None, :, :]
            gamma, beta = (board_film + hand_film).chunk(2, dim=-1)
            hidden = hidden * (1.0 + gamma[:, None, :, :]) + beta[:, None, :, :]
        return self.value_turn_equity_feature_head[2](hidden).squeeze(-1)

    def _river_range_equity_value(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
        equity: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if equity is not None:
            baseline, feature_values = equity
            baseline = baseline.to(dtype=dtype)
            feature_values = feature_values.to(dtype=dtype)
        else:
            baseline, feature_values = self._river_range_equity_features(
                player_beliefs,
                features,
                dtype,
            )
        if not self.value_river_range_equity_feature_head:
            return baseline
        return self.value_river_equity_feature_head(feature_values).squeeze(-1)

    def _river_range_equity_film_residual(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        value_state: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.value_river_range_equity_film_rank <= 0 or value_state is None:
            return None
        if value_state.dim() == 2:
            value_state = value_state[:, None, :].expand(-1, self.num_players, -1)
        if value_state.shape[:2] != (player_beliefs.shape[0], self.num_players):
            raise ValueError(
                "river equity FiLM value_state must have shape [B, num_players, D]"
            )
        _, feature_values = self._river_range_equity_features(
            player_beliefs,
            features,
            dtype,
        )
        latent = self.value_river_equity_film_state(value_state)
        film = self.value_river_equity_film(feature_values)
        gamma, beta = film.chunk(2, dim=-1)
        adapted = gamma * latent[:, :, None, :] + beta
        residual = self.value_river_equity_film_out(adapted).squeeze(-1)
        river_mask = ((features.street == 3) & (features.board >= 0).all(dim=1)).to(
            dtype=residual.dtype
        )
        return residual * river_mask[:, None, None]

    def _apply_river_range_equity_value(
        self,
        hand_values: torch.Tensor,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        equity: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self.value_river_range_equity_baseline:
            return hand_values
        return hand_values + self._river_range_equity_value(
            player_beliefs,
            features,
            hand_values.dtype,
            equity=equity,
        )

    def _apply_turn_range_equity_value(
        self,
        hand_values: torch.Tensor,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> torch.Tensor:
        if not self.value_turn_range_equity_baseline:
            return hand_values
        if os.environ.get("P2_TURN_EQUITY_PAIR_DIRECT_APPLY", "0") not in {"", "0"}:
            pair_applied = apply_turn_pair_operator_baseline_value(
                hand_values,
                player_beliefs,
                features,
                config=self._turn_range_equity_config(),
                board_cache=board_cache,
            )
            if pair_applied is not None:
                return pair_applied
        return hand_values + self._turn_range_equity_value(
            player_beliefs,
            features,
            hand_values.dtype,
            board_cache=board_cache,
        )

    def _river_canonical_value_residual(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
        equity: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if not self.value_river_canonical_head:
            return None
        residual = player_beliefs.new_zeros(
            player_beliefs.shape[0],
            self.num_players,
            NUM_HANDS,
            dtype=dtype,
        )
        river_mask = (features.street == 3) & (features.board >= 0).all(dim=1)
        if not river_mask.any():
            return residual
        rows = torch.where(river_mask)[0]
        n_rows = rows.shape[0]
        players = self.num_players
        bins = self.value_river_canonical_bins
        eps = 1e-8

        beliefs = player_beliefs[rows].float()  # [N, P, H]
        opponent_beliefs = beliefs.sum(dim=1, keepdim=True) - beliefs

        # --- Steps 1-2: canonical strength coordinate u and quantile bins k ---
        rank_groups = self._river_rank_groups(features.board[rows])  # [N, H]
        combined = beliefs.sum(dim=1)  # reference mass b0 + b1 per hand, [N, H]
        group_mass = beliefs.new_zeros(n_rows, NUM_HANDS)
        group_mass.scatter_add_(1, rank_groups, combined)
        cumulative_group = group_mass.cumsum(dim=1)
        total_ref = group_mass.sum(dim=1, keepdim=True).clamp_min(eps)
        u_group = (cumulative_group - 0.5 * group_mass) / total_ref  # [N, H]
        u = u_group.gather(1, rank_groups).clamp(0.0, 1.0)  # per-hand coordinate
        k = (u * bins).floor().clamp(max=bins - 1).long()  # [N, H]
        k_idx = k[:, None, :].expand(-1, players, -1)  # [N, P, H]

        # --- Exact per-hand equity score vs. opponent at rank-group resolution ---
        rank_idx = rank_groups[:, None, :].expand(-1, players, -1)
        opp_group_mass = beliefs.new_zeros(n_rows, players, NUM_HANDS)
        opp_group_mass.scatter_add_(2, rank_idx, opponent_beliefs)
        opp_cumulative = opp_group_mass.cumsum(dim=2)
        tie = opp_group_mass.gather(2, rank_idx)
        lower = opp_cumulative.gather(2, rank_idx) - tie
        opp_total = opp_group_mass.sum(dim=2, keepdim=True).clamp_min(eps)
        equity_score = (2.0 * lower + tie - opp_total) / opp_total  # [N, P, H]

        # --- Step 3: per-bin features per player ---
        bin_mass = beliefs.new_zeros(n_rows, players, bins)
        bin_mass.scatter_add_(2, k_idx, beliefs)
        bin_mass_safe = bin_mass.clamp_min(eps)

        bin_u_sum = beliefs.new_zeros(n_rows, players, bins)
        bin_u_sum.scatter_add_(2, k_idx, beliefs * u[:, None, :])
        bin_u_mean = bin_u_sum / bin_mass_safe

        bin_equity_sum = beliefs.new_zeros(n_rows, players, bins)
        bin_equity_sum.scatter_add_(2, k_idx, beliefs * equity_score)
        bin_equity = bin_equity_sum / bin_mass_safe

        if self.value_river_canonical_blocker_rows:
            card_a = self.hand_card_a.to(device=beliefs.device)
            card_b = self.hand_card_b.to(device=beliefs.device)
            card_a_idx = card_a.view(1, 1, NUM_HANDS).expand(n_rows, players, -1)
            card_b_idx = card_b.view(1, 1, NUM_HANDS).expand(n_rows, players, -1)
            flat_a = card_a_idx * bins + k_idx
            flat_b = card_b_idx * bins + k_idx
            card_bin_mass = beliefs.new_zeros(n_rows, players, 52 * bins)
            card_bin_mass.scatter_add_(2, flat_a, beliefs)
            card_bin_mass.scatter_add_(2, flat_b, beliefs)
            # [N, P, 52, K]
            card_bin_mass = card_bin_mass.view(n_rows, players, 52, bins)
            card_bin_mass_opp = (
                card_bin_mass.sum(dim=1, keepdim=True) - card_bin_mass
            )
            # B[p, k, j] = sum_c cbm[p, c, k] * cbm_opp[p, c, j]
            blocked = torch.einsum(
                "npck,npcj->npkj", card_bin_mass, card_bin_mass_opp
            )
            # Diagonal correction: subtract same-combo mass sum_{h in k} b_p * b_opp.
            diag = beliefs.new_zeros(n_rows, players, bins)
            diag.scatter_add_(2, k_idx, beliefs * opponent_beliefs)
            eye = torch.eye(bins, device=beliefs.device, dtype=blocked.dtype)
            blocked = blocked - diag[:, :, :, None] * eye[None, None]
            # Row-normalize by mass[p, k] * mass[opp, j].
            opp_bin_mass = bin_mass.sum(dim=1, keepdim=True) - bin_mass
            denom = (
                bin_mass_safe[:, :, :, None]
                * opp_bin_mass.clamp_min(eps)[:, :, None, :]
            )
            blocked_rows = blocked / denom  # [N, P, K, K]
        else:
            blocked_rows = beliefs.new_zeros(n_rows, players, bins, bins)

        # --- Optional: per-bin analytic-baseline value as a token feature ---
        # Anchors the head's correction so it can see the value it is adjusting
        # instead of reconstructing it blind. Uses the exact additive baseline
        # (blocker-corrected, pot-scaled, posneg-transformed).
        per_player_scalars = [
            bin_mass[:, :, :, None],
            bin_u_mean[:, :, :, None],
            bin_equity[:, :, :, None],
        ]
        if self.value_river_canonical_baseline_input:
            base_val = self._river_range_equity_baseline(
                player_beliefs, features, torch.float32, equity=equity
            )[rows].float()  # [N, P, H]
            bin_base_sum = beliefs.new_zeros(n_rows, players, bins)
            bin_base_sum.scatter_add_(2, k_idx, beliefs * base_val)
            bin_base_mean = bin_base_sum / bin_mass_safe
            per_player_scalars.append(bin_base_mean[:, :, :, None])

        # --- Assemble per-bin tokens: concat both players' features ---
        per_player = torch.cat(
            (*per_player_scalars, blocked_rows),
            dim=-1,
        )  # [N, P, K, S + K], S = 3 (+1 if baseline input)
        # concat over players -> [N, K, 2*(S + K)]
        token_features = per_player.permute(0, 2, 1, 3).reshape(
            n_rows, bins, players * per_player.shape[-1]
        )

        pot = features.context[rows, ValueScalarContext.POT.value].float()
        spr = self._player_spr_context(features.context[rows]).float()
        globals_features = torch.cat((pot[:, None], spr), dim=1)  # [N, 1 + P]

        nodal = self.value_river_canonical(
            token_features.to(dtype=dtype), globals_features.to(dtype=dtype)
        ).float()  # [N, P, K]
        nodal = nodal * pot[:, None, None]

        # --- Step 5: interpolate nodal values (bin midpoints) at coordinate u_h ---
        t = (u * bins - 0.5).clamp(0.0, bins - 1)  # [N, H]
        lo = t.floor().long().clamp(0, bins - 1)
        hi = (lo + 1).clamp(max=bins - 1)
        frac = (t - lo.float())[:, None, :]  # [N, 1, H]
        lo_idx = lo[:, None, :].expand(-1, players, -1)
        hi_idx = hi[:, None, :].expand(-1, players, -1)
        v_lo = nodal.gather(2, lo_idx)
        v_hi = nodal.gather(2, hi_idx)
        per_hand = v_lo * (1.0 - frac) + v_hi * frac  # [N, P, H]

        residual[rows] = per_hand.to(dtype=dtype)
        return residual

    def _river_exact_aux(
        self,
        player_beliefs: torch.Tensor,
        features: MLPFeatures,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        river_mask = (features.street == 3) & (features.board >= 0).all(dim=1)
        rank_percentile = features.beliefs.new_zeros(
            features.beliefs.shape[0], NUM_HANDS, dtype=torch.float32
        )
        if river_mask.any():
            rows = torch.where(river_mask)[0]
            rank_percentile[rows] = self._river_rank_percentile(features.board[rows])
        rank_score = (2.0 * rank_percentile - 1.0).to(dtype=dtype)
        pot_scale = features.context[:, ValueScalarContext.POT.value].to(dtype=dtype)
        showdown_baseline = rank_score[:, None, :] * pot_scale[:, None, None]

        aux = {
            "river_rank_score": rank_score,
            "showdown_baseline": showdown_baseline,
        }
        if self.value_exact_river_features:
            opp_belief = player_beliefs.sum(dim=1, keepdim=True) - player_beliefs
            opp_card_mass = self._card_mass(opp_belief)
            opp_unblocked = self._unblocked_mass_from_card_mass(
                opp_belief, opp_card_mass
            )
            exact_input = torch.stack(
                (
                    rank_score[:, None].expand(-1, self.num_players, -1),
                    showdown_baseline.expand(-1, self.num_players, -1),
                    opp_unblocked.to(dtype=dtype),
                    player_beliefs.to(dtype=dtype),
                ),
                dim=-1,
            )
            aux["exact_feature_residual"] = self.value_exact_feature_head(
                exact_input
            ).squeeze(-1)
        return aux

    def _value_latent_bucket_residual(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        features: MLPFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board_ctx = self._board_context(features.board).to(dtype=hand_emb.dtype)
        state = x[:, 0] if x.dim() == 3 else x
        return self.value_latent_bucket_residual(
            hand_emb,
            board_ctx,
            state,
            player_beliefs,
        )

    def _value_coarse_bucket_residual(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        features: MLPFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board_ctx = self._board_context(features.board).to(dtype=hand_emb.dtype)
        if self.value_strength_bucket_board_only:
            bet_ctx = torch.zeros(
                features.context.shape[0],
                self.strength_bet_ctx_proj.out_features,
                device=features.context.device,
                dtype=hand_emb.dtype,
            )
        else:
            bet_ctx = self._strength_bet_context(features.context).to(
                dtype=hand_emb.dtype
            )
        bucket_weights = self.strength_bucket_enc._bucket_weights(
            hand_emb,
            board_ctx,
            bet_ctx,
        )
        batch_size, num_players, num_hands = player_beliefs.shape
        if bucket_weights.shape[0] == 1 and batch_size != 1:
            bucket_weights = bucket_weights.expand(batch_size, -1, -1)
        if bucket_weights.shape[:2] != (batch_size, num_hands):
            raise ValueError("bucket weights must have shape [B, N, K]")

        opp_beliefs = player_beliefs.sum(dim=1, keepdim=True) - player_beliefs
        weighted = opp_beliefs[..., None].to(dtype=bucket_weights.dtype) * (
            bucket_weights[:, None, :, :]
        )
        opp_bucket = weighted.sum(dim=2)
        bucket_count = bucket_weights.shape[-1]
        opp_features = opp_bucket[:, :, None, :].expand(
            -1,
            -1,
            bucket_count,
            -1,
        )
        bucket_eye = torch.eye(
            bucket_count,
            device=bucket_weights.device,
            dtype=bucket_weights.dtype,
        ).view(1, 1, bucket_count, bucket_count)
        bucket_eye = bucket_eye.expand(batch_size, num_players, -1, -1)
        strat_input = torch.cat((opp_features, bucket_eye), dim=-1)
        if self.value_strength_bucket_relative:
            opp_share = opp_bucket / opp_bucket.sum(dim=-1, keepdim=True).clamp_min(
                1e-8
            )
            opp_share_features = opp_share[:, :, None, :].expand_as(opp_features)
            strat_input = torch.cat(
                (
                    strat_input,
                    opp_share_features,
                    opp_share_features - bucket_eye,
                ),
                dim=-1,
            )

        player_state = None
        if self.value_strength_bucket_film:
            if x.dim() == 3:
                player_state = x[:, 1:] if x.shape[1] == self.num_players + 1 else x
            else:
                player_state = x[:, None, :].expand(-1, self.num_players, -1)
        bucket_values = self.value_bucket_coarse_residual_head(
            strat_input,
            player_state,
        )
        hand_residual = torch.einsum(
            "bpk,bhk->bph",
            bucket_values,
            bucket_weights,
        )
        return hand_residual, bucket_weights

    def _value_stratification_residual(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        features: MLPFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.value_strength_bucket_count <= 0:
            return None
        if self.value_strength_bucket_coarse_residual:
            return self._value_coarse_bucket_residual(
                player_beliefs,
                x,
                hand_emb,
                features,
            )
        board_ctx = self._board_context(features.board).to(dtype=hand_emb.dtype)
        if self.value_strength_bucket_board_only:
            bet_ctx = torch.zeros(
                features.context.shape[0],
                self.strength_bet_ctx_proj.out_features,
                device=features.context.device,
                dtype=hand_emb.dtype,
            )
        else:
            bet_ctx = self._strength_bet_context(features.context).to(
                dtype=hand_emb.dtype
            )
        compat_bucket, hero_bucket = self.strength_bucket_enc(
            hand_emb,
            board_ctx,
            bet_ctx,
            player_beliefs,
            self.hand_card_a,
            self.hand_card_b,
            use_blockers=self.value_strength_bucket_blockers,
        )
        hero_bucket = hero_bucket[:, None, :, :].expand(
            -1,
            self.num_players,
            -1,
            -1,
        )
        strat_input = torch.cat((compat_bucket, hero_bucket), dim=-1)
        if self.value_strength_bucket_relative:
            compat_share = compat_bucket / compat_bucket.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8)
            strat_input = torch.cat(
                (
                    strat_input,
                    compat_share,
                    compat_share - hero_bucket,
                ),
                dim=-1,
            )
        player_state = None
        if self.value_strength_bucket_film:
            if x.dim() == 3:
                player_state = x[:, 1:] if x.shape[1] == self.num_players + 1 else x
            else:
                player_state = x[:, None, :].expand(-1, self.num_players, -1)
        return self.value_strat_head(strat_input, player_state), hero_bucket[:, 0]

    def static_feature_base(self, features: MLPFeatures) -> torch.Tensor:
        """Feature contribution that is fixed for a CFR leaf row."""
        return self.static_feature_base_from_prefix(
            self.static_feature_prefix(features.context, features.street),
            features.board,
        )

    def static_feature_prefix(
        self, context: torch.Tensor, street: torch.Tensor
    ) -> torch.Tensor:
        """Feature contribution from context and street before board expansion."""
        if context.shape[-1] > self.context_in_dim:
            context = context[..., : self.context_in_dim]
        elif context.shape[-1] < self.context_in_dim:
            pad = context.new_zeros(
                *context.shape[:-1], self.context_in_dim - context.shape[-1]
            )
            context = torch.cat((context, pad), dim=-1)
        return self.street_embedding(street) + self.context_encoder(context)

    def static_feature_base_from_prefix(
        self, prefix: torch.Tensor, board: torch.Tensor
    ) -> torch.Tensor:
        """Add board features to a precomputed context/street prefix."""
        return self._board_context(board) + prefix

    def _forward_base_from_static(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        player_beliefs = features.beliefs.view(-1, self.num_players, NUM_HANDS)
        board_context = (
            self._board_context(features.board)
            if (
                self.board_conditioned_hand_embedding_dim > 0
                or self.belief_low_rank_board_conditioned
                or self.belief_board_film
                or self.belief_board_bilinear_rank > 0
            )
            else None
        )
        hand_emb = self._hand_embedding(board_context)
        per_player_belief, per_player_variance = self._belief_moments(
            player_beliefs,
            hand_emb,
            board_context,
        )
        per_player_belief = self._apply_belief_board_film(
            per_player_belief,
            board_context,
        )
        belief_features = self.belief_proj(
            self._belief_projection_input(per_player_belief, per_player_variance)
        )

        flat_features = static_base_features + belief_features
        range_context_delta = self._range_context_delta(
            features.context, player_beliefs
        )
        if range_context_delta is not None:
            flat_features = flat_features + range_context_delta
        cross_features = self._cross_range_interaction(per_player_belief)
        if cross_features is not None:
            flat_features = flat_features + cross_features
        board_bilinear = self._belief_board_bilinear(per_player_belief, board_context)
        if board_bilinear is not None:
            flat_features = flat_features + board_bilinear
        board_mass_features = self._belief_board_mass_features(
            player_beliefs,
            features.board,
        )
        if board_mass_features is not None:
            flat_features = flat_features + board_mass_features
        showdown_range = self._river_showdown_range_features(
            player_beliefs, features, hand_emb
        )
        if showdown_range is not None:
            flat_features = flat_features + showdown_range.to(flat_features.dtype)
        showdown_dense = self._river_showdown_dense_features(
            player_beliefs, features
        )
        if showdown_dense is not None:
            flat_features = flat_features + showdown_dense.to(flat_features.dtype)
        river_equity_context = self._river_range_equity_context_delta(
            player_beliefs,
            features,
            flat_features.dtype,
        )
        trunk_game_features = static_base_features
        if river_equity_context is not None:
            flat_features = flat_features + river_equity_context
            trunk_game_features = trunk_game_features + river_equity_context
        board_stats = self._board_stats(features.board, player_beliefs.dtype)
        interaction_features = self._belief_board_interaction(
            player_beliefs, board_stats
        )
        if interaction_features is not None:
            flat_features = flat_features + interaction_features
        # assert flat_features.isfinite().all()

        x = (
            self._postflop_trunk_output(trunk_game_features, per_player_belief)
            if self.postflop_multi_token_trunk
            else self._postflop_trunk_output(flat_features, per_player_belief)
        )
        # assert x.isfinite().all()
        return (
            player_beliefs,
            flat_features,
            x,
            self.policy_hand_norm(hand_emb),
            board_stats,
        )

    def _forward_base(
        self,
        features: MLPFeatures,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        return self._forward_base_from_static(
            features, static_base_features=self.static_feature_base(features)
        )

    def forward_policy(
        self,
        features: MLPFeatures,
        latent=None,
    ) -> ModelOutput:
        player_beliefs, flat_features, x, hand_emb, board_stats = self._forward_base(
            features
        )
        policy_input = self._policy_input_from_base(flat_features, x)
        policy_logits = self._policy_logits(
            policy_input,
            player_beliefs,
            features.to_act,
            features.board,
            hand_emb,
            board_stats,
        )
        return ModelOutput(policy_logits=policy_logits)

    def forward_value(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> ModelOutput:
        """
        Value-only pass.

        apply_zero_sum controls where the zero-sum projection is applied, not
        whether it is required. If ``enforce_zero_sum`` is false this flag has no
        effect; if it is true and this flag is false, the caller must apply the
        projection after any value mixing.
        """
        player_beliefs, _, x, hand_emb, board_stats = self._forward_base(features)
        del board_stats
        return self._value_from_base(
            player_beliefs,
            x,
            hand_emb,
            features,
            apply_zero_sum=apply_zero_sum,
            turn_range_equity_board_cache=turn_range_equity_board_cache,
        )

    def forward_value_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        apply_zero_sum: bool = True,
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> ModelOutput:
        """Value-only pass for callers that precomputed static public features."""
        player_beliefs, _, x, hand_emb, board_stats = self._forward_base_from_static(
            features, static_base_features=static_base_features
        )
        del board_stats
        return self._value_from_base(
            player_beliefs,
            x,
            hand_emb,
            features,
            apply_zero_sum=apply_zero_sum,
            turn_range_equity_board_cache=turn_range_equity_board_cache,
        )

    def _value_from_base(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        features: MLPFeatures,
        apply_zero_sum: bool = True,
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> ModelOutput:
        equity = self._shared_river_range_equity(player_beliefs, features)
        hand_values_raw, aux = self._value_logits_and_aux_from_head(
            player_beliefs,
            x,
            hand_emb,
            self.hand_value_head,
            features,
            collect_aux=True,
            equity=equity,
        )
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            hand_values = hand_values_raw - hand_value_sums
        else:
            hand_values = hand_values_raw
        if self.value_showdown_baseline and aux is not None:
            baseline = aux.get("showdown_baseline")
            if baseline is not None:
                hand_values = hand_values + baseline.to(dtype=hand_values.dtype)
        hand_values = self._apply_river_range_equity_value(
            hand_values,
            player_beliefs,
            features,
            equity=equity,
        )
        hand_values = self._apply_turn_range_equity_value(
            hand_values,
            player_beliefs,
            features,
            board_cache=turn_range_equity_board_cache,
        )
        value = hand_values.mean(dim=-1)
        return ModelOutput(value=value, hand_values=hand_values, value_aux=aux)

    def _value_logits_from_head(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        head: nn.Module,
        features: MLPFeatures | None = None,
        equity: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hand_values, _ = self._value_logits_and_aux_from_head(
            player_beliefs,
            x,
            hand_emb,
            head,
            features,
            collect_aux=False,
            equity=equity,
        )
        return hand_values

    def _value_logits_and_aux_from_head(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        head: nn.Module,
        features: MLPFeatures | None = None,
        collect_aux: bool = False,
        equity: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        value_state: torch.Tensor | None = None
        if isinstance(head, CardTokenValueHead):
            value_state = x[:, 0] if x.dim() == 3 else x
            hand_values = head(
                value_state,
                player_beliefs,
                hand_emb,
                self.hand_card_a,
                self.hand_card_b,
            )
        elif isinstance(head, HandBasisValueHead):
            hand_values = head(x, hand_emb)
        else:
            if self.value_river_range_equity_film_rank > 0:
                hand_values, value_state = self._hand_value_logits_and_state_from_head(
                    x, head
                )
            else:
                hand_values = self._hand_value_logits_from_head(x, head)
        if self.value_per_hand_residual:
            correction = self._value_residual_correction(player_beliefs)
            hand_values = hand_values + correction.to(dtype=hand_values.dtype)
        aux: dict[str, torch.Tensor] | None = {} if collect_aux else None
        if features is not None:
            if self.value_latent_bucket_count > 0:
                latent_residual, latent_bucket_weights = (
                    self._value_latent_bucket_residual(
                        player_beliefs,
                        x,
                        hand_emb,
                        features,
                    )
                )
                hand_values = hand_values + latent_residual.to(
                    dtype=hand_values.dtype
                )
                if aux is not None:
                    aux["bucket_weights"] = latent_bucket_weights
            stratification_out = self._value_stratification_residual(
                player_beliefs,
                x,
                hand_emb,
                features,
            )
            if stratification_out is not None:
                stratification, bucket_weights = stratification_out
                hand_values = hand_values + stratification.to(dtype=hand_values.dtype)
                if aux is not None:
                    aux["bucket_weights"] = bucket_weights
            film_residual = self._river_range_equity_film_residual(
                player_beliefs,
                features,
                value_state,
                hand_values.dtype,
            )
            if film_residual is not None:
                hand_values = hand_values + film_residual.to(dtype=hand_values.dtype)
            canonical_residual = self._river_canonical_value_residual(
                player_beliefs,
                features,
                hand_values.dtype,
                equity=equity,
            )
            if canonical_residual is not None:
                if self.value_river_canonical_only:
                    # Canonical head is the *sole* river predictor: drop the
                    # trunk's per-hand (belief/board-derived) value on river
                    # rows so the river value depends only on the suit-invariant,
                    # card-agnostic rank-space tokens. Non-river rows keep the
                    # trunk head (canonical_residual is zero there anyway).
                    river_mask = (features.street == 3) & (
                        features.board >= 0
                    ).all(dim=1)
                    hand_values = torch.where(
                        river_mask[:, None, None],
                        canonical_residual.to(dtype=hand_values.dtype),
                        hand_values,
                    )
                else:
                    hand_values = hand_values + canonical_residual.to(
                        dtype=hand_values.dtype
                    )
            if (
                self.value_exact_river_features
                or (collect_aux and self.value_showdown_baseline)
            ):
                exact_aux = self._river_exact_aux(
                    player_beliefs, features, hand_values.dtype
                )
                exact_residual = exact_aux.get("exact_feature_residual")
                if exact_residual is not None:
                    hand_values = hand_values + exact_residual.to(
                        dtype=hand_values.dtype
                    )
                if aux is not None:
                    aux.update(exact_aux)
            if collect_aux and self.value_action_summary_head and aux is not None:
                state = x[:, 0] if x.dim() == 3 else x
                aux["action_summary"] = self.value_action_summary(state)
        return hand_values, (aux or None)

    def _value_aux_from_base(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        features: MLPFeatures,
        hand_values_dtype: torch.dtype,
    ) -> dict[str, torch.Tensor] | None:
        aux: dict[str, torch.Tensor] = {}
        if self.value_latent_bucket_count > 0:
            _, bucket_weights = self._value_latent_bucket_residual(
                player_beliefs,
                x,
                hand_emb,
                features,
            )
            aux["bucket_weights"] = bucket_weights
        if self.value_strength_bucket_count > 0:
            stratification_out = self._value_stratification_residual(
                player_beliefs,
                x,
                hand_emb,
                features,
            )
            if stratification_out is not None:
                _, bucket_weights = stratification_out
                aux["bucket_weights"] = bucket_weights
        if self.value_exact_river_features or self.value_showdown_baseline:
            aux.update(
                self._river_exact_aux(player_beliefs, features, hand_values_dtype)
            )
        if self.value_action_summary_head:
            state = x[:, 0] if x.dim() == 3 else x
            aux["action_summary"] = self.value_action_summary(state)
        return aux or None

    def _value_residual_correction(self, player_beliefs: torch.Tensor) -> torch.Tensor:
        card_mass = self._card_mass(player_beliefs)
        opp_belief = player_beliefs.sum(dim=1, keepdim=True) - player_beliefs
        opp_card_mass = card_mass.sum(dim=1, keepdim=True) - card_mass
        opp_unblocked = self._unblocked_mass_from_card_mass(
            opp_belief,
            opp_card_mass,
        )
        residual_input = torch.stack(
            [
                player_beliefs,
                player_beliefs.clamp_min(1e-8).log(),
                opp_unblocked.to(dtype=player_beliefs.dtype),
            ],
            dim=-1,
        )
        return self.value_residual(residual_input).squeeze(-1)

    def forward_both(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        player_beliefs, flat_features, x, hand_emb, board_stats = self._forward_base(
            features
        )
        policy_input = self._policy_input_from_base(flat_features, x)
        policy_logits = self._policy_logits(
            policy_input,
            player_beliefs,
            features.to_act,
            features.board,
            hand_emb,
            board_stats,
        )
        equity = self._shared_river_range_equity(player_beliefs, features)
        hand_values_raw = self._value_logits_from_head(
            player_beliefs, x, hand_emb, self.hand_value_head, features, equity=equity
        )
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            hand_values = hand_values_raw - hand_value_sums
        else:
            hand_values = hand_values_raw
        aux = self._value_aux_from_base(
            player_beliefs,
            x,
            hand_emb,
            features,
            hand_values.dtype,
        )
        if self.value_showdown_baseline and aux is not None:
            baseline = aux.get("showdown_baseline")
            if baseline is not None:
                hand_values = hand_values + baseline.to(dtype=hand_values.dtype)
        hand_values = self._apply_river_range_equity_value(
            hand_values,
            player_beliefs,
            features,
            equity=equity,
        )
        value = hand_values.mean(dim=-1)
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
            value_aux=aux,
        )

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        policy_logits = None
        value = None
        hand_values = None
        if include_policy:
            policy_logits = self.policy_model._call_forward_policy(features)
        if include_value:
            if value_head == "auto":
                value_output = self._forward_value_auto_split(
                    features,
                    latent=latent,
                    apply_zero_sum=apply_zero_sum,
                    static_base_features=static_base_features,
                )
            else:
                value_output = self.value_model._call_forward_value(
                    features,
                    latent=latent,
                    apply_zero_sum=apply_zero_sum,
                    static_base_features=static_base_features,
                    value_head=value_head,
                )
            value = value_output.value
            hand_values = value_output.hand_values
        if not include_policy and not include_value:
            raise ValueError(
                "At least one of include_policy/include_value must be true"
            )
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
        )
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        policy_logits = None
        value = None
        hand_values = None
        if include_policy:
            policy_logits = self.policy_model._call_forward_policy(features)
        if include_value:
            if value_head == "auto":
                value_output = self._forward_value_auto_split(
                    features,
                    latent=latent,
                    apply_zero_sum=apply_zero_sum,
                    static_base_features=static_base_features,
                )
            else:
                value_output = self.value_model._call_forward_value(
                    features,
                    latent=latent,
                    apply_zero_sum=apply_zero_sum,
                    static_base_features=static_base_features,
                    value_head=value_head,
                )
            value = value_output.value
            hand_values = value_output.hand_values
        if not include_policy and not include_value:
            raise ValueError(
                "At least one of include_policy/include_value must be true"
            )
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
        )
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
    ) -> ModelOutput:
        """Forward pass over flat feature vectors."""
        if include_policy and include_value:
            return self._call_forward_both(
                features,
                apply_zero_sum=apply_zero_sum,
            )
        if include_policy:
            return self._call_forward_policy(features)
        if include_value:
            if static_base_features is not None:
                return self._call_forward_value_static_base(
                    features,
                    static_base_features,
                    apply_zero_sum=apply_zero_sum,
                )
            return self._call_forward_value(
                features,
                apply_zero_sum=apply_zero_sum,
            )
        raise ValueError("At least one of include_policy/include_value must be true")

    def init_weights(self, rng: torch.Generator | None = None) -> None:
        """Initialize parameters following Xavier/RMSNorm defaults."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=rng)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=rng)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

        expansion_gain = math.sqrt(self.ffn_dim / self.hidden_dim)
        sequentials = [self.trunk]
        if hasattr(self, "policy_tower"):
            sequentials.append(self.policy_tower)
        if hasattr(self, "hand_value_head"):
            sequentials.append(self.hand_value_head)
        if hasattr(self, "pre_value_head"):
            sequentials.append(self.pre_value_head)
        if hasattr(self, "post_value_head"):
            sequentials.append(self.post_value_head)
        for sequential in sequentials:
            for block in sequential.modules():
                if not isinstance(block, ResidualBlock):
                    continue
                inner = block.inner
                if "swiglu" in dict(inner.named_children()):
                    swiglu = inner.get_submodule("swiglu")
                    nn.init.orthogonal_(
                        swiglu.gate.weight, expansion_gain, generator=rng
                    )
                    nn.init.orthogonal_(swiglu.up.weight, expansion_gain, generator=rng)
                else:
                    # 1.532 is the gain for GELU nonlinearity.
                    nn.init.orthogonal_(
                        inner.get_submodule("linear_in").weight,
                        1.532 * expansion_gain,
                        generator=rng,
                    )

        # Guess hand values are around stddev value_output_init_scale.
        for head_name in ("hand_value_head", "pre_value_head", "post_value_head"):
            head = getattr(self, head_name, None)
            if hasattr(head, "scale_output"):
                head.scale_output(self.value_output_init_scale)
            elif head is not None:
                head[-1].get_submodule("linear_out").weight.data.mul_(
                    self.value_output_init_scale
                )
        if self.board_interaction_dim > 0:
            if hasattr(self, "rank_board_interaction_out"):
                self.rank_board_interaction_out.weight.data.mul_(0.1)
            if hasattr(self, "suit_board_interaction_out"):
                self.suit_board_interaction_out.weight.data.mul_(0.1)
        if hasattr(self, "belief_phase_shift"):
            nn.init.zeros_(self.belief_phase_shift.weight)
        if hasattr(self, "value_latent_bucket_residual"):
            nn.init.normal_(
                self.value_latent_bucket_residual.bucket_query,
                std=0.02,
                generator=rng,
            )
            self.value_latent_bucket_residual.scale_output(0.1)
        if hasattr(self, "cross_proj"):
            self.cross_proj.weight.data.mul_(0.1)
        if hasattr(self, "card_offset_up"):
            self.card_offset_up.weight.data.mul_(0.1)
        if hasattr(self, "value_residual"):
            self.value_residual[-1].weight.data.mul_(0.1)
        if hasattr(self, "value_strat_head"):
            if hasattr(self.value_strat_head, "film"):
                nn.init.zeros_(self.value_strat_head.film.weight)
                nn.init.zeros_(self.value_strat_head.film.bias)
            self.value_strat_head.scale_output(0.1)
        if hasattr(self, "value_river_equity_feature_head"):
            self._init_river_equity_feature_head()
        if hasattr(self, "value_turn_equity_feature_head"):
            self._init_turn_equity_feature_head()
        if hasattr(self, "value_turn_equity_board_film_proj"):
            nn.init.zeros_(self.value_turn_equity_board_film_proj.weight)
            nn.init.zeros_(self.value_turn_equity_board_film_proj.bias)
        if hasattr(self, "value_turn_equity_hand_film_proj"):
            nn.init.zeros_(self.value_turn_equity_hand_film_proj.weight)
            nn.init.zeros_(self.value_turn_equity_hand_board_film_proj.weight)
            nn.init.zeros_(self.value_turn_equity_hand_board_film_proj.bias)
        if hasattr(self, "showdown_range_proj"):
            nn.init.zeros_(self.showdown_range_proj.weight)
        if hasattr(self, "showdown_perhand_fuse"):
            nn.init.zeros_(self.showdown_perhand_fuse.weight)
            nn.init.zeros_(self.showdown_perhand_fuse.bias)
        if hasattr(self, "value_river_equity_context_proj"):
            nn.init.zeros_(self.value_river_equity_context_proj.weight)
        if hasattr(self, "value_river_equity_film_out"):
            nn.init.zeros_(self.value_river_equity_film_out.weight)
        if hasattr(self, "value_river_canonical"):
            proj = self.value_river_canonical.output_proj
            nn.init.zeros_(proj.bias)
            scale = self.value_river_canonical_init_scale
            if scale > 0.0:
                # Break the residual-starts-at-baseline property on purpose:
                # scale is roughly the per-bin nodal residual std at init
                # (input to output_proj is RMSNorm'd to ~unit variance).
                fan_in = proj.weight.shape[1]
                nn.init.normal_(proj.weight, std=scale / math.sqrt(fan_in))
            else:
                nn.init.zeros_(proj.weight)

        # Start CFR warm-start policies close to uniform. The dominant low-rank
        # policy logit branch uses gain 0.1; auxiliary additive correction
        # branches start smaller because dynamic log-belief features have a
        # much larger raw magnitude.
        if hasattr(self, "policy_action_head"):
            self.policy_action_head.get_submodule("linear_out").weight.data.mul_(0.1)
            for projection in (
                self.policy_hand_bias_action,
                self.policy_dynamic_coeff,
                self.policy_action_bias,
            ):
                projection.get_submodule("linear_out").weight.data.mul_(0.01)

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterFeatureEncoder:
        return BetterFeatureEncoder(
            env=env,
            device=device,
            dtype=dtype,
        )

    def repeat(
        self,
        features: MLPFeatures,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self(
            features,
            include_policy=include_policy,
            include_value=include_value,
            apply_zero_sum=apply_zero_sum,
        )


class BetterPolicyFFN(BetterFFN):
    """BetterFFN policy path without value-head parameters."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        del self.hand_value_head

    def forward_policy(
        self,
        features: MLPFeatures,
        latent=None,
    ) -> torch.Tensor:
        player_beliefs, flat_features, x, hand_emb, board_stats = self._forward_base(
            features
        )
        policy_input = self._policy_input_from_base(flat_features, x)
        return self._policy_logits(
            policy_input,
            player_beliefs,
            features.to_act,
            features.board,
            hand_emb,
            board_stats,
        )

    def forward_value(
        self, features: MLPFeatures, latent=None, **kwargs
    ) -> ModelOutput:
        raise RuntimeError("BetterPolicyFFN does not provide value outputs")

    def forward_both(self, features: MLPFeatures, latent=None, **kwargs) -> ModelOutput:
        return ModelOutput(policy_logits=self._call_forward_policy(features))

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = False,
        **kwargs,
    ) -> ModelOutput:
        if include_value:
            raise RuntimeError("BetterPolicyFFN does not provide value outputs")
        if not include_policy:
            raise ValueError("BetterPolicyFFN requires include_policy=True")
        return ModelOutput(policy_logits=self._call_forward_policy(features))

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterPolicyFeatureEncoder:
        return BetterPolicyFeatureEncoder(
            env=env,
            device=device,
            dtype=dtype,
        )


class BetterStreetValueFFN(BetterFFN):
    """BetterFFN value path with deployed pre-chance and auxiliary post-chance heads."""

    def __init__(
        self,
        *args,
        value_heads: str | StreetValueHeads = StreetValueHeads.both,
        **kwargs,
    ) -> None:
        if not args:
            kwargs.setdefault("num_actions", 1)
        super().__init__(*args, **kwargs)
        value_heads = getattr(value_heads, "value", value_heads)
        if value_heads not in ("both", "pre", "post"):
            raise ValueError("value_heads must be one of: both, pre, post")
        self.value_heads = str(value_heads)

        del self.policy_tower
        del self.policy_hand_proj
        del self.policy_action_head
        del self.policy_hand_gate
        del self.policy_dynamic_coeff
        del self.policy_action_bias
        del self.policy_hand_bias
        del self.policy_hand_bias_action
        del self.policy_hand_norm

        base_value_head = self.hand_value_head
        del self.hand_value_head

        if self.value_heads in ("both", "pre"):
            self.pre_value_head = base_value_head
        if self.value_heads == "post":
            self.post_value_head = base_value_head
        elif self.value_heads == "both":
            self.post_value_head = self._make_value_head()

        # Directly conditions per-player belief summaries before belief_proj.
        self.belief_phase_shift = nn.Embedding(
            5 * 2, self.num_players * self.belief_feature_dim
        )

    def _make_value_head(self) -> nn.Module:
        return super()._make_value_head()

    def _phase_key(self, features: MLPFeatures) -> torch.Tensor:
        phase = features.context[:, ValueScalarContext.CHANCE_PHASE.value]
        phase = (
            phase.round()
            .long()
            .clamp(
                min=ChancePhase.POST_CHANCE.value,
                max=ChancePhase.PRE_CHANCE.value,
            )
        )
        return (features.street.long().clamp(min=0, max=4) * 2 + phase).clamp(
            min=0, max=9
        )

    def _forward_base_from_static(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        player_beliefs = features.beliefs.view(-1, self.num_players, NUM_HANDS)
        board_context = (
            self._board_context(features.board)
            if (
                self.board_conditioned_hand_embedding_dim > 0
                or self.belief_low_rank_board_conditioned
                or self.belief_board_film
                or self.belief_board_bilinear_rank > 0
            )
            else None
        )
        hand_emb = self._hand_embedding(board_context)
        per_player_belief, per_player_variance = self._belief_moments(
            player_beliefs,
            hand_emb,
            board_context,
        )
        per_player_belief = self._apply_belief_board_film(
            per_player_belief,
            board_context,
        )
        phase_shift = self.belief_phase_shift(self._phase_key(features)).view(
            -1, self.num_players, self.belief_feature_dim
        )
        per_player_belief = per_player_belief + phase_shift
        belief_features = self.belief_proj(
            self._belief_projection_input(per_player_belief, per_player_variance)
        )

        flat_features = static_base_features + belief_features
        range_context_delta = self._range_context_delta(
            features.context, player_beliefs
        )
        if range_context_delta is not None:
            flat_features = flat_features + range_context_delta
        cross_features = self._cross_range_interaction(per_player_belief)
        if cross_features is not None:
            flat_features = flat_features + cross_features
        board_bilinear = self._belief_board_bilinear(per_player_belief, board_context)
        if board_bilinear is not None:
            flat_features = flat_features + board_bilinear
        board_mass_features = self._belief_board_mass_features(
            player_beliefs,
            features.board,
        )
        if board_mass_features is not None:
            flat_features = flat_features + board_mass_features
        showdown_range = self._river_showdown_range_features(
            player_beliefs, features, hand_emb
        )
        if showdown_range is not None:
            flat_features = flat_features + showdown_range.to(flat_features.dtype)
        showdown_dense = self._river_showdown_dense_features(
            player_beliefs, features
        )
        if showdown_dense is not None:
            flat_features = flat_features + showdown_dense.to(flat_features.dtype)
        board_stats = self._board_stats(features.board, player_beliefs.dtype)
        interaction_features = self._belief_board_interaction(
            player_beliefs, board_stats
        )
        if interaction_features is not None:
            flat_features = flat_features + interaction_features
        trunk_game_features = static_base_features
        river_equity_context = self._river_range_equity_context_delta(
            player_beliefs,
            features,
            flat_features.dtype,
        )
        if river_equity_context is not None:
            flat_features = flat_features + river_equity_context
            trunk_game_features = trunk_game_features + river_equity_context

        x = (
            self._postflop_trunk_output(trunk_game_features, per_player_belief)
            if self.postflop_multi_token_trunk
            else self._postflop_trunk_output(flat_features, per_player_belief)
        )
        return player_beliefs, flat_features, x, hand_emb, board_stats

    def _hand_value_logits_from_head(
        self, value_input: torch.Tensor, head: nn.Module
    ) -> torch.Tensor:
        return super()._hand_value_logits_from_head(value_input, head)

    def _value_tensor_from_base(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        hand_emb: torch.Tensor,
        head: nn.Module,
        features: MLPFeatures,
        apply_zero_sum: bool = True,
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> torch.Tensor:
        equity = self._shared_river_range_equity(player_beliefs, features)
        hand_values_raw = self._value_logits_from_head(
            player_beliefs, x, hand_emb, head, features, equity=equity
        )
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            hand_values = hand_values_raw - hand_value_sums
        else:
            hand_values = hand_values_raw
        hand_values = self._apply_river_range_equity_value(
            hand_values,
            player_beliefs,
            features,
            equity=equity,
        )
        return self._apply_turn_range_equity_value(
            hand_values,
            player_beliefs,
            features,
            board_cache=turn_range_equity_board_cache,
        )

    def _forward_value_head(
        self,
        features: MLPFeatures,
        head: nn.Module,
        static_base_features: torch.Tensor | None = None,
        apply_zero_sum: bool = True,
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> torch.Tensor:
        if static_base_features is None:
            player_beliefs, _, x, hand_emb, _ = self._forward_base(features)
        else:
            player_beliefs, _, x, hand_emb, _ = self._forward_base_from_static(
                features, static_base_features=static_base_features
            )
        return self._value_tensor_from_base(
            player_beliefs,
            x,
            hand_emb,
            head,
            features,
            apply_zero_sum=apply_zero_sum,
            turn_range_equity_board_cache=turn_range_equity_board_cache,
        )

    def forward_pre(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor | None = None,
        apply_zero_sum: bool = True,
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> torch.Tensor:
        if not hasattr(self, "pre_value_head"):
            raise RuntimeError(
                "BetterStreetValueFFN was constructed without a pre head"
            )
        return self._forward_value_head(
            features,
            self.pre_value_head,
            static_base_features=static_base_features,
            apply_zero_sum=apply_zero_sum,
            turn_range_equity_board_cache=turn_range_equity_board_cache,
        )

    def forward_post(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor | None = None,
        apply_zero_sum: bool = True,
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> torch.Tensor:
        if not hasattr(self, "post_value_head"):
            raise RuntimeError(
                "BetterStreetValueFFN was constructed without a post head"
            )
        return self._forward_value_head(
            features,
            self.post_value_head,
            static_base_features=static_base_features,
            apply_zero_sum=apply_zero_sum,
            turn_range_equity_board_cache=turn_range_equity_board_cache,
        )

    def forward_policy(self, features: MLPFeatures, latent=None) -> torch.Tensor:
        raise RuntimeError("BetterStreetValueFFN does not provide policy outputs")

    def forward_value(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        value_head: str = "auto",
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> ModelOutput:
        if value_head == "pre":
            hand_values = self.forward_pre(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=turn_range_equity_board_cache,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)
        if value_head == "post":
            hand_values = self.forward_post(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=turn_range_equity_board_cache,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)
        if value_head != "auto":
            raise ValueError("value_head must be one of: auto, pre, post")
        if self.value_heads == "pre":
            hand_values = self.forward_pre(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=turn_range_equity_board_cache,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)
        if self.value_heads == "post":
            hand_values = self.forward_post(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=turn_range_equity_board_cache,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

        phase = features.context[:, ValueScalarContext.CHANCE_PHASE.value]
        pre_mask = (phase >= 0.5).view(-1, 1, 1)
        if torch.compiler.is_compiling() or _is_cuda_graph_capturing(features.context):
            pre = self.forward_pre(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=turn_range_equity_board_cache,
            )
            post = self.forward_post(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=turn_range_equity_board_cache,
            )
            hand_values = torch.where(pre_mask, pre, post)
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

        if static_base_features is None:
            player_beliefs, _, x, hand_emb, _ = self._forward_base(features)
        else:
            player_beliefs, _, x, hand_emb, _ = self._forward_base_from_static(
                features, static_base_features=static_base_features
            )
        hand_values = features.beliefs.new_empty(
            len(features), self.num_players, NUM_HANDS
        )
        pre_rows = torch.where(pre_mask[:, 0, 0])[0]
        post_rows = torch.where(~pre_mask[:, 0, 0])[0]
        if pre_rows.numel() > 0:
            hand_values[pre_rows] = self._value_tensor_from_base(
                player_beliefs[pre_rows],
                x[pre_rows],
                hand_emb[pre_rows] if hand_emb.dim() == 3 else hand_emb,
                self.pre_value_head,
                features[pre_rows],
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=(
                    None
                    if turn_range_equity_board_cache is None
                    else turn_range_equity_board_cache.slice(pre_rows)
                ),
            ).to(dtype=hand_values.dtype)
        if post_rows.numel() > 0:
            hand_values[post_rows] = self._value_tensor_from_base(
                player_beliefs[post_rows],
                x[post_rows],
                hand_emb[post_rows] if hand_emb.dim() == 3 else hand_emb,
                self.post_value_head,
                features[post_rows],
                apply_zero_sum=apply_zero_sum,
                turn_range_equity_board_cache=(
                    None
                    if turn_range_equity_board_cache is None
                    else turn_range_equity_board_cache.slice(post_rows)
                ),
            ).to(dtype=hand_values.dtype)
        return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

    def forward_value_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        apply_zero_sum: bool = True,
        value_head: str = "auto",
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> ModelOutput:
        return self.forward_value(
            features,
            latent=latent,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
            turn_range_equity_board_cache=turn_range_equity_board_cache,
        )

    def forward_both(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self.forward_value(
            features, latent=latent, apply_zero_sum=apply_zero_sum
        )

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = False,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        if include_policy:
            raise RuntimeError("BetterStreetValueFFN does not provide policy outputs")
        if not include_value:
            raise ValueError("BetterStreetValueFFN requires include_value=True")
        return self._call_forward_value(
            features,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
        )

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterStreetValueFeatureEncoder:
        return BetterStreetValueFeatureEncoder(
            env=env,
            device=device,
            dtype=dtype,
        )


def _preflop_class_ranks() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.arange(PREFLOP_HANDS, dtype=torch.long)
    row = ids // 13
    col = ids % 13
    pair = row == col
    suited = row > col
    hi = torch.where(suited | pair, row, col)
    lo = torch.where(suited | pair, col, row)
    return hi, lo, suited


def _preflop_bucket_projection() -> torch.Tensor:
    hi, lo, suited = _preflop_class_ranks()
    pair = hi == lo
    projection = torch.zeros(PREFLOP_HANDS, 16)
    projection.scatter_add_(
        1,
        hi[:, None],
        torch.ones(PREFLOP_HANDS, 1, dtype=projection.dtype),
    )
    projection.scatter_add_(
        1,
        lo[:, None],
        torch.ones(PREFLOP_HANDS, 1, dtype=projection.dtype),
    )
    projection[:, 13] = pair.to(projection.dtype)
    projection[:, 14] = suited.to(projection.dtype)
    projection[:, 15] = (hi == 12).to(projection.dtype)
    return projection.contiguous()


class _PreflopTokenEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        ffn_dim: int,
        nonlinearity: NonlinearityType,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by preflop_transformer_heads"
            )
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.attn_norm = nn.RMSNorm(dim, eps=1e-5)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn = ffn_block(dim, ffn_dim, dim, nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, dim = x.shape
        qkv = self.qkv(self.attn_norm(x)).view(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # The preflop token stream is tiny (1 game token + players). Explicit
        # attention avoids TorchInductor dispatching flash-attention kernels
        # that are fragile for these small dynamic shapes.
        scores = torch.matmul(qkv[0], qkv[1].transpose(-2, -1)) / math.sqrt(
            float(self.head_dim)
        )
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, qkv[2])
        attn = attn.transpose(1, 2).reshape(batch_size, token_count, dim)
        x = x + self.out(attn) / math.sqrt(2.0)
        return x + self.ffn(x) / math.sqrt(2.0)


class _PreflopGatedTokenMixerBlock(nn.Module):
    """Position-sensitive token mixer for the compact preflop token stream."""

    def __init__(
        self,
        dim: int,
        *,
        token_count: int,
        ffn_dim: int,
        nonlinearity: NonlinearityType,
    ) -> None:
        super().__init__()
        if token_count <= 1:
            raise ValueError("token_count must be greater than one")
        token_hidden = max(token_count, min(32, token_count * 4))
        self.token_count = int(token_count)
        self.token_norm = nn.RMSNorm(dim, eps=1e-5)
        self.token_mixer = nn.Sequential(
            OrderedDict(
                [
                    ("linear_in", nn.Linear(token_count, token_hidden, bias=False)),
                    ("activation", get_activation(nonlinearity)),
                    ("linear_out", nn.Linear(token_hidden, token_count, bias=False)),
                ]
            )
        )
        self.token_gate = nn.Linear(dim, dim, bias=True)
        self.ffn = ffn_block(dim, ffn_dim, dim, nonlinearity)

    def _can_use_token_triton_path(self, x: torch.Tensor) -> bool:
        linear_in = self.token_mixer.linear_in
        activation = self.token_mixer.activation
        linear_out = self.token_mixer.linear_out
        return (
            x.is_cuda
            and not self.training
            and not torch.is_grad_enabled()
            and isinstance(activation, nn.LeakyReLU)
            and activation.negative_slope == 0.01
            and x.shape[1] == 7
            and linear_in.weight.shape == (28, 7)
            and linear_out.weight.shape == (7, 28)
        )

    def _ffn_parts(
        self,
        x: torch.Tensor,
    ) -> tuple[nn.RMSNorm, nn.Linear, nn.Module, nn.Linear] | None:
        ffn_norm = getattr(self.ffn, "norm", None)
        ffn_linear_in = getattr(self.ffn, "linear_in", None)
        ffn_activation = getattr(self.ffn, "activation", None)
        ffn_linear_out = getattr(self.ffn, "linear_out", None)
        if (
            isinstance(ffn_norm, nn.RMSNorm)
            and isinstance(ffn_linear_in, nn.Linear)
            and ffn_activation is not None
            and isinstance(ffn_linear_out, nn.Linear)
            and ffn_norm.weight is not None
            and ffn_norm.normalized_shape == (x.shape[-1],)
        ):
            return ffn_norm, ffn_linear_in, ffn_activation, ffn_linear_out
        return None

    def _token_mixer_residual_from_norm(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        gate: torch.Tensor,
        ffn_norm: nn.RMSNorm | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        linear_in = self.token_mixer.linear_in
        linear_out = self.token_mixer.linear_out
        if self._can_use_token_triton_path(x):
            if (
                ffn_norm is not None
                and ffn_norm.weight is not None
                and x.shape[0] <= _PREFLOP_NEXT_NORM_MAX_BATCH
                and not torch.compiler.is_compiling()
            ):
                return _preflop_token_mixer_gate_residual_next_norm_triton(
                    x,
                    y,
                    gate,
                    linear_in.weight,
                    linear_out.weight,
                    ffn_norm.weight,
                    eps=ffn_norm.eps,
                    block_b=2,
                    num_warps=8,
                )
            token_out = _preflop_token_mixer_gate_residual_triton(
                x,
                y,
                gate,
                linear_in.weight,
                linear_out.weight,
            )
            return token_out, None

        mixed = self.token_mixer(y.transpose(1, 2)).transpose(1, 2)
        return x + mixed * torch.sigmoid(gate) / math.sqrt(2.0), None

    def _forward_from_token_norm(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.token_gate(y)
        ffn_parts = self._ffn_parts(x)
        ffn_norm = None if ffn_parts is None else ffn_parts[0]
        token_out, ffn_in = self._token_mixer_residual_from_norm(
            x,
            y,
            gate,
            ffn_norm,
        )
        if ffn_parts is None:
            return token_out + self.ffn(token_out) / math.sqrt(2.0)
        ffn_norm, ffn_linear_in, ffn_activation, ffn_linear_out = ffn_parts
        if ffn_in is None:
            ffn_in = ffn_norm(token_out)
        h = ffn_linear_in(ffn_in)
        h = ffn_activation(h)
        h = ffn_linear_out(h)
        return token_out + h / math.sqrt(2.0)

    def _forward_from_token_norm_with_next_token_norm(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        next_token_norm: nn.RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ffn_parts = self._ffn_parts(x)
        if (
            not self._can_use_token_triton_path(x)
            or ffn_parts is None
            or next_token_norm.weight is None
            or next_token_norm.normalized_shape != (x.shape[-1],)
        ):
            out = self._forward_from_token_norm(x, y)
            return out, next_token_norm(out)

        gate = self.token_gate(y)
        ffn_norm, ffn_linear_in, ffn_activation, ffn_linear_out = ffn_parts
        token_out, ffn_in = self._token_mixer_residual_from_norm(
            x,
            y,
            gate,
            ffn_norm,
        )
        if ffn_in is None:
            ffn_in = ffn_norm(token_out)
        h = ffn_linear_in(ffn_in)
        h = ffn_activation(h)
        ffn_out = ffn_linear_out(h)
        if (
            torch.compiler.is_compiling()
            and _preflop_compiled_ffn_boundary_enabled()
        ):
            out = token_out + ffn_out / math.sqrt(2.0)
            return out, next_token_norm(out)
        return _preflop_ffn_residual_next_token_norm_triton(
            token_out,
            ffn_out,
            next_token_norm.weight,
            eps=next_token_norm.eps,
            block_b=2,
            num_warps=8,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_from_token_norm(x, self.token_norm(x))


def _run_preflop_gated_token_mixer_blocks(
    blocks: nn.ModuleList,
    x: torch.Tensor,
) -> torch.Tensor:
    if len(blocks) == 0:
        return x
    if (
        len(blocks) <= 1
        or not x.is_cuda
        or torch.is_grad_enabled()
        or not all(isinstance(block, _PreflopGatedTokenMixerBlock) for block in blocks)
    ):
        for block in blocks:
            x = block(x)
        return x

    gated_blocks = list(blocks)
    if not all(
        not block.training
        and block._can_use_token_triton_path(x)
        and block._ffn_parts(x) is not None
        and isinstance(block.token_norm, nn.RMSNorm)
        and block.token_norm.weight is not None
        and block.token_norm.normalized_shape == (x.shape[-1],)
        for block in gated_blocks
    ):
        for block in blocks:
            x = block(x)
        return x

    precomputed_y: torch.Tensor | None = None
    for index, block in enumerate(gated_blocks):
        y = block.token_norm(x) if precomputed_y is None else precomputed_y
        if index + 1 < len(gated_blocks):
            x, precomputed_y = block._forward_from_token_norm_with_next_token_norm(
                x,
                y,
                gated_blocks[index + 1].token_norm,
            )
        else:
            x = block._forward_from_token_norm(x, y)
            precomputed_y = None
    return x


class _PreflopRangeSlotMomentPool(nn.Module):
    """Learned soft-slot moments over one player's compact preflop range.

    This intentionally avoids constructing per-hand hidden tokens. The only
    hand-axis tensors are tiny static slot bases/features; leaf-batch memory is
    proportional to ``slots`` rather than ``PREFLOP_HANDS * hidden_dim``.
    """

    def __init__(
        self,
        hand_feature_dim: int,
        player_context_dim: int,
        hidden_dim: int,
        slots: int,
        nonlinearity: NonlinearityType,
    ) -> None:
        super().__init__()
        if slots <= 0:
            raise ValueError("slots must be positive")
        if hand_feature_dim <= 0:
            raise ValueError("hand_feature_dim must be positive")
        self.hand_feature_dim = int(hand_feature_dim)
        self.slots = int(slots)
        self.slot_logits = nn.Parameter(torch.empty(PREFLOP_HANDS, slots))
        slot_width = max(16, min(64, hidden_dim // 4))
        slot_input_dim = self.hand_feature_dim + player_context_dim + 2
        self.slot_mlp = nn.Sequential(
            nn.Linear(slot_input_dim, slot_width, bias=False),
            nn.RMSNorm(slot_width, eps=1e-5),
            get_activation(nonlinearity),
            nn.Linear(slot_width, slot_width, bias=False),
            nn.RMSNorm(slot_width, eps=1e-5),
            get_activation(nonlinearity),
        )
        self.slot_embedding = nn.Parameter(torch.empty(slots, slot_width))
        self.output_proj = nn.Linear(slots * slot_width, hidden_dim, bias=False)
        nn.init.normal_(self.slot_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        player_beliefs: torch.Tensor,
        hand_features: torch.Tensor,
        player_context: torch.Tensor,
    ) -> torch.Tensor:
        if hand_features.shape != (PREFLOP_HANDS, self.hand_feature_dim):
            raise ValueError(
                "hand_features must have shape "
                f"({PREFLOP_HANDS}, {self.hand_feature_dim})"
            )
        dtype = player_beliefs.dtype
        assignment = torch.softmax(self.slot_logits, dim=-1).to(dtype)
        hand_features = hand_features.to(device=player_beliefs.device, dtype=dtype)
        slot_mass = player_beliefs @ assignment
        weighted_features = (
            assignment[:, :, None] * hand_features[:, None, :]
        ).reshape(PREFLOP_HANDS, self.slots * self.hand_feature_dim)
        slot_moments = (player_beliefs @ weighted_features).view(
            *player_beliefs.shape[:-1],
            self.slots,
            self.hand_feature_dim,
        )
        slot_moments = slot_moments / slot_mass.clamp_min(1.0e-8)[..., None]
        context = player_context[:, :, None, :].expand(
            -1,
            -1,
            self.slots,
            -1,
        )
        slot_input = torch.cat(
            (
                slot_mass[..., None],
                slot_mass.clamp_min(1.0e-8).log()[..., None],
                slot_moments,
                context,
            ),
            dim=-1,
        )
        slot_vecs = self.slot_mlp(slot_input)
        slot_vecs = slot_vecs + self.slot_embedding.to(dtype)[None, None, :, :]
        return self.output_proj(slot_vecs.flatten(2))


class _BetterPreflopCompactFFN(BaseMLPModel):
    """Shared compact 169-hand preflop MLP trunk."""

    hand_dim = PREFLOP_HANDS

    def __init__(
        self,
        num_actions: int,
        hidden_dim: int = 1024,
        range_hidden_dim: int = 256,
        ffn_dim: int = 1024,
        num_hidden_layers: int = 3,
        num_policy_layers: int = 3,
        num_value_layers: int = 3,
        num_players: int = 2,
        shared_trunk: bool = True,
        enforce_zero_sum: bool = True,
        board_interaction_dim: int = 0,
        policy_rank: int = 64,
        policy_hand_bias_rank: int = 32,
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
        context_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        _validate_internal_zero_sum(num_players, enforce_zero_sum)
        if range_hidden_dim < 0:
            raise ValueError("range_hidden_dim must be non-negative")
        if board_interaction_dim != 0:
            raise ValueError("compact preflop models do not support board interaction")
        if policy_rank <= 0:
            raise ValueError("policy_rank must be positive")
        if policy_hand_bias_rank <= 0:
            raise ValueError("policy_hand_bias_rank must be positive")

        self.num_actions = int(num_actions)
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_policy_layers = num_policy_layers
        self.num_value_layers = num_value_layers
        self.num_players = num_players
        self.shared_trunk = shared_trunk
        self.enforce_zero_sum = enforce_zero_sum
        self.board_interaction_dim = board_interaction_dim
        self.policy_rank = policy_rank
        self.policy_hand_bias_rank = policy_hand_bias_rank
        self.nonlinearity = nonlinearity

        self.street_embedding = nn.Embedding(5, hidden_dim)
        self.rank_embedding = nn.Embedding(13 + 1, hidden_dim, padding_idx=13)
        self.suit_embedding = nn.Embedding(4 + 1, hidden_dim, padding_idx=4)
        hi, lo, suited = _preflop_class_ranks()
        class_type = torch.where(
            hi == lo,
            torch.zeros_like(hi),
            torch.where(suited, torch.ones_like(hi), torch.full_like(hi, 2)),
        )
        self.register_buffer("class_hi_rank", hi, persistent=False)
        self.register_buffer("class_lo_rank", lo, persistent=False)
        self.register_buffer("class_suited", suited, persistent=False)
        self.register_buffer("class_type", class_type, persistent=False)
        self.register_buffer(
            "preflop_class_static_features",
            self._build_preflop_class_static_features(hi, lo, suited),
            persistent=False,
        )
        if range_hidden_dim == 0 and ffn_dim % num_players != 0:
            raise ValueError(
                "ffn_dim must be divisible by num_players when range_hidden_dim is 0"
            )
        effective_range_hidden_dim = (
            ffn_dim // num_players if range_hidden_dim == 0 else range_hidden_dim
        )
        self.range_summary_dim = int(effective_range_hidden_dim)
        self.class_hi_embedding = nn.Embedding(13, self.range_summary_dim)
        self.class_lo_embedding = nn.Embedding(13, self.range_summary_dim)
        self.class_type_embedding = nn.Embedding(3, self.range_summary_dim)
        self.class_feature_proj = nn.Linear(
            HAND_STATIC_FEATURE_DIM, self.range_summary_dim
        )
        self.belief_proj = output_projection(
            num_players * self.range_summary_dim,
            hidden_dim,
        )
        context_in_dim = (
            context_length(num_players) if context_in_dim is None else context_in_dim
        )
        self.context_in_dim = int(context_in_dim)
        self.context_encoder = ffn_block(
            self.context_in_dim, hidden_dim, hidden_dim, nonlinearity
        )

        alpha = 1 / math.sqrt(num_hidden_layers + max(1, num_value_layers))
        self.trunk = nn.Sequential(
            *[
                ResidualBlock(
                    ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), alpha
                )
                for _ in range(num_hidden_layers)
            ]
        )

    @staticmethod
    def _build_preflop_class_static_features(
        hi_rank: torch.Tensor,
        lo_rank: torch.Tensor,
        suited: torch.Tensor,
    ) -> torch.Tensor:
        hi = hi_rank.to(torch.float32)
        lo = lo_rank.to(torch.float32)
        pair = hi_rank == lo_rank
        gap = (hi - lo).clamp(min=0.0)
        return torch.stack(
            [
                pair.to(torch.float32),
                suited.to(torch.float32),
                gap / 12.0,
                hi / 12.0,
                lo / 12.0,
                (hi == 12).to(torch.float32),
                (lo >= 8).to(torch.float32),
                (gap <= 1).to(torch.float32),
            ],
            dim=-1,
        )

    def _class_static_features(self) -> torch.Tensor:
        return self.preflop_class_static_features

    def _hand_embedding(self) -> torch.Tensor:
        static = self._class_static_features().to(self.class_hi_embedding.weight.dtype)
        return (
            self.class_hi_embedding(self.class_hi_rank)
            + self.class_lo_embedding(self.class_lo_rank)
            + self.class_type_embedding(self.class_type)
            + self.class_feature_proj(static)
        )

    def static_feature_base(self, features: MLPFeatures) -> torch.Tensor:
        return self.static_feature_base_from_prefix(
            self.static_feature_prefix(features.context, features.street),
            features.board,
        )

    def static_feature_prefix(
        self, context: torch.Tensor, street: torch.Tensor
    ) -> torch.Tensor:
        context = context.to(dtype=self.street_embedding.weight.dtype)
        if context.shape[-1] > self.context_in_dim:
            context = context[..., : self.context_in_dim]
        elif context.shape[-1] < self.context_in_dim:
            pad = context.new_zeros(
                *context.shape[:-1], self.context_in_dim - context.shape[-1]
            )
            context = torch.cat((context, pad), dim=-1)
        return self.street_embedding(street) + self.context_encoder(context)

    def static_feature_base_from_prefix(
        self, prefix: torch.Tensor, board: torch.Tensor
    ) -> torch.Tensor:
        ranks = torch.where(board >= 0, board % 13, torch.full_like(board, 13))
        suits = torch.where(board >= 0, board // 13, torch.full_like(board, 4))
        board_features = self.rank_embedding(ranks) + self.suit_embedding(suits)
        return board_features.sum(dim=1) + prefix

    def _forward_base_from_static(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.hand_dim != PREFLOP_HANDS:
            raise ValueError(
                f"compact preflop model requires hand_dim={PREFLOP_HANDS}, "
                f"got {features.hand_dim}"
            )
        player_beliefs = features.beliefs.view(-1, self.num_players, PREFLOP_HANDS)
        hand_emb = self._hand_embedding()
        player_beliefs = player_beliefs.to(dtype=hand_emb.dtype)
        static_base_features = static_base_features.to(dtype=hand_emb.dtype)
        per_player_belief = player_beliefs @ hand_emb
        flat_features = static_base_features + self.belief_proj(
            per_player_belief.flatten(1)
        )
        x = self.trunk(flat_features)
        return player_beliefs, flat_features, x, hand_emb

    def _forward_base(
        self, features: MLPFeatures
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._forward_base_from_static(
            features, static_base_features=self.static_feature_base(features)
        )

    def init_weights(self, rng: torch.Generator | None = None) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=rng)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=rng)

        expansion_gain = math.sqrt(self.ffn_dim / self.hidden_dim)
        for sequential in (
            self.trunk,
            getattr(self, "policy_tower", None),
            getattr(self, "value_tower", None),
        ):
            if sequential is None:
                continue
            for block in sequential.modules():
                if not isinstance(block, ResidualBlock):
                    continue
                inner = block.inner
                if "swiglu" in dict(inner.named_children()):
                    swiglu = inner.get_submodule("swiglu")
                    nn.init.orthogonal_(
                        swiglu.gate.weight, expansion_gain, generator=rng
                    )
                    nn.init.orthogonal_(swiglu.up.weight, expansion_gain, generator=rng)
                else:
                    nn.init.orthogonal_(
                        inner.get_submodule("linear_in").weight,
                        1.532 * expansion_gain,
                        generator=rng,
                    )
        value_head = getattr(self, "value_head", None)
        if value_head is not None:
            value_head[-1].get_submodule("linear_out").weight.data.mul_(0.1)
        policy_action_head = getattr(self, "policy_action_head", None)
        if policy_action_head is not None:
            policy_action_head.get_submodule("linear_out").weight.data.mul_(0.1)

    def repeat(
        self,
        features: MLPFeatures,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self(
            features,
            include_policy=include_policy,
            include_value=include_value,
            apply_zero_sum=apply_zero_sum,
        )


class _BetterPreflopTransformerBase(BaseMLPModel):
    """Shared compact 169-hand preflop token encoder.

    The token stream is intentionally small: one game-state token from scalar
    public context and one token per player from that player's range and public
    player context.
    """

    hand_dim = PREFLOP_HANDS

    def __init__(
        self,
        num_actions: int,
        hidden_dim: int = 1024,
        range_hidden_dim: int = 256,
        ffn_dim: int = 1024,
        num_hidden_layers: int = 3,
        num_policy_layers: int = 3,
        num_value_layers: int = 3,
        num_players: int = 2,
        shared_trunk: bool = True,
        enforce_zero_sum: bool = True,
        board_interaction_dim: int = 0,
        policy_rank: int = 64,
        policy_hand_bias_rank: int = 32,
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
        transformer_heads: int = 8,
        range_slot_moment_slots: int = 0,
    ) -> None:
        super().__init__()
        _validate_internal_zero_sum(num_players, enforce_zero_sum)
        if range_hidden_dim < 0:
            raise ValueError("range_hidden_dim must be non-negative")
        if range_slot_moment_slots < 0:
            raise ValueError("range_slot_moment_slots must be non-negative")
        if board_interaction_dim != 0:
            raise ValueError("compact preflop models do not support board interaction")
        if policy_rank <= 0:
            raise ValueError("policy_rank must be positive")
        if policy_hand_bias_rank <= 0:
            raise ValueError("policy_hand_bias_rank must be positive")
        if transformer_heads <= 0:
            raise ValueError("preflop_transformer_heads must be positive")
        if self._uses_attention_heads() and hidden_dim % transformer_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by preflop_transformer_heads"
            )

        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_policy_layers = int(num_policy_layers)
        self.num_value_layers = int(num_value_layers)
        self.num_players = int(num_players)
        self.shared_trunk = bool(shared_trunk)
        self.enforce_zero_sum = bool(enforce_zero_sum)
        self.board_interaction_dim = int(board_interaction_dim)
        self.policy_rank = int(policy_rank)
        self.policy_hand_bias_rank = int(policy_hand_bias_rank)
        self.nonlinearity = nonlinearity
        self.transformer_heads = int(transformer_heads)
        self.range_slot_moment_slots = int(range_slot_moment_slots)

        scalar_schema, player_schema = context_schemas(num_players)
        self.scalar_context_dim = scalar_schema.NUM_SCALAR_CONTEXT.value
        self.player_context_dim = player_schema.NUM_PLAYER_CONTEXT.value
        hand_embed_dim = (
            max(1, ffn_dim // num_players)
            if range_hidden_dim == 0
            else range_hidden_dim
        )
        self.preflop_hand_embed_dim = int(hand_embed_dim)

        hi, lo, suited = _preflop_class_ranks()
        self.register_buffer("class_hi_rank", hi, persistent=False)
        self.register_buffer("class_lo_rank", lo, persistent=False)
        self.register_buffer("class_suited", suited, persistent=False)
        self.register_buffer(
            "preflop_class_static_features",
            self._build_preflop_class_static_features(hi, lo, suited),
            persistent=False,
        )
        self.register_buffer(
            "preflop_bucket_projection",
            _preflop_bucket_projection(),
            persistent=False,
        )
        self.register_buffer(
            "_preflop_eval_hand_embedding",
            torch.empty(PREFLOP_HANDS, hand_embed_dim),
            persistent=False,
        )
        self.register_buffer(
            "_preflop_eval_range_projection",
            torch.empty(PREFLOP_HANDS, hidden_dim),
            persistent=False,
        )
        self.register_buffer(
            "_preflop_eval_bucket_projection",
            torch.empty(PREFLOP_HANDS, 16),
            persistent=False,
        )
        self._preflop_eval_projection_cache_key = None

        self.street_embedding = nn.Embedding(5, hidden_dim)
        self.hand_encoder = nn.Sequential(
            nn.Linear(HAND_STATIC_FEATURE_DIM, hand_embed_dim),
            nn.RMSNorm(hand_embed_dim, eps=1e-5),
            get_activation(nonlinearity),
            nn.Linear(hand_embed_dim, hand_embed_dim),
            nn.RMSNorm(hand_embed_dim, eps=1e-5),
            get_activation(nonlinearity),
        )
        self.range_proj = nn.Linear(hand_embed_dim * 2, hidden_dim, bias=False)
        self.bucket_mass_proj = nn.Sequential(
            nn.Linear(16, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim, eps=1e-5),
            get_activation(nonlinearity),
        )
        self.range_slot_moment_pool = (
            _PreflopRangeSlotMomentPool(
                HAND_STATIC_FEATURE_DIM,
                self.player_context_dim,
                hidden_dim,
                self.range_slot_moment_slots,
                nonlinearity,
            )
            if self.range_slot_moment_slots > 0
            else None
        )
        self.game_context_proj = nn.Sequential(
            nn.Linear(self.scalar_context_dim, hidden_dim),
            nn.RMSNorm(hidden_dim, eps=1e-5),
            get_activation(nonlinearity),
        )
        self.player_context_proj = nn.Sequential(
            nn.Linear(self.player_context_dim, hidden_dim),
            nn.RMSNorm(hidden_dim, eps=1e-5),
            get_activation(nonlinearity),
        )
        self.encoder = nn.ModuleList(
            [self._make_token_encoder_block() for _ in range(num_hidden_layers)]
        )
        self.player_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.RMSNorm(hidden_dim, eps=1e-5),
            get_activation(nonlinearity),
        )

    def _uses_attention_heads(self) -> bool:
        return True

    def _make_token_encoder_block(self) -> nn.Module:
        return _PreflopTokenEncoderBlock(
            self.hidden_dim,
            num_heads=self.transformer_heads,
            ffn_dim=self.ffn_dim,
            nonlinearity=self.nonlinearity,
        )

    @staticmethod
    def _build_preflop_class_static_features(
        hi_rank: torch.Tensor,
        lo_rank: torch.Tensor,
        suited: torch.Tensor,
    ) -> torch.Tensor:
        hi = hi_rank.to(torch.float32)
        lo = lo_rank.to(torch.float32)
        pair = hi_rank == lo_rank
        gap = (hi - lo).clamp(min=0.0)
        return torch.stack(
            [
                pair.to(torch.float32),
                suited.to(torch.float32),
                gap / 12.0,
                hi / 12.0,
                lo / 12.0,
                (hi == 12).to(torch.float32),
                (lo >= 8).to(torch.float32),
                (gap <= 1).to(torch.float32),
            ],
            dim=-1,
        )

    def _class_static_features(self) -> torch.Tensor:
        return self.preflop_class_static_features

    def train(self, mode: bool = True):
        self._preflop_hand_embedding_cache = None
        self._preflop_eval_projection_cache_key = None
        if hasattr(self, "_preflop_eval_value_cache_key"):
            self._preflop_eval_value_cache_key = None
        return super().train(mode)

    def _preflop_projection_parameter_versions(self) -> tuple[int, ...]:
        return tuple(
            int(param._version)
            for module in (self.hand_encoder, self.range_proj)
            for param in module.parameters()
        )

    def _preflop_eval_cache_dtype(self) -> torch.dtype:
        device_type = self.class_hi_rank.device.type
        if device_type == "cuda" and torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return self.hand_encoder[0].weight.dtype

    def _store_preflop_eval_cache_tensor(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        value = value.detach().contiguous()
        existing = getattr(self, name)
        if (
            existing.shape == value.shape
            and existing.device == value.device
            and existing.dtype == value.dtype
        ):
            existing.copy_(value)
            return existing
        setattr(self, name, value)
        return value

    @torch.no_grad()
    def prepare_preflop_eval_cache(self) -> None:
        if self.training or not _preflop_eval_cache_enabled():
            self._preflop_eval_projection_cache_key = None
            return
        target_device = self.class_hi_rank.device
        target_dtype = self._preflop_eval_cache_dtype()
        cache_key = (
            target_device,
            target_dtype,
            self._preflop_projection_parameter_versions(),
        )
        cached_key = getattr(self, "_preflop_eval_projection_cache_key", None)
        if (
            cached_key == cache_key
            and self._preflop_eval_hand_embedding.device == target_device
            and self._preflop_eval_hand_embedding.dtype == target_dtype
        ):
            return
        hand_static = self._class_static_features().to(
            device=target_device,
            dtype=self.hand_encoder[0].weight.dtype,
        )
        hand_emb = self.hand_encoder(hand_static)
        dtype = hand_emb.dtype
        cache_key = (hand_emb.device, dtype, cache_key[2])
        bucket_projection = self.preflop_bucket_projection.to(
            device=hand_emb.device,
            dtype=dtype,
        )
        combined_projection = torch.cat((hand_emb, hand_emb.square()), dim=-1)
        range_projection = combined_projection.matmul(self.range_proj.weight.t())
        stored_hand_emb = self._store_preflop_eval_cache_tensor(
            "_preflop_eval_hand_embedding",
            hand_emb,
        )
        self._store_preflop_eval_cache_tensor(
            "_preflop_eval_range_projection",
            range_projection,
        )
        self._store_preflop_eval_cache_tensor(
            "_preflop_eval_bucket_projection",
            bucket_projection,
        )
        self._preflop_eval_projection_cache_key = cache_key
        self._preflop_hand_embedding_cache = (cache_key, stored_hand_emb)

    def _preflop_eval_projection_cache(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if (
            self.training
            or torch.is_grad_enabled()
            or not _preflop_eval_cache_enabled()
        ):
            return None
        if not torch.compiler.is_compiling():
            self.prepare_preflop_eval_cache()
        if getattr(self, "_preflop_eval_projection_cache_key", None) is None:
            return None
        return (
            self._preflop_eval_hand_embedding,
            self._preflop_eval_range_projection,
            self._preflop_eval_bucket_projection,
        )

    def _hand_embedding(self) -> torch.Tensor:
        cached_projection = self._preflop_eval_projection_cache()
        if cached_projection is not None:
            hand_emb, _, _ = cached_projection
            return hand_emb
        hand_static = self._class_static_features().to(
            device=self.class_hi_rank.device,
            dtype=self.hand_encoder[0].weight.dtype,
        )
        if torch.compiler.is_compiling():
            return self.hand_encoder(hand_static)
        if self.training or torch.is_grad_enabled():
            return self.hand_encoder(hand_static)
        cache_key = (
            hand_static.device,
            hand_static.dtype,
            tuple(int(p._version) for p in self.hand_encoder.parameters()),
        )
        cached = getattr(self, "_preflop_hand_embedding_cache", None)
        if cached is not None:
            cached_key, cached_value = cached
            if cached_key == cache_key:
                return cached_value
        hand_emb = self.hand_encoder(hand_static)
        self._preflop_hand_embedding_cache = (cache_key, hand_emb)
        return hand_emb

    def _split_context(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scalar_context = context[:, : self.scalar_context_dim]
        player_context = context[:, self.scalar_context_dim :].view(
            -1, self.player_context_dim, self.num_players
        )
        return scalar_context, player_context.transpose(1, 2).contiguous()

    def static_feature_prefix(
        self, context: torch.Tensor, street: torch.Tensor
    ) -> torch.Tensor:
        scalar_context, _ = self._split_context(context)
        dtype = self.street_embedding.weight.dtype
        return self.street_embedding(street) + self.game_context_proj(
            scalar_context.to(dtype)
        )

    def static_feature_base(self, features: MLPFeatures) -> torch.Tensor:
        scalar_context, player_context = self._split_context(features.context)
        dtype = self.street_embedding.weight.dtype
        game_token = self.street_embedding(features.street) + self.game_context_proj(
            scalar_context.to(dtype)
        )
        player_tokens = self.player_context_proj(player_context.to(dtype))
        return torch.cat((game_token[:, None, :], player_tokens), dim=1)

    def static_feature_base_from_prefix(
        self, prefix: torch.Tensor, board: torch.Tensor
    ) -> torch.Tensor:
        del board
        return prefix

    def _encode_base_tokens(
        self,
        features: MLPFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.hand_dim != PREFLOP_HANDS:
            raise ValueError(
                f"compact preflop transformer requires hand_dim={PREFLOP_HANDS}, "
                f"got {features.hand_dim}"
            )
        player_beliefs = features.beliefs.view(-1, self.num_players, PREFLOP_HANDS)
        hand_static = self._class_static_features().to(
            device=self.class_hi_rank.device,
            dtype=self.hand_encoder[0].weight.dtype,
        )
        eval_projection_cache = self._preflop_eval_projection_cache()
        if eval_projection_cache is None:
            hand_emb = self._hand_embedding()
            combined_projection = torch.cat((hand_emb, hand_emb.square()), dim=-1)
            range_projection = combined_projection.matmul(self.range_proj.weight.t())
            bucket_projection = self.preflop_bucket_projection.to(
                device=player_beliefs.device,
                dtype=hand_emb.dtype,
            )
        else:
            hand_emb, range_projection, bucket_projection = eval_projection_cache
        dtype = hand_emb.dtype
        player_beliefs = player_beliefs.to(dtype)
        bucket_mass = player_beliefs @ bucket_projection
        _, player_context = self._split_context(features.context)
        static_player_tokens = self.player_context_proj(player_context.to(dtype))
        player_tokens = (
            player_beliefs @ range_projection
            + self.bucket_mass_proj(bucket_mass)
            + static_player_tokens
        )
        if self.range_slot_moment_pool is not None:
            player_tokens = player_tokens + self.range_slot_moment_pool(
                player_beliefs,
                hand_static,
                player_context.to(dtype),
            )
        game_token = self.static_feature_prefix(features.context, features.street).to(
            dtype
        )
        encoded = torch.cat((game_token[:, None, :], player_tokens), dim=1)
        encoded = _run_preflop_gated_token_mixer_blocks(self.encoder, encoded)
        return player_beliefs, encoded, hand_emb

    def _encode_base_tokens_static(
        self,
        features: MLPFeatures,
        static_game_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.hand_dim != PREFLOP_HANDS:
            raise ValueError(
                f"compact preflop transformer requires hand_dim={PREFLOP_HANDS}, "
                f"got {features.hand_dim}"
            )
        player_beliefs = features.beliefs.view(-1, self.num_players, PREFLOP_HANDS)
        hand_static = self._class_static_features().to(
            device=self.class_hi_rank.device,
            dtype=self.hand_encoder[0].weight.dtype,
        )
        eval_projection_cache = self._preflop_eval_projection_cache()
        if eval_projection_cache is None:
            hand_emb = self._hand_embedding()
            combined_projection = torch.cat((hand_emb, hand_emb.square()), dim=-1)
            range_projection = combined_projection.matmul(self.range_proj.weight.t())
            bucket_projection = self.preflop_bucket_projection.to(
                device=player_beliefs.device,
                dtype=hand_emb.dtype,
            )
        else:
            hand_emb, range_projection, bucket_projection = eval_projection_cache
        dtype = hand_emb.dtype
        static_game_token = static_game_token.to(dtype)
        if static_game_token.ndim == 3:
            if static_game_token.shape[1] != self.num_players + 1:
                raise ValueError(
                    "static preflop token cache must have shape "
                    f"[batch, {self.num_players + 1}, hidden_dim], got "
                    f"{tuple(static_game_token.shape)}"
                )
            game_token = static_game_token[:, 0]
            static_player_tokens = static_game_token[:, 1:]
        elif static_game_token.ndim == 2:
            game_token = static_game_token
            _, player_context = self._split_context(features.context)
            static_player_tokens = self.player_context_proj(player_context.to(dtype))
        else:
            raise ValueError(
                "static preflop features must be either a game-token tensor "
                "or a cached token tensor"
            )
        player_context = None
        player_beliefs = player_beliefs.to(dtype)
        bucket_mass = player_beliefs @ bucket_projection
        player_tokens = (
            player_beliefs @ range_projection
            + self.bucket_mass_proj(bucket_mass)
            + static_player_tokens
        )
        if self.range_slot_moment_pool is not None:
            _, player_context = self._split_context(features.context)
            player_tokens = player_tokens + self.range_slot_moment_pool(
                player_beliefs,
                hand_static,
                player_context.to(dtype),
            )
        encoded = torch.cat((game_token[:, None, :], player_tokens), dim=1)
        encoded = _run_preflop_gated_token_mixer_blocks(self.encoder, encoded)
        return player_beliefs, encoded, hand_emb

    def _states_from_tokens(
        self, encoded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        game_state = encoded[:, 0]
        player_tokens = encoded[:, 1:]
        linear = self.player_state[0]
        norm = self.player_state[1]
        activation = self.player_state[2]
        if (
            isinstance(linear, nn.Linear)
            and isinstance(norm, nn.RMSNorm)
            and linear.in_features == self.hidden_dim * 2
            and linear.out_features == self.hidden_dim
        ):
            player_weight, game_weight = linear.weight.split(self.hidden_dim, dim=1)
            player_pre = player_tokens.flatten(0, 1).matmul(player_weight.t()).view(
                -1, self.num_players, self.hidden_dim
            )
            game_pre = game_state.matmul(game_weight.t())[:, None, :]
            player_pre = player_pre + game_pre
            if linear.bias is not None:
                player_pre = player_pre + linear.bias.to(dtype=player_pre.dtype)
            player_state = activation(norm(player_pre))
        else:
            player_state = self.player_state(
                torch.cat(
                    (
                        player_tokens,
                        game_state[:, None, :].expand(-1, self.num_players, -1),
                    ),
                    dim=-1,
                )
            )
        return game_state, player_state

    def _encode_tokens(
        self,
        features: MLPFeatures,
        static_game_token: torch.Tensor | None = None,
        extra_encoder: nn.ModuleList | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if static_game_token is None:
            player_beliefs, encoded, hand_emb = self._encode_base_tokens(features)
        else:
            player_beliefs, encoded, hand_emb = self._encode_base_tokens_static(
                features, static_game_token
            )
        if extra_encoder is not None:
            encoded = _run_preflop_gated_token_mixer_blocks(extra_encoder, encoded)
        game_state, player_state = self._states_from_tokens(encoded)
        return player_beliefs, game_state, player_state, hand_emb

    def init_weights(self, rng: torch.Generator | None = None) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=rng)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=rng)

        policy_action_head = getattr(self, "policy_action_head", None)
        if policy_action_head is not None:
            policy_action_head.get_submodule("linear_out").weight.data.mul_(0.1)
        value_scale = getattr(self, "value_scale", None)
        if value_scale is not None:
            value_scale.weight.data.mul_(0.1)

    def repeat(
        self,
        features: MLPFeatures,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self(
            features,
            include_policy=include_policy,
            include_value=include_value,
            apply_zero_sum=apply_zero_sum,
        )


class BetterPreflopTransformerValueFFN(_BetterPreflopTransformerBase):
    """Compact 169-hand preflop value model with game/player token attention."""

    def __init__(self, *args, value_heads=None, **kwargs) -> None:
        if not args:
            kwargs.setdefault("num_actions", 1)
        super().__init__(*args, **kwargs)
        del value_heads
        self.value_encoder = nn.ModuleList(
            [self._make_token_encoder_block() for _ in range(self.num_value_layers)]
        )
        self.value_hand_proj = nn.Linear(
            self.preflop_hand_embed_dim, self.hidden_dim, bias=False
        )
        self.value_scale = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value_bias = nn.Linear(self.hidden_dim, 1)
        self.register_buffer(
            "_preflop_eval_value_fused_weight",
            torch.empty(self.hidden_dim, PREFLOP_HANDS + 1),
            persistent=False,
        )
        self._preflop_eval_value_cache_key = None

    def _preflop_value_parameter_versions(self) -> tuple[int, ...]:
        return tuple(
            int(param._version)
            for module in (self.value_hand_proj, self.value_scale, self.value_bias)
            for param in module.parameters()
        )

    @torch.no_grad()
    def prepare_preflop_eval_cache(self) -> None:
        super().prepare_preflop_eval_cache()
        if self.training or not _preflop_eval_cache_enabled():
            self._preflop_eval_value_cache_key = None
            return
        projection_key = getattr(self, "_preflop_eval_projection_cache_key", None)
        if projection_key is None:
            self._preflop_eval_value_cache_key = None
            return
        cache_key = (projection_key, self._preflop_value_parameter_versions())
        cached_key = getattr(self, "_preflop_eval_value_cache_key", None)
        if (
            cached_key == cache_key
            and self._preflop_eval_value_fused_weight.device
            == self._preflop_eval_hand_embedding.device
            and self._preflop_eval_value_fused_weight.dtype
            == self._preflop_eval_hand_embedding.dtype
        ):
            return
        hand_value = self.value_hand_proj(self._preflop_eval_hand_embedding)
        combined_weight = self.value_scale.weight.t().matmul(hand_value.t())
        combined_weight = combined_weight / math.sqrt(float(self.hidden_dim))
        value_bias_weight = self.value_bias.weight.t().to(dtype=combined_weight.dtype)
        fused_weight = torch.cat((combined_weight, value_bias_weight), dim=1)
        self._store_preflop_eval_cache_tensor(
            "_preflop_eval_value_fused_weight",
            fused_weight,
        )
        self._preflop_eval_value_cache_key = cache_key

    def _preflop_eval_value_fused_weight_cache(self) -> torch.Tensor | None:
        if (
            self.training
            or torch.is_grad_enabled()
            or not _preflop_eval_cache_enabled()
        ):
            return None
        if not torch.compiler.is_compiling():
            self.prepare_preflop_eval_cache()
        if getattr(self, "_preflop_eval_value_cache_key", None) is None:
            return None
        return self._preflop_eval_value_fused_weight

    def _hand_values_from_tokens(
        self,
        player_beliefs: torch.Tensor,
        player_state: torch.Tensor,
        hand_emb: torch.Tensor,
        apply_zero_sum: bool = True,
    ) -> torch.Tensor:
        fused_weight = self._preflop_eval_value_fused_weight_cache()
        if fused_weight is None:
            hand_value = self.value_hand_proj(hand_emb)
            combined_weight = self.value_scale.weight.t().matmul(hand_value.t())
            combined_weight = combined_weight / math.sqrt(float(self.hidden_dim))
            value_bias_weight = self.value_bias.weight.t().to(
                dtype=combined_weight.dtype
            )
            fused_weight = torch.cat((combined_weight, value_bias_weight), dim=1)
        raw_and_bias = player_state.flatten(0, 1).matmul(fused_weight)
        hand_values_raw = raw_and_bias[:, :PREFLOP_HANDS].view(
            -1, self.num_players, PREFLOP_HANDS
        )
        value_bias = raw_and_bias[:, PREFLOP_HANDS:].view(-1, self.num_players, 1)
        if self.value_bias.bias is not None:
            value_bias = value_bias + self.value_bias.bias.to(dtype=value_bias.dtype)
        hand_values_raw = hand_values_raw + value_bias
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            return hand_values_raw - hand_value_sums
        return hand_values_raw

    def _value_from_tokens(
        self,
        player_beliefs: torch.Tensor,
        player_state: torch.Tensor,
        hand_emb: torch.Tensor,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        hand_values = self._hand_values_from_tokens(
            player_beliefs,
            player_state,
            hand_emb,
            apply_zero_sum=apply_zero_sum,
        )
        return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

    def forward_policy(self, features: MLPFeatures, latent=None) -> torch.Tensor:
        raise RuntimeError(
            "BetterPreflopTransformerValueFFN does not provide policy outputs"
        )

    def forward_value(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, value_head
        player_beliefs, _, player_state, hand_emb = self._encode_tokens(
            features,
            static_game_token=static_base_features,
            extra_encoder=self.value_encoder,
        )
        return self._value_from_tokens(
            player_beliefs,
            player_state,
            hand_emb,
            apply_zero_sum=apply_zero_sum,
        )

    def forward_value_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        apply_zero_sum: bool = True,
        value_head: str = "auto",
    ) -> ModelOutput:
        return self.forward_value(
            features,
            latent=latent,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
        )

    def forward_hand_values_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        apply_zero_sum: bool = True,
        value_head: str = "auto",
    ) -> torch.Tensor:
        del latent, value_head
        player_beliefs, _, player_state, hand_emb = self._encode_tokens(
            features,
            static_game_token=static_base_features,
            extra_encoder=self.value_encoder,
        )
        return self._hand_values_from_tokens(
            player_beliefs,
            player_state,
            hand_emb,
            apply_zero_sum=apply_zero_sum,
        )

    def forward_pre(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        return self.forward_value(features, **kwargs).hand_values

    def forward_post(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        return self.forward_value(features, **kwargs).hand_values

    def forward_both(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self.forward_value(
            features, latent=latent, apply_zero_sum=apply_zero_sum
        )

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = False,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        if include_policy:
            raise RuntimeError(
                "BetterPreflopTransformerValueFFN does not provide policy outputs"
            )
        if not include_value:
            raise ValueError(
                "BetterPreflopTransformerValueFFN requires include_value=True"
            )
        return self._call_forward_value(
            features,
            latent=latent,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
        )

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterPreflopValueFeatureEncoder:
        return BetterPreflopValueFeatureEncoder(
            env=env,
            device=device,
            dtype=dtype,
        )


class BetterPreflopTransformerPolicyFFN(_BetterPreflopTransformerBase):
    """Compact 169-hand preflop policy model with game/player token attention."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.policy_encoder = nn.ModuleList(
            [self._make_token_encoder_block() for _ in range(self.num_policy_layers)]
        )
        self.policy_hand_proj = output_projection(
            self.preflop_hand_embed_dim, self.policy_rank
        )
        self.policy_action_head = output_projection(
            self.hidden_dim, self.num_actions * self.policy_rank
        )
        self.policy_action_bias = output_projection(self.hidden_dim, self.num_actions)
        self.policy_hand_bias = output_projection(
            self.preflop_hand_embed_dim, self.policy_hand_bias_rank
        )
        self.policy_hand_bias_action = output_projection(
            self.hidden_dim, self.num_actions * self.policy_hand_bias_rank
        )

    def forward_policy(self, features: MLPFeatures, latent=None) -> torch.Tensor:
        _, encoded, hand_emb = self._encode_base_tokens(features)
        if not self.shared_trunk:
            encoded = encoded.detach()
        encoded = _run_preflop_gated_token_mixer_blocks(self.policy_encoder, encoded)
        game_state, player_state = self._states_from_tokens(encoded)
        actor = features.to_act.long().clamp(min=0, max=self.num_players - 1)
        actor_state = player_state.gather(
            1,
            actor[:, None, None].expand(-1, 1, self.hidden_dim),
        ).squeeze(1)
        policy_state = actor_state + game_state
        action_emb = self.policy_action_head(policy_state).view(
            -1, self.num_actions, self.policy_rank
        )
        hand_vec = self.policy_hand_proj(hand_emb)
        logits = torch.einsum("hr,bar->bha", hand_vec, action_emb)
        logits = logits / math.sqrt(self.policy_rank)
        hand_bias = self.policy_hand_bias(hand_emb)
        hand_bias_action = self.policy_hand_bias_action(policy_state).view(
            -1, self.num_actions, self.policy_hand_bias_rank
        )
        logits = logits + torch.einsum("hk,bak->bha", hand_bias, hand_bias_action)
        return logits + self.policy_action_bias(policy_state)[:, None, :]

    def forward_value(
        self, features: MLPFeatures, latent=None, **kwargs
    ) -> ModelOutput:
        raise RuntimeError(
            "BetterPreflopTransformerPolicyFFN does not provide value outputs"
        )

    def forward_both(self, features: MLPFeatures, latent=None, **kwargs) -> ModelOutput:
        return ModelOutput(policy_logits=self._call_forward_policy(features))

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = False,
        **kwargs,
    ) -> ModelOutput:
        if include_value:
            raise RuntimeError(
                "BetterPreflopTransformerPolicyFFN does not provide value outputs"
            )
        if not include_policy:
            raise ValueError(
                "BetterPreflopTransformerPolicyFFN requires include_policy=True"
            )
        return ModelOutput(policy_logits=self._call_forward_policy(features))

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterPreflopPolicyFeatureEncoder:
        return BetterPreflopPolicyFeatureEncoder(
            env=env,
            device=device,
            dtype=dtype,
        )


class _BetterPreflopGatedTokenMixerMixin:
    """Compact token encoder that mixes fixed token slots without attention."""

    def _uses_attention_heads(self) -> bool:
        return False

    def _make_token_encoder_block(self) -> nn.Module:
        return _PreflopGatedTokenMixerBlock(
            self.hidden_dim,
            token_count=self.num_players + 1,
            ffn_dim=self.ffn_dim,
            nonlinearity=self.nonlinearity,
        )


class BetterPreflopGatedTokenMixerValueFFN(
    _BetterPreflopGatedTokenMixerMixin,
    BetterPreflopTransformerValueFFN,
):
    """Compact 169-hand preflop value model with gated token mixing."""

    pass


class BetterPreflopGatedTokenMixerPolicyFFN(
    _BetterPreflopGatedTokenMixerMixin,
    BetterPreflopTransformerPolicyFFN,
):
    """Compact 169-hand preflop policy model with gated token mixing."""

    pass


class BetterPreflopValueFFN(_BetterPreflopCompactFFN):
    """Compact 169-hand preflop value model for `E_preflop`."""

    def __init__(self, *args, value_heads=None, **kwargs) -> None:
        if not args:
            kwargs.setdefault("num_actions", 1)
        super().__init__(*args, **kwargs)
        del value_heads
        alpha = 1 / math.sqrt(self.num_hidden_layers + self.num_value_layers)
        layers = [
            ResidualBlock(
                ffn_block(
                    self.hidden_dim, self.ffn_dim, nonlinearity=self.nonlinearity
                ),
                alpha,
            )
            for _ in range(self.num_value_layers)
        ]
        layers.append(
            output_projection(self.hidden_dim, self.num_players * PREFLOP_HANDS)
        )
        self.value_head = nn.Sequential(*layers)

    def _hand_values_from_base(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        apply_zero_sum: bool = True,
    ) -> torch.Tensor:
        hand_values_raw = self.value_head(x).view(-1, self.num_players, PREFLOP_HANDS)
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            return hand_values_raw - hand_value_sums
        return hand_values_raw

    def _value_from_base(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        hand_values = self._hand_values_from_base(
            player_beliefs,
            x,
            apply_zero_sum=apply_zero_sum,
        )
        return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

    def forward_policy(self, features: MLPFeatures, latent=None) -> torch.Tensor:
        raise RuntimeError("BetterPreflopValueFFN does not provide policy outputs")

    def forward_value(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        value_head: str = "auto",
    ) -> ModelOutput:
        del latent, value_head
        if static_base_features is None:
            player_beliefs, _, x, _ = self._forward_base(features)
        else:
            player_beliefs, _, x, _ = self._forward_base_from_static(
                features, static_base_features
            )
        return self._value_from_base(player_beliefs, x, apply_zero_sum=apply_zero_sum)

    def forward_value_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        apply_zero_sum: bool = True,
        value_head: str = "auto",
    ) -> ModelOutput:
        return self.forward_value(
            features,
            latent=latent,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
        )

    def forward_hand_values_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        apply_zero_sum: bool = True,
        value_head: str = "auto",
    ) -> torch.Tensor:
        del latent, value_head
        player_beliefs, _, x, _ = self._forward_base_from_static(
            features,
            static_base_features,
        )
        return self._hand_values_from_base(
            player_beliefs,
            x,
            apply_zero_sum=apply_zero_sum,
        )

    def forward_both(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self.forward_value(
            features, latent=latent, apply_zero_sum=apply_zero_sum
        )

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = False,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        if include_policy:
            raise RuntimeError("BetterPreflopValueFFN does not provide policy outputs")
        if not include_value:
            raise ValueError("BetterPreflopValueFFN requires include_value=True")
        return self._call_forward_value(
            features,
            latent=latent,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
        )

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterPreflopValueFeatureEncoder:
        return BetterPreflopValueFeatureEncoder(
            env=env,
            device=device,
            dtype=dtype,
        )


class BetterPreflopPolicyFFN(_BetterPreflopCompactFFN):
    """Compact 169-hand preflop policy model for `S_preflop`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        policy_alpha = (
            1 / math.sqrt(self.num_hidden_layers + max(1, self.num_value_layers))
            if self.shared_trunk
            else 1 / math.sqrt(self.num_policy_layers)
        )
        self.policy_tower = nn.Sequential(
            *[
                ResidualBlock(
                    ffn_block(
                        self.hidden_dim, self.ffn_dim, nonlinearity=self.nonlinearity
                    ),
                    policy_alpha,
                )
                for _ in range(self.num_policy_layers)
            ]
        )
        self.policy_hand_proj = output_projection(
            self.range_summary_dim, self.policy_rank
        )
        self.policy_action_head = output_projection(
            self.hidden_dim, self.num_actions * self.policy_rank
        )
        self.policy_action_bias = output_projection(self.hidden_dim, self.num_actions)
        self.policy_hand_bias = output_projection(
            self.range_summary_dim, self.policy_hand_bias_rank
        )
        self.policy_hand_bias_action = output_projection(
            self.hidden_dim, self.num_actions * self.policy_hand_bias_rank
        )
        self.policy_hand_norm = nn.RMSNorm(self.range_summary_dim, eps=1e-5)

    def forward_policy(self, features: MLPFeatures, latent=None) -> torch.Tensor:
        player_beliefs, flat_features, x, hand_emb = self._forward_base(features)
        del player_beliefs
        policy_input = x if self.shared_trunk else flat_features.detach()
        policy_state = self.policy_tower(policy_input)
        action_emb = self.policy_action_head(policy_state).view(
            -1, self.num_actions, self.policy_rank
        )
        hand_vec = self.policy_hand_proj(self.policy_hand_norm(hand_emb))
        logits = torch.einsum("hr,bar->bha", hand_vec, action_emb)
        logits = logits / math.sqrt(self.policy_rank)
        hand_bias = self.policy_hand_bias(hand_emb)
        hand_bias_action = self.policy_hand_bias_action(policy_state).view(
            -1, self.num_actions, self.policy_hand_bias_rank
        )
        logits = logits + torch.einsum("hk,bak->bha", hand_bias, hand_bias_action)
        return logits + self.policy_action_bias(policy_state)[:, None, :]

    def forward_value(
        self, features: MLPFeatures, latent=None, **kwargs
    ) -> ModelOutput:
        raise RuntimeError("BetterPreflopPolicyFFN does not provide value outputs")

    def forward_both(self, features: MLPFeatures, latent=None, **kwargs) -> ModelOutput:
        return ModelOutput(policy_logits=self._call_forward_policy(features))

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = False,
        **kwargs,
    ) -> ModelOutput:
        if include_value:
            raise RuntimeError("BetterPreflopPolicyFFN does not provide value outputs")
        if not include_policy:
            raise ValueError("BetterPreflopPolicyFFN requires include_policy=True")
        return ModelOutput(policy_logits=self._call_forward_policy(features))

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterPreflopPolicyFeatureEncoder:
        return BetterPreflopPolicyFeatureEncoder(
            env=env,
            device=device,
            dtype=dtype,
        )


class BetterSplitFFN(BaseMLPModel):
    """Container exposing separate Better policy and street-value modules."""

    def __init__(
        self,
        policy_model: BetterPolicyFFN,
        value_model: BetterStreetValueFFN,
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model
        self.hidden_dim = policy_model.hidden_dim
        self.hand_dim = policy_model.hand_dim
        self.num_players = policy_model.num_players
        self.num_actions = policy_model.num_actions
        self.enforce_zero_sum = value_model.enforce_zero_sum
        if value_model.hand_dim != self.hand_dim:
            raise ValueError(
                f"split policy/value hand_dim mismatch: {self.hand_dim} vs {value_model.hand_dim}"
            )

    def init_weights(self, rng: torch.Generator | None = None) -> None:
        self.policy_model.init_weights(rng)
        self.value_model.init_weights(rng)

    def compile_forward_modes(self, **kwargs):
        """Compile split child fixed-mode forwards used by the wrapper hot path."""
        kwargs = dict(kwargs)
        compile_policy = bool(kwargs.pop("policy_compile", True))
        policy_dynamic = bool(
            kwargs.pop("policy_dynamic", kwargs.get("dynamic", False))
        )
        dynamic_batch = bool(kwargs.get("dynamic", False))
        policy_kwargs = dict(kwargs)
        policy_kwargs["dynamic"] = policy_dynamic
        policy_model = self.policy_model
        policy_model._compiled_forward_dynamic_batch = (
            policy_dynamic if compile_policy else False
        )
        policy_model._compiled_forward_policy_dynamic_batch = (
            policy_dynamic if compile_policy else False
        )
        if compile_policy:
            policy_ns = {"policy_model": policy_model}
            exec(
                "def policy_forward_features_only(features):\n"
                "    return policy_model.forward_policy(features)\n",
                policy_ns,
            )

            self.policy_model._compiled_forward_policy = torch.compile(
                policy_ns["policy_forward_features_only"],
                **policy_kwargs,
            )
        else:
            self.policy_model._compiled_forward_policy = None
        value_model = self.value_model
        value_model._compiled_forward_dynamic_batch = dynamic_batch
        value_model._compiled_forward_value_dynamic_batch = dynamic_batch
        value_model._compiled_forward_value_static_base_dynamic_batch = False
        value_model._compiled_forward_both_dynamic_batch = dynamic_batch
        value_ns = {"value_model": value_model}
        exec(
            "def value_forward(features, latent=None, apply_zero_sum=True, "
            "static_base_features=None, value_head='auto'):\n"
            "    return value_model.forward_value(\n"
            "        features,\n"
            "        latent=latent,\n"
            "        apply_zero_sum=apply_zero_sum,\n"
            "        static_base_features=static_base_features,\n"
            "        value_head=value_head,\n"
            "    )\n",
            value_ns,
        )
        exec(
            "def value_forward_both(features, latent=None, apply_zero_sum=True):\n"
            "    return value_model.forward_both(\n"
            "        features,\n"
            "        latent=latent,\n"
            "        apply_zero_sum=apply_zero_sum,\n"
            "    )\n",
            value_ns,
        )
        exec(
            "def value_forward_static_base(features, static_base_features, "
            "latent=None, apply_zero_sum=True, value_head='auto'):\n"
            "    return value_model.forward_value_static_base(\n"
            "        features,\n"
            "        static_base_features,\n"
            "        latent=latent,\n"
            "        apply_zero_sum=apply_zero_sum,\n"
            "        value_head=value_head,\n"
            "    )\n",
            value_ns,
        )
        self.value_model._compiled_forward_value = torch.compile(
            value_ns["value_forward"], **kwargs
        )
        self.value_model._compiled_forward_both = torch.compile(
            value_ns["value_forward_both"], **kwargs
        )
        self.value_model._compiled_forward_value_static_base = torch.compile(
            value_ns["value_forward_static_base"], **kwargs
        )
        return super().compile_forward_modes(
            **kwargs,
            policy_dynamic=policy_dynamic,
            policy_compile=compile_policy,
        )

    def forward_policy(self, features: MLPFeatures, latent=None) -> ModelOutput:
        return ModelOutput(
            policy_logits=self.policy_model._call_forward_policy(features)
        )

    def forward_value(
        self, features: MLPFeatures, latent=None, **kwargs
    ) -> ModelOutput:
        if kwargs.get("value_head", "auto") == "auto":
            return self._forward_value_auto_split(features, latent=latent, **kwargs)
        return self.value_model._call_forward_value(features, latent=latent, **kwargs)

    def forward_value_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        **kwargs,
    ) -> ModelOutput:
        if kwargs.get("value_head", "auto") == "auto":
            return self._forward_value_auto_split(
                features,
                static_base_features=static_base_features,
                latent=latent,
                **kwargs,
            )
        return self.value_model._call_forward_value_static_base(
            features,
            static_base_features,
            latent=latent,
            **kwargs,
        )

    def forward_both(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self(
            features,
            include_policy=True,
            include_value=True,
            apply_zero_sum=apply_zero_sum,
            latent=latent,
        )

    def forward_pre(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        return self.value_model.forward_pre(features, **kwargs)

    def forward_post(self, features: MLPFeatures, **kwargs) -> torch.Tensor:
        return self.value_model.forward_post(features, **kwargs)

    def _forward_value_auto_split(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        value_head: str = "auto",
        turn_range_equity_board_cache: TurnRangeEquityBoardCache | None = None,
    ) -> ModelOutput:
        if value_head == "auto":
            return self.value_model.forward_value(
                features,
                latent=latent,
                apply_zero_sum=apply_zero_sum,
                static_base_features=static_base_features,
                value_head=value_head,
                turn_range_equity_board_cache=turn_range_equity_board_cache,
            )
        return self.value_model._call_forward_value(
            features,
            latent=latent,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
            turn_range_equity_board_cache=turn_range_equity_board_cache,
        )

    def static_feature_base(self, features: MLPFeatures) -> torch.Tensor:
        return self.value_model.static_feature_base(features)

    def static_feature_prefix(
        self, context: torch.Tensor, street: torch.Tensor
    ) -> torch.Tensor:
        return self.value_model.static_feature_prefix(context, street)

    def static_feature_base_from_prefix(
        self, prefix: torch.Tensor, board: torch.Tensor
    ) -> torch.Tensor:
        return self.value_model.static_feature_base_from_prefix(prefix, board)

    def prepare_preflop_eval_cache(self) -> None:
        for child in (self.policy_model, self.value_model):
            prepare = getattr(child, "prepare_preflop_eval_cache", None)
            if prepare is not None:
                prepare()

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        apply_zero_sum: bool = True,
        static_base_features: torch.Tensor | None = None,
        latent=None,
        value_head: str = "auto",
    ) -> ModelOutput:
        policy_logits = None
        value = None
        hand_values = None
        if include_policy:
            policy_logits = self.policy_model._call_forward_policy(features)
        if include_value:
            if value_head == "auto":
                value_output = self._forward_value_auto_split(
                    features,
                    latent=latent,
                    apply_zero_sum=apply_zero_sum,
                    static_base_features=static_base_features,
                )
            else:
                value_output = self.value_model._call_forward_value(
                    features,
                    latent=latent,
                    apply_zero_sum=apply_zero_sum,
                    static_base_features=static_base_features,
                    value_head=value_head,
                )
            value = value_output.value
            hand_values = value_output.hand_values
        if not include_policy and not include_value:
            raise ValueError(
                "At least one of include_policy/include_value must be true"
            )
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
        )

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterPolicyFeatureEncoder:
        return self.policy_model.create_feature_encoder(env, device=device, dtype=dtype)

    def repeat(
        self,
        features: MLPFeatures,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        return self(
            features,
            include_policy=include_policy,
            include_value=include_value,
            apply_zero_sum=apply_zero_sum,
        )
