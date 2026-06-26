from __future__ import annotations

import math
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
    ValueScalarContext,
    context_length,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.utils.profiling import profile


HAND_STATIC_FEATURE_DIM = 8
HAND_DYNAMIC_FEATURE_DIM = 15


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


def _preflop_token_mixer_gate_residual_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
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
    block_b = 8
    block_d = 32
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
        num_value_layers: int = 3,
        num_players: int = 2,
        shared_trunk: bool = True,
        enforce_zero_sum: bool = True,
        board_interaction_dim: int = 0,
        policy_rank: int = 64,
        policy_hand_bias_rank: int = 32,
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
        self.policy_rank = policy_rank
        self.policy_hand_bias_rank = policy_hand_bias_rank
        self.nonlinearity = nonlinearity

        if range_hidden_dim < 0:
            raise ValueError("range_hidden_dim must be non-negative")
        if board_interaction_dim < 0:
            raise ValueError("board_interaction_dim must be non-negative")
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
        self.register_buffer("hand_card_a", combos[:, 0].long(), persistent=False)
        self.register_buffer("hand_card_b", combos[:, 1].long(), persistent=False)
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
        belief_in_dim = num_players * hidden_dim
        belief_hidden_dim = num_players * effective_range_hidden_dim
        self.hand_feature_proj = nn.Linear(
            HAND_STATIC_FEATURE_DIM, hidden_dim, bias=False
        )
        self.belief_proj = ffn_block(
            belief_in_dim, belief_hidden_dim, hidden_dim, nonlinearity
        )
        context_in_dim = context_length(num_players)
        self.context_encoder = ffn_block(
            context_in_dim, hidden_dim, hidden_dim, nonlinearity
        )
        if board_interaction_dim > 0:
            self.rank_pair_low_embedding = nn.Embedding(91, board_interaction_dim)
            self.board_rank_low = nn.Linear(13, board_interaction_dim, bias=False)
            self.rank_board_interaction_out = nn.Linear(
                num_players * board_interaction_dim, hidden_dim, bias=False
            )
            self.suit_pair_low_embedding = nn.Embedding(10, board_interaction_dim)
            self.board_suit_low = nn.Linear(4, board_interaction_dim, bias=False)
            self.suit_board_interaction_out = nn.Linear(
                num_players * board_interaction_dim, hidden_dim, bias=False
            )

        # Build trunk
        # Default alpha is always based on hidden + value layers
        alpha = 1 / math.sqrt(num_hidden_layers + num_value_layers)
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

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), alpha
            )
            for _ in range(num_value_layers)
        ]
        layers.append(output_projection(hidden_dim, num_players * NUM_HANDS))
        self.hand_value_head = nn.Sequential(*layers)

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

    def _hand_embedding(self) -> torch.Tensor:
        """Per-hand exact-card embedding — shape [NUM_HANDS, hidden_dim]."""
        card_emb = self.card_embedding(self.hand_combos)
        static = self.hand_static_features.to(dtype=card_emb.dtype)
        return card_emb.sum(dim=1) + self.hand_feature_proj(static)

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
        return self.hand_value_head(value_input).view(-1, self.num_players, NUM_HANDS)

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
        logits = torch.einsum("hr,bar->bha", hand_vec, action_emb)
        logits = logits / math.sqrt(self.policy_rank)
        hand_bias = self.policy_hand_bias(hand_emb)
        hand_bias_action = self.policy_hand_bias_action(policy_state).view(
            -1, self.num_actions, self.policy_hand_bias_rank
        )
        logits = logits + torch.einsum("hk,bak->bha", hand_bias, hand_bias_action)

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
        rank_features = self.rank_board_interaction_out(
            (rank_pair_low * board_rank_low[:, None, :]).flatten(1)
        )

        suit_pair_mass = player_beliefs @ self.hand_suit_pair_one_hot.to(
            dtype=player_beliefs.dtype
        )
        suit_pair_low = suit_pair_mass @ self.suit_pair_low_embedding.weight
        board_suit_low = self.board_suit_low(board_suit_counts)
        suit_features = self.suit_board_interaction_out(
            (suit_pair_low * board_suit_low[:, None, :]).flatten(1)
        )

        return rank_features + suit_features

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
        return self.street_embedding(street) + self.context_encoder(context)

    def static_feature_base_from_prefix(
        self, prefix: torch.Tensor, board: torch.Tensor
    ) -> torch.Tensor:
        """Add board features to a precomputed context/street prefix."""
        ranks = torch.where(board >= 0, board % 13, torch.full_like(board, 13))
        suits = torch.where(board >= 0, board // 13, torch.full_like(board, 4))
        board_features = self.rank_embedding(ranks) + self.suit_embedding(suits)
        return board_features.sum(dim=1) + prefix

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
        hand_emb = self._hand_embedding()  # [NUM_HANDS, hidden_dim]
        per_player_belief = player_beliefs @ hand_emb  # [B, P, H]
        belief_features = self.belief_proj(per_player_belief.flatten(1))

        flat_features = static_base_features + belief_features
        board_stats = self._board_stats(features.board, player_beliefs.dtype)
        interaction_features = self._belief_board_interaction(
            player_beliefs, board_stats
        )
        if interaction_features is not None:
            flat_features = flat_features + interaction_features
        # assert flat_features.isfinite().all()

        x = self.trunk(flat_features)
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
        policy_input = x if self.shared_trunk else flat_features.detach()
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
    ) -> ModelOutput:
        """
        Value-only pass.

        apply_zero_sum controls where the zero-sum projection is applied, not
        whether it is required. If ``enforce_zero_sum`` is false this flag has no
        effect; if it is true and this flag is false, the caller must apply the
        projection after any value mixing.
        """
        player_beliefs, _, x, hand_emb, board_stats = self._forward_base(features)
        del hand_emb, board_stats
        return self._value_from_base(player_beliefs, x, apply_zero_sum=apply_zero_sum)

    def forward_value_static_base(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor,
        latent=None,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        """Value-only pass for callers that precomputed static public features."""
        player_beliefs, _, x, hand_emb, board_stats = self._forward_base_from_static(
            features, static_base_features=static_base_features
        )
        del hand_emb, board_stats
        return self._value_from_base(player_beliefs, x, apply_zero_sum=apply_zero_sum)

    def _value_from_base(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        hand_values_raw = self._hand_value_logits(x)
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            hand_values = hand_values_raw - hand_value_sums
        else:
            hand_values = hand_values_raw
        value = hand_values.mean(dim=-1)
        return ModelOutput(value=value, hand_values=hand_values)

    def forward_both(
        self,
        features: MLPFeatures,
        latent=None,
        apply_zero_sum: bool = True,
    ) -> ModelOutput:
        player_beliefs, flat_features, x, hand_emb, board_stats = self._forward_base(
            features
        )
        policy_input = x if self.shared_trunk else flat_features.detach()
        policy_logits = self._policy_logits(
            policy_input,
            player_beliefs,
            features.to_act,
            features.board,
            hand_emb,
            board_stats,
        )
        hand_values_raw = self._hand_value_logits(x)
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            hand_values = hand_values_raw - hand_value_sums
        else:
            hand_values = hand_values_raw
        value = hand_values.mean(dim=-1)
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
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
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

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

        # Guess hand values are around stddev 0.1.
        for head_name in ("hand_value_head", "pre_value_head", "post_value_head"):
            head = getattr(self, head_name, None)
            if head is not None:
                head[-1].get_submodule("linear_out").weight.data.mul_(0.1)
        if self.board_interaction_dim > 0:
            self.rank_board_interaction_out.weight.data.mul_(0.1)
            self.suit_board_interaction_out.weight.data.mul_(0.1)
        if hasattr(self, "belief_phase_shift"):
            nn.init.zeros_(self.belief_phase_shift.weight)

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
        policy_input = x if self.shared_trunk else flat_features.detach()
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
            5 * 2, self.num_players * self.hidden_dim
        )

    def _make_value_head(self) -> nn.Sequential:
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
        hand_emb = self._hand_embedding()
        per_player_belief = player_beliefs @ hand_emb
        phase_shift = self.belief_phase_shift(self._phase_key(features)).view(
            -1, self.num_players, self.hidden_dim
        )
        per_player_belief = per_player_belief + phase_shift
        belief_features = self.belief_proj(per_player_belief.flatten(1))

        flat_features = static_base_features + belief_features
        board_stats = self._board_stats(features.board, player_beliefs.dtype)
        interaction_features = self._belief_board_interaction(
            player_beliefs, board_stats
        )
        if interaction_features is not None:
            flat_features = flat_features + interaction_features

        x = self.trunk(flat_features)
        return player_beliefs, flat_features, x, hand_emb, board_stats

    def _hand_value_logits_from_head(
        self, value_input: torch.Tensor, head: nn.Module
    ) -> torch.Tensor:
        return head(value_input).view(-1, self.num_players, NUM_HANDS)

    def _value_tensor_from_base(
        self,
        player_beliefs: torch.Tensor,
        x: torch.Tensor,
        head: nn.Module,
        apply_zero_sum: bool = True,
    ) -> torch.Tensor:
        hand_values_raw = self._hand_value_logits_from_head(x, head)
        if self.enforce_zero_sum and apply_zero_sum:
            hand_value_sums = (
                (hand_values_raw * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            return hand_values_raw - hand_value_sums
        return hand_values_raw

    def _forward_value_head(
        self,
        features: MLPFeatures,
        head: nn.Module,
        static_base_features: torch.Tensor | None = None,
        apply_zero_sum: bool = True,
    ) -> torch.Tensor:
        if static_base_features is None:
            player_beliefs, _, x, _, _ = self._forward_base(features)
        else:
            player_beliefs, _, x, _, _ = self._forward_base_from_static(
                features, static_base_features=static_base_features
            )
        return self._value_tensor_from_base(
            player_beliefs, x, head, apply_zero_sum=apply_zero_sum
        )

    def forward_pre(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor | None = None,
        apply_zero_sum: bool = True,
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
        )

    def forward_post(
        self,
        features: MLPFeatures,
        static_base_features: torch.Tensor | None = None,
        apply_zero_sum: bool = True,
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
    ) -> ModelOutput:
        if value_head == "pre":
            hand_values = self.forward_pre(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)
        if value_head == "post":
            hand_values = self.forward_post(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)
        if value_head != "auto":
            raise ValueError("value_head must be one of: auto, pre, post")
        if self.value_heads == "pre":
            hand_values = self.forward_pre(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)
        if self.value_heads == "post":
            hand_values = self.forward_post(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
            )
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

        phase = features.context[:, ValueScalarContext.CHANCE_PHASE.value]
        pre_mask = (phase >= 0.5).view(-1, 1, 1)
        if torch.compiler.is_compiling() or _is_cuda_graph_capturing(features.context):
            pre = self.forward_pre(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
            )
            post = self.forward_post(
                features,
                static_base_features=static_base_features,
                apply_zero_sum=apply_zero_sum,
            )
            hand_values = torch.where(pre_mask, pre, post)
            return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

        if static_base_features is None:
            player_beliefs, _, x, _, _ = self._forward_base(features)
        else:
            player_beliefs, _, x, _, _ = self._forward_base_from_static(
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
                self.pre_value_head,
                apply_zero_sum=apply_zero_sum,
            )
        if post_rows.numel() > 0:
            hand_values[post_rows] = self._value_tensor_from_base(
                player_beliefs[post_rows],
                x[post_rows],
                self.post_value_head,
                apply_zero_sum=apply_zero_sum,
            )
        return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x)
        linear_in = self.token_mixer.linear_in
        activation = self.token_mixer.activation
        linear_out = self.token_mixer.linear_out
        gate = self.token_gate(y)
        if (
            x.is_cuda
            and isinstance(activation, nn.LeakyReLU)
            and activation.negative_slope == 0.01
            and x.shape[1] == 7
            and linear_in.weight.shape == (28, 7)
            and linear_out.weight.shape == (7, 28)
        ):
            x = _preflop_token_mixer_gate_residual_triton(
                x,
                y,
                gate,
                linear_in.weight,
                linear_out.weight,
            )
        else:
            mixed = self.token_mixer(y.transpose(1, 2)).transpose(1, 2)
            x = x + mixed * torch.sigmoid(gate) / math.sqrt(2.0)
        return x + self.ffn(x) / math.sqrt(2.0)


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

        self.scalar_context_dim = context_length(num_players) - num_players * 13
        self.player_context_dim = 13
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
        return super().train(mode)

    def _hand_embedding(self) -> torch.Tensor:
        hand_static = self._class_static_features().to(
            device=self.class_hi_rank.device,
            dtype=self.hand_encoder[0].weight.dtype,
        )
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
        static_game_token: torch.Tensor | None = None,
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
        hand_emb = self._hand_embedding()
        dtype = hand_emb.dtype
        game_token = None
        static_player_tokens = None
        if static_game_token is not None:
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
            else:
                raise ValueError(
                    "static preflop features must be either a game-token tensor "
                    "or a cached token tensor"
                )
        player_context = None
        player_beliefs = player_beliefs.to(dtype)
        bucket_projection = self.preflop_bucket_projection.to(
            device=player_beliefs.device,
            dtype=dtype,
        )
        combined_projection = torch.cat((hand_emb, hand_emb.square()), dim=-1)
        range_projection = combined_projection.matmul(self.range_proj.weight.t())
        bucket_mass = player_beliefs @ bucket_projection
        if static_player_tokens is None:
            _, player_context = self._split_context(features.context)
            static_player_tokens = self.player_context_proj(player_context.to(dtype))
        player_tokens = (
            player_beliefs @ range_projection
            + self.bucket_mass_proj(bucket_mass)
            + static_player_tokens
        )
        if self.range_slot_moment_pool is not None:
            if player_context is None:
                _, player_context = self._split_context(features.context)
            player_tokens = player_tokens + self.range_slot_moment_pool(
                player_beliefs,
                hand_static,
                player_context.to(dtype),
            )
        if game_token is None:
            game_token = self.static_feature_prefix(
                features.context, features.street
            ).to(dtype)
        encoded = torch.cat((game_token[:, None, :], player_tokens), dim=1)
        for block in self.encoder:
            encoded = block(encoded)
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
        player_beliefs, encoded, hand_emb = self._encode_base_tokens(
            features, static_game_token=static_game_token
        )
        if extra_encoder is not None:
            for block in extra_encoder:
                encoded = block(encoded)
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

    def _hand_values_from_tokens(
        self,
        player_beliefs: torch.Tensor,
        player_state: torch.Tensor,
        hand_emb: torch.Tensor,
        apply_zero_sum: bool = True,
    ) -> torch.Tensor:
        hand_value = self.value_hand_proj(hand_emb)
        combined_weight = self.value_scale.weight.t().matmul(hand_value.t())
        combined_weight = combined_weight / math.sqrt(float(self.hidden_dim))
        value_bias_weight = self.value_bias.weight.t().to(dtype=combined_weight.dtype)
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
        for block in self.policy_encoder:
            encoded = block(encoded)
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
        self.policy_model._compiled_forward_policy = torch.compile(
            self.policy_model.forward_policy, **kwargs
        )
        self.value_model._compiled_forward_value = torch.compile(
            self.value_model.forward_value, **kwargs
        )
        self.value_model._compiled_forward_both = torch.compile(
            self.value_model.forward_both, **kwargs
        )
        self.value_model._compiled_forward_value_static_base = torch.compile(
            self.value_model.forward_value_static_base, **kwargs
        )
        return super().compile_forward_modes(**kwargs)

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
    ) -> ModelOutput:
        if value_head == "auto":
            return self.value_model.forward_value(
                features,
                latent=latent,
                apply_zero_sum=apply_zero_sum,
                static_base_features=static_base_features,
                value_head=value_head,
            )
        return self.value_model._call_forward_value(
            features,
            latent=latent,
            apply_zero_sum=apply_zero_sum,
            static_base_features=static_base_features,
            value_head=value_head,
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
