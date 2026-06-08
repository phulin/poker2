from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    hand_combos_tensor,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
)


HAND_FEATURE_DIM = 8
PLAYER_FEATURE_DIM = 8
POT_LAYER_SCALAR_FEATURE_DIM = 7
RANGE_BUCKET_FEATURE_DIM = 20
PREFLOP_169_BUCKET_FEATURE_DIM = 16
OUTPUT_HEAD_INIT_SCALE = 1.464
OUTPUT_HEAD_INIT_BIAS = 0.382


def _preflop_unblocked_projection() -> torch.Tensor:
    multiplicity = preflop_class_multiplicity_tensor()
    compatibility = preflop_class_compatibility_counts_tensor()
    return (compatibility.T * multiplicity.reciprocal()[:, None]).contiguous()


def _preflop_bucket_projection() -> torch.Tensor:
    class_ids = torch.arange(PREFLOP_HANDS)
    row = torch.div(class_ids, 13, rounding_mode="floor")
    col = class_ids.remainder(13)
    hi = torch.maximum(row, col)
    lo = torch.minimum(row, col)
    pair = row == col
    suited = row > col
    projection = torch.zeros(PREFLOP_HANDS, PREFLOP_169_BUCKET_FEATURE_DIM)
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


class _LeakyRMSBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, negative_slope: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.up(y)
        y = self.activation(y)
        y = self.down(y)
        return x + y / math.sqrt(2.0)


class _TokenEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        hidden_dim: int,
        negative_slope: float,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.attn_norm = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = nn.RMSNorm(dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, token_count, dim = x.shape
        qkv = self.qkv(self.attn_norm(x)).view(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask)
        attn = attn.transpose(1, 2).reshape(batch_size, token_count, dim)
        x = x + self.out(attn) / math.sqrt(2.0)
        y = self.ffn_norm(x)
        y = self.up(y)
        y = self.activation(y)
        y = self.down(y)
        return x + y / math.sqrt(2.0)


class PreflopAllInEquityModel(nn.Module):
    """Browser-friendly preflop all-in terminal value model.

    The model predicts chip-normalized all-in values for every player and every
    private hand. It intentionally uses only Linear, RMSNorm, LeakyReLU, basic
    tensor ops, and fixed buffers so it can be ported to browser runtimes. The
    output head includes an optional low-rank FiLM residual controlled by
    ``film_rank``.
    """

    def __init__(
        self,
        players: int = 4,
        hidden_dim: int = 512,
        hand_dim: int = 128,
        num_layers: int = 4,
        film_rank: int = 64,
        dense_belief_residual: bool = False,
        dense_output_residual: bool = False,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if players < 2:
            raise ValueError("players must be at least 2")
        if film_rank < 0:
            raise ValueError("film_rank must be non-negative")
        self.players = int(players)
        self.hidden_dim = int(hidden_dim)
        self.hand_dim = int(hand_dim)
        self.num_layers = int(num_layers)
        self.film_rank = int(film_rank)
        self.dense_belief_residual = bool(dense_belief_residual)
        self.dense_output_residual = bool(dense_output_residual)
        self.negative_slope = float(negative_slope)

        combos = hand_combos_tensor()
        ranks = combos % 13
        suits = combos // 13
        self.register_buffer("hand_features", self._hand_features(ranks, suits))
        self.register_buffer("hand_card_a", combos[:, 0].long(), persistent=False)
        self.register_buffer("hand_card_b", combos[:, 1].long(), persistent=False)
        self.register_buffer("hand_rank_a", ranks[:, 0].long(), persistent=False)
        self.register_buffer("hand_rank_b", ranks[:, 1].long(), persistent=False)
        self.register_buffer("hand_suit_a", suits[:, 0].long(), persistent=False)
        self.register_buffer("hand_suit_b", suits[:, 1].long(), persistent=False)

        self.hand_encoder = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hand_dim, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.range_proj = nn.Linear(hand_dim * 2, hidden_dim, bias=False)
        self.card_mass_proj = nn.Sequential(
            nn.Linear(52, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.bucket_mass_proj = nn.Sequential(
            nn.Linear(RANGE_BUCKET_FEATURE_DIM, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.player_proj = nn.Sequential(
            nn.Linear(PLAYER_FEATURE_DIM, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        if self.dense_belief_residual:
            self.dense_belief_proj = nn.Sequential(
                nn.Linear(players * NUM_HANDS, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
        self.input_proj = nn.Sequential(
            nn.Linear(players * hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.trunk = nn.Sequential(
            *[
                _LeakyRMSBlock(hidden_dim, hidden_dim * 2, negative_slope)
                for _ in range(num_layers)
            ]
        )
        self.player_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.value_hand_proj = nn.Linear(hand_dim, hidden_dim, bias=False)
        self.value_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.blocker_proj = nn.Linear(players, hidden_dim, bias=False)
        self.blocker_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_bias = nn.Linear(hidden_dim, 1)
        if self.dense_output_residual:
            self.dense_output_proj = nn.Linear(hidden_dim, NUM_HANDS)
        if self.film_rank > 0:
            self.value_film_hand_proj = nn.Linear(hand_dim, self.film_rank, bias=False)
            self.value_film_state = nn.Linear(hidden_dim, self.film_rank, bias=False)

    @staticmethod
    def _hand_features(ranks: torch.Tensor, suits: torch.Tensor) -> torch.Tensor:
        rank_a = ranks[:, 0].to(torch.float32)
        rank_b = ranks[:, 1].to(torch.float32)
        suit_a = suits[:, 0]
        suit_b = suits[:, 1]
        hi = torch.maximum(rank_a, rank_b)
        lo = torch.minimum(rank_a, rank_b)
        gap = (hi - lo).clamp_min(0.0)
        return torch.stack(
            (
                (rank_a == rank_b).to(torch.float32),
                (suit_a == suit_b).to(torch.float32),
                gap / 12.0,
                hi / 12.0,
                lo / 12.0,
                (hi == 12).to(torch.float32),
                (lo >= 8).to(torch.float32),
                (gap <= 1).to(torch.float32),
            ),
            dim=-1,
        )

    @staticmethod
    def _max_eligible_to_win(
        committed: torch.Tensor,
        folded_mask: torch.Tensor,
    ) -> torch.Tensor:
        levels = committed.sort(dim=1).values
        previous = torch.cat((torch.zeros_like(levels[:, :1]), levels[:, :-1]), dim=1)
        widths = (levels - previous).clamp_min(0.0)
        participants = committed[:, None, :] >= levels[:, :, None]
        layer_amount = widths * participants.to(committed.dtype).sum(dim=2)
        eligible = participants & (~folded_mask[:, None, :])
        return (layer_amount[:, :, None] * eligible.to(committed.dtype)).sum(dim=1)

    def _player_scalar_features(
        self,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        allin_mask: torch.Tensor,
        folded_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = starting_stacks.mean(dim=1, keepdim=True).clamp_min(1.0)
        live = (~folded_mask).to(beliefs.dtype)
        allin = allin_mask.to(beliefs.dtype)
        folded = folded_mask.to(beliefs.dtype)
        pot = committed.sum(dim=1, keepdim=True) / scale
        max_eligible_to_win = (
            self._max_eligible_to_win(
                committed,
                folded_mask,
            )
            / scale
        )
        return torch.stack(
            (
                starting_stacks / scale,
                committed / scale,
                stacks_after / scale,
                allin,
                folded,
                live,
                pot.expand(-1, self.players),
                max_eligible_to_win,
            ),
            dim=-1,
        )

    def _range_mass_features(
        self, beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, players, _ = beliefs.shape
        dtype = beliefs.dtype
        card_mass = torch.zeros(
            batch_size, players, 52, device=beliefs.device, dtype=dtype
        )
        card_a = self.hand_card_a.to(beliefs.device)
        card_b = self.hand_card_b.to(beliefs.device)
        card_mass.scatter_add_(
            2,
            card_a[None, None, :].expand(batch_size, players, -1),
            beliefs,
        )
        card_mass.scatter_add_(
            2,
            card_b[None, None, :].expand(batch_size, players, -1),
            beliefs,
        )

        rank_mass = torch.zeros(
            batch_size, players, 13, device=beliefs.device, dtype=dtype
        )
        rank_a = self.hand_rank_a.to(beliefs.device)
        rank_b = self.hand_rank_b.to(beliefs.device)
        rank_mass.scatter_add_(
            2,
            rank_a[None, None, :].expand(batch_size, players, -1),
            beliefs,
        )
        rank_mass.scatter_add_(
            2,
            rank_b[None, None, :].expand(batch_size, players, -1),
            beliefs,
        )

        suit_mass = torch.zeros(
            batch_size, players, 4, device=beliefs.device, dtype=dtype
        )
        suit_a = self.hand_suit_a.to(beliefs.device)
        suit_b = self.hand_suit_b.to(beliefs.device)
        suit_mass.scatter_add_(
            2,
            suit_a[None, None, :].expand(batch_size, players, -1),
            beliefs,
        )
        suit_mass.scatter_add_(
            2,
            suit_b[None, None, :].expand(batch_size, players, -1),
            beliefs,
        )

        pocket_pair_mass = beliefs[..., self.hand_rank_a == self.hand_rank_b].sum(
            dim=-1, keepdim=True
        )
        suited_mass = beliefs[..., self.hand_suit_a == self.hand_suit_b].sum(
            dim=-1, keepdim=True
        )
        ace_mass = beliefs[
            ..., (self.hand_rank_a == 12) | (self.hand_rank_b == 12)
        ].sum(dim=-1, keepdim=True)
        bucket_features = torch.cat(
            (rank_mass, suit_mass, pocket_pair_mass, suited_mass, ace_mass),
            dim=-1,
        )
        return card_mass, bucket_features

    def _blocker_features(
        self,
        beliefs: torch.Tensor,
        card_mass: torch.Tensor,
        folded_mask: torch.Tensor,
    ) -> torch.Tensor:
        card_a = self.hand_card_a.to(beliefs.device)
        card_b = self.hand_card_b.to(beliefs.device)
        blocked = (
            1.0 - card_mass[:, :, card_a] - card_mass[:, :, card_b] + beliefs
        ).clamp_min(0.0)
        players = beliefs.shape[1]
        player_ids = torch.arange(players, device=beliefs.device)
        non_self = player_ids[None, :, None] != player_ids[None, None, :]
        live_opp = (~folded_mask).to(beliefs.dtype)[:, None, :]
        return (
            blocked[:, None, :, :]
            * non_self[:, :, :, None].to(beliefs.dtype)
            * live_opp[:, :, :, None]
        ).permute(0, 1, 3, 2)

    def forward(
        self,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        allin_mask: torch.Tensor,
        folded_mask: torch.Tensor,
        *,
        hardcode_folded_values: bool = True,
    ) -> torch.Tensor:
        if beliefs.shape[1] != self.players:
            raise ValueError(f"expected {self.players} players, got {beliefs.shape[1]}")
        scale = starting_stacks.mean(dim=1, keepdim=True).clamp_min(1.0)
        player_features = self._player_scalar_features(
            beliefs,
            starting_stacks,
            committed,
            stacks_after,
            allin_mask,
            folded_mask,
        )

        hand_emb = self.hand_encoder(
            self.hand_features.to(beliefs.device, beliefs.dtype)
        )
        beliefs_f = beliefs.to(hand_emb.dtype)
        card_mass, bucket_features = self._range_mass_features(beliefs_f)
        range_summary = torch.cat(
            (
                beliefs_f @ hand_emb,
                beliefs_f @ hand_emb.square(),
            ),
            dim=-1,
        )
        per_player = (
            self.range_proj(range_summary)
            + self.card_mass_proj(card_mass)
            + self.bucket_mass_proj(bucket_features)
            + self.player_proj(player_features)
        )
        global_state = self.input_proj(per_player.flatten(1))
        if self.dense_belief_residual:
            global_state = global_state + self.dense_belief_proj(beliefs_f.flatten(1))
        global_state = self.trunk(global_state)
        state = self.player_state(
            torch.cat(
                [
                    per_player,
                    global_state[:, None, :].expand(-1, self.players, -1),
                ],
                dim=-1,
            )
        )
        hand_value = self.value_hand_proj(hand_emb)
        state_value = self.value_scale(state)
        values = torch.einsum("bpd,hd->bph", state_value, hand_value)
        values = values / math.sqrt(float(self.hidden_dim))
        blocker_features = self._blocker_features(beliefs_f, card_mass, folded_mask)
        blocker_state = self.blocker_proj(blocker_features)
        blocker_scale = self.blocker_scale(state)
        blocker_values = torch.einsum("bphd,bpd->bph", blocker_state, blocker_scale)
        values = values + blocker_values / math.sqrt(float(self.hidden_dim))
        if self.film_rank > 0:
            film_hand = self.value_film_hand_proj(hand_emb)
            film_state = self.value_film_state(state)
            film_values = torch.einsum("bpr,hr->bph", film_state, film_hand)
            values = values + film_values / math.sqrt(float(self.film_rank))
        values = values + self.value_bias(state).expand(-1, -1, NUM_HANDS)
        if self.dense_output_residual:
            values = values + self.dense_output_proj(state)
        if hardcode_folded_values:
            folded_value = (stacks_after - starting_stacks) / scale
            values = torch.where(
                folded_mask[:, :, None],
                folded_value[:, :, None],
                values,
            )
        return values.to(beliefs.dtype)

    def init_weights(self, generator: torch.Generator | None = None) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        self.value_scale.weight.data.mul_(OUTPUT_HEAD_INIT_SCALE)
        self.blocker_scale.weight.data.mul_(OUTPUT_HEAD_INIT_SCALE)
        self.value_bias.weight.data.mul_(OUTPUT_HEAD_INIT_SCALE)
        self.value_bias.bias.data.fill_(OUTPUT_HEAD_INIT_BIAS)
        if self.film_rank > 0:
            self.value_film_state.weight.data.zero_()
        if self.dense_output_residual:
            self.dense_output_proj.weight.data.zero_()
            self.dense_output_proj.bias.data.zero_()


class PreflopAllIn169EquityModel(nn.Module):
    """Native rank-class preflop all-in model.

    The public model boundary is hand-major compact tensors:
    ``beliefs`` is ``[batch, 169, players]`` and the return value has the same
    shape. No combo expansion is performed inside the model.
    """

    def __init__(
        self,
        players: int = 4,
        hidden_dim: int = 512,
        hand_dim: int = 128,
        num_layers: int = 4,
        film_rank: int = 64,
        dense_belief_residual: bool = False,
        dense_output_residual: bool = False,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if players < 2:
            raise ValueError("players must be at least 2")
        if film_rank < 0:
            raise ValueError("film_rank must be non-negative")
        self.players = int(players)
        self.hidden_dim = int(hidden_dim)
        self.hand_dim = int(hand_dim)
        self.num_layers = int(num_layers)
        self.film_rank = int(film_rank)
        self.dense_belief_residual = bool(dense_belief_residual)
        self.dense_output_residual = bool(dense_output_residual)
        self.negative_slope = float(negative_slope)

        class_ids = torch.arange(PREFLOP_HANDS)
        row = torch.div(class_ids, 13, rounding_mode="floor")
        col = class_ids.remainder(13)
        hi = torch.maximum(row, col)
        lo = torch.minimum(row, col)
        pair = row == col
        suited = row > col
        self.register_buffer(
            "hand_features",
            self._preflop_class_features(hi, lo, pair, suited),
        )
        self.register_buffer("hand_hi_rank", hi.long(), persistent=False)
        self.register_buffer("hand_lo_rank", lo.long(), persistent=False)
        self.register_buffer("hand_pair", pair, persistent=False)
        self.register_buffer("hand_suited", suited, persistent=False)
        self.register_buffer(
            "preflop_unblocked_projection",
            _preflop_unblocked_projection(),
            persistent=False,
        )
        self.register_buffer(
            "preflop_bucket_projection",
            _preflop_bucket_projection(),
            persistent=False,
        )

        self.hand_encoder = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hand_dim, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.range_proj = nn.Linear(hand_dim * 2, hidden_dim, bias=False)
        self.bucket_mass_proj = nn.Sequential(
            nn.Linear(PREFLOP_169_BUCKET_FEATURE_DIM, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.player_proj = nn.Sequential(
            nn.Linear(PLAYER_FEATURE_DIM, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        if self.dense_belief_residual:
            self.dense_belief_proj = nn.Sequential(
                nn.Linear(players * PREFLOP_HANDS, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
        self.input_proj = nn.Sequential(
            nn.Linear(players * hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.trunk = nn.Sequential(
            *[
                _LeakyRMSBlock(hidden_dim, hidden_dim * 2, negative_slope)
                for _ in range(num_layers)
            ]
        )
        self.player_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.value_hand_proj = nn.Linear(hand_dim, hidden_dim, bias=False)
        self.value_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.blocker_proj = nn.Linear(players, hidden_dim, bias=False)
        self.blocker_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_bias = nn.Linear(hidden_dim, 1)
        if self.dense_output_residual:
            self.dense_output_proj = nn.Linear(hidden_dim, PREFLOP_HANDS)
        if self.film_rank > 0:
            self.value_film_hand_proj = nn.Linear(hand_dim, self.film_rank, bias=False)
            self.value_film_state = nn.Linear(hidden_dim, self.film_rank, bias=False)

    _max_eligible_to_win = staticmethod(PreflopAllInEquityModel._max_eligible_to_win)

    @staticmethod
    def _preflop_class_features(
        hi: torch.Tensor,
        lo: torch.Tensor,
        pair: torch.Tensor,
        suited: torch.Tensor,
    ) -> torch.Tensor:
        hi_f = hi.to(torch.float32)
        lo_f = lo.to(torch.float32)
        gap = (hi_f - lo_f).clamp_min(0.0)
        return torch.stack(
            (
                pair.to(torch.float32),
                suited.to(torch.float32),
                gap / 12.0,
                hi_f / 12.0,
                lo_f / 12.0,
                (hi == 12).to(torch.float32),
                (lo >= 8).to(torch.float32),
                (gap <= 1).to(torch.float32),
            ),
            dim=-1,
        )

    def _range_mass_features(self, beliefs: torch.Tensor) -> torch.Tensor:
        projection = self.preflop_bucket_projection.to(
            device=beliefs.device,
            dtype=beliefs.dtype,
        )
        return beliefs @ projection

    def _blocker_features(
        self,
        beliefs: torch.Tensor,
        folded_mask: torch.Tensor,
    ) -> torch.Tensor:
        projection = self.preflop_unblocked_projection.to(
            device=beliefs.device,
            dtype=beliefs.dtype,
        )
        unblocked = beliefs @ projection
        player_ids = torch.arange(self.players, device=beliefs.device)
        non_self = player_ids[None, :, None] != player_ids[None, None, :]
        live_opp = (~folded_mask).to(beliefs.dtype)[:, None, :]
        return (
            unblocked[:, None, :, :]
            * non_self[:, :, :, None].to(beliefs.dtype)
            * live_opp[:, :, :, None]
        ).permute(0, 1, 3, 2)

    def forward(
        self,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        allin_mask: torch.Tensor,
        folded_mask: torch.Tensor,
        *,
        hardcode_folded_values: bool = True,
    ) -> torch.Tensor:
        if beliefs.shape[1:] != (PREFLOP_HANDS, self.players):
            raise ValueError(
                "compact all-in model expects beliefs shaped "
                f"[batch, {PREFLOP_HANDS}, {self.players}], got {tuple(beliefs.shape)}"
            )
        beliefs_player = beliefs.transpose(1, 2).contiguous()
        scale = starting_stacks.mean(dim=1, keepdim=True).clamp_min(1.0)
        player_features = PreflopAllInEquityModel._player_scalar_features(
            self,
            beliefs_player,
            starting_stacks,
            committed,
            stacks_after,
            allin_mask,
            folded_mask,
        )

        hand_emb = self.hand_encoder(
            self.hand_features.to(beliefs.device, beliefs.dtype)
        )
        beliefs_f = beliefs_player.to(hand_emb.dtype)
        bucket_features = self._range_mass_features(beliefs_f)
        range_summary = torch.cat(
            (
                beliefs_f @ hand_emb,
                beliefs_f @ hand_emb.square(),
            ),
            dim=-1,
        )
        per_player = (
            self.range_proj(range_summary)
            + self.bucket_mass_proj(bucket_features)
            + self.player_proj(player_features)
        )
        global_state = self.input_proj(per_player.flatten(1))
        if self.dense_belief_residual:
            global_state = global_state + self.dense_belief_proj(beliefs_f.flatten(1))
        global_state = self.trunk(global_state)
        state = self.player_state(
            torch.cat(
                (
                    per_player,
                    global_state[:, None, :].expand(-1, self.players, -1),
                ),
                dim=-1,
            )
        )

        hand_value = self.value_hand_proj(hand_emb)
        state_value = self.value_scale(state)
        values = torch.einsum("bpd,hd->bph", state_value, hand_value)
        values = values / math.sqrt(float(self.hidden_dim))
        blocker_features = self._blocker_features(beliefs_f, folded_mask)
        blocker_state = self.blocker_proj(blocker_features)
        blocker_scale = self.blocker_scale(state)
        blocker_values = torch.einsum("bphd,bpd->bph", blocker_state, blocker_scale)
        values = values + blocker_values / math.sqrt(float(self.hidden_dim))
        if self.film_rank > 0:
            film_hand = self.value_film_hand_proj(hand_emb)
            film_state = self.value_film_state(state)
            film_values = torch.einsum("bpr,hr->bph", film_state, film_hand)
            values = values + film_values / math.sqrt(float(self.film_rank))
        values = values + self.value_bias(state).expand(-1, -1, PREFLOP_HANDS)
        if self.dense_output_residual:
            values = values + self.dense_output_proj(state)
        if hardcode_folded_values:
            folded_value = (stacks_after - starting_stacks) / scale
            values = torch.where(
                folded_mask[:, :, None],
                folded_value[:, :, None],
                values,
            )
        return values.transpose(1, 2).contiguous().to(beliefs.dtype)

    def init_weights(self, generator: torch.Generator | None = None) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        self.value_scale.weight.data.mul_(OUTPUT_HEAD_INIT_SCALE)
        self.blocker_scale.weight.data.mul_(OUTPUT_HEAD_INIT_SCALE)
        self.value_bias.weight.data.mul_(OUTPUT_HEAD_INIT_SCALE)
        self.value_bias.bias.data.fill_(OUTPUT_HEAD_INIT_BIAS)
        if self.film_rank > 0:
            self.value_film_state.weight.data.zero_()
        if self.dense_output_residual:
            self.dense_output_proj.weight.data.zero_()
            self.dense_output_proj.bias.data.zero_()


class PreflopAllIn169TransformerModel(PreflopAllIn169EquityModel):
    """Player-token preflop all-in transformer for native 169-class ranges."""

    def __init__(
        self,
        players: int = 4,
        hidden_dim: int = 512,
        hand_dim: int = 128,
        num_layers: int = 4,
        film_rank: int = 64,
        transformer_heads: int = 8,
        dense_belief_residual: bool = False,
        dense_output_residual: bool = False,
        mask_folded_player_tokens: bool = False,
        negative_slope: float = 0.01,
    ) -> None:
        nn.Module.__init__(self)
        if players < 2:
            raise ValueError("players must be at least 2")
        if film_rank < 0:
            raise ValueError("film_rank must be non-negative")
        if transformer_heads <= 0:
            raise ValueError("transformer_heads must be positive")
        if hidden_dim % transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        self.players = int(players)
        self.hidden_dim = int(hidden_dim)
        self.hand_dim = int(hand_dim)
        self.num_layers = int(num_layers)
        self.film_rank = int(film_rank)
        self.transformer_heads = int(transformer_heads)
        self.dense_belief_residual = bool(dense_belief_residual)
        self.dense_output_residual = bool(dense_output_residual)
        self.mask_folded_player_tokens = bool(mask_folded_player_tokens)
        self.negative_slope = float(negative_slope)

        class_ids = torch.arange(PREFLOP_HANDS)
        row = torch.div(class_ids, 13, rounding_mode="floor")
        col = class_ids.remainder(13)
        hi = torch.maximum(row, col)
        lo = torch.minimum(row, col)
        pair = row == col
        suited = row > col
        self.register_buffer(
            "hand_features",
            self._preflop_class_features(hi, lo, pair, suited),
        )
        self.register_buffer("hand_hi_rank", hi.long(), persistent=False)
        self.register_buffer("hand_lo_rank", lo.long(), persistent=False)
        self.register_buffer("hand_pair", pair, persistent=False)
        self.register_buffer("hand_suited", suited, persistent=False)
        self.register_buffer(
            "preflop_unblocked_projection",
            _preflop_unblocked_projection(),
            persistent=False,
        )
        self.register_buffer(
            "preflop_bucket_projection",
            _preflop_bucket_projection(),
            persistent=False,
        )

        self.hand_encoder = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hand_dim, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.range_proj = nn.Linear(hand_dim * 2, hidden_dim, bias=False)
        self.bucket_mass_proj = nn.Sequential(
            nn.Linear(PREFLOP_169_BUCKET_FEATURE_DIM, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(players * PLAYER_FEATURE_DIM, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pot_layer_proj = nn.Sequential(
            nn.Linear(POT_LAYER_SCALAR_FEATURE_DIM + players * 2, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        if self.dense_belief_residual:
            self.dense_belief_proj = nn.Sequential(
                nn.Linear(players * PREFLOP_HANDS, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
        self.encoder = nn.ModuleList(
            [
                _TokenEncoderBlock(
                    hidden_dim,
                    num_heads=transformer_heads,
                    hidden_dim=hidden_dim * 2,
                    negative_slope=negative_slope,
                )
                for _ in range(num_layers)
            ]
        )
        self.player_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.value_hand_proj = nn.Linear(hand_dim, hidden_dim, bias=False)
        self.value_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.blocker_proj = nn.Linear(players, hidden_dim, bias=False)
        self.blocker_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_bias = nn.Linear(hidden_dim, 1)
        if self.dense_output_residual:
            self.dense_output_proj = nn.Linear(hidden_dim, PREFLOP_HANDS)
        if self.film_rank > 0:
            self.value_film_hand_proj = nn.Linear(hand_dim, self.film_rank, bias=False)
            self.value_film_state = nn.Linear(hidden_dim, self.film_rank, bias=False)

    def _pot_layer_features(
        self,
        beliefs: torch.Tensor,
        committed: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        levels = committed.sort(dim=1).values
        previous = torch.cat((torch.zeros_like(levels[:, :1]), levels[:, :-1]), dim=1)
        widths = (levels - previous).clamp_min(0.0)
        participants = committed[:, None, :] >= levels[:, :, None]
        eligible = participants & (~folded_mask[:, None, :])
        participant_count = participants.to(beliefs.dtype).sum(dim=2)
        eligible_count = eligible.to(beliefs.dtype).sum(dim=2)
        layer_amount = widths * participant_count
        positive = widths > 1.0e-6
        eligible_live = eligible_count >= 2.0
        layer_scalars = torch.stack(
            (
                levels / scale,
                widths / scale,
                layer_amount / scale,
                participant_count / float(self.players),
                eligible_count / float(self.players),
                positive.to(beliefs.dtype),
                (positive & eligible_live).to(beliefs.dtype),
            ),
            dim=-1,
        )
        layer_features = torch.cat(
            (
                layer_scalars,
                participants.to(beliefs.dtype),
                eligible.to(beliefs.dtype),
            ),
            dim=-1,
        )
        return layer_features, eligible & positive[:, :, None]

    def _token_attention_mask(
        self,
        player_pot_eligible: torch.Tensor,
        folded_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = player_pot_eligible.shape[0]
        token_count = 1 + self.players + self.players
        attn_mask = torch.zeros(
            batch_size,
            1,
            token_count,
            token_count,
            device=player_pot_eligible.device,
            dtype=torch.float32,
        )
        player_queries = slice(1, 1 + self.players)
        pot_keys = slice(1 + self.players, token_count)
        disallowed = ~player_pot_eligible.transpose(1, 2)
        attn_mask[:, :, player_queries, pot_keys] = torch.where(
            disallowed[:, None],
            torch.full((), float("-inf"), device=attn_mask.device),
            torch.zeros((), device=attn_mask.device),
        )
        if self.mask_folded_player_tokens and folded_mask is not None:
            player_keys = slice(1, 1 + self.players)
            folded_penalty = torch.where(
                folded_mask[:, None, None, :],
                torch.full((), float("-inf"), device=attn_mask.device),
                torch.zeros((), device=attn_mask.device),
            )
            attn_mask[:, :, :, player_keys] = folded_penalty
        return attn_mask

    def forward(
        self,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        allin_mask: torch.Tensor,
        folded_mask: torch.Tensor,
        *,
        hardcode_folded_values: bool = True,
    ) -> torch.Tensor:
        if beliefs.shape[1:] != (PREFLOP_HANDS, self.players):
            raise ValueError(
                "compact all-in transformer expects beliefs shaped "
                f"[batch, {PREFLOP_HANDS}, {self.players}], got {tuple(beliefs.shape)}"
            )
        beliefs_player = beliefs.transpose(1, 2).contiguous()
        scale = starting_stacks.mean(dim=1, keepdim=True).clamp_min(1.0)
        player_features = PreflopAllInEquityModel._player_scalar_features(
            self,
            beliefs_player,
            starting_stacks,
            committed,
            stacks_after,
            allin_mask,
            folded_mask,
        )

        hand_emb = self.hand_encoder(
            self.hand_features.to(beliefs.device, beliefs.dtype)
        )
        beliefs_f = beliefs_player.to(hand_emb.dtype)
        bucket_features = self._range_mass_features(beliefs_f)
        range_summary = torch.cat(
            (
                beliefs_f @ hand_emb,
                beliefs_f @ hand_emb.square(),
            ),
            dim=-1,
        )
        player_tokens = self.range_proj(range_summary) + self.bucket_mass_proj(
            bucket_features
        )
        pot_layer_features, player_pot_eligible = self._pot_layer_features(
            beliefs_f,
            committed,
            folded_mask,
            scale,
        )
        pot_tokens = self.pot_layer_proj(pot_layer_features)
        context_token = self.context_proj(player_features.flatten(1))
        if self.dense_belief_residual:
            context_token = context_token + self.dense_belief_proj(beliefs_f.flatten(1))
        tokens = torch.cat((context_token[:, None, :], player_tokens, pot_tokens), dim=1)
        attn_mask = self._token_attention_mask(player_pot_eligible, folded_mask)
        encoded = tokens
        for block in self.encoder:
            encoded = block(encoded, attn_mask)
        context_state = encoded[:, 0]
        player_state = encoded[:, 1 : 1 + self.players]
        state = self.player_state(
            torch.cat(
                (
                    player_state,
                    context_state[:, None, :].expand(-1, self.players, -1),
                ),
                dim=-1,
            )
        )

        hand_value = self.value_hand_proj(hand_emb)
        state_value = self.value_scale(state)
        values = torch.einsum("bpd,hd->bph", state_value, hand_value)
        values = values / math.sqrt(float(self.hidden_dim))
        blocker_features = self._blocker_features(beliefs_f, folded_mask)
        blocker_state = self.blocker_proj(blocker_features)
        blocker_scale = self.blocker_scale(state)
        blocker_values = torch.einsum("bphd,bpd->bph", blocker_state, blocker_scale)
        values = values + blocker_values / math.sqrt(float(self.hidden_dim))
        if self.film_rank > 0:
            film_hand = self.value_film_hand_proj(hand_emb)
            film_state = self.value_film_state(state)
            film_values = torch.einsum("bpr,hr->bph", film_state, film_hand)
            values = values + film_values / math.sqrt(float(self.film_rank))
        values = values + self.value_bias(state).expand(-1, -1, PREFLOP_HANDS)
        if self.dense_output_residual:
            values = values + self.dense_output_proj(state)
        if hardcode_folded_values:
            folded_value = (stacks_after - starting_stacks) / scale
            values = torch.where(
                folded_mask[:, :, None],
                folded_value[:, :, None],
                values,
            )
        return values.transpose(1, 2).contiguous().to(beliefs.dtype)


class PreflopAllInTransformerModel(PreflopAllInEquityModel):
    """Player-token preflop all-in model.

    The token stream contains one context token projected from scalar stack/pot
    state and one range token per player. Encoder blocks mix the tokens before
    the same hand-conditioned value head used by the MLP model.
    """

    def __init__(
        self,
        players: int = 4,
        hidden_dim: int = 512,
        hand_dim: int = 128,
        num_layers: int = 4,
        film_rank: int = 64,
        transformer_heads: int = 8,
        dense_belief_residual: bool = False,
        dense_output_residual: bool = False,
        mask_folded_player_tokens: bool = False,
        negative_slope: float = 0.01,
    ) -> None:
        nn.Module.__init__(self)
        if players < 2:
            raise ValueError("players must be at least 2")
        if film_rank < 0:
            raise ValueError("film_rank must be non-negative")
        if transformer_heads <= 0:
            raise ValueError("transformer_heads must be positive")
        if hidden_dim % transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        self.players = int(players)
        self.hidden_dim = int(hidden_dim)
        self.hand_dim = int(hand_dim)
        self.num_layers = int(num_layers)
        self.film_rank = int(film_rank)
        self.transformer_heads = int(transformer_heads)
        self.dense_belief_residual = bool(dense_belief_residual)
        self.dense_output_residual = bool(dense_output_residual)
        self.mask_folded_player_tokens = bool(mask_folded_player_tokens)
        self.negative_slope = float(negative_slope)

        combos = hand_combos_tensor()
        ranks = combos % 13
        suits = combos // 13
        self.register_buffer("hand_features", self._hand_features(ranks, suits))
        self.register_buffer("hand_card_a", combos[:, 0].long(), persistent=False)
        self.register_buffer("hand_card_b", combos[:, 1].long(), persistent=False)
        self.register_buffer("hand_rank_a", ranks[:, 0].long(), persistent=False)
        self.register_buffer("hand_rank_b", ranks[:, 1].long(), persistent=False)
        self.register_buffer("hand_suit_a", suits[:, 0].long(), persistent=False)
        self.register_buffer("hand_suit_b", suits[:, 1].long(), persistent=False)

        self.hand_encoder = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hand_dim, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.range_proj = nn.Linear(hand_dim * 2, hidden_dim, bias=False)
        self.card_mass_proj = nn.Sequential(
            nn.Linear(52, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.bucket_mass_proj = nn.Sequential(
            nn.Linear(RANGE_BUCKET_FEATURE_DIM, hidden_dim, bias=False),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(players * PLAYER_FEATURE_DIM, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.pot_layer_proj = nn.Sequential(
            nn.Linear(POT_LAYER_SCALAR_FEATURE_DIM + players * 2, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        if self.dense_belief_residual:
            self.dense_belief_proj = nn.Sequential(
                nn.Linear(players * NUM_HANDS, hidden_dim, bias=False),
                nn.RMSNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
        self.encoder = nn.ModuleList(
            [
                _TokenEncoderBlock(
                    hidden_dim,
                    num_heads=transformer_heads,
                    hidden_dim=hidden_dim * 2,
                    negative_slope=negative_slope,
                )
                for _ in range(num_layers)
            ]
        )
        self.player_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.value_hand_proj = nn.Linear(hand_dim, hidden_dim, bias=False)
        self.value_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.blocker_proj = nn.Linear(players, hidden_dim, bias=False)
        self.blocker_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_bias = nn.Linear(hidden_dim, 1)
        if self.dense_output_residual:
            self.dense_output_proj = nn.Linear(hidden_dim, NUM_HANDS)
        if self.film_rank > 0:
            self.value_film_hand_proj = nn.Linear(hand_dim, self.film_rank, bias=False)
            self.value_film_state = nn.Linear(hidden_dim, self.film_rank, bias=False)

    def _pot_layer_features(
        self,
        beliefs: torch.Tensor,
        committed: torch.Tensor,
        folded_mask: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        levels = committed.sort(dim=1).values
        previous = torch.cat((torch.zeros_like(levels[:, :1]), levels[:, :-1]), dim=1)
        widths = (levels - previous).clamp_min(0.0)
        participants = committed[:, None, :] >= levels[:, :, None]
        eligible = participants & (~folded_mask[:, None, :])
        participant_count = participants.to(beliefs.dtype).sum(dim=2)
        eligible_count = eligible.to(beliefs.dtype).sum(dim=2)
        layer_amount = widths * participant_count
        positive = widths > 1.0e-6
        eligible_live = eligible_count >= 2.0
        layer_scalars = torch.stack(
            (
                levels / scale,
                widths / scale,
                layer_amount / scale,
                participant_count / float(self.players),
                eligible_count / float(self.players),
                positive.to(beliefs.dtype),
                (positive & eligible_live).to(beliefs.dtype),
            ),
            dim=-1,
        )
        layer_features = torch.cat(
            (
                layer_scalars,
                participants.to(beliefs.dtype),
                eligible.to(beliefs.dtype),
            ),
            dim=-1,
        )
        return layer_features, eligible & positive[:, :, None]

    def _token_attention_mask(
        self,
        player_pot_eligible: torch.Tensor,
        folded_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = player_pot_eligible.shape[0]
        token_count = 1 + self.players + self.players
        attn_mask = torch.zeros(
            batch_size,
            1,
            token_count,
            token_count,
            device=player_pot_eligible.device,
            dtype=torch.float32,
        )
        player_queries = slice(1, 1 + self.players)
        pot_keys = slice(1 + self.players, token_count)
        disallowed = ~player_pot_eligible.transpose(1, 2)
        attn_mask[:, :, player_queries, pot_keys] = torch.where(
            disallowed[:, None],
            torch.full((), float("-inf"), device=attn_mask.device),
            torch.zeros((), device=attn_mask.device),
        )
        if self.mask_folded_player_tokens and folded_mask is not None:
            player_keys = slice(1, 1 + self.players)
            folded_penalty = torch.where(
                folded_mask[:, None, None, :],
                torch.full((), float("-inf"), device=attn_mask.device),
                torch.zeros((), device=attn_mask.device),
            )
            attn_mask[:, :, :, player_keys] = folded_penalty
        return attn_mask

    def forward(
        self,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        allin_mask: torch.Tensor,
        folded_mask: torch.Tensor,
        *,
        hardcode_folded_values: bool = True,
    ) -> torch.Tensor:
        if beliefs.shape[1] != self.players:
            raise ValueError(f"expected {self.players} players, got {beliefs.shape[1]}")
        scale = starting_stacks.mean(dim=1, keepdim=True).clamp_min(1.0)
        player_features = self._player_scalar_features(
            beliefs,
            starting_stacks,
            committed,
            stacks_after,
            allin_mask,
            folded_mask,
        )

        hand_emb = self.hand_encoder(
            self.hand_features.to(beliefs.device, beliefs.dtype)
        )
        beliefs_f = beliefs.to(hand_emb.dtype)
        card_mass, bucket_features = self._range_mass_features(beliefs_f)
        range_summary = torch.cat(
            (
                beliefs_f @ hand_emb,
                beliefs_f @ hand_emb.square(),
            ),
            dim=-1,
        )
        player_tokens = (
            self.range_proj(range_summary)
            + self.card_mass_proj(card_mass)
            + self.bucket_mass_proj(bucket_features)
        )
        pot_layer_features, player_pot_eligible = self._pot_layer_features(
            beliefs_f,
            committed,
            folded_mask,
            scale,
        )
        pot_tokens = self.pot_layer_proj(pot_layer_features)
        context_token = self.context_proj(player_features.flatten(1))
        if self.dense_belief_residual:
            context_token = context_token + self.dense_belief_proj(beliefs_f.flatten(1))
        tokens = torch.cat((context_token[:, None, :], player_tokens, pot_tokens), dim=1)
        attn_mask = self._token_attention_mask(player_pot_eligible, folded_mask)
        encoded = tokens
        for block in self.encoder:
            encoded = block(encoded, attn_mask)
        context_state = encoded[:, 0]
        player_state = encoded[:, 1 : 1 + self.players]
        state = self.player_state(
            torch.cat(
                (
                    player_state,
                    context_state[:, None, :].expand(-1, self.players, -1),
                ),
                dim=-1,
            )
        )

        hand_value = self.value_hand_proj(hand_emb)
        state_value = self.value_scale(state)
        values = torch.einsum("bpd,hd->bph", state_value, hand_value)
        values = values / math.sqrt(float(self.hidden_dim))
        blocker_features = self._blocker_features(beliefs_f, card_mass, folded_mask)
        blocker_state = self.blocker_proj(blocker_features)
        blocker_scale = self.blocker_scale(state)
        blocker_values = torch.einsum("bphd,bpd->bph", blocker_state, blocker_scale)
        values = values + blocker_values / math.sqrt(float(self.hidden_dim))
        if self.film_rank > 0:
            film_hand = self.value_film_hand_proj(hand_emb)
            film_state = self.value_film_state(state)
            film_values = torch.einsum("bpr,hr->bph", film_state, film_hand)
            values = values + film_values / math.sqrt(float(self.film_rank))
        values = values + self.value_bias(state).expand(-1, -1, NUM_HANDS)
        if self.dense_output_residual:
            values = values + self.dense_output_proj(state)
        if hardcode_folded_values:
            folded_value = (stacks_after - starting_stacks) / scale
            values = torch.where(
                folded_mask[:, :, None],
                folded_value[:, :, None],
                values,
            )
        return values.to(beliefs.dtype)
