from __future__ import annotations

import math

import torch
import torch.nn as nn

from p2.env.card_utils import NUM_HANDS, hand_combos_tensor


HAND_FEATURE_DIM = 8
PLAYER_FEATURE_DIM = 9


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


class PreflopAllInEquityModel(nn.Module):
    """Browser-friendly preflop all-in terminal value model.

    The model predicts chip-normalized all-in values for every player and every
    private hand. It intentionally uses only Linear, RMSNorm, LeakyReLU, basic
    tensor ops, and fixed buffers so it can be ported to browser runtimes.
    """

    def __init__(
        self,
        players: int = 4,
        hidden_dim: int = 512,
        hand_dim: int = 128,
        num_layers: int = 4,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if players < 2:
            raise ValueError("players must be at least 2")
        self.players = int(players)
        self.hidden_dim = int(hidden_dim)
        self.hand_dim = int(hand_dim)
        self.num_layers = int(num_layers)
        self.negative_slope = float(negative_slope)

        combos = hand_combos_tensor()
        ranks = combos % 13
        suits = combos // 13
        self.register_buffer("hand_features", self._hand_features(ranks, suits))

        self.hand_encoder = nn.Sequential(
            nn.Linear(HAND_FEATURE_DIM, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hand_dim, hand_dim),
            nn.RMSNorm(hand_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.range_proj = nn.Linear(hand_dim * 2, hidden_dim, bias=False)
        self.player_proj = nn.Sequential(
            nn.Linear(PLAYER_FEATURE_DIM, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
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
        self.value_bias = nn.Linear(hidden_dim, 1)

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

    def forward(
        self,
        beliefs: torch.Tensor,
        starting_stacks: torch.Tensor,
        committed: torch.Tensor,
        stacks_after: torch.Tensor,
        allin_mask: torch.Tensor,
        folded_mask: torch.Tensor,
    ) -> torch.Tensor:
        if beliefs.shape[1] != self.players:
            raise ValueError(f"expected {self.players} players, got {beliefs.shape[1]}")
        scale = starting_stacks.mean(dim=1, keepdim=True).clamp_min(1.0)
        live = (~folded_mask).to(beliefs.dtype)
        allin = allin_mask.to(beliefs.dtype)
        folded = folded_mask.to(beliefs.dtype)
        player_idx = torch.linspace(
            0.0,
            1.0,
            self.players,
            device=beliefs.device,
            dtype=beliefs.dtype,
        )[None, :].expand(beliefs.shape[0], -1)
        pot = committed.sum(dim=1, keepdim=True) / scale
        max_eligible_to_win = self._max_eligible_to_win(
            committed,
            folded_mask,
        ) / scale
        player_features = torch.stack(
            (
                starting_stacks / scale,
                committed / scale,
                stacks_after / scale,
                allin,
                folded,
                live,
                player_idx,
                pot.expand(-1, self.players),
                max_eligible_to_win,
            ),
            dim=-1,
        )

        hand_emb = self.hand_encoder(self.hand_features.to(beliefs.device, beliefs.dtype))
        beliefs_f = beliefs.to(hand_emb.dtype)
        range_summary = torch.cat(
            (
                beliefs_f @ hand_emb,
                beliefs_f @ hand_emb.square(),
            ),
            dim=-1,
        )
        per_player = self.range_proj(range_summary) + self.player_proj(player_features)
        global_state = self.input_proj(per_player.flatten(1))
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
        values = values + self.value_bias(state).expand(-1, -1, NUM_HANDS)
        folded_value = (stacks_after - starting_stacks) / scale
        values = torch.where(folded_mask[:, :, None], folded_value[:, :, None], values)
        return values.to(beliefs.dtype)

    def init_weights(self, generator: torch.Generator | None = None) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        self.value_scale.weight.data.mul_(0.1)
        self.value_bias.weight.data.mul_(0.1)
