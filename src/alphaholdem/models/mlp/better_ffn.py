from __future__ import annotations

import torch
import torch.nn as nn

from alphaholdem.core.interfaces import Model
from alphaholdem.models.mlp.better_features import context_length
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS
from alphaholdem.models.model_output import ModelOutput


class ResidualBlock(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + x


def ffn_block(in_dim: int, hidden_dim: int, out_dim: int | None = None) -> nn.Module:
    if out_dim is None:
        out_dim = in_dim
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class BetterFFN(nn.Module, Model):
    """Better PBS feed-forward poker model."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 1024,
        range_hidden_dim: int = 128,
        ffn_dim: int = 2048,
        num_hidden_layers: int = 4,
        num_players: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_players = num_players

        # Encoder for beliefs
        self.street_embedding = nn.Embedding(5, hidden_dim)
        self.rank_embedding = nn.Embedding(13 + 1, hidden_dim, padding_idx=13)
        self.suit_embedding = nn.Embedding(4 + 1, hidden_dim, padding_idx=4)
        self.belief_encoder = nn.Sequential(
            nn.Linear(num_players * NUM_HANDS, num_players * range_hidden_dim),
            nn.GELU(),
            nn.Linear(num_players * range_hidden_dim, hidden_dim),
        )
        self.context_encoder = ffn_block(
            context_length(num_players), hidden_dim, hidden_dim
        )

        # Build trunk
        layers: list[nn.Module] = []
        for _ in range(num_hidden_layers):
            layers.append(ResidualBlock(ffn_block(hidden_dim, ffn_dim)))
        self.trunk = nn.Sequential(*layers)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # Heads
        self.policy_head = ffn_block(hidden_dim, ffn_dim, num_actions * NUM_HANDS)
        self.hand_value_head = ffn_block(hidden_dim, ffn_dim, num_players * NUM_HANDS)

    def forward(self, features: MLPFeatures) -> ModelOutput:
        """
        Forward pass over flat feature vectors.

        Args:
            features: MLPFeatures

        Returns:
            ModelOutput with policy logits and value predictions.
        """

        board = torch.where(
            features.board > 0, features.board, torch.full_like(features.board, 52)
        )
        ranks = board % 13
        suits = board // 13
        board_features = self.rank_embedding(ranks) + self.suit_embedding(suits)

        street_features = self.street_embedding(features.street)
        context_features = self.context_encoder(features.context)
        belief_features = self.belief_encoder(
            features.beliefs.view(-1, self.num_players * NUM_HANDS)
        )

        # TODO: some kind of positional encoding for board features?
        flat_features = (
            board_features.sum(dim=1)
            + street_features
            + context_features
            + belief_features
        )

        x = self.trunk(flat_features)
        x = self.post_norm(x)

        policy_logits = self.policy_head(x).view(-1, NUM_HANDS, self.num_actions)
        hand_values = self.hand_value_head(x).view(-1, self.num_players, NUM_HANDS)
        value = hand_values.mean(dim=-1)

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
        )

    def init_weights(self, rng: torch.Generator | None = None) -> None:
        """Initialize parameters following Xavier/LayerNorm defaults."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, generator=rng)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def adjust_scale(self, weight_scale: float, bias_adjustment: float) -> None:
        """
        Apply PopArt scaling to the final value head.

        No-op for quantile value heads.
        """

        last_linear = self.hand_value_head[-1]
        last_linear.weight.data.mul_(weight_scale)
        if last_linear.bias is not None:
            last_linear.bias.data.mul_(weight_scale).add_(bias_adjustment)
