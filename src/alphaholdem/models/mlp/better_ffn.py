from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from alphaholdem.core.structured_config import NonlinearityType
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.models.base_mlp_model import BaseMLPModel
from alphaholdem.models.activation_utils import get_activation
from alphaholdem.models.mlp.better_features import context_length
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.utils.profiling import profile


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
    return nn.Sequential(
        OrderedDict(
            [
                ("norm", nn.LayerNorm(in_dim)),
                ("linear_in", nn.Linear(in_dim, hidden_dim, bias=False)),
                ("activation", get_activation(nonlinearity)),
                ("linear_out", nn.Linear(hidden_dim, out_dim)),
            ]
        )
    )


class BetterFFN(BaseMLPModel):
    """Better PBS feed-forward poker model."""

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
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_players = num_players
        self.shared_trunk = shared_trunk
        self.enforce_zero_sum = enforce_zero_sum

        self.street_embedding = nn.Embedding(5, hidden_dim)
        self.rank_embedding = nn.Embedding(13 + 1, hidden_dim, padding_idx=13)
        self.suit_embedding = nn.Embedding(4 + 1, hidden_dim, padding_idx=4)
        self.belief_encoder = ffn_block(
            num_players * NUM_HANDS,
            num_players * range_hidden_dim,
            hidden_dim,
            nonlinearity,
        )
        self.context_encoder = ffn_block(
            context_length(num_players), hidden_dim, hidden_dim, nonlinearity
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
            for _ in range(num_policy_layers - 1)
        ]
        layers.append(
            ffn_block(hidden_dim, ffn_dim, num_actions * NUM_HANDS, nonlinearity)
        )
        self.policy_head = nn.Sequential(*layers)

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), alpha
            )
            for _ in range(num_value_layers - 1)
        ]
        layers.append(
            ffn_block(hidden_dim, ffn_dim, num_players * NUM_HANDS, nonlinearity)
        )
        self.hand_value_head = nn.Sequential(*layers)

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        latent=None,
    ) -> ModelOutput:
        """
        Forward pass over flat feature vectors.

        Args:
            features: MLPFeatures

        Returns:
            ModelOutput with policy logits and value predictions.
        """

        board = features.board
        ranks = torch.where(board >= 0, board % 13, torch.full_like(board, 13))
        suits = torch.where(board >= 0, board // 13, torch.full_like(board, 4))
        board_features = self.rank_embedding(ranks) + self.suit_embedding(suits)

        street_features = self.street_embedding(features.street)
        context_features = self.context_encoder(features.context)
        belief_features = self.belief_encoder(features.beliefs)
        player_beliefs = features.beliefs.view(-1, self.num_players, NUM_HANDS)

        flat_features = (
            board_features.sum(dim=1)
            + street_features
            + context_features
            + belief_features
        )
        # assert flat_features.isfinite().all()

        x = self.trunk(flat_features)
        # assert x.isfinite().all()

        policy_input = x if self.shared_trunk else flat_features.detach()

        if include_policy:
            policy_logits = self.policy_head(policy_input).view(
                -1, NUM_HANDS, self.num_actions
            )
        else:
            policy_logits = None
        hand_values = None
        value = None
        if include_value:
            hand_values_raw = self.hand_value_head(x).view(
                -1, self.num_players, NUM_HANDS
            )
            if self.enforce_zero_sum:
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

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterFeatureEncoder:
        return BetterFeatureEncoder(env=env, device=device, dtype=dtype)

        for sequential in [self.trunk, self.policy_head, self.hand_value_head]:
            for block in sequential.modules():
                if not isinstance(block, ResidualBlock):
                    continue
                # 1.532 is the gain for GELU nonlinearity.
                nn.init.orthogonal_(
                    block.inner.get_submodule("linear_in").weight,
                    1.532 * math.sqrt(self.ffn_dim / self.hidden_dim),
                    generator=rng,
                )

        # Guess hand values are around stddev 0.1.
        self.hand_value_head[-1].get_submodule("linear_out").weight.data.mul_(0.1)

    def repeat(
        self,
        features: MLPFeatures,
        count: int,
        include_policy: bool = False,
        include_value: bool = True,
    ) -> ModelOutput:
        return self(
            features, include_policy=include_policy, include_value=include_value
        )
