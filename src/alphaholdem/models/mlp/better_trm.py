from __future__ import annotations

from dataclasses import dataclass
import math
from collections import OrderedDict

import torch
import torch.nn as nn

from alphaholdem.core.structured_config import NonlinearityType
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.models.activation_utils import SwiGLU, get_activation
from alphaholdem.models.mlp.better_features import context_length
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.model_output import ModelOutput, TRMLatent
from alphaholdem.utils.profiling import profile


def trunc_normal_(
    tensor: torch.Tensor,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Truncated normal initializer with correct std (mirrors JAX implementation)."""
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()

        sqrt2 = math.sqrt(2.0)
        lower = math.erf(a / sqrt2)
        upper = math.erf(b / sqrt2)
        z = (upper - lower) / 2.0

        c = (2.0 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * a * a)
        pdf_l = c * math.exp(-0.5 * b * b)
        comp_std = std / math.sqrt(
            1.0 - (b * pdf_u - a * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
        )

        tensor.uniform_(lower, upper, generator=generator)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(a * comp_std, b * comp_std)
    return tensor


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
                    ("norm", nn.LayerNorm(in_dim)),
                    ("swiglu", SwiGLU(in_dim, out_dim)),
                ]
            )
        )
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


class BetterTRM(nn.Module):
    """Better PBS TRM poker model."""

    def __init__(
        self,
        num_actions: int,
        hidden_dim: int = 512,
        range_hidden_dim: int = 256,
        ffn_dim: int = 1024,
        num_policy_layers: int = 1,
        num_value_layers: int = 1,
        num_players: int = 2,
        num_recursions: int = 6,
        num_iterations: int = 3,
        shared_trunk: bool = False,
        enforce_zero_sum: bool = True,
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_players = num_players
        self.num_recursions = num_recursions  # Number of TRM recursions n
        self.num_iterations = num_iterations  # Number of TRM iterations T
        self.shared_trunk = shared_trunk
        self.enforce_zero_sum = enforce_zero_sum
        self.nonlinearity = nonlinearity

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

        self.register_buffer("y_init", torch.zeros(hidden_dim))
        self.register_buffer("z_init", torch.zeros(hidden_dim))

        # Build trunk
        # Fixed 2-layer for recursion
        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), 1.0
            )
            for _ in range(2)
        ]
        layers.append(nn.LayerNorm(hidden_dim))
        self.trunk = nn.Sequential(*layers)

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), 1.0
            )
            for _ in range(num_value_layers - 1)
        ]
        layers.append(
            ffn_block(hidden_dim, ffn_dim, num_players * NUM_HANDS, nonlinearity)
        )
        self.hand_value_head = nn.Sequential(*layers)

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), 1.0
            )
            for _ in range(num_policy_layers - 1)
        ]
        layers.append(
            ffn_block(hidden_dim, ffn_dim, num_actions * NUM_HANDS, nonlinearity)
        )
        self.policy_head = nn.Sequential(*layers)

    def latent_recursion(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.num_recursions):
            z = self.trunk(x + y + z)
        y = self.trunk(y + z)
        return y, z

    @profile
    def forward(
        self,
        features: MLPFeatures,
        include_policy: bool = True,
        include_value: bool = True,
        latent: TRMLatent | None = None,
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

        x = (
            board_features.sum(dim=1)
            + street_features
            + context_features
            + belief_features
        )
        # assert flat_features.isfinite().all()

        y = (
            latent.y
            if latent is not None
            else self.y_init.clone()[None, :].expand(x.shape[0], -1)
        )
        z = (
            latent.z
            if latent is not None
            else self.z_init.clone()[None, :].expand(x.shape[0], -1)
        )
        with torch.no_grad():
            for j in range(self.num_iterations - 1):
                y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)

        if include_policy:
            policy_input = y if self.shared_trunk else y.detach()
            policy_logits = self.policy_head(policy_input).view(
                -1, NUM_HANDS, self.num_actions
            )
        else:
            policy_logits = None

        if include_value:
            hand_values_raw = self.hand_value_head(y).view(
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
        else:
            hand_values = None
            value = None

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
            latent=TRMLatent(y=y.detach(), z=z.detach()),
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

        if self.nonlinearity == NonlinearityType.gelu:
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

        trunc_normal_(self.y_init, std=1.0, generator=rng)
        trunc_normal_(self.z_init, std=1.0, generator=rng)

        # Guess hand values are around stddev 0.1.
        if self.nonlinearity == NonlinearityType.swiglu:
            self.hand_value_head[-1].get_submodule("swiglu").V.weight.data.mul_(0.1)
        else:
            self.hand_value_head[-1].get_submodule("linear_out").weight.data.mul_(0.1)

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
