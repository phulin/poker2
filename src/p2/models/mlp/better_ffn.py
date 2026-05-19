from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from p2.core.structured_config import NonlinearityType
from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.models.activation_utils import get_activation, SwiGLU
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp.better_feature_encoder import BetterFeatureEncoder
from p2.models.mlp.better_features import context_length
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.utils.profiling import profile


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


def direct_projection_block(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        OrderedDict(
            [
                ("norm", nn.RMSNorm(in_dim, eps=1e-5)),
                ("linear", nn.Linear(in_dim, out_dim)),
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

        if range_hidden_dim < 0:
            raise ValueError("range_hidden_dim must be non-negative")

        self.street_embedding = nn.Embedding(5, hidden_dim)
        self.rank_embedding = nn.Embedding(13 + 1, hidden_dim, padding_idx=13)
        self.suit_embedding = nn.Embedding(4 + 1, hidden_dim, padding_idx=4)
        # Hand-aware belief encoder: project per-player belief vectors through a
        # hand embedding tied to the rank/suit embeddings, then fuse across
        # players. Gives each "hand axis" learned card structure for free
        # instead of treating beliefs as an unstructured 1326-dim vector.
        combos = hand_combos_tensor()  # [NUM_HANDS, 2]
        self.register_buffer("hand_combos", combos, persistent=False)
        self.register_buffer("hand_ranks", combos % 13, persistent=False)
        self.register_buffer("hand_suits", combos // 13, persistent=False)
        self._hand_embedding_cache: torch.Tensor | None = None
        self._hand_embedding_cache_key: tuple[int, int, int, int] | None = None
        self._skip_hand_embedding_cache_when_compiling = False
        belief_in_dim = num_players * hidden_dim
        self.belief_proj = (
            direct_projection_block(belief_in_dim, hidden_dim)
            if range_hidden_dim == 0
            else ffn_block(
                belief_in_dim,
                num_players * range_hidden_dim,
                hidden_dim,
                nonlinearity,
            )
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
            ffn_block(
                hidden_dim, ffn_dim, num_actions * NUM_HANDS, NonlinearityType.leaky_relu
            )
        )
        self.policy_head = nn.Sequential(*layers)

        layers = [
            ResidualBlock(
                ffn_block(hidden_dim, ffn_dim, nonlinearity=nonlinearity), alpha
            )
            for _ in range(num_value_layers - 1)
        ]
        layers.append(
            ffn_block(
                hidden_dim, ffn_dim, num_players * NUM_HANDS, NonlinearityType.leaky_relu
            )
        )
        self.hand_value_head = nn.Sequential(*layers)

    def _hand_embedding(self) -> torch.Tensor:
        """Per-hand embedding tied to rank/suit embeddings — shape [NUM_HANDS, hidden_dim]."""
        use_cache = not torch.is_grad_enabled()
        if self._skip_hand_embedding_cache_when_compiling and torch.compiler.is_compiling():
            use_cache = False
        if use_cache:
            key = (
                int(self.rank_embedding.weight.data_ptr()),
                int(self.rank_embedding.weight._version),
                int(self.suit_embedding.weight.data_ptr()),
                int(self.suit_embedding.weight._version),
            )
            if self._hand_embedding_cache_key == key and self._hand_embedding_cache is not None:
                return self._hand_embedding_cache
        card_emb = self.rank_embedding(self.hand_ranks) + self.suit_embedding(
            self.hand_suits
        )
        out = card_emb.sum(dim=1)
        if use_cache:
            self._hand_embedding_cache = out
            self._hand_embedding_cache_key = key
        return out

    def static_feature_base(self, features: MLPFeatures) -> torch.Tensor:
        """Feature contribution that is fixed for a CFR leaf row."""
        board = features.board
        ranks = torch.where(board >= 0, board % 13, torch.full_like(board, 13))
        suits = torch.where(board >= 0, board // 13, torch.full_like(board, 4))
        board_features = self.rank_embedding(ranks) + self.suit_embedding(suits)
        return (
            board_features.sum(dim=1)
            + self.street_embedding(features.street)
            + self.context_encoder(features.context)
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
        """
        Forward pass over flat feature vectors.

        Args:
            features: MLPFeatures

        Returns:
            ModelOutput with policy logits and value predictions.
        """

        player_beliefs = features.beliefs.view(-1, self.num_players, NUM_HANDS)
        hand_emb = self._hand_embedding()  # [NUM_HANDS, hidden_dim]
        per_player_belief = player_beliefs @ hand_emb  # [B, P, H]
        belief_features = self.belief_proj(per_player_belief.flatten(1))

        if static_base_features is None:
            static_base_features = self.static_feature_base(features)
        flat_features = static_base_features + belief_features
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
        for sequential in [self.trunk, self.policy_head, self.hand_value_head]:
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
        self.hand_value_head[-1].get_submodule("linear_out").weight.data.mul_(0.1)

    def create_feature_encoder(
        self,
        env,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> BetterFeatureEncoder:
        return BetterFeatureEncoder(env=env, device=device, dtype=dtype)

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
