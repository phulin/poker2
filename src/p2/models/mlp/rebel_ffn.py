from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from p2.core.structured_config import NonlinearityType
from p2.env.card_utils import NUM_HANDS
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.activation_utils import get_activation
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from p2.models.model_output import ModelOutput


@dataclass
class RebelFFNConfig:
    """Lightweight container for default ReBeL-style FFN hyperparameters."""

    input_dim: int
    num_actions: int
    hidden_dim: int = 1536
    num_hidden_layers: int = 6
    value_head_type: str = "scalar"
    value_head_num_quantiles: int = 1
    detach_value_head: bool = True
    num_players: int = 2


class _FFNBlock(nn.Module):
    """Single feed-forward block with LayerNorm + activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = get_activation(nonlinearity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class RebelFFN(BaseMLPModel):
    """ReBeL-inspired feed-forward poker model."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 1536,
        num_hidden_layers: int = 6,
        detach_value_head: bool = False,
        num_players: int = 2,
        nonlinearity: NonlinearityType = NonlinearityType.gelu,
        enforce_zero_sum: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.detach_value_head = detach_value_head
        self.num_players = num_players
        self.enforce_zero_sum = enforce_zero_sum

        # Build trunk
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(_FFNBlock(in_dim, hidden_dim, nonlinearity))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # Heads
        self.policy_head = nn.Linear(hidden_dim, num_actions * NUM_HANDS)
        self.hand_value_head = nn.Linear(hidden_dim, num_players * NUM_HANDS)

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
            features: MLPFeatures with flat tensor or Tensor of shape [batch, input_dim]

        Returns:
            ModelOutput with policy logits and value predictions.
        """
        board_features = torch.where(features.board >= 0, features.board / 51.0, -1.0)
        features_tensor = torch.cat(
            [features.context[:, :4], board_features, features.beliefs],
            dim=-1,
        )

        if features_tensor.dim() != 2 or features_tensor.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input [batch, {self.input_dim}], got {tuple(features_tensor.shape)}"
            )

        x = self.trunk(features_tensor)
        x = self.post_norm(x)

        if include_policy:
            policy_logits = self.policy_head(x).reshape(-1, NUM_HANDS, self.num_actions)
        else:
            policy_logits = None

        hand_values = None
        value = None
        if include_value:
            hand_value_input = x.detach() if self.detach_value_head else x
            hand_values = self.hand_value_head(hand_value_input)
            hand_values = hand_values.view(
                features_tensor.shape[0], self.num_players, NUM_HANDS
            )
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
    ) -> RebelFeatureEncoder:
        return RebelFeatureEncoder(env=env, device=device, dtype=dtype)

    def get_model_info(self) -> dict[str, any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_type": "rebel_ffn",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "num_actions": self.num_actions,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

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
