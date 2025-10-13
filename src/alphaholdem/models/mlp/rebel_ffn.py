from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from alphaholdem.core.interfaces import Model
from alphaholdem.core.structured_config import ValueHeadType
from alphaholdem.models.model_output import ModelOutput


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
    belief_dim: int = 1326
    num_players: int = 2


class _FFNBlock(nn.Module):
    """Single feed-forward block with LayerNorm + GeLU."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class RebelFFN(nn.Module, Model):
    """ReBeL-inspired feed-forward poker model."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dim: int = 1536,
        num_hidden_layers: int = 6,
        value_head_type: str = "scalar",
        value_head_num_quantiles: int = 1,
        detach_value_head: bool = True,
        belief_dim: int = 1326,
        num_players: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.detach_value_head = detach_value_head
        self.belief_dim = belief_dim
        self.num_players = num_players

        self.value_head_type = value_head_type
        self.num_value_quantiles = int(value_head_num_quantiles)
        if (
            self.value_head_type == ValueHeadType.quantile
            and self.num_value_quantiles <= 1
        ):
            self.num_value_quantiles = 2

        # Build trunk
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(_FFNBlock(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # Heads
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        value_out_dim = (
            self.num_value_quantiles
            if self.value_head_type == ValueHeadType.quantile
            else 1
        )
        self.value_head = nn.Linear(hidden_dim, value_out_dim)
        self.hand_value_head = nn.Linear(hidden_dim, self.num_players * self.belief_dim)

    def forward(self, features: torch.Tensor, *_: Any, **__: Any) -> ModelOutput:
        """
        Forward pass over flat feature vectors.

        Args:
            features: Tensor of shape [batch, input_dim]

        Returns:
            ModelOutput with policy logits and value predictions.
        """
        if features.dim() != 2 or features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input [batch, {self.input_dim}], got {tuple(features.shape)}"
            )

        x = self.trunk(features)
        x = self.post_norm(x)

        policy_logits = self.policy_head(x)

        value_input = x.detach() if self.detach_value_head else x
        if self.value_head_type == ValueHeadType.quantile:
            value_quantiles = self.value_head(value_input)
            value = value_quantiles.mean(dim=-1)
        else:
            value_quantiles = None
            value = self.value_head(value_input).squeeze(-1)

        hand_value_input = x.detach() if self.detach_value_head else x
        hand_values = self.hand_value_head(hand_value_input)
        hand_values = hand_values.view(
            features.shape[0], self.num_players, self.belief_dim
        )

        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            value_quantiles=value_quantiles,
            hand_values=hand_values,
        )

    def init_weights(self, rng: Optional[torch.Generator] = None) -> None:
        """Initialize parameters following Xavier/LayerNorm defaults."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=rng)
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
        if self.value_head_type == ValueHeadType.quantile:
            return

        last_linear = self.value_head
        last_linear.weight.data.mul_(weight_scale)
        if last_linear.bias is not None:
            last_linear.bias.data.mul_(weight_scale).add_(bias_adjustment)

    def get_model_info(self) -> Dict[str, Any]:
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
