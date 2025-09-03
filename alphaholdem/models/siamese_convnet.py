from __future__ import annotations

from typing import Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.interfaces import Model
from ..core.registry import register_model


class ConvTrunk(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.net(x)
        return y.flatten(1)


@register_model("siamese_convnet_v1")
class SiameseConvNetV1(nn.Module, Model):
    def __init__(
        self,
        cards_channels: int = 6,
        actions_channels: int = 24,
        fusion_hidden: int = 256,
        num_actions: int = 9,
    ):
        super().__init__()
        self.cards_trunk = ConvTrunk(cards_channels, hidden=64)
        self.actions_trunk = ConvTrunk(actions_channels, hidden=64)
        fusion_in = 64 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.LayerNorm(fusion_hidden),  # Use LayerNorm instead of BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Add dropout for regularization
        )
        self.policy_head = nn.Linear(fusion_hidden, num_actions)
        self.value_head = nn.Linear(fusion_hidden, 1)

    def forward(
        self, cards_tensor: torch.Tensor, actions_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expect input shapes: (B, C, 4, 13) for both
        x_cards = self.cards_trunk(cards_tensor)
        x_actions = self.actions_trunk(actions_tensor)
        x = torch.cat([x_cards, x_actions], dim=1)
        h = self.fusion(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value
