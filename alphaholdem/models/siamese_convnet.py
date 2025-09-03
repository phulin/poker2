from __future__ import annotations

from typing import Any, Tuple, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.interfaces import Model
from ..core.registry import register_model


class ConvTrunk(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden),
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
        fusion_hidden: int | Sequence[int] = (2048, 2048),
        num_actions: int = 8,
    ):
        super().__init__()
        self.cards_trunk = ConvTrunk(cards_channels, hidden=256)
        self.actions_trunk = ConvTrunk(actions_channels, hidden=256)
        fusion_in = 256 + 256
        # Build fusion MLP: accept single int or a sequence of hidden sizes
        hidden_sizes = (
            [fusion_hidden] if isinstance(fusion_hidden, int) else list(fusion_hidden)
        )
        layers: list[nn.Module] = []
        in_dim = fusion_in
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            in_dim = h
        self.fusion = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, num_actions)
        self.value_head = nn.Linear(in_dim, 1)

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
