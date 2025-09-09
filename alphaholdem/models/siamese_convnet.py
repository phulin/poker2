from __future__ import annotations

from typing import Tuple, Sequence
import torch
import torch.nn as nn

from ..core.interfaces import Model
from ..core.registry import register_model


class CardsConvTrunk(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        # First conv projects to hidden if needed
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=(3, 1), padding=1)
        self.norm1 = nn.GroupNorm(16, hidden)
        self.relu1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=(1, 3), padding=1)
        self.norm2 = nn.GroupNorm(16, hidden)
        self.relu2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=2, padding=1)
        self.norm3 = nn.GroupNorm(16, hidden)
        self.relu3 = nn.SiLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Input projection for residual connection
        if in_channels != hidden:
            self.input_proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        else:
            self.input_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for final residual connection
        input_residual = x
        if self.input_proj is not None:
            input_residual = self.input_proj(input_residual)

        # First block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        # Residual block 1
        identity = out
        out2 = self.conv2(out)
        out2 = self.norm2(out2)
        out2 = self.relu2(out2)
        out2 = self.conv3(out2)
        out2 = self.norm3(out2)
        # Residual connection
        out = self.relu3(out2 + identity)

        # Final residual connection from input
        out = out + input_residual

        # Pool
        out = self.pool(out)
        return out.flatten(1)


class ActionsConvTrunk(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        # Simpler approach for actions - focus on local patterns
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=2, padding=0)
        self.norm1 = nn.GroupNorm(16, hidden)
        self.relu1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=2, padding=0)
        self.norm2 = nn.GroupNorm(16, hidden)
        self.relu2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=2, padding=0)
        self.norm3 = nn.GroupNorm(16, hidden)
        self.relu3 = nn.SiLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Input projection for residual connection
        if in_channels != hidden:
            self.input_proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        else:
            self.input_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for final residual connection
        input_residual = x
        if self.input_proj is not None:
            input_residual = self.input_proj(input_residual)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        identity = out
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu3(out + identity)

        # Final residual connection from input
        out = out + input_residual

        out = self.pool(out)
        return out.flatten(1)


@register_model("siamese_convnet_v1")
class SiameseConvNetV1(nn.Module, Model):
    def __init__(
        self,
        cards_channels,
        actions_channels,
        cards_hidden: int,
        actions_hidden: int,
        fusion_hidden: int | Sequence[int],
        num_actions: int,
    ):
        super().__init__()
        self.cards_trunk = CardsConvTrunk(cards_channels, hidden=256)
        self.actions_trunk = ActionsConvTrunk(actions_channels, hidden=256)
        fusion_in = cards_hidden + actions_hidden
        # Build fusion MLP: accept single int or a sequence of hidden sizes
        hidden_sizes = (
            [fusion_hidden] if isinstance(fusion_hidden, int) else list(fusion_hidden)
        )

        layers: list[nn.Module] = []
        in_dim = fusion_in
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU(inplace=True))
            if i < len(hidden_sizes) - 1:
                layers.append(nn.Dropout(0.1))
            in_dim = h
        self.fusion = nn.Sequential(*layers)

        self.policy_head = nn.Sequential(
            nn.Linear(in_dim, 512),  # Same size as value head
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),  # Very light dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, num_actions),
        )
        # Critic MLP head for more stable value learning
        self.value_head = nn.Sequential(
            nn.Linear(in_dim, 512),  # Larger hidden
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.05),  # Very light dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

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
