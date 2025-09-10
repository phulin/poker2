from __future__ import annotations

from typing import Tuple, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ..core.interfaces import Model
from ..core.registry import register_model


def _resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad x with zeros on right/bottom to match ref's spatial size. Error if x is larger.

    This function never crops. It only pads when x's spatial dims are <= ref's.
    """
    if x.shape[2:] == ref.shape[2:]:
        return x
    x_h, x_w = x.shape[2], x.shape[3]
    r_h, r_w = ref.shape[2], ref.shape[3]
    if x_h > r_h or x_w > r_w:
        raise ValueError(
            f"Cannot align: source spatial {(x_h, x_w)} is larger than ref {(r_h, r_w)}"
        )
    pad_h = r_h - x_h
    pad_w = r_w - x_w
    # Pad order for F.pad with 4D tensors is (w_left, w_right, h_top, h_bottom)
    return F.pad(x, (0, pad_w, 0, pad_h))


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
        out3 = self.conv3(out2)
        out3 = self.norm3(out3)
        # Residual connection with spatial alignment (pad only, no crop)
        out = self.relu3(out3 + _resize_to(identity, out3))

        # Final residual connection from input with spatial alignment (pad only)
        out = out + _resize_to(input_residual, out)

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
        # Identity is larger here; pad conv output up to identity size (no crop)
        out = self.relu3(_resize_to(out, identity) + identity)

        # Final residual connection: pad to the larger (input_residual) to avoid cropping
        out = _resize_to(out, input_residual) + input_residual

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
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.cards_trunk = CardsConvTrunk(cards_channels, hidden=cards_hidden)
        self.actions_trunk = ActionsConvTrunk(actions_channels, hidden=actions_hidden)
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
        # Use gradient checkpointing for memory-intensive conv trunks if enabled
        if self.use_gradient_checkpointing:
            x_cards = checkpoint.checkpoint(self.cards_trunk, cards_tensor)
            x_actions = checkpoint.checkpoint(self.actions_trunk, actions_tensor)
        else:
            x_cards = self.cards_trunk(cards_tensor)
            x_actions = self.actions_trunk(actions_tensor)

        x = torch.cat([x_cards, x_actions], dim=1)
        h = self.fusion(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value
