from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch


@lru_cache(maxsize=1)
def hand_combos_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [1326, 2] tensor of sorted hole-card index pairs."""
    combos = []
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            combos.append((c1, c2))
    tensor = torch.tensor(combos, dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


@lru_cache(maxsize=1)
def combo_lookup_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [52, 52] tensor mapping unordered card pairs to combo indices."""
    lookup = torch.full((52, 52), -1, dtype=torch.long)
    combos = hand_combos_tensor()
    for idx, (c1, c2) in enumerate(combos.tolist()):
        lookup[c1, c2] = idx
        lookup[c2, c1] = idx
    if device is not None:
        lookup = lookup.to(device)
    return lookup


def combo_index(card_a: int, card_b: int) -> int:
    """Return 1326-hand index for unordered (card_a, card_b)."""
    a, b = sorted((int(card_a), int(card_b)))
    lookup = combo_lookup_tensor()
    idx = int(lookup[a, b].item())
    if idx < 0:
        raise ValueError(f"Invalid combo for cards {card_a}, {card_b}")
    return idx


def mask_conflicting_combos(
    occupied_cards: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Return [1326] bool mask where True means combo does NOT intersect occupied cards.

    Args:
        occupied_cards: 1-D tensor of unique card indices to exclude (-1 ignored).
        device: Target device for mask tensor.
    """
    combos = hand_combos_tensor(device=device)  # [1326, 2]
    occupied = occupied_cards[occupied_cards >= 0]
    if occupied.numel() == 0:
        return torch.ones(combos.shape[0], dtype=torch.bool, device=device)
    # Broadcast compare combos against occupied cards and check membership
    intersects = torch.isin(combos, occupied).any(dim=1)
    return ~intersects


@lru_cache(maxsize=1)
def combo_blocking_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [1326, 1326] tensor of blocked hands for each combo."""
    combos = hand_combos_tensor(device=device)  # [1326, 2]
    combo_onehot = torch.zeros(1326, 52, dtype=torch.float32, device=device)
    idx = torch.arange(1326, device=device)
    combo_onehot[idx, combos[:, 0]] = 1
    combo_onehot[idx, combos[:, 1]] = 1
    return (combo_onehot @ combo_onehot.T).clamp_(0, 1).to(torch.bool)
