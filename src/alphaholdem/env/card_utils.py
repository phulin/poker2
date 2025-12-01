from __future__ import annotations

from functools import lru_cache
from itertools import permutations

import torch

NUM_HANDS = 1326

# Rank mapping: A=12, K=11, Q=10, J=9, T=8, 9=7, 8=6, 7=5, 6=4, 5=3, 4=2, 3=1, 2=0
IDX_TO_RANK = "23456789TJQKA"
RANK_TO_IDX = {rank: idx for idx, rank in enumerate(IDX_TO_RANK)}

HAND_EQUITY_ORDERING = (
    "AA,KK,QQ,JJ,AKs,AQs,TT,AKo,AJs,KQs,99,ATs,AQo,KJs,88,QJs,KTs,AJo,A9s,QTs,"
    "77,KQo,JTs,A8s,K9s,ATo,A7s,A5s,66,KJo,A4s,Q9s,T9s,J9s,A6s,QJo,55,A3s,KTo,"
    "K8s,A2s,K7s,T8s,98s,QTo,Q8s,87s,44,A9o,JTo,J8s,76s,K6s,97s,K5s,K4s,T7s,"
    "Q7s,33,A8o,K9o,J7s,86s,65s,K3s,K2s,Q9o,Q6s,J9o,T9o,54s,22,Q5s,T8o,96s,75s,"
    "64s,A7o,Q4s,J8o,T7o,98o,97o,K8o,K7o,Q8o,Q3s,J6s,J5s,J4s,T6o,T6s,86o,85o,"
    "85s,76o,75o,74s,63s,53s,A6o,A5o,A4o,K6o,Q7o,Q2s,J7o,J6o,T5o,T5s,T4o,T3o,"
    "T2o,96o,95o,95s,94o,93o,92o,87o,84o,83o,82o,74o,73o,72o,65o,64o,63o,62o,"
    "53o,52o,42o,A3o,K5o,K4o,Q6o,Q5o,Q4o,Q3o,Q2o,J5o,J4o,J3o,J3s,J2o,T4s,T3s,"
    "84s,54o,43o,43s,K3o,K2o,J2s,T2s,93s,92s,82s,73s,62s,52s,42s,32s,A2o,94s,"
    "83s,72s,32o"
).split(",")


def parse_hand_name(hand_name: str) -> tuple[int, int]:
    """Parse a poker hand name to get card indices.

    Args:
        hand_name: Like 'AA', 'AKs', 'KQo'
                   - Pairs: 'AA' means pair of aces (any suits)
                   - Suited: 'AKs' means AK suited
                   - Offsuit: 'AKo' means AK offsuit

    Returns:
        Tuple of (card1, card2) card indices
    """
    if len(hand_name) == 2 and hand_name[0] == hand_name[1]:
        # Pair - use first two suits (0, 1)
        rank = RANK_TO_IDX[hand_name[0]]
        return rank, rank + 13
    elif len(hand_name) >= 3:
        rank1 = RANK_TO_IDX[hand_name[0]]
        rank2 = RANK_TO_IDX[hand_name[1]]
        is_suited = hand_name[2] == "s"

        # Suited hands use same suit, offsuit use different suits
        if is_suited:
            return rank1, rank2
        else:
            return rank1, rank2 + 13
    else:
        raise ValueError(f"Invalid hand name: {hand_name}")


@lru_cache(maxsize=2)
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


@lru_cache(maxsize=2)
def combo_lookup_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [52, 52] long tensor mapping unordered card pairs to combo indices."""
    lookup = torch.full((52, 52), -1, dtype=torch.long)
    combos = hand_combos_tensor(device=device)
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
    device: torch.device | None = None,
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


@lru_cache(maxsize=2)
def combo_to_onehot_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [1326, 52] bool tensor of one-hot encoded combos."""
    combos = hand_combos_tensor(device=device)  # [1326, 2]
    combo_onehot = torch.zeros(1326, 52, dtype=bool, device=device)
    idx = torch.arange(1326, device=device)
    combo_onehot[idx, combos[:, 0]] = True
    combo_onehot[idx, combos[:, 1]] = True
    return combo_onehot


@lru_cache(maxsize=2)
def combo_to_range_grid(device: torch.device | None = None) -> torch.Tensor:
    """Return [1326, 2] tensor of range grid index (suited/offsuit) for each combo."""
    combos = hand_combos_tensor(device=device)  # [1326, 2]
    combos_suited = combos[:, 0] // 13 == combos[:, 1] // 13
    combo_ranks = combos % 13
    combo_ranks_lower = combo_ranks.sort(dim=1).values
    combo_ranks_upper = combo_ranks_lower.flip(dims=(1,))
    return 12 - torch.where(
        combos_suited[:, None], combo_ranks_upper, combo_ranks_lower
    )


@lru_cache(maxsize=2)
def combo_blocking_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [1326, 1326] tensor of blocked hands for each combo."""
    combo_onehot = combo_to_onehot_tensor(device=device).float()
    return (combo_onehot @ combo_onehot.T) > 0.5


@lru_cache(maxsize=2)
def combo_compatible_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [1326, 1326] tensor of blocked hands for each combo."""
    combo_onehot = combo_to_onehot_tensor(device=device).float()
    return (combo_onehot @ combo_onehot.T) < 0.5


@lru_cache(maxsize=2)
def suit_permutations_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [24, 4] tensor enumerating all suit permutations."""
    perms = torch.tensor(tuple(permutations(range(4))), dtype=torch.long)
    if device is not None:
        perms = perms.to(device)
    return perms


@lru_cache(maxsize=2)
def combo_suit_permutation_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [24, 1326] tensor: hole-card index permutations for each suit permutation."""
    combos = hand_combos_tensor(device=device)  # [1326, 2]
    lookup = combo_lookup_tensor(device=device)  # [52, 52]

    ranks = combos % 13  # [1326, 2]
    suits = combos // 13  # [1326, 2]

    suit_perms = suit_permutations_tensor(device=device)  # [24, 4]
    permuted_suits = suit_perms[:, suits]  # [24, 1326, 2]
    ranks_expand = ranks.unsqueeze(0).expand(permuted_suits.shape[0], -1, -1)
    permuted_cards = permuted_suits * 13 + ranks_expand  # [24, 1326, 2]
    permuted_cards_sorted = permuted_cards.sort(dim=2).values

    permuted_indices = lookup[
        permuted_cards_sorted[:, :, 0],
        permuted_cards_sorted[:, :, 1],
    ]

    return permuted_indices


@lru_cache(maxsize=2)
def combo_suit_permutation_inverse_tensor(
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return [24, 1326] tensor mapping canonical combos back to original ordering."""
    permuted = combo_suit_permutation_tensor()
    inverse = torch.argsort(permuted, dim=1)
    if device is not None:
        inverse = inverse.to(device)
    return inverse


def calculate_unblocked_mass(
    target: torch.Tensor,
) -> torch.Tensor:
    """Calculate unblocked mass for each hand (generally for getting opponent unblocked mass).
    Equivalent to multiplying by the compatibility matrix ~combo_blocking_tensor().
    Used for finding matchup. See DEVN paper for details. CFV = matchup * EV.

    Note that blocking = combo_onehot @ combo_onehot.T - torch.eye(1326).
    Optimization: compatible = ~blocking = 1 - blocking
    = 1 - (combo_onehot @ combo_onehot.T) + torch.eye(1326)

    BUT WE ARE NOT USING THIS OPTIMIZATION.
    It's numerically unstable and gives incorrect results.

    Args:
        target: [..., 1326] tensor of reach weights for each node.
        device: Device for the combo_onehot tensor. If None, inferred from target.

    Returns:
        [..., 1326] tensor of unblocked mass for each hand.
    """
    target_batched = target.view(-1, NUM_HANDS)
    compatible = combo_compatible_tensor(device=target.device).float()
    multiply = target_batched @ compatible
    # Make sure it's min-0 (sometimes get numerical precision issues)
    return multiply.view_as(target).clamp(min=0.0)
