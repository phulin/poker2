from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import torch

NUM_HANDS = 1326
PREFLOP_HANDS = 169

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


@lru_cache(maxsize=2)
def combo_to_preflop_class_tensor(
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return [1326] class ids for the 169 rank/suit preflop abstraction.

    Diagonal entries are pairs. For non-pairs, ``hi * 13 + lo`` is suited and
    ``lo * 13 + hi`` is offsuit, matching the conventional 13x13 grid split.
    """
    combos = hand_combos_tensor(device=device)
    ranks = combos % 13
    suits = combos // 13
    rank_a = ranks[:, 0]
    rank_b = ranks[:, 1]
    hi = torch.maximum(rank_a, rank_b)
    lo = torch.minimum(rank_a, rank_b)
    suited = suits[:, 0] == suits[:, 1]
    class_ids = torch.where(suited, hi * 13 + lo, lo * 13 + hi)
    if device is not None:
        class_ids = class_ids.to(device)
    return class_ids.long()


@lru_cache(maxsize=2)
def preflop_class_multiplicity_tensor(
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return [169] combo counts per preflop rank class.

    Pairs have multiplicity 6, suited non-pairs 4, and offsuit non-pairs 12.
    The tensor sums to 1326.
    """
    class_ids = combo_to_preflop_class_tensor(device=device)
    weights = torch.ones(NUM_HANDS, dtype=torch.float32, device=device)
    multiplicity = torch.zeros(PREFLOP_HANDS, dtype=torch.float32, device=device)
    multiplicity.scatter_add_(0, class_ids, weights)
    return multiplicity


@lru_cache(maxsize=2)
def preflop_class_compatibility_counts_tensor(
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return [169, 169] disjoint-combo counts between preflop classes.

    Entry ``[hero_class, opp_class]`` is the number of combos in
    ``opp_class`` that do not share a card with any fixed combo from
    ``hero_class``. Suit symmetry makes this count invariant within each
    hero class, so the table exactly reproduces combo-level unblocked-mass
    projection for class-constant preflop ranges.
    """
    class_ids = combo_to_preflop_class_tensor(device=device)
    combo_cards = combo_to_onehot_tensor(device=device).to(torch.float32)
    compatible = (combo_cards @ combo_cards.T) < 0.5

    opp_class_onehot = torch.zeros(
        NUM_HANDS, PREFLOP_HANDS, dtype=torch.float32, device=device
    )
    opp_class_onehot.scatter_(1, class_ids[:, None], 1.0)
    per_combo_counts = compatible.to(torch.float32) @ opp_class_onehot

    counts = torch.zeros(
        PREFLOP_HANDS, PREFLOP_HANDS, dtype=torch.float32, device=device
    )
    counts.scatter_add_(
        0,
        class_ids[:, None].expand(-1, PREFLOP_HANDS),
        per_combo_counts,
    )
    multiplicity = preflop_class_multiplicity_tensor(device=device).to(
        dtype=counts.dtype
    )
    return counts / multiplicity[:, None]


def preflop_class_unblocked_mass(class_mass: torch.Tensor) -> torch.Tensor:
    """Project class mass through exact preflop card-removal compatibility.

    ``class_mass`` is a tensor whose final axis is the 169-class preflop
    abstraction and whose entries are probability/reach mass per class. The
    returned tensor has the same shape; each output class is the combo-level
    mass compatible with a fixed combo in that class, assuming uniform mass
    inside every preflop class.
    """
    if class_mass.shape[-1] != PREFLOP_HANDS:
        raise ValueError(
            f"expected final axis {PREFLOP_HANDS}, got {class_mass.shape[-1]}"
        )
    multiplicity = preflop_class_multiplicity_tensor(device=class_mass.device).to(
        dtype=class_mass.dtype
    )
    compatibility = preflop_class_compatibility_counts_tensor(
        device=class_mass.device
    ).to(dtype=class_mass.dtype)
    combo_mass_per_class = class_mass / multiplicity
    return combo_mass_per_class @ compatibility.T


def expand_169_to_1326(
    values_169: torch.Tensor,
    *,
    divide_by_multiplicity: bool = False,
) -> torch.Tensor:
    """Expand a tensor whose final axis is 169 preflop classes to 1326 combos.

    Set ``divide_by_multiplicity=True`` for probability mass beliefs that should
    be distributed uniformly within each class. Leave it false for class values
    or logits that should be copied to every combo in the class.
    """
    if values_169.shape[-1] != PREFLOP_HANDS:
        raise ValueError(
            f"expected final axis {PREFLOP_HANDS}, got {values_169.shape[-1]}"
        )
    class_ids = combo_to_preflop_class_tensor(device=values_169.device)
    expanded = values_169.index_select(-1, class_ids)
    if divide_by_multiplicity:
        multiplicity = preflop_class_multiplicity_tensor(
            device=values_169.device
        ).to(dtype=values_169.dtype)
        expanded = expanded / multiplicity.index_select(0, class_ids)
    return expanded


def collapse_1326_to_169(
    values_1326: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Collapse a tensor whose final axis is 1326 combos to 169 classes.

    ``reduction="mean"`` is for class values/targets. ``reduction="sum"`` is
    for probability mass beliefs.
    """
    if values_1326.shape[-1] != NUM_HANDS:
        raise ValueError(
            f"expected final axis {NUM_HANDS}, got {values_1326.shape[-1]}"
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    class_ids = combo_to_preflop_class_tensor(device=values_1326.device)
    out = values_1326.new_zeros(*values_1326.shape[:-1], PREFLOP_HANDS)
    index = class_ids.expand(*values_1326.shape[:-1], NUM_HANDS)
    out.scatter_add_(-1, index, values_1326)
    if reduction == "mean":
        multiplicity = preflop_class_multiplicity_tensor(
            device=values_1326.device
        ).to(dtype=values_1326.dtype)
        out = out / multiplicity
    return out


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


def board_allowed_hands(board: torch.Tensor) -> torch.Tensor:
    """Return a bool mask of hands compatible with each public board.

    Args:
        board: Tensor with shape ``(..., board_cards)`` containing card ids and
            ``-1`` padding for undealt board slots.

    Returns:
        Bool tensor with shape ``(..., 1326)`` where True means the private hand
        does not use any public board card.
    """
    if board.shape[-1] == 0:
        return torch.ones(
            *board.shape[:-1], NUM_HANDS, dtype=torch.bool, device=board.device
        )

    flat_board = board.reshape(-1, board.shape[-1]).long()
    board_onehot = torch.zeros(
        flat_board.shape[0], 52, dtype=torch.bool, device=board.device
    )
    valid = flat_board >= 0
    if valid.any():
        rows = torch.arange(flat_board.shape[0], device=board.device)[
            :, None
        ].expand_as(flat_board)
        board_onehot[rows[valid], flat_board[valid]] = True

    blocked_count = (
        combo_to_onehot_tensor(device=board.device).float() @ board_onehot.float().T
    )
    allowed = blocked_count.T < 0.5
    return allowed.reshape(*board.shape[:-1], NUM_HANDS)


@lru_cache(maxsize=2)
def combo_to_onehot_tensor(device: torch.device | None = None) -> torch.Tensor:
    """Return [1326, 52] bool tensor of one-hot encoded combos."""
    combos = hand_combos_tensor(device=device)  # [1326, 2]
    combo_onehot = torch.zeros(1326, 52, dtype=torch.bool, device=device)
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


def _canonical_suit_key(cards: tuple[int, ...]) -> tuple[int, ...]:
    best: tuple[int, ...] | None = None
    for perm in permutations(range(4)):
        mapped = tuple(sorted(perm[card // 13] * 13 + card % 13 for card in cards))
        if best is None or mapped < best:
            best = mapped
    if best is None:
        raise ValueError("cards must be non-empty")
    return best


@lru_cache(maxsize=1)
def _canonical_flops_with_weights_cpu() -> tuple[torch.Tensor, torch.Tensor]:
    counts: dict[tuple[int, ...], int] = {}
    for flop in combinations(range(52), 3):
        key = _canonical_suit_key(flop)
        counts[key] = counts.get(key, 0) + 1

    flops = torch.tensor(tuple(sorted(counts)), dtype=torch.long)
    weights = torch.tensor(
        [counts[tuple(flop.tolist())] for flop in flops], dtype=torch.float32
    )
    if flops.shape != (1755, 3):
        raise RuntimeError(f"Expected 1755 canonical flops, got {flops.shape[0]}")
    return flops, weights


def canonical_flops_with_weights(
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical flop representatives and raw-flop orbit weights."""
    flops, weights = _canonical_flops_with_weights_cpu()
    if device is not None:
        flops = flops.to(device)
        weights = weights.to(device)
    return flops, weights


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

    This uses the equivalent inclusion-exclusion form:
    unblocked[(a,b)] = total - mass_with_card[a] - mass_with_card[b] + target[(a,b)].

    Args:
        target: [..., 1326] tensor of reach weights for each node.
        device: Device for the combo_onehot tensor. If None, inferred from target.

    Returns:
        [..., 1326] tensor of unblocked mass for each hand.
    """
    target_batched = target.view(-1, NUM_HANDS).float()
    combos = hand_combos_tensor(device=target.device)
    card_a = combos[:, 0].long()
    card_b = combos[:, 1].long()

    total = target_batched.sum(dim=-1, keepdim=True)
    cardsum = torch.zeros(
        target_batched.shape[0],
        52,
        dtype=target_batched.dtype,
        device=target_batched.device,
    )
    card_a_idx = card_a[None, :].expand(target_batched.shape[0], -1)
    card_b_idx = card_b[None, :].expand(target_batched.shape[0], -1)
    cardsum.scatter_add_(1, card_a_idx, target_batched)
    cardsum.scatter_add_(1, card_b_idx, target_batched)

    multiply = total - cardsum[:, card_a] - cardsum[:, card_b] + target_batched
    # Make sure it's min-0 (sometimes get numerical precision issues)
    return multiply.view_as(target).clamp(min=0.0)
