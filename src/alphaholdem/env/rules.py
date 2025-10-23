from __future__ import annotations

import random
from enum import Enum
from typing import List

import torch

from alphaholdem.env.card_utils import hand_combos_tensor
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS

# Card encoding: 0..51, rank = c % 13 (2..A), suit = c // 13 (0..3)

RANKS = list(range(13))  # 0..12 means 2..A
SUITS = list(range(4))


class HandType(Enum):
    NUM_HAND_TYPES = 10
    STRAIGHT_FLUSH = 9
    FOUR_OF_A_KIND = 8
    FULL_HOUSE = 7
    FLUSH = 6
    STRAIGHT = 5
    THREE_OF_A_KIND = 4
    TWO_PAIR = 3
    ONE_PAIR = 2
    HIGH_CARD = 1
    EMPTY = 0


def new_shuffled_deck(rng: random.Random) -> List[int]:
    deck = list(range(52))
    rng.shuffle(deck)
    return deck


def rank(card: int) -> int:
    return card % 13


def suit(card: int) -> int:
    return card // 13


# Tensor helpers


def cards_to_onehot_indices(cards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert numeric card tensor to (suit_idx, rank_idx) tensors.

    cards: tensor of any shape with integral dtype; returns tensors of same shape
    containing suit indices [0..3] and rank indices [0..12].
    """
    cards = cards.to(torch.int)
    suits = cards // 13
    ranks = cards % 13
    return suits, ranks


def unfold_conv1d_ones(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Simple unfold-based convolution with ones kernel"""
    unfolded = input_tensor.unfold(-1, kernel_size, 1)
    return torch.sum(unfolded, dim=-1)


def rank_hands(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank all 1326 hands on a [N, 1326] batch of boards.

    Args:
        board: [N, 5] - batch of board card indices

    Returns:
        [N, 1326] - batch of hand ranks (higher number means stronger hand)
        [N, 1326] - batch of sorted indices
    """
    N = board.size(0)
    device = board.device

    # Enumerate all possible 2-card poker hands
    all_combos = torch.arange(NUM_HANDS, device=device)
    # Get the [1326, 2] tensor mapping combo index to two card indices
    combo_to_hand = hand_combos_tensor(device=device)
    # Convert to suit and rank tensors for each hole card
    combo_to_suits, combo_to_ranks = cards_to_onehot_indices(combo_to_hand)

    # Construct a tensor of zeros, to be filled with ones for each card in each hand plus the board
    hand_vectors = torch.zeros(N, NUM_HANDS, 4, 13, device=device)
    # For each hand, set the relevant entries (hand holecards) to 1
    for card_idx in range(2):
        hand_vectors[
            :, all_combos, combo_to_suits[:, card_idx], combo_to_ranks[:, card_idx]
        ] = 1

    # For each board, set the relevant entries (board cards) in all hand vectors to 1
    board_suits, board_ranks = cards_to_onehot_indices(board)
    hand_vectors[:, :, board_suits, board_ranks] = 1

    # Compute the 26-field (4 bits per field) comparison vector for every hand+board in the batch
    comparison_vectors = create_comparison_vector(hand_vectors)

    # Add 1 so that -1 becomes 0 and 0..13 => 1..14 (so all values are >= 0 for bit packing)
    cv = comparison_vectors.to(torch.int)  # [N, 1326, 6]

    # Prepare offsets for bit-packing the 6-field group of 4 bits, most significant fields first
    offsets = 4 * (5 - torch.arange(6, device=cv.device, dtype=torch.int))
    packed = (cv << offsets[None, None, :]).sum(dim=-1)

    # Sort hands primarily by lo, then by hi (breaking ties)
    sorted_indices = torch.argsort(packed, dim=1, stable=True)

    # Sort lo and hi values with final permutation, for group identification
    sorted_packed = packed.gather(1, sorted_indices)

    # A hand is the start of a new group (unique hand strength) if lo or hi changes from previous
    is_group_start = sorted_packed[:, :-1] != sorted_packed[:, 1:]

    # Always start with a group at the first position
    is_group_start = torch.cat([torch.ones(N, 1, device=device), is_group_start], dim=1)

    # Assign a unique increasing label (group id, i.e. rank) to each group
    group_ids = torch.cumsum(is_group_start, dim=1, dtype=torch.int)

    # Scatter those ranks (group_ids) back to the original hand order for each batch
    result = torch.zeros(N, NUM_HANDS, dtype=torch.int, device=device)
    result.scatter_(1, sorted_indices, group_ids)

    return result, sorted_indices


def create_comparison_vector(ab_batch: torch.Tensor) -> torch.Tensor:
    """Create comparison vector for poker hand evaluation.

    Args:
        ab_batch: [N, P, 4, 13] - batch of hands for P players

    Returns:
        compare: [N, P, 26] - comparison vector with hand strengths
    """
    N = ab_batch.size(0)
    P = ab_batch.size(1)
    dtype = torch.int
    device = ab_batch.device

    # 9 hands, 1 slot for hand type, 5 slots for kickers
    result = torch.zeros(
        N, P, HandType.NUM_HAND_TYPES.value, 6, device=device, dtype=dtype
    )

    # Sort ranks by count (descending), then by rank (descending)
    # Use topk instead of full argsort for better performance
    rank_sum = ab_batch.sum(dim=2, dtype=dtype)  # [N, P, 13]
    composite_key = (rank_sum << 8) | torch.arange(13, device=device, dtype=dtype)
    top_ranks_values, top_ranks_indices = torch.topk(
        composite_key, k=5, dim=2
    )  # [N, P, 5] - only need top 5 for all hand types
    top_ranks_values = top_ranks_values >> 8  # Convert back to actual counts

    # == 9. STRAIGHT FLUSH ==
    # Stack to [N, P, 4, 13] and check each for straight flush using a convolution with [1, 1, 1, 1]
    # Create a second tensor for straight-checking which copies the last rank column back to the beginning
    # For straight checking, pad/copy last rank (Ace) to the front for wheel
    ab_ext_straight = torch.cat(
        [ab_batch[..., 12:13], ab_batch], dim=-1
    )  # [N, P, 4, 14]
    conv_sf = unfold_conv1d_ones(ab_ext_straight, 5)  # [N, P, 4, 10]
    sf_win_max = torch.amax(
        conv_sf, dim=2
    )  # [N, P, 10] - max convolution values per window
    sf_win_mask = sf_win_max == 5  # [N, P, 10] - which windows have straight flush
    windows = torch.arange(10, device=device, dtype=dtype)  # [10] - window indices
    sf_last = (sf_win_mask.to(windows.dtype) * windows).amax(
        dim=2, keepdim=True
    )  # [N, P, 1]
    has_sf = sf_win_mask.any(dim=2)
    result[:, :, HandType.STRAIGHT_FLUSH.value, 0] = torch.where(
        has_sf, HandType.STRAIGHT_FLUSH.value, 0
    )
    result[:, :, HandType.STRAIGHT_FLUSH.value, 1:2] = torch.where(
        has_sf[:, :, None], sf_last, -1
    )  # [N, P, 1]

    # == 8. FOUR OF A KIND ==
    has_quads_0 = (top_ranks_values[:, :, 0] >= 4).view(N, P, 1)
    result[:, :, HandType.FOUR_OF_A_KIND.value, 0:1].masked_fill_(
        has_quads_0, HandType.FOUR_OF_A_KIND.value
    )
    result[:, :, HandType.FOUR_OF_A_KIND.value, 1:3] = torch.where(
        has_quads_0,
        top_ranks_indices[:, :, :2],
        0,
    )  # [N, P, 2]

    # == 7. FULL HOUSE ==
    has_triple_0 = (top_ranks_values[:, :, 0] >= 3).view(N, P, 1)
    has_pair_0 = (top_ranks_values[:, :, 0] >= 2).view(N, P, 1)
    has_pair_1 = (top_ranks_values[:, :, 1] >= 2).view(N, P, 1)
    has_full_house = has_triple_0 & has_pair_1
    result[:, :, HandType.FULL_HOUSE.value, 0:1].masked_fill_(
        has_full_house, HandType.FULL_HOUSE.value
    )
    result[:, :, HandType.FULL_HOUSE.value, 1:3] = torch.where(
        has_full_house,
        top_ranks_indices[:, :, :2],
        0,
    )

    # == 4. FLUSH ==
    # Flush: sum along suit dimension (rank axis=2), check for >= 5
    suit_sum = ab_batch.sum(dim=3, dtype=dtype)  # [N, P, 4]
    flush_mask = (
        torch.amax(suit_sum, dim=2) >= 5
    )  # [N, P] - use numeric amax instead of boolean any
    flush_suit = suit_sum.argmax(dim=2)  # [N, P]
    # ab_batch shape: [N, P, 4, 13]
    # flush_suit shape: [N, P]
    # To select the cards of the flush suit for each hand, use torch.gather:
    # First, expand flush_suit to match ab_batch's shape for the suit dimension
    flush_suit_expanded = flush_suit.unsqueeze(-1).expand(-1, -1, 13)  # [N, P, 13]
    # Gather along the suit dimension (dim=2)
    top_suit_cards = ab_batch.gather(2, flush_suit_expanded.unsqueeze(2)).squeeze(
        2
    )  # [N, P, 13]
    top_suit_ranks = top_suit_cards.int().argsort(dim=2, descending=True)[
        :, :, :5
    ]  # [N, P, 5]
    result[:, :, HandType.FLUSH.value, 0].masked_fill_(flush_mask, HandType.FLUSH.value)
    result[:, :, HandType.FLUSH.value, 1:6] = torch.where(
        flush_mask[:, :, None],
        top_suit_ranks,
        0,
    )

    # == 5. STRAIGHT ==
    # Straight: take rank presence, convolve like with straight-flush
    rank_presence_straight = ab_ext_straight.sum(dim=2, dtype=dtype).clamp(
        0, 1
    )  # [N, P, 14]
    conv_straight = unfold_conv1d_ones(rank_presence_straight, 5)
    straight_mask = torch.amax(conv_straight, dim=2) == 5  # [N, P]
    straight_low = conv_straight.argmax(dim=2)  # [N, P]
    result[:, :, HandType.STRAIGHT.value, 0].masked_fill_(
        straight_mask, HandType.STRAIGHT.value
    )
    result[:, :, HandType.STRAIGHT.value, 1] = torch.where(
        straight_mask,
        straight_low,
        0,
    )

    # == 4. THREE OF A KIND ==
    has_three = (top_ranks_values[:, :, 0] >= 3).view(N, P, 1)
    result[:, :, HandType.THREE_OF_A_KIND.value, 0:1].masked_fill_(
        has_three, HandType.THREE_OF_A_KIND.value
    )
    result[:, :, HandType.THREE_OF_A_KIND.value, 1:4] = torch.where(
        has_three,
        top_ranks_indices[:, :, :3],
        0,
    )

    # == 3. TWO PAIR ==
    has_two_pair = (top_ranks_values[:, :, 0] >= 2).view(N, P, 1) & (
        top_ranks_values[:, :, 1] >= 2
    ).view(N, P, 1)
    result[:, :, HandType.TWO_PAIR.value, 0:1].masked_fill_(
        has_two_pair, HandType.TWO_PAIR.value
    )
    result[:, :, HandType.TWO_PAIR.value, 1:4] = torch.where(
        has_two_pair,
        top_ranks_indices[:, :, :3],
        0,
    )

    # == 2. ONE PAIR ==
    has_pair_0 = (top_ranks_values[:, :, 0] >= 2).view(N, P, 1)
    result[:, :, HandType.ONE_PAIR.value, 0:1].masked_fill_(
        has_pair_0, HandType.ONE_PAIR.value
    )
    result[:, :, HandType.ONE_PAIR.value, 1:5] = torch.where(
        has_pair_0,
        top_ranks_indices[:, :, :4],
        0,
    )

    # == 1. HIGH CARD ==
    high_card_with_kickers = top_ranks_indices[:, :, :5]
    result[:, :, HandType.HIGH_CARD.value, 0] = 1
    result[:, :, HandType.HIGH_CARD.value, 1:6] = high_card_with_kickers

    first_nonzero_row = result[:, :, :, 0].argmax(dim=2)
    return result.gather(
        2, first_nonzero_row[:, :, None, None].expand(-1, -1, -1, 6)
    ).squeeze(2, 3)


def debug_comparison_vector(compare: torch.Tensor, hand_index: int = 0) -> str:
    """Debug function to explain what a comparison vector represents.

    Args:
        compare: [N, 2, 26] - comparison vector from create_comparison_vector
        hand_index: which hand in the batch to debug

    Returns:
        String explanation of the hand strengths
    """
    hand_data = compare[hand_index]  # [2, 26]

    # Hand type names and their positions in the comparison vector
    # Dimensions: sf_ranks(1) + four_with_kickers(2) + full_house(2) + flush(5) + straight(1) +
    #             three_with_kickers(3) + two_pair_with_kickers(3) + one_pair_with_kickers(4) + high_card_with_kickers(5) = 26
    hand_types = [
        ("Straight Flush", 0, 1),  # 0-0
        ("Four of a Kind", 1, 2),  # 1-2
        ("Full House", 3, 2),  # 3-4
        ("Flush", 5, 5),  # 5-9
        ("Straight", 10, 1),  # 10-10
        ("Three of a Kind", 11, 3),  # 11-13
        ("Two Pair", 14, 3),  # 14-16
        ("One Pair", 17, 4),  # 17-20
        ("High Card", 21, 5),  # 21-25
    ]

    rank_names = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

    result = f"Hand {hand_index} Comparison:\n"
    result += "=" * 50 + "\n"

    for player in range(2):
        result += f"\nPlayer {player + 1}:\n"
        result += "-" * 20 + "\n"

        for hand_type, start_idx, length in hand_types:
            end_idx = start_idx + length
            hand_values = hand_data[player, start_idx:end_idx]

            # Check if this hand type is present (not all -1)
            if not torch.all(hand_values == -1):
                result += f"{hand_type} [{' '.join([str(n) for n in hand_values.tolist()])}]: "
                rank_strs = []
                for val in hand_values:
                    if val != -1:
                        rank_strs.append(rank_names[val.item()])
                    else:
                        rank_strs.append("X")
                result += " ".join(rank_strs) + "\n"
            else:
                result += f"{hand_type}: Not present\n"

    return result


def compare_7(a_cards: List[int], b_cards: List[int]) -> int:
    """Compare two 7-card hands and return comparison result.

    Args:
        a_cards: List of 7 card integers for hand A
        b_cards: List of 7 card integers for hand B

    Returns:
        int: 1 if hand A wins, -1 if hand B wins, 0 if tie
    """
    assert len(a_cards) == 7, f"Expected 7 cards for hand A, got {len(a_cards)}"
    assert len(b_cards) == 7, f"Expected 7 cards for hand B, got {len(b_cards)}"

    # Convert card lists to one-hot planes
    a_onehot = torch.zeros(4, 13, dtype=torch.int)
    b_onehot = torch.zeros(4, 13, dtype=torch.int)

    for card in a_cards:
        suit_idx = card // 13
        rank_idx = card % 13
        a_onehot[suit_idx, rank_idx] = 1

    for card in b_cards:
        suit_idx = card // 13
        rank_idx = card % 13
        b_onehot[suit_idx, rank_idx] = 1

    # Use batch comparison with single hand
    result = compare_7_batches(a_onehot.unsqueeze(0), b_onehot.unsqueeze(0))
    return int(result[0].item())


def compare_7_single_batch(ab_batch: torch.Tensor) -> torch.Tensor:
    """Compare two 7-card hands and return comparison result.

    Args:
        ab_batch: [N, 2, 4, 13] - batch of hands for 2 players

    Returns:
        compare: [N] - comparison vector with winner for each hand
    """
    assert ab_batch.dim() == 4
    assert ab_batch.shape[1:] == (2, 4, 13)
    assert ab_batch.dtype == torch.bool

    # Create comparison vector
    compare = create_comparison_vector(ab_batch)  # [N, 2, 6]

    offsets = 4 * (5 - torch.arange(6, device=compare.device, dtype=torch.int))
    packed = (compare << offsets[None, None, :]).sum(dim=-1)

    return (packed[:, 0] - packed[:, 1]).clamp(-1, 1)


def compare_7_batches(
    a_batch: torch.Tensor,
    b_batch: torch.Tensor,
) -> torch.Tensor:
    """Vectorized comparison for batches of 7-card hands using one-hot planes.

    Inputs must be one-hot planes [N,4,13]; returns [N] in {1, -1, 0}.
    """
    assert a_batch.shape == b_batch.shape
    assert a_batch.dim() == 3
    assert a_batch.shape[1:] == (4, 13)

    ab_batch = torch.stack([a_batch, b_batch], dim=1).bool()  # [N, 2, 4, 13]

    return compare_7_single_batch(ab_batch)
