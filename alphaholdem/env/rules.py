from __future__ import annotations

from typing import Iterable, List, Tuple, Sequence
import random

try:
    from line_profiler import profile
except ImportError:  # pragma: no cover

    def profile(f):
        return f


import torch

# Card encoding: 0..51, rank = c % 13 (2..A), suit = c // 13 (0..3)

RANKS = list(range(13))  # 0..12 means 2..A
SUITS = list(range(4))


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
    cards = cards.to(torch.long)
    suits = cards // 13
    ranks = cards % 13
    return suits, ranks


def unfold_conv1d_ones(input_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Simple unfold-based convolution with ones kernel"""
    unfolded = input_tensor.unfold(-1, kernel_size, 1)
    return torch.sum(unfolded, dim=-1)


@profile
def create_comparison_vector(ab_batch: torch.Tensor) -> torch.Tensor:
    """Create comparison vector for poker hand evaluation.

    Args:
        ab_batch: [N, P, 4, 13] - batch of hands for P players

    Returns:
        compare: [N, P, 26] - comparison vector with hand strengths
    """
    N = ab_batch.size(0)
    P = ab_batch.size(1)
    device = ab_batch.device

    # Sort ranks by count (descending), then by rank (descending)
    rank_sum = ab_batch.sum(dim=2)  # [N, P, 13]
    composite_key = rank_sum * 1000 + torch.arange(13, device=device)
    top_ranks_indices = torch.argsort(
        composite_key, dim=2, descending=True
    )  # [N, P, 13]
    top_ranks_values = torch.gather(rank_sum, 2, top_ranks_indices)  # [N, P, 13]

    # == 1. STRAIGHT FLUSH ==
    # Stack to [N, P, 4, 13] and check each for straight flush using a convolution with [1, 1, 1, 1]
    # Create a second tensor for straight-checking which copies the last rank column back to the beginning
    # For straight checking, pad/copy last rank (Ace) to the front for wheel
    ab_ext_straight = torch.cat(
        [ab_batch[..., 12:13], ab_batch], dim=-1
    )  # [N, P, 4, 14]
    conv_sf = unfold_conv1d_ones(ab_ext_straight, 5)  # [N, P, 4, 10]
    sf_mask = (conv_sf == 5).any(dim=2)  # [N, P, 10]
    sf_ranks_spread = torch.where(
        sf_mask,
        torch.arange(10, device=device).view(1, 1, 10).expand_as(sf_mask),
        -1,
    )  # [N, P, 10]
    sf_ranks = sf_ranks_spread.max(dim=2).values.view(N, P, 1)  # [N, P, 1]

    # == 2. FOUR OF A KIND ==
    four_with_kickers = torch.where(
        (top_ranks_values[:, :, 0] >= 4).view(N, P, 1),
        top_ranks_indices[:, :, :2],
        -1,
    )

    # == 3. FULL HOUSE ==
    full_house = torch.where(
        ((top_ranks_values[:, :, 0] >= 3) & (top_ranks_values[:, :, 1] >= 2)).view(
            N, P, 1
        ),
        top_ranks_indices[:, :, :2],
        -1,
    )

    # == 4. FLUSH ==
    # Flush: sum along suit dimension (rank axis=2), check for >= 5
    suit_sum = ab_batch.sum(dim=3)  # [N, P, 4]
    flush_mask = (suit_sum >= 5).any(dim=2)  # [N, P]
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
    top_suit_ranks = top_suit_cards.argsort(dim=2, descending=True)[
        :, :, :5
    ]  # [N, P, 5]
    flush = torch.where(
        flush_mask.view(N, P, 1),
        top_suit_ranks,
        -1,
    )

    # == 5. STRAIGHT ==
    # Straight: take rank presence, convolve like with straight-flush
    rank_presence_straight = ab_ext_straight.sum(dim=2).clamp(0, 1)  # [N, P, 14]
    conv_straight = unfold_conv1d_ones(rank_presence_straight, 5)
    straight_mask = (conv_straight == 5).any(dim=2)  # [N, P]
    straight_high = conv_straight.argmax(dim=2)  # [N, P]
    straight = torch.where(
        straight_mask,
        straight_high,
        -1,
    ).view(N, P, 1)

    # == 6. THREE OF A KIND ==
    three_with_kickers = torch.where(
        (top_ranks_values[:, :, 0] >= 3).view(N, P, 1),
        top_ranks_indices[:, :, :3],
        -1,
    )

    # == 7. TWO PAIR ==
    two_pair_with_kickers = torch.where(
        ((top_ranks_values[:, :, 0] >= 2) & (top_ranks_values[:, :, 1] >= 2)).view(
            N, P, 1
        ),
        top_ranks_indices[:, :, :3],
        -1,
    )

    # == 8. ONE PAIR ==
    one_pair_with_kickers = torch.where(
        (top_ranks_values[:, :, 0] >= 2).view(N, P, 1),
        top_ranks_indices[:, :, :4],
        -1,
    )

    # == 9. HIGH CARD ==
    high_card_with_kickers = top_ranks_indices[:, :, :5]

    # Concatenate all hand types
    compare = torch.cat(
        [
            sf_ranks,
            four_with_kickers,
            full_house,
            flush,
            straight,
            three_with_kickers,
            two_pair_with_kickers,
            one_pair_with_kickers,
            high_card_with_kickers,
        ],
        dim=2,
    )

    return compare


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
                result += f"{hand_type} [{" ".join([str(n) for n in hand_values.tolist()])}]: "
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
    a_onehot = torch.zeros(4, 13, dtype=torch.long)
    b_onehot = torch.zeros(4, 13, dtype=torch.long)

    for card in a_cards:
        suit_idx = card // 13
        rank_idx = card % 13
        a_onehot[suit_idx, rank_idx] = 1

    for card in b_cards:
        suit_idx = card // 13
        rank_idx = card % 13
        b_onehot[suit_idx, rank_idx] = 1

    # Use batch comparison with single hand
    result = compare_7_batch(a_onehot.unsqueeze(0), b_onehot.unsqueeze(0))
    return int(result[0].item())


def compare_7_batch(
    a_batch: torch.Tensor,
    b_batch: torch.Tensor,
) -> torch.Tensor:
    """Vectorized comparison for batches of 7-card hands using one-hot planes.

    Inputs must be one-hot planes [N,4,13]; returns [N] in {1, -1, 0}.
    The score layout per hand is [category, t1, t2, t3, t4, t5] where
    category is in {8:SF, 7:4K, 6:FH, 5:F, 4:S, 3:3K, 2:2P, 1:1P, 0:HC} and
    t1..t5 are tiebreaker ranks in descending priority for that category.
    """
    assert isinstance(a_batch, torch.Tensor) and isinstance(b_batch, torch.Tensor)
    assert a_batch.shape == b_batch.shape
    assert a_batch.dim() == 3
    assert a_batch.shape[1:] == (4, 13)
    assert a_batch.dtype == torch.long
    assert b_batch.dtype == torch.long

    ab_batch = torch.stack([a_batch, b_batch], dim=1)  # [N,2,4,13]

    # Create comparison vector
    compare = create_comparison_vector(ab_batch)  # [N, 2, 20]
    # print(debug_comparison_vector(compare))

    # Compare hands
    diff = compare[:, 0, :] - compare[:, 1, :]  # [N, 20]
    diff_nonzero = (diff != 0).long()  # [N, 20]
    first_nonzero = diff_nonzero.argmax(dim=1)  # [N]
    has_nonzero = diff_nonzero.any(dim=1)  # [N]
    # Gather the first nonzero value for each row
    first_values = torch.gather(diff, 1, first_nonzero.unsqueeze(1)).squeeze(1)  # [N]
    # If no nonzero, set to 0
    return torch.where(has_nonzero, first_values.clamp(-1, 1), 0)
