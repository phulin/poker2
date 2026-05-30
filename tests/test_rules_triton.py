from __future__ import annotations

import pytest
import torch

from p2.env.rules import compare_7_single_batch


def C(r: int, s: int) -> int:
    return s * 13 + r


def _cards_to_plane(cards: list[int]) -> torch.Tensor:
    out = torch.zeros(4, 13, dtype=torch.bool, device="cuda")
    card_tensor = torch.tensor(cards, dtype=torch.long, device="cuda")
    out[card_tensor // 13, card_tensor % 13] = True
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton test")
def test_compare_7_single_batch_triton_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.env.rules_triton import (
        compare_7_cards_single_batch_triton,
        compare_7_single_batch_triton,
    )

    generator = torch.Generator(device="cuda").manual_seed(1234)
    batch = torch.zeros((2048, 2, 4, 13), dtype=torch.bool, device="cuda")
    cards_batch = torch.empty((2048, 2, 7), dtype=torch.long, device="cuda")
    for row in range(batch.shape[0]):
        for player in range(2):
            cards = torch.randperm(52, generator=generator, device="cuda")[:7]
            cards_batch[row, player] = cards
            batch[row, player, cards // 13, cards % 13] = True

    expected = compare_7_single_batch(batch)
    actual = compare_7_single_batch_triton(batch)
    assert torch.equal(expected.to(torch.int32), actual)

    cards_actual = compare_7_cards_single_batch_triton(cards_batch)
    assert torch.equal(expected.to(torch.int32), cards_actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton test")
def test_rank_7_cards_matches_comparator() -> None:
    """The batched rank scorer must induce the same ordering as the
    separately-validated comparator kernel for distinct-card hands."""
    pytest.importorskip("triton")
    from p2.env.rules_triton import (
        compare_7_cards_single_batch_triton,
        rank_7_cards_single_batch_triton,
    )

    generator = torch.Generator(device="cuda").manual_seed(2024)
    n = 4096
    cards_batch = torch.empty((n, 2, 7), dtype=torch.long, device="cuda")
    for row in range(n):
        for player in range(2):
            cards_batch[row, player] = torch.randperm(52, generator=generator, device="cuda")[:7]

    expected_sign = compare_7_cards_single_batch_triton(cards_batch)  # [n] in {-1,0,1}
    scores = rank_7_cards_single_batch_triton(cards_batch.reshape(-1, 7)).view(n, 2)
    diff = scores[:, 0] - scores[:, 1]
    actual_sign = torch.sign(diff).to(torch.int32)
    assert torch.equal(expected_sign, actual_sign)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton test")
def test_rank_hands_triton_matches_per_hand_scorer() -> None:
    """The fused (board+combos gather) path must agree with scoring the
    explicitly-materialized 7-card hands, on legal (distinct-card) combos."""
    pytest.importorskip("triton")
    from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
    from p2.env.rules_triton import (
        rank_7_cards_single_batch_triton,
        rank_hand_scores_triton,
        rank_hands_triton,
    )

    generator = torch.Generator(device="cuda").manual_seed(7)
    m = 64
    board = torch.empty((m, 5), dtype=torch.long, device="cuda")
    for i in range(m):
        board[i] = torch.randperm(52, generator=generator, device="cuda")[:5]

    combos = hand_combos_tensor(device="cuda")  # [1326, 2]
    holes = combos[None].expand(m, NUM_HANDS, 2)
    boards_b = board[:, None, :].expand(m, NUM_HANDS, 5)
    cards = torch.cat([holes, boards_b], dim=-1)  # [m, 1326, 7]
    cards_flat = cards.reshape(-1, 7)

    ref = rank_7_cards_single_batch_triton(cards_flat).view(m, NUM_HANDS)
    scores_only = rank_hand_scores_triton(board)
    fused, _ = rank_hands_triton(board)

    # Blocked combos (hole card shares with board) are undefined; exclude them.
    sorted_cards, _ = cards.sort(dim=-1)
    legal = (sorted_cards[..., 1:] != sorted_cards[..., :-1]).all(dim=-1)
    assert torch.equal(ref[legal], scores_only[legal])
    assert torch.equal(ref[legal], fused[legal])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton test")
def test_compare_7_single_batch_triton_matches_pytorch_edge_cases() -> None:
    pytest.importorskip("triton")
    from p2.env.rules_triton import (
        compare_7_cards_single_batch_triton,
        compare_7_single_batch_triton,
    )

    cases = [
        (
            [C(12, 0), C(0, 1), C(1, 2), C(2, 3), C(3, 0), C(4, 1), C(5, 2)],
            [C(12, 1), C(0, 2), C(1, 3), C(2, 0), C(3, 1), C(9, 2), C(10, 3)],
        ),
        (
            [C(0, 0), C(0, 1), C(0, 2), C(0, 3), C(12, 0), C(4, 1), C(5, 2)],
            [C(0, 0), C(0, 1), C(0, 2), C(0, 3), C(11, 0), C(11, 1), C(10, 2)],
        ),
    ]
    batch = torch.stack(
        [
            torch.stack([_cards_to_plane(a), _cards_to_plane(b)], dim=0)
            for a, b in cases
        ],
        dim=0,
    )
    cards_batch = torch.tensor(cases, dtype=torch.long, device="cuda")

    expected = compare_7_single_batch(batch)
    actual = compare_7_single_batch_triton(batch)
    assert torch.equal(expected.to(torch.int32), actual)

    cards_actual = compare_7_cards_single_batch_triton(cards_batch)
    assert torch.equal(expected.to(torch.int32), cards_actual)
