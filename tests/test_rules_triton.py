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
