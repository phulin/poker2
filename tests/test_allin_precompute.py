from __future__ import annotations

import torch
import pytest

from p2.allin.precompute import (
    _board_chunks,
    allin_2p_169_share0_from_combo_payoff,
    precompute_allin_3p_share0,
    precompute_allin_169_tensors,
    precompute_allin_class_shares,
)
from p2.env.card_utils import (
    PREFLOP_HANDS,
    board_allowed_hands,
    combo_to_preflop_class_tensor,
    hand_combos_tensor,
)
from p2.env.rules import rank_hands


def test_allin_2p_169_share0_from_combo_payoff_collapses_by_compatible_pairs() -> None:
    class_ids = combo_to_preflop_class_tensor()
    payoff = (class_ids[:, None] - class_ids[None, :]).to(torch.float32)

    collapsed, counts = allin_2p_169_share0_from_combo_payoff(payoff)

    assert collapsed.shape == (PREFLOP_HANDS, PREFLOP_HANDS)
    assert counts.shape == (PREFLOP_HANDS, PREFLOP_HANDS)
    assert torch.all(counts > 0)
    expected = (
        torch.arange(PREFLOP_HANDS)[:, None] - torch.arange(PREFLOP_HANDS)[None, :]
    ).to(torch.float32)
    expected = 0.5 * (expected + 1.0)
    torch.testing.assert_close(collapsed, expected)


def _brute_class_shares(
    *,
    players: int,
    class_tuple: tuple[int, ...],
    boards: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    class_ids = combo_to_preflop_class_tensor()
    combos = hand_combos_tensor()
    combos_by_class = [
        torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        for class_id in class_tuple
    ]
    ranks, _ = rank_hands(boards.int())
    allowed = board_allowed_hands(boards)
    share_sum = torch.zeros(players, dtype=torch.float64)
    count = 0

    if players == 2:
        iterable = (
            (int(h0), int(h1))
            for h0 in combos_by_class[0].tolist()
            for h1 in combos_by_class[1].tolist()
        )
    else:
        iterable = (
            (int(h0), int(h1), int(h2))
            for h0 in combos_by_class[0].tolist()
            for h1 in combos_by_class[1].tolist()
            for h2 in combos_by_class[2].tolist()
        )
    for hand_tuple in iterable:
        used_cards: set[int] = set()
        compatible = True
        for hand in hand_tuple:
            card0, card1 = (int(c) for c in combos[hand].tolist())
            if card0 in used_cards or card1 in used_cards:
                compatible = False
                break
            used_cards.add(card0)
            used_cards.add(card1)
        if not compatible:
            continue

        hand_idx = torch.tensor(hand_tuple, dtype=torch.long)
        board_ok = allowed.index_select(1, hand_idx).all(dim=1)
        if not bool(board_ok.any()):
            continue
        tuple_ranks = ranks.index_select(1, hand_idx)
        max_rank = tuple_ranks.max(dim=1, keepdim=True).values
        winners = tuple_ranks == max_rank
        shares = winners.to(torch.float64) / winners.sum(dim=1, keepdim=True).clamp_min(1)
        shares = torch.where(board_ok[:, None], shares, 0.0)
        share_sum += shares.sum(dim=0)
        count += int(board_ok.sum().item())
    return share_sum / max(count, 1), count


def _sample_boards(*, count: int, seed: int) -> torch.Tensor:
    return next(_board_chunks(sample_boards=count, seed=seed, board_chunk=count))


def test_precompute_allin_class_shares_2p_matches_brute_force_sample_boards() -> None:
    class_ids = [0, 1, 14]
    shares, counts, metadata = precompute_allin_class_shares(
        players=2,
        device="cpu",
        sample_boards=5,
        seed=123,
        board_chunk=2,
        class_chunk=3,
        tuple_chunk=32,
        class_ids=class_ids,
        use_triton=False,
    )
    boards = _sample_boards(count=5, seed=123)

    assert shares.shape == (2, PREFLOP_HANDS, PREFLOP_HANDS)
    assert counts.shape == (PREFLOP_HANDS, PREFLOP_HANDS)
    assert metadata["exact"] is False

    for class_tuple in ((0, 1), (1, 14), (14, 0)):
        expected, count = _brute_class_shares(
            players=2,
            class_tuple=class_tuple,
            boards=boards,
        )
        actual = shares[(slice(None), *class_tuple)]
        torch.testing.assert_close(actual, expected.to(torch.float32))
        assert int(counts[class_tuple].item()) == count


def test_precompute_allin_169_tensors_zeroes_uncomputed_2p_entries() -> None:
    result = precompute_allin_169_tensors(
        players=(2,),
        device="cpu",
        sample_boards=2,
        seed=321,
        board_chunk=1,
        class_chunk=1,
        tuple_chunk=16,
        class_ids=[0],
        use_triton=False,
    )

    assert result.allin_2p_share0 is not None
    assert result.allin_2p_count is not None
    assert int(result.allin_2p_count[0, 0].item()) > 0
    assert result.allin_2p_share0[0, 1].item() == 0.0
    assert result.allin_2p_count[0, 1].item() == 0


def test_precompute_allin_class_shares_3p_matches_brute_force_sample_boards() -> None:
    class_ids = [0, 1, 14]
    shares, counts, metadata = precompute_allin_class_shares(
        players=3,
        device="cpu",
        sample_boards=4,
        seed=456,
        board_chunk=2,
        class_chunk=4,
        tuple_chunk=32,
        class_ids=class_ids,
        use_triton=False,
    )
    boards = _sample_boards(count=4, seed=456)

    assert shares.shape == (3, PREFLOP_HANDS, PREFLOP_HANDS, PREFLOP_HANDS)
    assert counts.shape == (PREFLOP_HANDS, PREFLOP_HANDS, PREFLOP_HANDS)
    assert metadata["players"] == 3

    for class_tuple in ((0, 1, 14), (14, 1, 0), (1, 1, 14)):
        expected, count = _brute_class_shares(
            players=3,
            class_tuple=class_tuple,
            boards=boards,
        )
        actual = shares[(slice(None), *class_tuple)]
        torch.testing.assert_close(actual, expected.to(torch.float32))
        assert int(counts[class_tuple].item()) == count


def test_precompute_allin_3p_share0_matches_generic_share0() -> None:
    class_ids = [0, 1, 14]
    generic, generic_counts, _ = precompute_allin_class_shares(
        players=3,
        device="cpu",
        sample_boards=3,
        seed=789,
        board_chunk=2,
        class_chunk=4,
        tuple_chunk=32,
        class_ids=class_ids,
        use_triton=False,
    )
    share0, counts, metadata = precompute_allin_3p_share0(
        device="cpu",
        sample_boards=3,
        seed=789,
        board_chunk=2,
        class_chunk=4,
        tuple_chunk=32,
        class_ids=class_ids,
        use_triton=False,
    )

    assert metadata["share_accumulator"] == "fp32 weighted sixths"
    torch.testing.assert_close(share0, generic[0])
    torch.testing.assert_close(counts, generic_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_precompute_allin_3p_share0_triton_matches_torch_cuda() -> None:
    pytest.importorskip("triton")
    kwargs = dict(
        device="cuda",
        sample_boards=3,
        seed=987,
        board_chunk=2,
        class_chunk=4,
        tuple_chunk=64,
        class_ids=[0, 1, 14],
        use_triton=True,
        compile_inner=False,
    )

    torch_path = precompute_allin_3p_share0(
        **kwargs,
        use_accumulation_kernel=False,
    )
    triton_path = precompute_allin_3p_share0(
        **kwargs,
        use_accumulation_kernel=True,
        triton_block_t=16,
        triton_block_b=2,
    )

    assert triton_path[2]["accumulation_kernel"] == "triton"
    torch.testing.assert_close(triton_path[0], torch_path[0])
    torch.testing.assert_close(triton_path[1], torch_path[1])


def test_precompute_allin_169_tensors_stores_canonical_3p_share0_only() -> None:
    result = precompute_allin_169_tensors(
        players=(3,),
        device="cpu",
        sample_boards=2,
        seed=654,
        board_chunk=1,
        class_chunk=2,
        tuple_chunk=16,
        class_ids=[0, 1],
        use_triton=False,
    )

    assert result.allin_3p_share0 is not None
    assert result.allin_3p_count is not None
    assert result.allin_3p_share0.shape == (PREFLOP_HANDS, PREFLOP_HANDS, PREFLOP_HANDS)
    assert result.payload()["allin_3p_share0"].shape == (
        PREFLOP_HANDS,
        PREFLOP_HANDS,
        PREFLOP_HANDS,
    )
    assert "allin_3p_share" not in result.payload()
