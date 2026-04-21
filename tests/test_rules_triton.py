from __future__ import annotations

import pytest
import torch

from p2.env.rules import compare_7_single_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton test")
def test_compare_7_single_batch_triton_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.env.rules_triton import compare_7_single_batch_triton

    generator = torch.Generator(device="cuda").manual_seed(1234)
    batch = torch.zeros((2048, 2, 4, 13), dtype=torch.bool, device="cuda")
    for row in range(batch.shape[0]):
        for player in range(2):
            cards = torch.randperm(52, generator=generator, device="cuda")[:7]
            batch[row, player, cards // 13, cards % 13] = True

    expected = compare_7_single_batch(batch)
    actual = compare_7_single_batch_triton(batch)
    assert torch.equal(expected.to(torch.int32), actual)
