from __future__ import annotations

import os
import re
import sys

import pytest
import torch

from alphaholdem.env import analyze_tensor_env as ate
from alphaholdem.env.analyze_tensor_env import (
    _grid_coords_for_hand,
    create_1326_hand_combinations,
    get_preflop_betting_grid,
    get_preflop_range_grid,
    get_preflop_value_grid,
    get_probabilities,
)
from alphaholdem.models.model_outputs import ModelOutput
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.tokens import (
    HOLE0_INDEX,
    HOLE1_INDEX,
    Context,
    Special,
    get_card_token_id_offset,
)

# Ensure repo root on path for direct pytest runs
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


RANKS: list[str] = [
    "A",
    "K",
    "Q",
    "J",
    "T",
    "9",
    "8",
    "7",
    "6",
    "5",
    "4",
    "3",
    "2",
]


def _expected_count_for_cell(i: int, j: int) -> int:
    if i == j:
        return 6
    if i < j:
        return 4
    return 12


def test_1326_to_169_bucket_counts() -> None:
    combos = create_1326_hand_combinations()
    counts = torch.zeros((13, 13), dtype=torch.int32)
    for c1, c2 in combos:
        i, j = _grid_coords_for_hand(c1, c2)
        counts[i, j] += 1

    for i in range(13):
        for j in range(13):
            assert counts[i, j].item() == _expected_count_for_cell(i, j)

    assert counts.sum().item() == 1326


class DummyTransformerModel(torch.nn.Module):
    """Dummy model that returns fixed logits/values for 169 states.

    Detected as a transformer by name. Ignores inputs and outputs shapes that
    match expectations of analyze_tensor_env helpers.
    """

    def __init__(self, logits: torch.Tensor, values: torch.Tensor):
        super().__init__()
        self._logits = logits
        self._values = values

    def forward(self, embedding_data):  # embedding_data is unused
        return ModelOutput(policy_logits=self._logits, value=self._values)


def _parse_grid_values(grid: str) -> list[list[str]]:
    """Extract the 13x13 cell strings from a grid string."""
    lines = grid.splitlines()
    # Skip header (2 lines)
    data_lines = [ln for ln in lines[2:] if ln.strip()]
    table: list[list[str]] = []
    for ln in data_lines:
        # Expect format like " A | 12 34 ..."
        if "|" not in ln:
            continue
        parts = ln.split("|")
        if len(parts) < 2:
            continue
        cells_str = parts[1].strip()
        # Cells are 13 two-character numbers possibly with leading spaces
        cells = re.findall(r"\d{1,4}", cells_str)
        if len(cells) == 13:
            table.append(cells)
    return table


def test_preflop_range_grid_allin_high():
    # num_bet_bins = 8 by default (5 presets + fold/check + all-in)
    N = len(create_1326_hand_combinations())
    B = 8
    logits = torch.full((N, B), -10.0)
    # Make all-in (bin 7) extremely likely
    logits[:, B - 1] = 10.0
    values = torch.zeros(N)

    model = DummyTransformerModel(logits, values)
    device = torch.device("cpu")
    grid = get_preflop_range_grid(model, bin_index=B - 1, device=device)

    table = _parse_grid_values(grid)
    # 13 rows x 13 cols
    assert len(table) == 13
    assert all(len(row) == 13 for row in table)
    # With all-in logit dominant for all hands, each cell should be capped "99"
    assert all(cell == "99" for row in table for cell in row)


def test_preflop_betting_grid_prefers_bets():
    N = len(create_1326_hand_combinations())
    B = 8
    logits = torch.full((N, B), -10.0)
    # Prefer betting bins 2..6 equally
    logits[:, 2 : B - 1] = 10.0
    # De-emphasize fold, call, all-in
    logits[:, 0] = -10.0
    logits[:, 1] = -10.0
    logits[:, B - 1] = -10.0
    values = torch.zeros(N)

    model = DummyTransformerModel(logits, values)
    device = torch.device("cpu")
    grid = get_preflop_betting_grid(model, device=device)

    table = _parse_grid_values(grid)
    assert len(table) == 13
    assert all(len(row) == 13 for row in table)
    # All mass on betting bins -> sum should be ~100% per hand (capped to 99)
    assert all(cell == "99" for row in table for cell in row)


class DummyStateEncoder:
    """Minimal encoder that exposes hole card indices as token_ids [M,2]."""

    def __init__(self, env, device: torch.device):
        self.env = env
        self.device = device

    def encode_tensor_states(
        self, seat: int, idxs: torch.Tensor
    ) -> StructuredEmbeddingData:
        # Extract hole indices for the specified seat: shape [M, 2]
        hole = self.env.hole_indices[idxs, seat, :].to(self.device)
        M = hole.shape[0]
        # Build minimal StructuredEmbeddingData; model will only read token_ids
        L = 2
        token_ids = hole.clone()
        zeros = torch.zeros(M, L, dtype=torch.long, device=self.device)
        legal = torch.ones(M, L, 8, dtype=torch.bool, device=self.device)
        ctx = torch.zeros(
            M, L, Context.NUM_RAW_CONTEXT.value, dtype=torch.int16, device=self.device
        )
        lengths = torch.full((M,), L, dtype=torch.uint8, device=self.device)
        return StructuredEmbeddingData(
            token_ids=token_ids,
            token_streets=zeros,
            card_ranks=zeros,
            card_suits=zeros,
            action_actors=zeros,
            action_legal_masks=legal,
            context_features=ctx,
            lengths=lengths,
        )


class DummyValueModel(torch.nn.Module):
    """Model that sets value based on ranks of the two hole cards.

    value = (rank1 + rank2) / 24.0 in [0,1]. policy logits are zeros.
    """

    def __init__(self):
        super().__init__()

    def forward(self, embedding_data: StructuredEmbeddingData):
        token_ids = embedding_data.token_ids  # [N, 2]
        ranks = token_ids % 13
        suits = token_ids // 13
        values = (ranks[:, 0] + ranks[:, 1] + suits[:, 0] * suits[:, 1]).to(
            torch.float32
        ) / 30.0  # [N]
        logits = torch.zeros(
            values.shape[0], 8, dtype=torch.float32, device=values.device
        )
        return ModelOutput(policy_logits=logits, value=values)


def test_preflop_value_grid_varies_with_rank_sum(monkeypatch):
    # Monkeypatch factory to use DummyStateEncoder so model sees hole indices
    monkeypatch.setattr(
        "alphaholdem.env.analyze_tensor_env.create_state_encoder_for_model",
        lambda model, env, device: DummyStateEncoder(env, device),
    )

    model = DummyValueModel()
    device = torch.device("cpu")
    grid = get_preflop_value_grid(model, device=device)

    # Symbol imported at module top

    table = _parse_grid_values(grid)
    # Convert value cells (scaled by 1000) to ints
    value_rows = [[int(cell) for cell in row] for row in table]

    # Coordinates: [row=A..2][col=A..2]
    # AA should be highest
    aa = value_rows[0][0]
    kk = value_rows[1][1]
    qq = value_rows[2][2]
    aks = value_rows[0][1]
    ako = value_rows[1][0]
    twotwo = value_rows[12][12]

    assert aa == 861  # [12 + 12 + (1 * 2 + 1 * 3 + 2 * 3) / 6] / 30.0
    assert kk == 794
    assert qq == 728
    assert aks == 883  # [12 + 11 + (1 * 1 + 2 * 2 + 3 * 3) / 4] / 30.0
    assert ako == 828  # [12 + 11 + (1 * 2 + 1 * 3 + 2 * 3) / 6] / 30.0
    assert twotwo == 61


class DummyModelFoldAAKKAKs(PokerTransformerV1):
    def __init__(self):
        super().__init__(
            max_sequence_length=50,
            d_model=128,
            n_layers=2,
            n_heads=4,
            num_bet_bins=8,
            dropout=0.1,
            use_gradient_checkpointing=False,
        )

    def forward(self, embedding_data: StructuredEmbeddingData):
        N = embedding_data.batch_size
        device = embedding_data.device
        logits = torch.full((N, 8), -10.0, device=device)
        logits[:, 1] = 10.0

        cards = embedding_data.token_ids[:, HOLE0_INDEX : HOLE1_INDEX + 1]
        ranks = cards % 13
        suits = cards // 13
        r1, r2 = ranks[:, 0], ranks[:, 1]
        s1, s2 = suits[:, 0], suits[:, 1]

        is_pair = r1 == r2
        is_aa = is_pair & (r1 == 12)
        is_kk = is_pair & (r1 == 11)
        is_ak_unordered = ((r1 == 12) & (r2 == 11)) | ((r1 == 11) & (r2 == 12))
        is_suited = s1 == s2
        is_aks = is_ak_unordered & is_suited
        fold_mask = is_aa | is_kk | is_aks
        print("fold", fold_mask.sum().item())

        logits[fold_mask, :] = -10.0
        logits[fold_mask, 0] = 10.0

        value = torch.zeros(N, device=device)
        return ModelOutput(policy_logits=logits, value=value)


def test_dummy_model_range_grid_call_bin() -> None:
    model = DummyModelFoldAAKKAKs()

    device = torch.device("cpu")
    env, encoder = ate.create_169_hand_analysis_setup(
        model=model,
        button=0,
        starting_stack=1000,
        sb=5,
        bb=10,
        bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0],
        device=device,
        rng=torch.Generator(device=device),
        flop_showdown=False,
    )

    probs, _, _ = get_probabilities(model, encoder, env, seat=0, device=device)
    call_probs = probs[:, 1]

    grid_sums = torch.zeros(13, 13, dtype=call_probs.dtype, device=device)
    grid_counts = torch.zeros(13, 13, dtype=torch.long, device=device)
    combos = create_1326_hand_combinations()
    for idx, (c1, c2) in enumerate(combos):
        i, j = _grid_coords_for_hand(c1, c2)
        grid_sums[i, j] += call_probs[idx]
        grid_counts[i, j] += 1

    averaged = torch.where(
        grid_counts > 0,
        grid_sums / grid_counts.clamp(min=1),
        torch.zeros_like(grid_sums),
    )

    a_idx = RANKS.index("A")
    k_idx = RANKS.index("K")
    q_idx = RANKS.index("Q")
    assert averaged[a_idx, a_idx].item() == pytest.approx(0.0, abs=1e-6)
    assert averaged[k_idx, k_idx].item() == pytest.approx(0.0, abs=1e-6)
    assert averaged[a_idx, k_idx].item() == pytest.approx(0.0, abs=1e-6)
    assert averaged[k_idx, a_idx].item() == pytest.approx(1.0, abs=1e-6)
    assert averaged[:, q_idx:].mean().item() == pytest.approx(1.0, abs=1e-6)
    assert averaged[q_idx:, :].mean().item() == pytest.approx(1.0, abs=1e-6)


class CapturingTransformerModel(PokerTransformerV1):
    """Transformer that stores the last StructuredEmbeddingData it saw."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_embedding = None

    def forward(self, embedding_data):
        self.last_embedding = embedding_data
        return super().forward(embedding_data)


def test_token_sequence_order_cls_game_hole_hole_context() -> None:
    model = CapturingTransformerModel(
        max_sequence_length=50,
        d_model=128,
        n_layers=2,
        n_heads=4,
        num_bet_bins=8,
        dropout=0.1,
        use_gradient_checkpointing=False,
    )
    # Create 1326-hand environment and state encoder
    env, state_encoder = ate.create_169_hand_analysis_setup(model, button=0)

    # Get model probabilities for all 1326 combos
    ate.get_probabilities(model, state_encoder, env, 0, env.device)

    emb = model.last_embedding
    assert emb is not None

    # Verify token order and contents for a few envs
    token_ids = emb.token_ids  # [N, L]
    L = emb.lengths.max().item()
    assert L >= 5  # CLS, GAME, HOLE0, HOLE1, CONTEXT

    # CLS at position 0
    assert torch.all(token_ids[:, 0] == Special.CLS.value)
    # GAME at position 1
    assert torch.all(token_ids[:, 1] == Special.GAME.value)
    # HOLE0 and HOLE1 at positions 2 and 3
    hole0 = env.hole_indices[:, 0, 0]
    hole1 = env.hole_indices[:, 0, 1]
    expected_hole0_tokens = get_card_token_id_offset() + hole0
    expected_hole1_tokens = get_card_token_id_offset() + hole1
    assert torch.all(token_ids[:, 2] == expected_hole0_tokens)
    assert torch.all(token_ids[:, 3] == expected_hole1_tokens)
    assert torch.all(emb.card_ranks[:, 2] == hole0 % 13)
    assert torch.all(emb.card_suits[:, 2] == hole0 // 13)
    assert torch.all(emb.card_ranks[:, 3] == hole1 % 13)
    assert torch.all(emb.card_suits[:, 3] == hole1 // 13)
    # CONTEXT at position 4
    assert torch.all(token_ids[:, 4] == Special.CONTEXT.value)


if __name__ == "__main__":
    pytest.main([__file__])
