from __future__ import annotations

import re

import pytest
import torch

from alphaholdem.env.analyze_tensor_env import (
    PreflopAnalyzer,
    RebelPreflopAnalyzer,
)
from alphaholdem.env.card_utils import hand_combos_tensor
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.models.model_output import ModelOutput
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

# NB the indices are REVERSED in the grid from our internal representation
RANKS = "23456789TJQKA"
SUITS = "shdc"
GRID_RANKS = RANKS[::-1]


def _grid_coords_for_hand(card1: int, card2: int) -> tuple[int, int]:
    s1, s2 = card1 // 13, card2 // 13

    # Grid is stored internally in [rank, rank] position.
    # We reverse it later (on output) so AA is at top right.
    i, j = card1 % 13, card2 % 13

    # Suited if same suit and not pair → top-right triangle; else bottom-left
    if s1 == s2:
        # suited → place at (min(i,j), max(i,j)) where higher rank is column
        return (min(i, j), max(i, j))
    else:
        # offsuit → place at (max(i,j), min(i,j))
        return (max(i, j), min(i, j))


def test_1326_to_169_bucket_counts() -> None:
    expected_counts = torch.tensor(
        [
            [6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [12, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [12, 12, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [12, 12, 12, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [12, 12, 12, 12, 6, 4, 4, 4, 4, 4, 4, 4, 4],
            [12, 12, 12, 12, 12, 6, 4, 4, 4, 4, 4, 4, 4],
            [12, 12, 12, 12, 12, 12, 6, 4, 4, 4, 4, 4, 4],
            [12, 12, 12, 12, 12, 12, 12, 6, 4, 4, 4, 4, 4],
            [12, 12, 12, 12, 12, 12, 12, 12, 6, 4, 4, 4, 4],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 4, 4, 4],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 4, 4],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 4],
            [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6],
        ],
        dtype=torch.int32,
    )
    combos = hand_combos_tensor()
    counts = torch.zeros((13, 13), dtype=torch.int32)
    for c1, c2 in combos.tolist():
        i, j = _grid_coords_for_hand(c1, c2)
        counts[i, j] += 1

    assert counts.sum().item() == 1326
    torch.testing.assert_close(counts, expected_counts)


def test_rebel_preflop_analyzer_uses_canonical_combo_order() -> None:
    model = RebelFFN(
        input_dim=RebelFeatureEncoder.feature_dim,
        num_actions=8,
        hidden_dim=32,
        num_hidden_layers=1,
    )
    analyzer = RebelPreflopAnalyzer(model, device=torch.device("cpu"))
    expected_combos = hand_combos_tensor()

    assert torch.equal(analyzer.all_hands.cpu(), expected_combos)
    for (card_a, card_b), (idx_a, idx_b) in zip(
        analyzer.all_hands, expected_combos.tolist()
    ):
        assert card_a == idx_a
        assert card_b == idx_b


def test_rebel_allin_response_has_call_and_fold() -> None:
    model = RebelFFN(
        input_dim=RebelFeatureEncoder.feature_dim,
        num_actions=8,
        hidden_dim=32,
        num_hidden_layers=1,
    )
    analyzer = RebelPreflopAnalyzer(model, device=torch.device("cpu"))
    analyzer.reset(1)
    analyzer.step_sb_action("allin")

    probs, _, _ = analyzer.get_probabilities(0)
    # Only fold/call should be available for reachable combos.
    reachable = probs.sum(dim=1) > 1e-6
    assert reachable.any()
    torch.testing.assert_close(
        probs[reachable, 2:], torch.zeros_like(probs[reachable, 2:])
    )
    # There must be both folding and calling hands among reachable combos.
    assert torch.any(probs[reachable, 0] > 0.5)
    assert torch.any(probs[reachable, 1] > 0.5)


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


def _grid_strs_to_ints(table: list[list[str]]) -> list[list[int]]:
    return [[int(cell) for cell in row] for row in table]


def test_preflop_range_grid_allin_high():
    # num_bet_bins = 8 by default (5 presets + fold/check + all-in)
    N = 1326
    B = 8
    logits = torch.full((N, B), -10.0)
    # Make all-in (bin 7) extremely likely
    logits[:, B - 1] = 10.0
    values = torch.zeros(N)

    model = DummyTransformerModel(logits, values)
    device = torch.device("cpu")

    analyzer = PreflopAnalyzer(model, button=0, device=device)
    grid = analyzer.get_preflop_range_grid(bin_index=B - 1)

    table = _parse_grid_values(grid)
    # 13 rows x 13 cols
    assert len(table) == 13
    assert all(len(row) == 13 for row in table)
    # With all-in logit dominant for all hands, each cell should be capped "99"
    assert all(cell == "99" for row in table for cell in row)


def test_preflop_betting_grid_prefers_bets():
    N = 1326
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

    analyzer = PreflopAnalyzer(model, button=0, device=device)
    grid = analyzer.get_preflop_betting_grid()

    table = _parse_grid_values(grid)
    assert len(table) == 13
    assert all(len(row) == 13 for row in table)
    # All mass on betting bins -> sum should be ~100% per hand (capped to 99)
    assert all(cell == "99" for row in table for cell in row)


def test_preflop_grids_reflect_probabilities_and_values():
    combos = hand_combos_tensor()
    N = combos.shape[0]
    B = 8
    logits = torch.full((N, B), -10.0)

    # Emphasize betting bin 3 for AA combos only.
    ranks = combos % 13
    aa_mask = (ranks[:, 0] == 12) & (ranks[:, 1] == 12)

    # Make all other bins very unlikely for all combinations
    logits[:, :3] = -20.0  # fold, call, bet bins
    logits[:, 4:] = -20.0  # other bet bins
    logits[aa_mask, 3] = 10.0  # AA gets high probability for bin 3
    logits[~aa_mask, 3] = -20.0  # Non-AA gets very low probability for bin 3

    values = torch.zeros(N)
    values[aa_mask] = 2.0  # average should be 2.0 for AA bucket

    model = DummyTransformerModel(logits, values)
    analyzer = PreflopAnalyzer(model, button=0, device=torch.device("cpu"))
    grids = analyzer.get_preflop_grids()

    range_table = _grid_strs_to_ints(_parse_grid_values(grids["ranges"][3]))
    # AA should have high probability (99%)
    assert range_table[0][0] == 99
    # Non-AA should have low probability (around 14% due to softmax normalization)
    # Check that non-AA positions have much lower values than AA
    for i in range(13):
        for j in range(13):
            if not (i == 0 and j == 0):
                assert range_table[i][j] < 50  # Much less than AA's 99%

    value_table = _grid_strs_to_ints(_parse_grid_values(grids["value"]))
    assert value_table[0][0] == 2000
    assert all(
        (i == 0 and j == 0) or value_table[i][j] == 0
        for i in range(13)
        for j in range(13)
    )

    suited_vs_offsuit = grids["suited_vs_offsuit"][3]
    # AA only affects the pair bucket, so suited/off-suit averages should be low (~0.14).
    # Both suited and offsuit should have similar low values due to softmax normalization
    assert torch.allclose(suited_vs_offsuit[0], suited_vs_offsuit[1], atol=1e-3)
    assert suited_vs_offsuit[0] < 0.2  # Should be low due to softmax normalization


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
            action_amounts=zeros.to(torch.int32),
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
        lambda model, env, device, num_bet_bins: DummyStateEncoder(env, device),
    )

    model = DummyValueModel()
    device = torch.device("cpu")

    analyzer = PreflopAnalyzer(model, button=0, device=device)
    grid = analyzer.get_preflop_value_grid()

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

    # Values are calculated as (rank1 + rank2 + suit1 * suit2) / 30.0
    assert aa == 861  # AA pairs
    assert kk == 794  # KK pairs
    assert qq == 728  # QQ pairs
    assert aks == 828  # AK suited
    assert ako == 883  # AK offsuit
    assert twotwo == 61  # 22 pairs


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

        cards = (
            embedding_data.token_ids[:, HOLE0_INDEX : HOLE1_INDEX + 1]
            - get_card_token_id_offset()
        )
        ranks = cards % 13
        suits = cards // 13
        r1, r2 = ranks[:, 0], ranks[:, 1]
        s1, s2 = suits[:, 0], suits[:, 1]

        is_pair = r1 == r2
        is_aa = is_pair & (r1 == 0)  # A is now rank 0
        is_kk = is_pair & (r1 == 1)  # K is now rank 1
        is_ak_unordered = r1 + r2 == 1  # A(0) + K(1) = 1
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

    analyzer = PreflopAnalyzer(
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

    probs, _, _ = analyzer.get_probabilities(seat=0)
    call_probs = probs[:, 1]
    averaged = analyzer.convert_1326_to_169_tensor(call_probs)

    a_idx = GRID_RANKS.index("A")
    k_idx = GRID_RANKS.index("K")
    q_idx = GRID_RANKS.index("Q")
    assert averaged[a_idx, a_idx].item() == pytest.approx(0.0, abs=1e-6)
    assert averaged[k_idx, k_idx].item() == pytest.approx(0.0, abs=1e-6)
    assert averaged[a_idx, k_idx].item() == pytest.approx(
        0.0, abs=1e-6
    )  # AK suited folds
    assert averaged[k_idx, a_idx].item() == pytest.approx(
        1.0, abs=1e-6
    )  # AK offsuit calls
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

    analyzer = PreflopAnalyzer(model, button=0)

    # Get model probabilities for all 1326 combos
    analyzer.get_probabilities(0)

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
    hole0 = analyzer.env.hole_indices[:, 0, 0]
    hole1 = analyzer.env.hole_indices[:, 0, 1]
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
