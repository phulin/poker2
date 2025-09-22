from __future__ import annotations

import re

import torch

from alphaholdem.env.analyze_tensor_env import (
    create_1326_hand_combinations,
    get_preflop_betting_grid,
    get_preflop_range_grid,
    get_preflop_value_grid,
)
from alphaholdem.env import analyze_tensor_env as ate
from alphaholdem.models.model_outputs import ModelOutput
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.tokens import Context


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
    env = ate.create_169_hand_analysis_setup(model, button=0, device=device)
    state_encoder = ate.create_state_encoder_for_model(model, env, device)
    grid = get_preflop_range_grid(
        model, bin_index=B - 1, state_encoder=state_encoder, device=device
    )

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
    env = ate.create_169_hand_analysis_setup(model, button=0, device=device)
    state_encoder = ate.create_state_encoder_for_model(model, env, device)
    grid = get_preflop_betting_grid(model, state_encoder=state_encoder, device=device)

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
    env = ate.create_169_hand_analysis_setup(model, button=0, device=device)
    state_encoder = ate.create_state_encoder_for_model(model, env, device)
    grid = get_preflop_value_grid(model, state_encoder=state_encoder, device=device)

    # Import inside to keep top-level imports minimal for pytest collection
    from alphaholdem.env.analyze_tensor_env import _create_169_grid  # noqa: F401

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
