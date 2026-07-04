import torch

from p2.env.card_utils import NUM_HANDS, combo_compatible_tensor, hand_combos_tensor
from p2.env.rules import rank_hands
from p2.models.mlp.better_features import ValueScalarContext, context_length
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.mlp_features import MLPFeatures


def _features_for_turn(board: torch.Tensor, beliefs: torch.Tensor, pot: float) -> MLPFeatures:
    context = torch.zeros(board.shape[0], context_length(beliefs.shape[1]))
    context[:, ValueScalarContext.POT.value] = pot
    return MLPFeatures(
        context=context,
        street=torch.full((board.shape[0],), 2, dtype=torch.long),
        to_act=torch.zeros(board.shape[0], dtype=torch.long),
        board=board,
        beliefs=beliefs.reshape(board.shape[0], -1),
    )


def _reference_turn_equity(
    board4: torch.Tensor,
    beliefs: torch.Tensor,
    *,
    blockers: bool,
) -> torch.Tensor:
    combos = hand_combos_tensor()
    compatible = combo_compatible_tensor()
    board_ok = (
        (combos[:, 0, None] != board4[None, :])
        & (combos[:, 1, None] != board4[None, :])
    ).all(dim=1)
    score_sum = torch.zeros_like(beliefs)
    total_sum = torch.zeros_like(beliefs)
    for river in range(52):
        if bool((board4 == river).any()):
            continue
        full_board = torch.cat((board4, torch.tensor([river]))).view(1, 5)
        ranks, _ = rank_hands(full_board)
        ranks = ranks.squeeze(0)
        river_ok = board_ok & (combos != river).all(dim=1)
        payoff = torch.sign((ranks[:, None] - ranks[None, :]).float())
        valid = river_ok[:, None] & river_ok[None, :]
        if blockers:
            valid = valid & compatible
        valid_f = valid.to(dtype=beliefs.dtype)
        payoff = payoff * valid_f
        for player in range(beliefs.shape[0]):
            opp = beliefs[1 - player]
            score_sum[player] += payoff.matmul(opp)
            total_sum[player] += valid_f.matmul(opp)
    return torch.where(
        total_sum > 0.0,
        score_sum / total_sum.clamp_min(1e-8),
        torch.zeros_like(score_sum),
    )


def test_turn_range_equity_matches_reference_without_blockers() -> None:
    generator = torch.Generator().manual_seed(123)
    board = torch.tensor([[0, 14, 28, 42, -1]], dtype=torch.long)
    beliefs = torch.rand(1, 2, NUM_HANDS, generator=generator)
    model = BetterFFN(
        num_actions=3,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=16,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        value_turn_range_equity_baseline=True,
        value_turn_range_equity_baseline_scale=1.0,
        value_turn_range_equity_rank_bins=NUM_HANDS,
        value_turn_range_equity_chunk_size=1,
    )
    features = _features_for_turn(board, beliefs, pot=2.5)

    baseline, feature_values = model._turn_range_equity_features(
        beliefs,
        features,
        torch.float32,
    )

    expected = _reference_turn_equity(board[0, :4], beliefs[0], blockers=False) * 2.5
    torch.testing.assert_close(baseline[0], expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(feature_values[0, :, :, 0], expected, rtol=1e-5, atol=1e-5)


def test_turn_range_equity_matches_reference_with_blockers() -> None:
    generator = torch.Generator().manual_seed(124)
    board = torch.tensor([[1, 15, 29, 43, -1]], dtype=torch.long)
    beliefs = torch.rand(1, 2, NUM_HANDS, generator=generator)
    model = BetterFFN(
        num_actions=3,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=16,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        value_turn_range_equity_baseline=True,
        value_turn_range_equity_baseline_scale=1.0,
        value_turn_range_equity_blockers=True,
        value_turn_range_equity_rank_bins=NUM_HANDS,
        value_turn_range_equity_chunk_size=1,
    )
    features = _features_for_turn(board, beliefs, pot=1.0)

    baseline, _ = model._turn_range_equity_features(
        beliefs,
        features,
        torch.float32,
    )

    expected = _reference_turn_equity(board[0, :4], beliefs[0], blockers=True)
    torch.testing.assert_close(baseline[0], expected, rtol=1e-5, atol=1e-5)
