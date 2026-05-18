from types import SimpleNamespace

import torch
from torch.testing import assert_close

from p2.env.card_utils import NUM_HANDS
from p2.rl.pbs_games import _two_prior_river_payoffs
from p2.rl.trueskill_tracker import TSSnapshot, TrueSkillTracker


def _tracker_with_alloc_config(
    *,
    min_games: int = 1,
    max_games: int = 64,
    recency_tau_frac: float = 0.25,
    opponents_per_eval: int = 10,
) -> TrueSkillTracker:
    tracker = object.__new__(TrueSkillTracker)
    tracker.ts_cfg = SimpleNamespace(
        min_games_per_opponent=min_games,
        max_games_per_opponent=max_games,
        recency_tau_frac=recency_tau_frac,
        opponents_per_eval=opponents_per_eval,
    )
    tracker.opponents_per_eval = opponents_per_eval
    tracker.device = torch.device("cpu")
    tracker.generator = torch.Generator(device="cpu").manual_seed(0)
    return tracker


def test_sample_opponent_indices_uses_all_when_pool_under_cap() -> None:
    tracker = _tracker_with_alloc_config(opponents_per_eval=10)

    sampled = tracker._sample_opponent_indices(n_opponents=4)

    assert sampled == [0, 1, 2, 3]


def test_sample_opponent_indices_caps_large_pool() -> None:
    tracker = _tracker_with_alloc_config(opponents_per_eval=10)

    sampled = tracker._sample_opponent_indices(n_opponents=25)

    assert len(sampled) == 10
    assert sampled == sorted(sampled)
    assert len(set(sampled)) == 10
    assert all(0 <= idx < 25 for idx in sampled)


def test_compute_games_per_opponent_splits_eval_budget_across_sampled_opponents() -> (
    None
):
    ts_cfg = SimpleNamespace(
        game_budget_frac=0.03,
        actions_per_game=16,
        eval_solves_per_action=2.0,
        eval_inefficiency_factor=2.0,
        min_games_per_opponent=1,
        max_games_per_opponent=64,
    )

    games = TrueSkillTracker._compute_games_per_opponent(
        total_steps=5000,
        num_envs=1024,
        n_snapshots=100,
        ts_cfg=ts_cfg,
        opponents_per_eval=10,
    )

    assert games == 38


def test_compute_games_per_opponent_respects_per_opponent_cap() -> None:
    ts_cfg = SimpleNamespace(
        game_budget_frac=0.03,
        actions_per_game=16,
        eval_solves_per_action=2.0,
        eval_inefficiency_factor=2.0,
        min_games_per_opponent=1,
        max_games_per_opponent=64,
    )

    games = TrueSkillTracker._compute_games_per_opponent(
        total_steps=5000,
        num_envs=4096,
        n_snapshots=100,
        ts_cfg=ts_cfg,
        opponents_per_eval=10,
    )

    assert games == 64


def test_compute_games_per_eval_accounts_for_eval_cost_multipliers() -> None:
    ts_cfg = SimpleNamespace(
        game_budget_frac=0.03,
        actions_per_game=16,
        eval_solves_per_action=2.0,
        eval_inefficiency_factor=2.0,
    )

    games = TrueSkillTracker._compute_games_per_eval(
        total_steps=5000,
        num_envs=1024,
        n_snapshots=100,
        ts_cfg=ts_cfg,
    )

    assert games == 384


def test_mu_step_regression_metrics_perfect_positive_trend() -> None:
    snapshots = [
        TSSnapshot(step=10, mu=20.0, sigma=1.0),
        TSSnapshot(step=20, mu=25.0, sigma=1.0),
        TSSnapshot(step=30, mu=30.0, sigma=1.0),
    ]

    metrics = TrueSkillTracker._mu_step_regression_metrics(snapshots)

    assert_close(
        torch.tensor(metrics["trueskill/mu_step_correlation"]),
        torch.tensor(1.0),
    )
    assert_close(torch.tensor(metrics["trueskill/mu_step_r2"]), torch.tensor(1.0))


def test_mu_step_regression_metrics_preserves_negative_correlation_sign() -> None:
    snapshots = [
        TSSnapshot(step=10, mu=30.0, sigma=1.0),
        TSSnapshot(step=20, mu=25.0, sigma=1.0),
        TSSnapshot(step=30, mu=20.0, sigma=1.0),
    ]

    metrics = TrueSkillTracker._mu_step_regression_metrics(snapshots)

    assert_close(
        torch.tensor(metrics["trueskill/mu_step_correlation"]),
        torch.tensor(-1.0),
    )
    assert_close(torch.tensor(metrics["trueskill/mu_step_r2"]), torch.tensor(1.0))


def test_mu_step_regression_metrics_handles_degenerate_variance() -> None:
    snapshots = [
        TSSnapshot(step=10, mu=25.0, sigma=1.0),
        TSSnapshot(step=20, mu=25.0, sigma=1.0),
    ]

    metrics = TrueSkillTracker._mu_step_regression_metrics(snapshots)

    assert metrics["trueskill/mu_step_correlation"] == 0.0
    assert metrics["trueskill/mu_step_r2"] == 0.0


class _FakeShowdownEvaluator:
    def __init__(self, showdown_indices: list[int], scale: float) -> None:
        self.device = torch.device("cpu")
        self.total_nodes = 16
        self.showdown_indices = torch.tensor(showdown_indices, dtype=torch.long)
        self.scale = scale

    def _showdown_value(self, beliefs: torch.Tensor, hero: int) -> torch.Tensor:
        assert hero == 0
        row_scale = (
            torch.arange(
                beliefs.shape[0],
                dtype=torch.float32,
                device=beliefs.device,
            )
            + 1.0
        ) * self.scale
        hand_scale = torch.linspace(-0.5, 0.5, NUM_HANDS, device=beliefs.device)
        opp_mass = beliefs[:, 1].sum(dim=1)
        return row_scale[:, None] * hand_scale[None, :] + 0.01 * opp_mass[:, None]


def _abs_belief(child_belief: torch.Tensor, last_actor: int) -> torch.Tensor:
    belief = torch.empty_like(child_belief)
    belief[1 - last_actor] = child_belief[0]
    belief[last_actor] = child_belief[1]
    return belief


def _scalar_showdown_payoff(
    ev: _FakeShowdownEvaluator,
    child: torch.Tensor,
    abs_belief: torch.Tensor,
) -> torch.Tensor:
    matches = (ev.showdown_indices == child).nonzero(as_tuple=True)[0]
    row = matches[0]
    showdown_beliefs = abs_belief.unsqueeze(0).expand(
        ev.showdown_indices.numel(), -1, -1
    )
    showdown_values = ev._showdown_value(showdown_beliefs, 0)
    return (abs_belief[0] * showdown_values[row]).sum()


def test_two_prior_river_payoffs_batches_scalar_showdown_reference() -> None:
    ev_a = _FakeShowdownEvaluator([2, 4, 7], scale=1.0)
    ev_b = _FakeShowdownEvaluator([3, 5, 8], scale=1.7)
    a_children = torch.tensor([2, 7, 4], dtype=torch.long)
    b_children = torch.tensor([8, 3, 5], dtype=torch.long)
    last_actor = torch.tensor([0, 1, 0], dtype=torch.long)

    raw = torch.arange(3 * 2 * NUM_HANDS, dtype=torch.float32).reshape(3, 2, NUM_HANDS)
    bel_a_post = (raw.remainder(23) + 1.0) / 1000.0
    bel_b_post = (raw.flip(0).remainder(29) + 1.0) / 1100.0

    batched = _two_prior_river_payoffs(
        ev_a,
        ev_b,
        a_children,
        b_children,
        bel_a_post,
        bel_b_post,
        last_actor,
    )

    expected = torch.stack(
        [
            0.5
            * (
                _scalar_showdown_payoff(
                    ev_a,
                    a_children[i],
                    _abs_belief(bel_a_post[i], int(last_actor[i])),
                )
                + _scalar_showdown_payoff(
                    ev_b,
                    b_children[i],
                    _abs_belief(bel_b_post[i], int(last_actor[i])),
                )
            )
            for i in range(a_children.numel())
        ]
    )

    assert_close(batched, expected)
