from types import SimpleNamespace

import torch
from torch.testing import assert_close

from p2.env.card_utils import NUM_HANDS
from p2.rl.pbs_games import _two_prior_river_payoffs
from p2.rl.trueskill_tracker import TrueSkillTracker


def _tracker_with_alloc_config(
    *,
    min_games: int = 1,
    max_games: int = 64,
    recency_tau_frac: float = 0.25,
) -> TrueSkillTracker:
    tracker = object.__new__(TrueSkillTracker)
    tracker.ts_cfg = SimpleNamespace(
        min_games_per_opponent=min_games,
        max_games_per_opponent=max_games,
        recency_tau_frac=recency_tau_frac,
    )
    return tracker


def test_allocate_games_respects_budget_when_opponents_exceed_games() -> None:
    tracker = _tracker_with_alloc_config(min_games=1, max_games=64)

    alloc = tracker._allocate_games(n_opponents=10, total_games=3)

    assert sum(alloc) == 3
    assert alloc[:7] == [0] * 7
    assert alloc[-3:] == [1, 1, 1]


def test_allocate_games_respects_per_opponent_cap() -> None:
    tracker = _tracker_with_alloc_config(min_games=1, max_games=4)

    alloc = tracker._allocate_games(n_opponents=3, total_games=100)

    assert alloc == [4, 4, 4]


def test_allocate_games_uses_budget_with_minimums_when_feasible() -> None:
    tracker = _tracker_with_alloc_config(min_games=2, max_games=4)

    alloc = tracker._allocate_games(n_opponents=5, total_games=17)

    assert sum(alloc) == 17
    assert all(2 <= games <= 4 for games in alloc)


def test_allocate_games_supports_zero_minimum_without_single_opponent_collapse() -> (
    None
):
    tracker = _tracker_with_alloc_config(min_games=0, max_games=64)

    alloc = tracker._allocate_games(n_opponents=10, total_games=3)

    assert sum(alloc) == 3
    assert alloc[:7] == [0] * 7
    assert all(games > 0 for games in alloc[-3:])


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
