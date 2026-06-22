from __future__ import annotations

from p2.core.action_schedule import make_action_schedule
from p2.core.structured_config import Config


SCHEDULE = [
    [0.25, 0.5, 0.75, 1.0, 1.5],
    [0.5, 1.0],
    [1.0],
    [1.0],
    [1.0],
    [],
]


def test_action_schedule_global_superset_is_sorted() -> None:
    schedule = make_action_schedule(
        [0.5, 1.0],
        bet_bins_by_depth=SCHEDULE,
        allin_by_depth=[True, True, True, True, True, False],
    )

    assert schedule.bet_bins == (0.25, 0.5, 0.75, 1.0, 1.5)
    assert schedule.num_actions == 8
    assert schedule.num_depths == 6


def test_config_derives_action_space_from_depth_schedule() -> None:
    cfg = Config.from_dict(
        {
            "env": {"bet_bins": [0.5, 1.0]},
            "search": {
                "bet_bins_by_depth": SCHEDULE,
                "allin_by_depth": [True, True, True, True, True, False],
            },
        }
    )

    assert cfg.env.bet_bins == [0.25, 0.5, 0.75, 1.0, 1.5]
    assert cfg.model.num_actions == 8
    assert cfg.search.depth == 6


def test_config_rejects_removed_policy_factor_scale() -> None:
    try:
        Config.from_dict({"model": {"policy_factor_scale": 0.05}})
    except TypeError as exc:
        assert "policy_factor_scale" in str(exc)
    else:
        raise AssertionError("removed model fields should fail fast")
