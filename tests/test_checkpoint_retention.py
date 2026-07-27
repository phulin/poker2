from __future__ import annotations

from p2.core.structured_config import Config
from p2.rl.rebel_loop import checkpoints_to_delete


def _step(n: int) -> str:
    return f"/ckpt/rebel_step_{n}.pt"


def _simulate_run(
    total_steps: int,
    *,
    checkpoint_interval: int,
    anchor_interval: int,
) -> list[str]:
    """Run cleanup after every checkpoint save and return the surviving files."""
    on_disk: list[str] = []
    for step in range(checkpoint_interval, total_steps + 1, checkpoint_interval):
        current = _step(step)
        on_disk.append(current)
        for path in checkpoints_to_delete(
            on_disk, current, anchor_interval=anchor_interval
        ):
            on_disk.remove(path)
    return on_disk


def test_anchors_survive_repeated_cleanups() -> None:
    survivors = _simulate_run(1000, checkpoint_interval=50, anchor_interval=200)

    assert survivors == [
        _step(200),
        _step(400),
        _step(600),
        _step(800),
        _step(1000),
    ]


def test_non_anchor_checkpoints_are_deleted() -> None:
    paths = [_step(100), _step(150), _step(200), _step(250)]

    # 200 is an anchor and 250 is both current and newest; 100/150 are not.
    assert checkpoints_to_delete(paths, _step(250), anchor_interval=200) == [
        _step(100),
        _step(150),
    ]


def test_anchor_interval_zero_matches_legacy_behavior() -> None:
    paths = [_step(100), _step(200), _step(300), _step(400)]

    assert checkpoints_to_delete(paths, _step(400), anchor_interval=0) == [
        _step(100),
        _step(200),
        _step(300),
    ]

    survivors = _simulate_run(1000, checkpoint_interval=50, anchor_interval=0)
    assert survivors == [_step(1000)]


def test_malformed_names_are_ignored_safely() -> None:
    malformed = [
        "/ckpt/rebel_step_.pt",
        "/ckpt/rebel_step_abc.pt",
        "/ckpt/rebel_step_12x.pt",
        "/ckpt/rebel_final.pt",
        "/ckpt/rebel_latest.pt",
    ]
    paths = [*malformed, _step(100), _step(200)]

    assert checkpoints_to_delete(paths, _step(200), anchor_interval=200) == [
        _step(100),
    ]


def test_newest_and_best_and_latest_are_never_deleted() -> None:
    paths = [
        "/ckpt/best_model.pt",
        "/ckpt/latest_model.pt",
        _step(100),
        _step(150),
        _step(175),
    ]

    # Newest (175) survives even though it is not an anchor, and so does the
    # current path (150) while it is being written.
    assert checkpoints_to_delete(paths, _step(150), anchor_interval=200) == [
        _step(100),
    ]

    # Same protection with anchoring disabled.
    assert checkpoints_to_delete(paths, _step(150), anchor_interval=0) == [
        _step(100),
    ]


def test_current_path_is_protected_even_when_not_newest() -> None:
    paths = [_step(10), _step(20), _step(30)]

    assert checkpoints_to_delete(paths, _step(10), anchor_interval=0) == [_step(20)]


def test_empty_and_single_file_inputs() -> None:
    assert checkpoints_to_delete([], _step(1), anchor_interval=2000) == []
    assert checkpoints_to_delete([_step(1)], _step(1), anchor_interval=2000) == []


def test_default_anchor_interval_is_configured() -> None:
    assert Config().checkpoint_anchor_interval == 2000
