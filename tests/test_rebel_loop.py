from __future__ import annotations

from types import SimpleNamespace

import torch

from p2.rl import rebel_loop


class _FakeTrueSkillTracker:
    def __init__(self) -> None:
        self.snapshots = []

    def should_snapshot(self, step: int) -> bool:
        return step == 2

    def snapshot_and_evaluate(self, *, step, candidate_weights, wandb_run) -> None:
        self.snapshots.append((step, candidate_weights, wandb_run))


class _FakeTrainer:
    def __init__(self) -> None:
        self.steps = []
        self.saved = []
        self.trueskill_tracker = _FakeTrueSkillTracker()

    def train_step(self, step: int) -> dict:
        self.steps.append(step)
        return {
            "step": step,
            "loss": 1.0,
            "policy_loss": 0.5,
            "value_loss": 0.25,
            "local_exploitability": 0.125,
            "local_exploitability_mbbg": 12.5,
            "evaluator_street": 3.0,
        }

    def save_checkpoint(
        self,
        path: str,
        step: int,
        wandb_run_id: str | None = None,
        save_optimizer: bool = True,
        save_dtype: torch.dtype | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.saved.append(
            (path, step, wandb_run_id, save_optimizer, save_dtype, metadata)
        )

    def trueskill_snapshot_weights(self) -> dict:
        return {"weight": 1.0}


def test_run_training_loop_preserves_checkpoint_and_snapshot_flow(
    monkeypatch, tmp_path
) -> None:
    grid_calls = []
    monkeypatch.setattr(
        rebel_loop,
        "print_preflop_range_grid",
        lambda trainer, step, **kwargs: grid_calls.append((step, kwargs)),
    )

    trainer = _FakeTrainer()
    cfg = SimpleNamespace(
        use_wandb=False,
        wandb_project="unused",
        wandb_name=None,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=2,
        economize_checkpoints=False,
    )

    last_step = rebel_loop.run_training_loop(
        trainer,
        cfg,
        None,
        start_step=0,
        stop_step=3,
        checkpoint_metadata={"curriculum_substep": "river"},
    )

    assert last_step == 2
    assert trainer.steps == [0, 1, 2]
    assert trainer.saved == [
        (
            str(tmp_path / "rebel_step_2.pt"),
            1,
            None,
            False,
            torch.bfloat16,
            {"curriculum_substep": "river"},
        ),
        (
            str(tmp_path / "rebel_latest.pt"),
            1,
            None,
            True,
            None,
            {"curriculum_substep": "river"},
        ),
        (
            str(tmp_path / "rebel_final.pt"),
            3,
            None,
            False,
            None,
            {"curriculum_substep": "river"},
        ),
    ]
    assert trainer.trueskill_tracker.snapshots == [(2, {"weight": 1.0}, None)]
    assert grid_calls == [
        (1, {"rebel": True}),
        (3, {"title": "Final Preflop Range Grid", "rebel": True}),
    ]


def test_run_training_loop_can_skip_preflop_analyzer(monkeypatch, tmp_path) -> None:
    grid_calls = []
    monkeypatch.setattr(
        rebel_loop,
        "print_preflop_range_grid",
        lambda trainer, step, **kwargs: grid_calls.append((step, kwargs)),
    )

    trainer = _FakeTrainer()
    cfg = SimpleNamespace(
        use_wandb=False,
        wandb_project="unused",
        wandb_name=None,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=1,
        economize_checkpoints=False,
    )

    last_step = rebel_loop.run_training_loop(
        trainer,
        cfg,
        None,
        start_step=0,
        stop_step=2,
        print_preflop_analyzer=False,
    )

    assert last_step == 1
    assert trainer.steps == [0, 1]
    assert [saved[0] for saved in trainer.saved] == [
        str(tmp_path / "rebel_step_1.pt"),
        str(tmp_path / "rebel_latest.pt"),
        str(tmp_path / "rebel_step_2.pt"),
        str(tmp_path / "rebel_latest.pt"),
        str(tmp_path / "rebel_final.pt"),
    ]
    assert grid_calls == []
