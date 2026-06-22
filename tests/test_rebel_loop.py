from __future__ import annotations

import torch

from p2.core.structured_config import Config
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

    def save_value_checkpoint(
        self,
        path: str,
        step: int,
        wandb_run_id: str | None = None,
        save_optimizer: bool = True,
        save_dtype: torch.dtype | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.saved.append(
            (
                f"value:{path}",
                step,
                wandb_run_id,
                save_optimizer,
                save_dtype,
                metadata,
            )
        )

    def trueskill_snapshot_weights(self) -> dict:
        return {"weight": 1.0}


class _FakeRun:
    id = "fake-run"

    def __init__(self) -> None:
        self.logged = []

    def log(self, metrics: dict, step: int) -> None:
        self.logged.append((dict(metrics), step))


def _loop_cfg(
    tmp_path,
    *,
    use_wandb: bool = False,
    checkpoint_interval: int = 2,
) -> Config:
    return Config(
        device="cpu",
        use_wandb=use_wandb,
        wandb_project="unused",
        wandb_name=None,
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=checkpoint_interval,
        economize_checkpoints=False,
    )


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
    cfg = _loop_cfg(tmp_path)

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
    cfg = _loop_cfg(tmp_path, checkpoint_interval=1)

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


def test_value_checkpoint_pair_uses_value_only_saver(tmp_path) -> None:
    trainer = _FakeTrainer()
    run = _FakeRun()
    cfg = _loop_cfg(tmp_path)

    path = rebel_loop.save_rebel_checkpoint_pair(
        trainer,
        cfg,
        run,
        step=4,
        checkpoint_metadata={"curriculum_kind": "distill"},
        value_only=True,
    )

    assert path == str(tmp_path / "rebel_step_5.pt")
    assert trainer.saved == [
        (
            f"value:{tmp_path / 'rebel_step_5.pt'}",
            4,
            "fake-run",
            False,
            torch.bfloat16,
            {"curriculum_kind": "distill"},
        ),
        (
            f"value:{tmp_path / 'rebel_latest.pt'}",
            4,
            "fake-run",
            True,
            None,
            {"curriculum_kind": "distill"},
        ),
    ]


def test_run_training_loop_logs_validation_set_metrics(monkeypatch, tmp_path) -> None:
    validation_calls = []

    class FakeValidationSetEvaluator:
        def __init__(
            self,
            *,
            trainer,
            cfg,
            dataset_path,
            batch_size,
            max_examples,
        ) -> None:
            validation_calls.append(
                ("init", trainer, cfg, dataset_path, batch_size, max_examples)
            )

        def evaluate(self) -> dict:
            validation_calls.append(("evaluate",))
            return {
                "validation_value_loss": 0.125,
                "validation_examples": 16,
                "validation_num_batches": 2,
                "validation_batch_loss_mean": 0.125,
                "validation_element_mean_weighted_square_error": 0.1,
                "validation_dataset": "validation-data",
            }

    monkeypatch.setattr(
        rebel_loop,
        "RebelValueValidationSetEvaluator",
        FakeValidationSetEvaluator,
    )
    monkeypatch.setattr(
        rebel_loop,
        "print_preflop_range_grid",
        lambda *args, **kwargs: None,
    )

    trainer = _FakeTrainer()
    run = _FakeRun()
    cfg = _loop_cfg(tmp_path, use_wandb=True, checkpoint_interval=10)
    cfg.validation_set.enabled = True
    cfg.validation_set.dataset = "validation-data"
    cfg.validation_set.interval = 2
    cfg.validation_set.batch_size = 8
    cfg.validation_set.max_examples = 16

    rebel_loop.run_training_loop(
        trainer,
        cfg,
        run,
        start_step=0,
        stop_step=3,
        print_preflop_analyzer=False,
    )

    assert validation_calls[0] == (
        "init",
        trainer,
        cfg,
        "validation-data",
        8,
        16,
    )
    assert validation_calls[1:] == [("evaluate",)]
    assert [step for _, step in run.logged] == [0, 1, 2]
    assert "validation_value_loss" not in run.logged[0][0]
    assert run.logged[1][0]["validation_value_loss"] == 0.125
    assert "validation_time_s" in run.logged[1][0]
    assert "validation_value_loss" not in run.logged[2][0]
