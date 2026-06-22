from __future__ import annotations

import json

import torch
from omegaconf import OmegaConf

from p2.config.rebel_load import load_rebel_experiment_config
from p2.core.structured_config import Config
from p2.runtime import training_run


class _FakeWandbRun:
    id = "new-run"

    def __init__(self) -> None:
        self.summary = {}
        self.watched = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def watch(self, model, *, log_freq: int) -> None:
        self.watched.append((model, log_freq))


def test_wandb_run_id_from_checkpoint_reads_on_cpu(tmp_path) -> None:
    checkpoint = tmp_path / "rebel_latest.pt"
    torch.save({"wandb_run_id": "abc123"}, checkpoint)

    assert training_run.wandb_run_id_from_checkpoint(str(checkpoint)) == "abc123"
    assert training_run.wandb_run_id_from_checkpoint(str(tmp_path / "missing.pt")) is None


def test_training_run_initializes_wandb_with_resolved_config(
    monkeypatch, tmp_path
) -> None:
    captured = {}
    fake_run = _FakeWandbRun()

    def fake_init(**kwargs):
        captured.update(kwargs)
        return fake_run

    checkpoint = tmp_path / "rebel_latest.pt"
    torch.save({"wandb_run_id": "resume-me"}, checkpoint)
    monkeypatch.setattr(training_run.wandb, "Run", _FakeWandbRun)
    monkeypatch.setattr(training_run.wandb, "init", fake_init)
    monkeypatch.setattr(
        training_run,
        "setup_torch_runtime",
        lambda cfg, device, *, recompile_limit: None,
    )

    cfg = Config(
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        use_wandb=True,
        wandb_project="project",
        wandb_name="base",
        wandb_tags=["tag"],
        resume_from=str(checkpoint),
    )

    with training_run.training_run(
        cfg,
        group="group",
        name="stage-name",
        stage="river",
    ) as ctx:
        assert ctx.run is fake_run
        assert ctx.device == torch.device("cpu")

    assert captured["project"] == "project"
    assert captured["name"] == "stage-name"
    assert captured["group"] == "group"
    assert captured["tags"] == ["tag"]
    assert captured["id"] == "resume-me"
    assert captured["resume"] == "must"
    assert captured["config"]["stage"] == {"name": "river"}
    assert captured["config"]["resolved_config"]["checkpoint_dir"] == str(
        tmp_path / "checkpoints"
    )
    resolved_config = tmp_path / "checkpoints" / "resolved_config.json"
    assert resolved_config.exists()
    assert json.loads(resolved_config.read_text())["checkpoint_dir"] == str(
        tmp_path / "checkpoints"
    )


def test_wandb_run_can_skip_checkpoint_resume_and_add_metadata(
    monkeypatch, tmp_path
) -> None:
    captured = {}
    fake_run = _FakeWandbRun()

    def fake_init(**kwargs):
        captured.update(kwargs)
        return fake_run

    checkpoint = tmp_path / "rebel_latest.pt"
    torch.save({"wandb_run_id": "training-run"}, checkpoint)
    monkeypatch.setattr(training_run.wandb, "init", fake_init)

    cfg = Config(
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        use_wandb=True,
        wandb_project="project",
        resume_from=str(checkpoint),
    )

    with training_run.wandb_run(
        cfg,
        stage="river-probe",
        extra_config={"diagnostic": {"spots": 3}},
        resume_from_checkpoint=False,
    ) as run:
        assert run is fake_run

    assert "id" not in captured
    assert "resume" not in captured
    assert captured["config"]["stage"] == {"name": "river-probe"}
    assert captured["config"]["diagnostic"] == {"spots": 3}


def test_write_resolved_config_accepts_explicit_directory(tmp_path) -> None:
    cfg = Config(device="cpu", checkpoint_dir=str(tmp_path / "checkpoints"))

    path = training_run.write_resolved_config(cfg, tmp_path / "stage")

    assert path == tmp_path / "stage" / "resolved_config.json"
    payload = json.loads(path.read_text())
    assert payload["checkpoint_dir"] == str(tmp_path / "checkpoints")
    assert payload["device"] == "cpu"


def test_write_resolved_config_accepts_rebel_experiment_payload(tmp_path) -> None:
    experiment = load_rebel_experiment_config(
        OmegaConf.create(
            {
                "device": "cpu",
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "wandb_tags": ["rebel"],
                "model": {"name": "BetterFFN"},
            }
        )
    )
    cfg = experiment.to_trainer_config()

    path = training_run.write_resolved_config(
        cfg,
        tmp_path / "stage",
        resolved_config=experiment,
    )

    payload = json.loads(path.read_text())
    assert payload["run"]["device"] == "cpu"
    assert payload["checkpoint"]["checkpoint_dir"] == str(tmp_path / "checkpoints")
    assert payload["logging"]["wandb_tags"] == ["rebel"]
    assert payload["train"]["optimizer"] == "adamw"
    assert "ppo_eps" not in payload["train"]
    assert "kl_type" not in payload["train"]
    assert "lr_scaling_init_value" not in payload["train"]
    assert payload["model"]["name"] == "BetterFFN"
    assert payload["model"]["preflop_model_type"] == "ffn"
    assert "cards_channels" not in payload["model"]
    assert "max_sequence_length" not in payload["model"]
    assert "value_rank" not in payload["model"]
    assert "opponent_pool_type" not in payload
    assert "exploiter" not in payload


def test_training_run_context_logs_model_summary(monkeypatch) -> None:
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(training_run.wandb, "Run", _FakeWandbRun)
    monkeypatch.setattr(
        training_run,
        "count_model_parameters",
        lambda model: {"total_parameters": 3, "trainable_parameters": 2},
    )
    ctx = training_run.TrainingRunContext(
        cfg=Config(device="cpu"),
        device=torch.device("cpu"),
        run=fake_run,
    )
    model = torch.nn.Linear(1, 1)

    ctx.log_model_parameter_summary(model)
    ctx.watch_model(model, log_freq=7)

    assert fake_run.summary == {"total_parameters": 3, "trainable_parameters": 2}
    assert fake_run.watched == [(model, 7)]


def test_training_run_disables_wandb_on_init_failure(monkeypatch, tmp_path) -> None:
    def fake_init(**kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(training_run.wandb, "init", fake_init)
    monkeypatch.setattr(
        training_run,
        "setup_torch_runtime",
        lambda cfg, device, *, recompile_limit: None,
    )
    cfg = Config(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        use_wandb=True,
        wandb_project="project",
    )

    with training_run.training_run(cfg) as ctx:
        assert ctx.run is None

    assert cfg.use_wandb is False
