from __future__ import annotations

from contextlib import nullcontext
import json
import os

import pytest
import torch

from p2.cli import train_rebel_curriculum as curriculum_cli
from p2.core.structured_config import (
    Config,
    CurriculumSubstepConfig,
    ModelScope,
    StreetValueHeads,
)


class _FakeTrainer:
    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = torch.nn.Linear(1, 1)
        self.loaded = []

    def load_checkpoint(self, path: str) -> int:
        self.loaded.append(path)
        return 4


def test_stage_wandb_name_only_uses_explicit_base_name() -> None:
    cfg = Config(wandb_name=None)
    assert curriculum_cli._stage_wandb_name(cfg, "river") is None

    cfg.wandb_name = "postflop"
    assert curriculum_cli._stage_wandb_name(cfg, "river") == "postflop-river"


def test_curriculum_distill_substep_uses_pre_only_value_head() -> None:
    cfg = Config()
    substep = CurriculumSubstepConfig(
        kind="distill",
        net="E_turn",
        from_net="S_river",
        num_steps=1,
    )

    stage_cfg = curriculum_cli._stage_config(
        cfg,
        "distill_E_turn",
        substep,
        resume_from=None,
        promoted={},
    )

    assert stage_cfg.model.street_value_heads == StreetValueHeads.pre


def test_curriculum_train_substeps_do_not_run_preflop_analyzer() -> None:
    assert curriculum_cli._should_print_preflop_analyzer("E_preflop") is False
    assert curriculum_cli._should_print_preflop_analyzer("S_0") is False
    assert curriculum_cli._should_print_preflop_analyzer("S_preflop") is False
    assert curriculum_cli._should_print_preflop_analyzer("S_river") is False
    assert curriculum_cli._should_print_preflop_analyzer("S_turn") is False
    assert curriculum_cli._should_print_preflop_analyzer("S_flop") is False
    assert curriculum_cli._should_print_preflop_analyzer("E_turn") is False


class _FakeSplitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_model = torch.nn.Linear(1, 1)
        self.value_model = torch.nn.Linear(1, 1)


class _FakeSplitTrainer:
    def __init__(self) -> None:
        self.model = _FakeSplitModel()
        self.cfg = Config(device="cpu")
        self.cfg.strict_model_loading = True
        self.synced = False

    def _load_closing_leaf_model(self, checkpoint_path: str) -> torch.nn.Module:
        del checkpoint_path
        source = _FakeSplitModel()
        with torch.no_grad():
            source.policy_model.weight.fill_(3.0)
            source.policy_model.bias.fill_(4.0)
        return source

    def _sync_inference_model(self) -> None:
        self.synced = True


def test_initialize_policy_from_checkpoint_copies_only_policy() -> None:
    trainer = _FakeSplitTrainer()
    original_value = {
        key: value.clone()
        for key, value in trainer.model.value_model.state_dict().items()
    }

    curriculum_cli._initialize_policy_from_checkpoint(
        trainer,
        "unused.pt",
        substep_name="turn",
    )

    assert torch.equal(
        trainer.model.policy_model.weight,
        torch.full_like(trainer.model.policy_model.weight, 3.0),
    )
    assert torch.equal(
        trainer.model.policy_model.bias,
        torch.full_like(trainer.model.policy_model.bias, 4.0),
    )
    for key, value in trainer.model.value_model.state_dict().items():
        assert torch.equal(value, original_value[key])
    assert trainer.synced is True


def test_curriculum_train_substep_uses_stage_dir_and_metadata(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_run_loop(trainer, cfg, run, **kwargs):
        calls.append((trainer, cfg, run, kwargs))
        final_path = os.path.join(cfg.checkpoint_dir, "rebel_final.pt")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        torch.save(
            {"model": trainer.model.state_dict(), "step": cfg.num_steps}, final_path
        )
        return cfg.num_steps - 1

    monkeypatch.setattr(curriculum_cli, "RebelCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(curriculum_cli, "run_training_loop", fake_run_loop)
    monkeypatch.setattr(curriculum_cli, "_init_wandb", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        curriculum_cli, "_log_model_parameter_summary", lambda model, run: None
    )

    cfg = Config(device="cpu", checkpoint_dir=str(tmp_path), use_wandb=False)
    cfg.curriculum.stages = ["river"]
    cfg.curriculum.wandb_group = "group-a"
    cfg.curriculum.substeps = {
        "river": CurriculumSubstepConfig(
            kind="train",
            net="S_river",
            num_steps=3,
            closing_checkpoint="outputs/E_turn.pt",
            data_overrides={"live_root_source": "random_river"},
            train_overrides={"batch_size": 77, "learning_rate": 0.08},
            search_overrides={"iterations": 17},
        )
    }

    curriculum_cli.train_rebel_curriculum(cfg)

    assert len(calls) == 1
    _, stage_cfg, run, kwargs = calls[0]
    assert run is None
    assert stage_cfg.num_steps == 3
    assert stage_cfg.checkpoint_dir == str(tmp_path / "river")
    assert stage_cfg.search.closing_leaf_checkpoint == "outputs/E_turn.pt"
    assert stage_cfg.search.model_scope == ModelScope.end_of_street
    assert stage_cfg.data.live_root_source == "random_river"
    assert stage_cfg.train.batch_size == 77
    assert stage_cfg.train.learning_rate == 0.08
    assert stage_cfg.search.iterations == 17
    assert stage_cfg.trueskill.enabled is False
    assert kwargs["start_step"] == 0
    assert kwargs["stop_step"] == 3
    assert kwargs["stage_tag"] == "river"
    assert kwargs["print_preflop_analyzer"] is False
    assert kwargs["checkpoint_metadata"] == {
        "curriculum_substep": "river",
        "curriculum_kind": "train",
        "curriculum_net": "S_river",
        "curriculum_closing_checkpoint": "outputs/E_turn.pt",
    }
    promoted_path = tmp_path / "promoted" / "S_river.pt"
    assert promoted_path.exists()
    state = json.loads((tmp_path / "promoted" / "curriculum_state.json").read_text())
    assert state == {"promoted": {"S_river": str(promoted_path)}}


def test_curriculum_train_substep_initializes_policy_from_promoted_source(
    monkeypatch, tmp_path
) -> None:
    init_calls = []

    def fake_run_loop(trainer, cfg, run, **kwargs):
        del run, kwargs
        final_path = os.path.join(cfg.checkpoint_dir, "rebel_final.pt")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        torch.save(
            {"model": trainer.model.state_dict(), "step": cfg.num_steps}, final_path
        )
        return cfg.num_steps - 1

    def fake_init_policy(trainer, checkpoint_path, *, substep_name):
        init_calls.append((trainer, checkpoint_path, substep_name))

    source_path = tmp_path / "promoted" / "S_river.pt"
    source_path.parent.mkdir()
    source_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(curriculum_cli, "RebelCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(curriculum_cli, "run_training_loop", fake_run_loop)
    monkeypatch.setattr(curriculum_cli, "_init_wandb", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        curriculum_cli, "_initialize_policy_from_checkpoint", fake_init_policy
    )
    monkeypatch.setattr(
        curriculum_cli, "_log_model_parameter_summary", lambda model, run: None
    )

    cfg = Config(device="cpu", checkpoint_dir=str(tmp_path), use_wandb=False)
    substep = CurriculumSubstepConfig(
        kind="train", net="S_turn", from_net="S_river", num_steps=2
    )

    curriculum_cli._run_train_substep(
        cfg,
        "turn",
        substep,
        device=torch.device("cpu"),
        resume_from=None,
        promoted={"S_river": str(source_path)},
    )

    assert len(init_calls) == 1
    _, checkpoint_path, substep_name = init_calls[0]
    assert checkpoint_path == str(source_path)
    assert substep_name == "turn"


def test_curriculum_train_substep_allows_hybrid_holdout_mode(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_run_loop(trainer, cfg, run, **kwargs):
        del run, kwargs
        calls.append((trainer, cfg))
        final_path = os.path.join(cfg.checkpoint_dir, "rebel_final.pt")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        torch.save(
            {"model": trainer.model.state_dict(), "step": cfg.num_steps}, final_path
        )
        return cfg.num_steps - 1

    monkeypatch.setattr(curriculum_cli, "RebelCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(curriculum_cli, "run_training_loop", fake_run_loop)
    monkeypatch.setattr(curriculum_cli, "_init_wandb", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        curriculum_cli, "_log_model_parameter_summary", lambda model, run: None
    )

    cfg = Config(device="cpu", checkpoint_dir=str(tmp_path), use_wandb=False)
    cfg.data.mode = "hybrid"
    cfg.curriculum.stages = ["river"]
    cfg.curriculum.substeps = {
        "river": CurriculumSubstepConfig(
            kind="train",
            net="S_river",
            num_steps=3,
            data_overrides={"live_root_source": "random_river"},
        )
    }

    curriculum_cli.train_rebel_curriculum(cfg)

    assert len(calls) == 1
    _, stage_cfg = calls[0]
    assert stage_cfg.data.mode == "hybrid"
    assert stage_cfg.data.live_root_source == "random_river"


def test_curriculum_distill_substep_uses_promoted_source_checkpoint(
    monkeypatch, tmp_path
) -> None:
    calls = []
    promoted_dir = tmp_path / "promoted"
    promoted_dir.mkdir()
    source_path = promoted_dir / "S_river.pt"
    source_path.write_bytes(b"checkpoint")
    (promoted_dir / "curriculum_state.json").write_text(
        json.dumps({"promoted": {"S_river": str(source_path)}})
    )

    def fake_run_distill(cfg, substep_name, substep, **kwargs):
        calls.append((cfg, substep_name, substep, kwargs))
        promoted_path = promoted_dir / f"{substep.net}.pt"
        promoted_path.write_bytes(b"distilled")
        return str(promoted_path)

    monkeypatch.setattr(curriculum_cli, "_run_distill_substep", fake_run_distill)

    cfg = Config(device="cpu", checkpoint_dir=str(tmp_path), use_wandb=False)
    cfg.curriculum.stages = ["distill_E_turn"]
    cfg.curriculum.substeps = {
        "distill_E_turn": CurriculumSubstepConfig(
            kind="distill",
            net="E_turn",
            from_net="S_river",
            chance="single_card",
            num_steps=10,
        )
    }

    curriculum_cli.train_rebel_curriculum(cfg)

    assert len(calls) == 1
    _, substep_name, substep, kwargs = calls[0]
    assert substep_name == "distill_E_turn"
    assert substep.net == "E_turn"
    assert kwargs["promoted"]["S_river"] == str(source_path)
    state = json.loads((promoted_dir / "curriculum_state.json").read_text())
    assert state == {
        "promoted": {
            "S_river": str(source_path),
            "E_turn": str(promoted_dir / "E_turn.pt"),
        }
    }


def test_curriculum_resume_restarts_recorded_substep(monkeypatch, tmp_path) -> None:
    calls = []
    promoted_dir = tmp_path / "promoted"
    promoted_dir.mkdir()
    promoted = {
        "S_river": str(promoted_dir / "S_river.pt"),
        "E_turn": str(promoted_dir / "E_turn.pt"),
    }
    for path in promoted.values():
        torch.save({"model": {}, "metadata": {}}, path)
    (promoted_dir / "curriculum_state.json").write_text(
        json.dumps({"promoted": promoted})
    )
    resume_path = tmp_path / "turn" / "rebel_latest.pt"
    resume_path.parent.mkdir()
    torch.save(
        {"model": {}, "metadata": {"curriculum_substep": "turn"}},
        resume_path,
    )

    def fake_run_train(cfg, substep_name, substep, **kwargs):
        calls.append((substep_name, kwargs["resume_from"], dict(kwargs["promoted"])))
        promoted_path = promoted_dir / f"{substep.net}.pt"
        promoted_path.write_bytes(b"promoted")
        return str(promoted_path)

    monkeypatch.setattr(curriculum_cli, "_run_train_substep", fake_run_train)
    monkeypatch.setattr(
        curriculum_cli,
        "_run_distill_substep",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("resume should skip earlier distill substeps")
        ),
    )

    cfg = Config(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        use_wandb=False,
        resume_from=str(resume_path),
    )
    cfg.curriculum.stages = ["river", "distill_E_turn", "turn"]
    cfg.curriculum.substeps = {
        "river": CurriculumSubstepConfig(kind="train", net="S_river", num_steps=3),
        "distill_E_turn": CurriculumSubstepConfig(
            kind="distill",
            net="E_turn",
            from_net="S_river",
            num_steps=2,
        ),
        "turn": CurriculumSubstepConfig(
            kind="train",
            net="S_turn",
            closing_net="E_turn",
            num_steps=5,
        ),
    }

    curriculum_cli.train_rebel_curriculum(cfg)

    assert calls == [("turn", str(resume_path), promoted)]


def test_curriculum_train_resume_recovers_closing_checkpoint_from_metadata(
    monkeypatch, tmp_path
) -> None:
    calls = []
    resume_path = tmp_path / "turn" / "rebel_latest.pt"
    resume_path.parent.mkdir()
    torch.save(
        {
            "model": {},
            "metadata": {
                "curriculum_substep": "turn",
                "curriculum_closing_checkpoint": "outputs/E_turn.pt",
            },
        },
        resume_path,
    )

    def fake_run_loop(trainer, cfg, run, **kwargs):
        calls.append((cfg, kwargs))
        final_path = os.path.join(cfg.checkpoint_dir, "rebel_final.pt")
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        torch.save({"model": trainer.model.state_dict(), "step": 5}, final_path)
        return 4

    monkeypatch.setattr(curriculum_cli, "RebelCFRTrainer", _FakeTrainer)
    monkeypatch.setattr(curriculum_cli, "run_training_loop", fake_run_loop)
    monkeypatch.setattr(curriculum_cli, "_init_wandb", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        curriculum_cli, "_log_model_parameter_summary", lambda model, run: None
    )

    cfg = Config(device="cpu", checkpoint_dir=str(tmp_path), use_wandb=False)
    substep = CurriculumSubstepConfig(
        kind="train",
        net="S_turn",
        closing_net="E_turn",
        num_steps=5,
    )

    curriculum_cli._run_train_substep(
        cfg,
        "turn",
        substep,
        device=torch.device("cpu"),
        resume_from=str(resume_path),
        promoted={},
    )

    assert len(calls) == 1
    stage_cfg, kwargs = calls[0]
    assert stage_cfg.search.closing_leaf_checkpoint == "outputs/E_turn.pt"
    assert stage_cfg.search.model_scope == ModelScope.end_of_street
    assert kwargs["checkpoint_metadata"]["curriculum_closing_checkpoint"] == (
        "outputs/E_turn.pt"
    )


def test_curriculum_distill_resume_recovers_source_checkpoint_from_metadata() -> None:
    substep = CurriculumSubstepConfig(
        kind="distill",
        net="E_turn",
        from_net="S_river",
        num_steps=5,
    )

    source = curriculum_cli._source_checkpoint(
        substep,
        promoted={},
        resume_metadata={"curriculum_source_checkpoint": "outputs/S_river.pt"},
    )

    assert source == "outputs/S_river.pt"


def test_curriculum_resume_rejects_non_curriculum_checkpoint(tmp_path) -> None:
    resume_path = tmp_path / "plain.pt"
    torch.save({"model": {}, "metadata": {}}, resume_path)

    cfg = Config(
        device="cpu",
        checkpoint_dir=str(tmp_path),
        use_wandb=False,
        resume_from=str(resume_path),
    )
    cfg.curriculum.stages = ["river"]
    cfg.curriculum.substeps = {
        "river": CurriculumSubstepConfig(kind="train", net="S_river", num_steps=1)
    }

    with pytest.raises(ValueError, match="missing metadata field"):
        curriculum_cli.train_rebel_curriculum(cfg)


def test_curriculum_substep_rejects_unknown_overrides(tmp_path) -> None:
    cfg = Config(device="cpu", checkpoint_dir=str(tmp_path), use_wandb=False)
    cfg.curriculum.substeps = {
        "river": CurriculumSubstepConfig(
            kind="train",
            net="S_river",
            num_steps=1,
            data_overrides={"not_a_data_field": "x"},
        )
    }

    with pytest.raises(ValueError, match="Unknown river.data override"):
        curriculum_cli._stage_config(
            cfg,
            "river",
            cfg.curriculum.substeps["river"],
            resume_from=None,
        )

    cfg.curriculum.substeps = {
        "river": CurriculumSubstepConfig(
            kind="train",
            net="S_river",
            num_steps=1,
            train_overrides={"not_a_train_field": "x"},
        )
    }

    with pytest.raises(ValueError, match="Unknown river.train override"):
        curriculum_cli._stage_config(
            cfg,
            "river",
            cfg.curriculum.substeps["river"],
            resume_from=None,
        )
