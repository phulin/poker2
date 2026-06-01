from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from p2.cli import train_rebel_curriculum as curriculum_cli
from p2.core.structured_config import Config, CurriculumSubstepConfig


class _FakeTrainer:
    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = torch.nn.Linear(1, 1)
        self.loaded = []

    def load_checkpoint(self, path: str) -> int:
        self.loaded.append(path)
        return 4


def test_curriculum_train_substep_uses_stage_dir_and_metadata(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_run_loop(trainer, cfg, run, **kwargs):
        calls.append((trainer, cfg, run, kwargs))
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
        )
    }

    curriculum_cli.train_rebel_curriculum(cfg)

    assert len(calls) == 1
    _, stage_cfg, run, kwargs = calls[0]
    assert run is None
    assert stage_cfg.num_steps == 3
    assert stage_cfg.checkpoint_dir == str(tmp_path / "river")
    assert stage_cfg.search.closing_leaf_checkpoint == "outputs/E_turn.pt"
    assert kwargs["start_step"] == 0
    assert kwargs["stop_step"] == 3
    assert kwargs["stage_tag"] == "river"
    assert kwargs["checkpoint_metadata"] == {
        "curriculum_substep": "river",
        "curriculum_kind": "train",
        "curriculum_net": "S_river",
    }


def test_curriculum_distill_substep_is_explicitly_not_implemented(tmp_path) -> None:
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

    with pytest.raises(NotImplementedError, match="E_X distiller"):
        curriculum_cli.train_rebel_curriculum(cfg)
