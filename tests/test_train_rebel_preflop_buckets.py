from __future__ import annotations

from pathlib import Path

import hydra
import pytest
from omegaconf import DictConfig

from p2.cli import train_rebel_preflop_buckets as preflop_cli
from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path) -> Config:
    cfg = Config(device="cpu", use_wandb=False, wandb_project="preflop-project")
    cfg.wandb_tags = ["preflop", "test"]
    cfg.seed = 123
    cfg.preflop_buckets.state_dataset = str(tmp_path / "states")
    cfg.preflop_buckets.base_checkpoint = str(tmp_path / "base.pt")
    cfg.preflop_buckets.output_dir = str(tmp_path / "out")
    return cfg


def test_config_rebel_preflop_buckets_resolves() -> None:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"),
        version_base=None,
    ):
        dict_config: DictConfig = hydra.compose(
            config_name="config_rebel_preflop_buckets",
            overrides=[
                "device=cpu",
                "use_wandb=false",
                "preflop_buckets.state_dataset=/tmp/states",
                "preflop_buckets.base_checkpoint=/tmp/base.pt",
            ],
        )

    cfg = load_rebel_config(dict_config)

    assert cfg.device == "cpu"
    assert cfg.use_wandb is False
    assert cfg.wandb_project == "poker-rebel-preflop-backward-induction"
    assert cfg.preflop_buckets.command == "train_specialists"
    assert cfg.preflop_buckets.presolve_bucket == "actions_12_15"
    assert cfg.preflop_buckets.state_dataset == "/tmp/states"
    assert cfg.preflop_buckets.base_checkpoint == "/tmp/base.pt"
    assert cfg.preflop_buckets.distill_checkpoints.checkpoint_12_15 is None


def test_preflop_hydra_cli_dispatches_train_specialists(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_train(args, *, base_template):
        calls.append(("train", args, base_template))

    monkeypatch.setattr(preflop_cli.preflop_bi, "run_train_specialists", fake_train)
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.command = "train_specialists"
    cfg.preflop_buckets.cfr_batch_size = 17
    cfg.preflop_buckets.actions_12_15_cfr_batch_size = 9

    preflop_cli.train_rebel_preflop_buckets(cfg)

    assert len(calls) == 1
    kind, args, base_template = calls[0]
    assert kind == "train"
    assert base_template is cfg
    assert args.command == "train-specialists"
    assert args.state_dataset == str(tmp_path / "states")
    assert args.base_checkpoint == str(tmp_path / "base.pt")
    assert args.cfr_batch_size == 17
    assert args.actions_12_15_cfr_batch_size == 9
    assert args.use_wandb is False
    assert args.wandb_project == "preflop-project"


def test_preflop_hydra_cli_dispatches_distill(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_distill(args, *, base_template):
        calls.append(("distill", args, base_template))

    monkeypatch.setattr(preflop_cli.preflop_bi, "run_distill", fake_distill)
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.command = "distill"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_12_15 = "ckpt-12.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_8_11 = "ckpt-8.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_4_7 = "ckpt-4.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_0_3 = "ckpt-0.pt"

    preflop_cli.train_rebel_preflop_buckets(cfg)

    assert len(calls) == 1
    kind, args, base_template = calls[0]
    assert kind == "distill"
    assert base_template is cfg
    assert args.command == "distill"
    assert args.checkpoint_12_15 == "ckpt-12.pt"
    assert args.checkpoint_8_11 == "ckpt-8.pt"
    assert args.checkpoint_4_7 == "ckpt-4.pt"
    assert args.checkpoint_0_3 == "ckpt-0.pt"


def test_preflop_hydra_cli_dispatches_presolve_values(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_presolve(args, *, base_template):
        calls.append(("presolve", args, base_template))

    monkeypatch.setattr(
        preflop_cli.preflop_bi,
        "run_presolve_values",
        fake_presolve,
    )
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.command = "presolve_values"
    cfg.preflop_buckets.presolve_bucket = "actions_8_11"

    preflop_cli.train_rebel_preflop_buckets(cfg)

    assert len(calls) == 1
    kind, args, base_template = calls[0]
    assert kind == "presolve"
    assert base_template is cfg
    assert args.command == "presolve-values"
    assert args.presolve_bucket == "actions_8_11"


def test_preflop_hydra_cli_requires_distill_checkpoints(tmp_path) -> None:
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.command = "distill"

    with pytest.raises(ValueError, match="distill_checkpoints"):
        preflop_cli.train_rebel_preflop_buckets(cfg)


def test_preflop_hydra_cli_requires_source_paths(tmp_path) -> None:
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.base_checkpoint = None

    with pytest.raises(ValueError, match="base_checkpoint"):
        preflop_cli.train_rebel_preflop_buckets(cfg)
