from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import DictConfig

from p2.cli import train_rebel_preflop_buckets as preflop_cli
from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.env.card_utils import PREFLOP_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _base_config(tmp_path: Path) -> Config:
    cfg = Config(device="cpu", use_wandb=False, wandb_project="preflop-project")
    cfg.wandb_tags = ["preflop", "test"]
    cfg.seed = 123
    cfg.preflop_buckets.state_dataset = str(tmp_path / "states")
    cfg.preflop_buckets.base_checkpoint = str(tmp_path / "base.pt")
    cfg.preflop_buckets.output_dir = str(tmp_path / "out")
    return cfg


def _policy_batch(rows: int) -> RebelBatch:
    return RebelBatch(
        features=MLPFeatures(
            context=torch.arange(rows * 4, dtype=torch.float32).view(rows, 4),
            street=torch.zeros(rows, dtype=torch.long),
            to_act=torch.zeros(rows, dtype=torch.long),
            board=torch.full((rows, 5), -1, dtype=torch.long),
            beliefs=torch.ones(rows, 2 * PREFLOP_HANDS),
            hand_dim=PREFLOP_HANDS,
        ),
        legal_masks=torch.ones(rows, 2, dtype=torch.bool),
        policy_targets=torch.full((rows, PREFLOP_HANDS, 2), 0.5),
        value_targets=None,
    )


def test_preflop_policy_minibatch_padding_zero_weights_duplicates() -> None:
    batch = _policy_batch(3)
    part = batch[1:3]

    padded = preflop_cli.preflop_bi._pad_policy_minibatch(part, batch, 5)

    assert len(padded) == 5
    torch.testing.assert_close(
        padded.statistics["policy_loss_weight"],
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(padded.features.context[:2], part.features.context)
    torch.testing.assert_close(padded.features.context[2:], batch.features.context)


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
                "resume_from=/tmp/resume.pt",
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
    assert cfg.preflop_buckets.train_bucket is None
    assert cfg.preflop_buckets.state_dataset == "/tmp/states"
    assert cfg.preflop_buckets.base_checkpoint == "/tmp/base.pt"
    assert cfg.resume_from == "/tmp/resume.pt"
    assert cfg.trueskill.enabled is False
    assert cfg.preflop_buckets.snapshot_interval_steps == 50
    assert cfg.preflop_buckets.distill_buckets is None
    assert cfg.preflop_buckets.distill_checkpoints.checkpoint_12_15 is None
    assert cfg.preflop_buckets.states_per_bucket == 5_000_000
    assert cfg.preflop_buckets.train_batch_size == 512
    assert cfg.preflop_buckets.policy_train_batch_size == 2048
    assert cfg.preflop_buckets.cfr_iterations == 300
    assert cfg.preflop_buckets.actions_12_15_cfr_batch_size == 8192
    assert cfg.preflop_buckets.actions_8_11_cfr_batch_size == 2048
    assert cfg.preflop_buckets.actions_12_15_epochs == 10
    assert cfg.preflop_buckets.compile == "static"
    assert cfg.preflop_buckets.actions_12_15_depth == 7
    assert cfg.preflop_buckets.actions_4_7_depth == 4
    assert cfg.train.learning_rate == 0.006
    assert cfg.train.learning_rate_final == 0.0009
    assert cfg.train.adamw_learning_rate == 0.004
    assert str(cfg.train.lr_schedule) == "LrSchedule.wsd"
    assert cfg.train.lr_wsd_decay_fraction == 0.6
    assert cfg.train.optimizer == "muon"
    assert cfg.model.preflop_model_type.value == "gated_token_mixer"
    assert cfg.model.hidden_dim == 192
    assert cfg.model.range_hidden_dim == 256
    assert cfg.model.ffn_dim == 256
    assert cfg.model.num_policy_layers == 4
    assert cfg.model.num_value_layers == 5
    assert cfg.env.num_players == 6
    assert str(cfg.search.model_scope) == "ModelScope.mixed_street"
    assert cfg.search.closing_leaf_checkpoint is not None
    assert cfg.search.closing_leaf_checkpoint.endswith("distilled_final.pt")
    assert cfg.search.allin_call_terminal_abstraction is True
    assert cfg.search.preflop_allin_169_model_checkpoint is not None


def test_preflop_hydra_cli_dispatches_train_specialists(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_train(args, *, base_template):
        calls.append(("train", args, base_template))

    monkeypatch.setattr(preflop_cli.preflop_bi, "run_train_specialists", fake_train)
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.command = "train_specialists"
    cfg.preflop_buckets.train_bucket = "actions_12_15"
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
    assert args.resume_from is None
    assert args.train_bucket == "actions_12_15"
    assert args.cfr_batch_size == 17
    assert args.actions_12_15_cfr_batch_size == 9
    assert args.snapshot_interval_steps == 50
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
    assert args.distill_buckets is None
    assert args.distill_train_value is True


def test_preflop_hydra_cli_distill_allows_bucket_subset(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_distill(args, *, base_template):
        calls.append(("distill", args, base_template))

    monkeypatch.setattr(preflop_cli.preflop_bi, "run_distill", fake_distill)
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.command = "distill"
    cfg.preflop_buckets.distill_buckets = [
        "actions_12_15",
        "actions_8_11",
        "actions_4_7",
    ]
    cfg.preflop_buckets.distill_checkpoints.checkpoint_12_15 = "ckpt-12.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_8_11 = "ckpt-8.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_4_7 = "ckpt-4.pt"

    preflop_cli.train_rebel_preflop_buckets(cfg)

    assert len(calls) == 1
    kind, args, _ = calls[0]
    assert kind == "distill"
    assert args.distill_buckets == ("actions_12_15", "actions_8_11", "actions_4_7")
    assert args.checkpoint_0_3 is None


def test_preflop_hydra_cli_distill_can_disable_value_training(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_distill(args, *, base_template):
        calls.append(("distill", args, base_template))

    monkeypatch.setattr(preflop_cli.preflop_bi, "run_distill", fake_distill)
    cfg = _base_config(tmp_path)
    cfg.preflop_buckets.command = "distill"
    cfg.preflop_buckets.distill_train_value = False
    cfg.preflop_buckets.distill_checkpoints.checkpoint_12_15 = "ckpt-12.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_8_11 = "ckpt-8.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_4_7 = "ckpt-4.pt"
    cfg.preflop_buckets.distill_checkpoints.checkpoint_0_3 = "ckpt-0.pt"

    preflop_cli.train_rebel_preflop_buckets(cfg)

    assert len(calls) == 1
    kind, args, _ = calls[0]
    assert kind == "distill"
    assert args.distill_train_value is False


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
