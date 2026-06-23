from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from p2.core.structured_config import Config
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.stages.preflop_backward_induction import (
    _estimate_train_updates,
    _init_wandb,
    _load_model_weights,
    _pot_relative_value_error_metrics,
)
from p2.stages.preflop_buckets import (
    PreflopBucketExecutionConfig,
    build_run_config,
)


def _execution_config(**overrides) -> PreflopBucketExecutionConfig:
    values = {
        "command": "train-specialists",
        "state_dataset": "/tmp/states",
        "base_checkpoint": "/tmp/base.pt",
        "output_dir": "/tmp/out",
        "presolve_bucket": "actions_12_15",
        "train_bucket": None,
        "device": "cpu",
        "seed": 123,
        "depth": 4,
        "cfr_iterations": 400,
        "warm_start_iterations": 0,
        "sparse_fused": True,
        "compile": "off",
        "belief_mode": "random",
        "states_per_bucket": 100_000,
        "train_batch_size": 128,
        "cfr_batch_size": 512,
        "actions_12_15_cfr_batch_size": None,
        "actions_8_11_cfr_batch_size": None,
        "actions_12_15_epochs": 1,
        "validation_items": 4096,
        "validation_cfr_iterations": 10_000,
        "validation_interval_steps": 10,
        "validation_eval_batch_size": 1024,
        "replay_buffer_batches": 3,
        "storage_dtype": "bfloat16",
        "write_solved_shards": True,
        "allow_partial": False,
        "overwrite": False,
        "progress_roots": 10_000,
        "snapshot_interval_steps": 250,
        "use_wandb": False,
        "wandb_project": "preflop-project",
        "wandb_name": "preflop-run",
        "wandb_group": None,
        "wandb_tags": ("preflop", "test"),
        "student_init": None,
        "distill_batch_size": 1024,
        "distill_buckets": None,
        "distill_train_value": True,
        "checkpoint_12_15": None,
        "checkpoint_8_11": None,
        "checkpoint_4_7": None,
        "checkpoint_0_3": None,
    }
    values.update(overrides)
    return PreflopBucketExecutionConfig(**values)


def test_build_run_config_uses_base_config_not_checkpoint(tmp_path) -> None:
    base = Config(device="cuda", num_envs=999, num_steps=999)
    base.data.mode = "pregenerated"
    base.data.live_root_source = "random_river"
    base.train.batch_size = 2048
    base.train.replay_buffer_batches = 99
    base.search.depth = 9
    base.search.iterations = 1234
    base.search.iterations_final = 5678
    base.search.warm_start_iterations = 15
    base.search.sparse = False
    base.search.sparse_fused = False
    base.model.compile = "default"

    cfg = build_run_config(
        base,
        _execution_config(),
        checkpoint_dir=tmp_path / "checkpoints",
        num_steps=42,
        num_envs=64,
    )

    assert cfg is not base
    assert cfg.device == "cpu"
    assert cfg.num_envs == 64
    assert cfg.num_steps == 42
    assert cfg.checkpoint_dir == str(tmp_path / "checkpoints")
    assert cfg.use_wandb is False
    assert cfg.wandb_project == "preflop-project"
    assert cfg.wandb_name == "preflop-run"
    assert cfg.wandb_tags == ["preflop", "test"]
    assert cfg.data.mode == "live"
    assert cfg.data.live_root_source == "self_play"
    assert cfg.data.warmup_self_play_roots is False
    assert cfg.train.batch_size == 128
    assert cfg.train.episodes_per_step == 1
    assert cfg.train.replay_buffer_batches == 3
    assert cfg.train.save_replay_buffers is False
    assert cfg.search.depth == 4
    assert cfg.search.iterations == 400
    assert cfg.search.iterations_final is None
    assert cfg.search.warm_start_iterations == 0
    assert cfg.search.sparse is True
    assert cfg.search.sparse_fused is True
    assert cfg.model.compile == "off"

    assert base.data.mode == "pregenerated"
    assert base.search.iterations_final == 5678
    assert base.model.compile == "default"


def test_estimate_train_updates_counts_cfr_root_batches(tmp_path) -> None:
    manifest = {
        "buckets": [
            {
                "label": label,
                "num_rows": 1024,
                "shards": [{"path": f"{label}.pt"}],
            }
            for label in (
                "actions_12_15",
                "actions_8_11",
                "actions_4_7",
                "actions_0_3",
            )
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    args = _execution_config(
        state_dataset=str(tmp_path),
        states_per_bucket=1024,
        train_batch_size=128,
        cfr_batch_size=512,
        actions_12_15_cfr_batch_size=8192,
        actions_8_11_cfr_batch_size=2048,
        actions_12_15_epochs=2,
    )

    # One logical training step is one solved CFR-root batch. The deepest and
    # 8-11 buckets fit in one override batch, while 4-7 and 0-3 use two 512-root
    # batches: 1*2 + 1 + 2 + 2.
    assert _estimate_train_updates(args) == 7

    # Restricting the specialist run to one bucket estimates only that bucket.
    assert _estimate_train_updates(
        _execution_config(
            state_dataset=str(tmp_path),
            train_bucket="actions_12_15",
            states_per_bucket=1024,
            train_batch_size=128,
            actions_12_15_cfr_batch_size=8192,
            actions_12_15_epochs=2,
        )
    ) == 2


def test_preflop_wandb_init_passes_stage_payload(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_wandb_run(cfg, **kwargs):
        captured["cfg"] = cfg
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "p2.stages.preflop_backward_induction.wandb_run",
        fake_wandb_run,
    )

    args = _execution_config(use_wandb=True, wandb_group="bucket-group")
    cfg = build_run_config(
        Config(device="cpu"),
        args,
        checkpoint_dir=tmp_path / "checkpoints",
        num_steps=1,
        num_envs=2,
    )

    assert (
        _init_wandb(
            args,
            cfg,
            name="bucket-run",
            stage="preflop_bucket_specialists",
        )
        is None
    )

    assert captured["cfg"] is cfg
    assert captured["group"] == "bucket-group"
    assert captured["name"] == "preflop-run"
    assert captured["stage"] == "preflop_bucket_specialists"
    assert captured["resolved_config"].run.num_envs == 2


def test_pot_relative_value_error_metrics_uses_value_weights() -> None:
    output = SimpleNamespace(
        hand_values=torch.tensor([[[12.0, 8.0]], [[15.0, 5.0]]])
    )
    batch = SimpleNamespace(
        value_targets=torch.tensor([[[10.0, 10.0]], [[10.0, 10.0]]]),
        statistics={
            "pot": torch.tensor([10.0, 20.0]),
            "scale": torch.tensor([100.0, 40.0]),
        },
    )
    loss_dict = {
        "value_weights": torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]]),
    }

    metrics = _pot_relative_value_error_metrics(output, batch, loss_dict)

    # Weighted relative absolute errors are:
    # 2 * 100 / 10, 5 * 40 / 20, and 5 * 40 / 20.
    assert torch.allclose(
        metrics["pot_relative_mae"],
        torch.tensor((20.0 + 10.0 + 10.0) / 3.0),
    )


class _FakeSplitLeaf(torch.nn.Linear):
    hand_dim = NUM_HANDS

    def __init__(self) -> None:
        super().__init__(1, 1)
        self.hidden_dim = 1
        self.num_players = 2
        self.num_actions = 2
        self.enforce_zero_sum = False


class _FakeTrainer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.float_dtype = torch.float32
        self.cfg = Config(device="cpu")
        self.cfg.strict_model_loading = True
        self.model = BetterSplitFFN(
            policy_model=_FakeSplitLeaf(),
            value_model=_FakeSplitLeaf(),
        )
        self.synced = False
        self.target_step = None

    def sync_inference_model(self) -> None:
        self.synced = True

    def sync_cfr_target_model(self, step: int) -> None:
        self.target_step = step


def test_load_model_weights_loads_value_only_split_checkpoint(tmp_path) -> None:
    trainer = _FakeTrainer()
    original_policy = {
        key: value.clone()
        for key, value in trainer.model.policy_model.state_dict().items()
    }
    model_state = trainer.model.value_model.state_dict()
    model_state = {
        key: torch.full_like(value, 5.0) for key, value in model_state.items()
    }
    path = tmp_path / "value.pt"
    torch.save(
        {
            "model": model_state,
            "model_component": "value_model",
            "step": 12,
        },
        path,
    )

    step = _load_model_weights(trainer, str(path))

    assert step == 12
    for value in trainer.model.value_model.state_dict().values():
        assert torch.equal(value, torch.full_like(value, 5.0))
    for key, value in trainer.model.policy_model.state_dict().items():
        assert torch.equal(value, original_policy[key])
    assert trainer.synced is True
    assert trainer.target_step == 12
