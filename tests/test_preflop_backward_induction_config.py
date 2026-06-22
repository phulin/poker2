from __future__ import annotations

from pathlib import Path

from p2.core.structured_config import Config
from p2.stages.preflop_buckets import (
    PreflopBucketExecutionConfig,
    build_run_config,
    load_base_config,
)


def _execution_config(**overrides) -> PreflopBucketExecutionConfig:
    values = {
        "command": "train-specialists",
        "config_name": "config_rebel_cfr",
        "config_overrides": (),
        "state_dataset": "/tmp/states",
        "base_checkpoint": "/tmp/base.pt",
        "output_dir": "/tmp/out",
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
        "use_wandb": False,
        "wandb_project": "preflop-project",
        "wandb_name": "preflop-run",
        "wandb_group": None,
        "wandb_tags": ("preflop", "test"),
        "student_init": None,
        "distill_batch_size": 1024,
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
        _execution_config().run_config(),
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
    assert cfg.data.include_pre_chance_value_batches is False
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


def test_load_base_config_accepts_hydra_overrides() -> None:
    cfg = load_base_config(
        repo_root=Path(__file__).resolve().parents[1],
        config_name="config_rebel_cfr",
        overrides=("device=cpu", "train.batch_size=321", "search.iterations=77"),
    )

    assert cfg.device == "cpu"
    assert cfg.train.batch_size == 321
    assert cfg.search.iterations == 77
