from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from p2.config.rebel_load import load_rebel_config, load_rebel_experiment_config


def test_load_rebel_config_applies_rebel_logging_defaults() -> None:
    cfg = load_rebel_config(
        OmegaConf.create(
            {
                "device": "cpu",
                "model": {"name": "BetterFFN"},
                "search": {"enabled": True},
            }
        )
    )

    assert cfg.wandb_tags == ["rebel", "cfr"]
    assert cfg.device == "cpu"
    assert cfg.model.name.value == "BetterFFN"
    assert cfg.search.enabled is True


def test_load_rebel_config_rejects_kbest_top_level_fields() -> None:
    with pytest.raises(ValueError, match="PPO/K-best"):
        load_rebel_config(
            OmegaConf.create(
                {
                    "device": "cpu",
                    "opponent_pool_type": "k_best",
                    "exploiter": {"enabled": False},
                }
            )
        )


def test_rebel_experiment_config_round_trips_to_trainer_config() -> None:
    experiment = load_rebel_experiment_config(
        OmegaConf.create(
            {
                "num_steps": 17,
                "num_envs": 9,
                "device": "cpu",
                "checkpoint_dir": "checkpoints-test",
                "checkpoint_interval": 3,
                "resume_from": "resume.pt",
                "use_wandb": True,
                "wandb_project": "project",
                "wandb_name": "run",
                "wandb_tags": ["custom"],
                "train": {"batch_size": 11},
                "data": {"mode": "pregenerated"},
                "curriculum": {"stages": ["river"]},
            }
        )
    )
    cfg = experiment.to_trainer_config()

    assert experiment.run.num_steps == 17
    assert experiment.run.num_envs == 9
    assert experiment.checkpoint.checkpoint_dir == "checkpoints-test"
    assert experiment.logging.wandb_tags == ["custom"]
    assert cfg.num_steps == 17
    assert cfg.num_envs == 9
    assert cfg.checkpoint_interval == 3
    assert cfg.resume_from == "resume.pt"
    assert cfg.train.batch_size == 11
    assert cfg.data.mode == "pregenerated"
    assert cfg.curriculum.stages == ["river"]
