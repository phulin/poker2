from __future__ import annotations

import torch
from omegaconf import OmegaConf

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from p2.models.cnn import SiameseConvNetV1
from p2.models.policy import CategoricalPolicyV1
from p2.rl.self_play import SelfPlayTrainer


def test_trainer_builds_components_from_config():
    # Create a Hydra config instance with proper initialization
    cfg = Config(
        train=TrainingConfig(),
        model=ModelConfig(),
        env=EnvConfig(),
        device="cpu",  # Set device to cpu for testing
    )

    # Set device for testing
    device = torch.device("cpu")

    trainer = SelfPlayTrainer(
        cfg=cfg,
        device=device,
    )

    # For transformer default config, card/action encoders may be None
    # Ensure model and policy exist
    assert trainer.model is not None
    assert isinstance(trainer.model, SiameseConvNetV1)
    assert isinstance(trainer.policy, CategoricalPolicyV1)
    assert trainer.num_bet_bins == 8


def test_rebel_curriculum_and_data_config_parse_from_hydra_shape():
    cfg = Config.from_dict_config(
        OmegaConf.create(
            {
                "data": {
                    "mode": "pregenerated",
                    "pregenerated": {
                        "value_batch_size": 32,
                        "policy_batch_size": 64,
                        "direct_sample": True,
                        "datasets": [
                            {
                                "path": "outputs/rebel_postflop/river_v1",
                                "value_weight": 0.5,
                                "policy_weight": 2.0,
                                "min_step": 3,
                                "max_step": 7,
                            }
                        ],
                    },
                },
                "curriculum": {
                    "stages": ["river", "distill_E_turn"],
                    "wandb_group": "rebel_postflop_curriculum",
                    "substeps": {
                        "river": {
                            "kind": "train",
                            "net": "S_river",
                            "num_steps": 100,
                        },
                        "distill_E_turn": {
                            "kind": "distill",
                            "net": "E_turn",
                            "from_net": "S_river",
                            "chance": "single_card",
                            "num_steps": 10,
                        },
                    },
                },
            }
        )
    )

    assert cfg.data.mode == "pregenerated"
    assert cfg.data.live_root_source == "self_play"
    assert cfg.data.pregenerated.value_batch_size == 32
    assert cfg.data.pregenerated.direct_sample is True
    assert cfg.data.pregenerated.datasets[0].path == "outputs/rebel_postflop/river_v1"
    assert cfg.data.pregenerated.datasets[0].policy_weight == 2.0
    assert cfg.curriculum.stages == ["river", "distill_E_turn"]
    assert cfg.curriculum.substeps["river"].net == "S_river"
    assert cfg.curriculum.substeps["distill_E_turn"].from_net == "S_river"


def test_rebel_curriculum_river_config_loads_from_yaml():
    cfg = load_rebel_config(OmegaConf.load("conf/config_rebel_curriculum_river.yaml"))

    assert cfg.data.mode == "live"
    assert cfg.data.live_root_source == "random_river"
    assert cfg.curriculum.stages == ["river"]
    assert cfg.curriculum.substeps["river"].kind == "train"
    assert cfg.curriculum.substeps["river"].net == "S_river"
    assert cfg.model.num_value_layers == 7
    assert cfg.model.value_output_init_scale == 0.0
    assert cfg.validation_set.enabled is True
    assert cfg.validation_set.interval == 50


def test_rebel_curriculum_turn_and_flop_configs_load_from_yaml():
    turn_cfg = load_rebel_config(
        OmegaConf.load("conf/config_rebel_curriculum_turn.yaml")
    )
    flop_cfg = load_rebel_config(
        OmegaConf.load("conf/config_rebel_curriculum_flop.yaml")
    )
    distill_train_overrides = {
        "learning_rate": 0.08,
        "learning_rate_final": 0.008,
        "lr_schedule": "linear",
        "adamw_learning_rate": 0.08,
        "batch_size": 1024,
    }
    turn_distill_train_overrides = {
        "learning_rate": 0.02,
        "learning_rate_final": 0.002,
        "lr_schedule": "linear",
        "adamw_learning_rate": 0.02,
        "batch_size": 1024,
    }
    street_train_overrides = {
        "learning_rate": 0.004,
        "learning_rate_final": 0.0004,
        "lr_schedule": "cosine",
        "adamw_learning_rate": 0.0004,
        "batch_size": 4096,
        "value_replay_mode": "streaming_epoch",
        "value_replay_epochs": 3,
        "value_epoch_block_batches": 20,
    }

    assert turn_cfg.data.live_root_source == "random_turn"
    assert turn_cfg.curriculum.stages == ["distill_E_turn", "turn"]
    assert turn_cfg.curriculum.substeps["distill_E_turn"].kind == "distill"
    assert turn_cfg.curriculum.substeps["distill_E_turn"].net == "E_turn"
    assert turn_cfg.curriculum.substeps["distill_E_turn"].from_net == "S_river"
    assert turn_cfg.curriculum.substeps["distill_E_turn"].chance == "single_card"
    assert (
        turn_cfg.curriculum.substeps["distill_E_turn"].train_overrides
        == turn_distill_train_overrides
    )
    assert turn_cfg.model.num_value_layers == 7
    assert turn_cfg.model.value_output_init_scale == 0.0
    assert turn_cfg.curriculum.substeps["turn"].from_net is None
    assert turn_cfg.curriculum.substeps["turn"].closing_net == "E_turn"
    assert turn_cfg.curriculum.substeps["turn"].closing_checkpoint is None
    assert (
        turn_cfg.curriculum.substeps["turn"].train_overrides == street_train_overrides
    )
    assert flop_cfg.data.live_root_source == "random_flop"
    assert flop_cfg.curriculum.stages == [
        "distill_E_flop",
        "flop",
        "distill_E_preflop",
    ]
    assert flop_cfg.curriculum.substeps["distill_E_flop"].kind == "distill"
    assert flop_cfg.curriculum.substeps["distill_E_flop"].net == "E_flop"
    assert flop_cfg.curriculum.substeps["distill_E_flop"].from_net == "S_turn"
    assert flop_cfg.curriculum.substeps["distill_E_flop"].chance == "single_card"
    assert (
        flop_cfg.curriculum.substeps["distill_E_flop"].train_overrides
        == distill_train_overrides
    )
    assert flop_cfg.curriculum.substeps["flop"].from_net == "S_turn"
    assert flop_cfg.curriculum.substeps["flop"].closing_net == "E_flop"
    assert flop_cfg.curriculum.substeps["flop"].closing_checkpoint is None
    assert (
        flop_cfg.curriculum.substeps["flop"].train_overrides == street_train_overrides
    )
    assert flop_cfg.curriculum.substeps["distill_E_preflop"].net == "E_preflop"
    assert flop_cfg.curriculum.substeps["distill_E_preflop"].from_net == "S_flop"
    assert flop_cfg.curriculum.substeps["distill_E_preflop"].chance == "sample_flops"
    assert flop_cfg.curriculum.substeps["distill_E_preflop"].model_overrides == {
        "preflop_hand_dim": 169
    }
    assert (
        flop_cfg.curriculum.substeps["distill_E_preflop"].train_overrides
        == distill_train_overrides
    )
    assert flop_cfg.preflop_validation.enabled is True
    assert flop_cfg.preflop_validation.interval == 100
    assert flop_cfg.preflop_validation.examples == 1024
    assert flop_cfg.preflop_validation.batch_size == 128


def test_rebel_curriculum_postflop_config_loads_fixed_schedule_from_yaml():
    cfg = load_rebel_config(
        OmegaConf.load("conf/config_rebel_curriculum_postflop.yaml")
    )
    distill_train_overrides = {
        "learning_rate": 0.08,
        "learning_rate_final": 0.008,
        "lr_schedule": "linear",
        "adamw_learning_rate": 0.08,
        "batch_size": 1024,
    }
    turn_distill_train_overrides = {
        "learning_rate": 0.02,
        "learning_rate_final": 0.002,
        "lr_schedule": "linear",
        "adamw_learning_rate": 0.02,
        "batch_size": 1024,
    }
    street_train_overrides = {
        "learning_rate": 0.004,
        "learning_rate_final": 0.0004,
        "lr_schedule": "cosine",
        "adamw_learning_rate": 0.0004,
        "batch_size": 4096,
        "value_replay_mode": "streaming_epoch",
        "value_replay_epochs": 3,
        "value_epoch_block_batches": 20,
    }

    assert cfg.data.mode == "live"
    assert cfg.curriculum.stages == [
        "river",
        "distill_E_turn",
        "turn",
        "distill_E_flop",
        "flop",
        "distill_E_preflop",
    ]
    assert cfg.model.num_value_layers == 7
    assert cfg.model.value_output_init_scale == 0.0
    assert cfg.curriculum.substeps["river"].num_steps == 2000
    assert cfg.curriculum.substeps["distill_E_turn"].num_steps == 100000
    assert cfg.curriculum.substeps["turn"].num_steps == 150000
    assert cfg.curriculum.substeps["distill_E_flop"].num_steps == 20000
    assert cfg.curriculum.substeps["flop"].num_steps == 150000
    assert cfg.curriculum.substeps["distill_E_preflop"].num_steps == 30000
    assert cfg.curriculum.substeps["river"].data_overrides == {
        "mode": "bootstrap_pregenerated",
        "live_root_source": "random_river",
        "pregenerated": {
            "value_batch_size": 1024,
            "policy_batch_size": 0,
            "shuffle": True,
            "direct_sample": False,
            "pin_memory": True,
            "async_shard_prefetch": True,
            "validate_manifest": True,
            "datasets": [
                {
                    "path": "outputs/rebel_postflop/river_value_500steps_512000_300it_20260630",
                    "value_weight": 1.0,
                    "policy_weight": 0.0,
                }
            ],
        },
    }
    assert cfg.curriculum.substeps["turn"].data_overrides == {
        "live_root_source": "random_turn"
    }
    assert cfg.curriculum.substeps["flop"].data_overrides == {
        "live_root_source": "random_flop"
    }
    assert cfg.curriculum.substeps["turn"].closing_net == "E_turn"
    assert cfg.curriculum.substeps["turn"].from_net is None
    assert cfg.curriculum.substeps["flop"].closing_net == "E_flop"
    assert cfg.curriculum.substeps["flop"].from_net == "S_turn"
    assert cfg.curriculum.substeps["distill_E_preflop"].chance == "sample_flops"
    assert cfg.curriculum.substeps["distill_E_preflop"].model_overrides == {
        "preflop_hand_dim": 169
    }
    assert cfg.curriculum.substeps["river"].train_overrides == {}
    assert (
        cfg.curriculum.substeps["distill_E_turn"].train_overrides
        == turn_distill_train_overrides
    )
    assert cfg.curriculum.substeps["turn"].train_overrides == street_train_overrides
    assert (
        cfg.curriculum.substeps["distill_E_flop"].train_overrides
        == distill_train_overrides
    )
    assert cfg.curriculum.substeps["flop"].train_overrides == street_train_overrides
    assert (
        cfg.curriculum.substeps["distill_E_preflop"].train_overrides
        == distill_train_overrides
    )
    assert cfg.preflop_validation.enabled is True


def test_rebel_postflop_hybrid_holdout_config_loads_from_yaml():
    cfg = load_rebel_config(
        OmegaConf.load("conf/config_rebel_postflop_hybrid_holdout.yaml")
    )

    assert cfg.data.mode == "hybrid"
    assert cfg.data.live_root_source == "random_river"
    assert cfg.data.pregenerated.value_batch_size == 4096
    assert cfg.data.pregenerated.policy_batch_size == 4096
    assert cfg.data.pregenerated.shuffle is False
    assert cfg.data.pregenerated.direct_sample is False
    assert len(cfg.data.pregenerated.datasets) == 1
    assert cfg.data.pregenerated.datasets[0].path == "outputs/rebel_postflop/river_val"


def test_rebel_pregenerate_postflop_config_loads_from_yaml():
    cfg = load_rebel_config(
        OmegaConf.load("conf/config_rebel_pregenerate_postflop.yaml")
    )

    assert cfg.data.mode == "live"
    assert cfg.data.live_root_source == "random_river"
    assert cfg.rebel_pregenerate.output_dir == "outputs/rebel_postflop/river_v1"
    assert cfg.rebel_pregenerate.stage == "river"
    assert cfg.rebel_pregenerate.root_source == "random_river"
    assert cfg.rebel_pregenerate.value_target_min == 100000
    assert cfg.rebel_pregenerate.policy_target_min == 1000000
