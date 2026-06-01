from __future__ import annotations

import torch
from omegaconf import OmegaConf

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
                    "river": {
                        "kind": "train",
                        "net": "S_river",
                        "num_steps": 100,
                    },
                    "distill_E_turn": {
                        "kind": "distill",
                        "net": "E_turn",
                        "from": "S_river",
                        "chance": "single_card",
                        "num_steps": 10,
                    },
                },
            }
        )
    )

    assert cfg.data.mode == "pregenerated"
    assert cfg.data.live_root_source == "self_play"
    assert cfg.data.pregenerated.value_batch_size == 32
    assert cfg.data.pregenerated.datasets[0].path == "outputs/rebel_postflop/river_v1"
    assert cfg.data.pregenerated.datasets[0].policy_weight == 2.0
    assert cfg.curriculum.stages == ["river", "distill_E_turn"]
    assert cfg.curriculum.substeps["river"].net == "S_river"
    assert cfg.curriculum.substeps["distill_E_turn"].from_net == "S_river"
