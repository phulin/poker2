from __future__ import annotations

import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.models.cnn import SiameseConvNetV1
from alphaholdem.models.policy import CategoricalPolicyV1
from alphaholdem.rl.self_play import SelfPlayTrainer


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
