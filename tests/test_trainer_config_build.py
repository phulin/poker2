from __future__ import annotations

import torch
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.encoding.cards_encoder import CardsPlanesV1
from alphaholdem.encoding.actions_encoder import ActionsHUEncoderV1
from alphaholdem.models.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.heads import CategoricalPolicyV1
from alphaholdem.core.structured_config import (
    Config,
    TrainingConfig,
    ModelConfig,
    EnvConfig,
)


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

    # Expect default config components
    assert isinstance(trainer.cards_encoder, CardsPlanesV1)
    assert isinstance(trainer.actions_encoder, ActionsHUEncoderV1)
    assert isinstance(trainer.model, SiameseConvNetV1)
    assert isinstance(trainer.policy, CategoricalPolicyV1)
    assert trainer.num_bet_bins == 8
