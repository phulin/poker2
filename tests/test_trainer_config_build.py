from __future__ import annotations

import os
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.encoding.cards_encoder import CardsPlanesV1
from alphaholdem.encoding.actions_encoder import ActionsHUEncoderV1
from alphaholdem.models.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.heads import CategoricalPolicyV1


def test_trainer_builds_components_from_config():
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml"
    )

    trainer = SelfPlayTrainer(
        batch_size=8,
        learning_rate=3e-4,
        config=cfg_path,
    )

    # Expect default config components
    assert isinstance(trainer.cards_encoder, CardsPlanesV1)
    assert isinstance(trainer.actions_encoder, ActionsHUEncoderV1)
    assert isinstance(trainer.model, SiameseConvNetV1)
    assert isinstance(trainer.policy, CategoricalPolicyV1)
    assert trainer.num_bet_bins == 8
