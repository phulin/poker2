from __future__ import annotations

from alphaholdem.core import registry
from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.models import cnn, heads  # noqa: F401


def test_load_default_config_values():
    # Create a default Hydra config instance with proper initialization
    cfg = Config(train=TrainingConfig(), model=ModelConfig(), env=EnvConfig())

    # Test training parameters
    assert cfg.train.ppo_eps == 0.2
    assert cfg.train.ppo_delta1 == 3.0
    assert cfg.train.gae_lambda == 0.95
    assert cfg.train.gamma == 0.999
    assert cfg.train.entropy_coef == 0.01
    assert cfg.train.value_coef == 0.05
    assert cfg.train.grad_clip == 1.0

    # Test environment parameters
    assert cfg.env.bet_bins == [0.5, 0.75, 1.0, 1.5, 2.0]

    # Test component configurations
    assert cfg.env.card_encoder["name"] == "cards_planes_v1"
    assert cfg.env.action_encoder["name"] == "actions_hu_v1"
    assert cfg.model.name == "siamese_convnet_v1"
    assert cfg.model.policy["name"] == "categorical_v1"


def test_registries_have_required_components():
    # Modules are imported at top to trigger registration

    # Names in default.yaml should be registered
    assert "cards_planes_v1" in registry.CARD_ENCODERS
    assert "actions_hu_v1" in registry.ACTION_ENCODERS
    assert "siamese_convnet_v1" in registry.MODELS
    assert "categorical_v1" in registry.POLICIES
