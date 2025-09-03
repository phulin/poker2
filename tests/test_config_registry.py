from __future__ import annotations

import os
from alphaholdem.core.config import load_config
from alphaholdem.core import registry


def test_load_default_config_values():
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml")
    cfg = load_config(path=cfg_path)

    assert cfg.nb == 9
    assert cfg.ppo_eps == 0.2
    assert cfg.ppo_delta1 == 3.0
    assert cfg.gae_lambda == 0.95
    assert cfg.gamma == 0.999
    assert cfg.entropy_coef == 0.01
    assert cfg.value_coef == 0.5
    assert cfg.grad_clip == 1.0

    assert cfg.card_encoder.name == "cards_planes_v1"
    assert cfg.action_encoder.name == "actions_hu_v1"
    assert cfg.model.name == "siamese_convnet_v1"
    assert cfg.policy.name == "categorical_v1"


def test_registries_have_required_components():
    # Names in default.yaml should be registered
    assert "cards_planes_v1" in registry.CARD_ENCODERS
    assert "actions_hu_v1" in registry.ACTION_ENCODERS
    assert "siamese_convnet_v1" in registry.MODELS
    assert "categorical_v1" in registry.POLICIES


