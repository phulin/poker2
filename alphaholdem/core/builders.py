from __future__ import annotations

from typing import Any, Tuple

from .config import RootConfig
from .config_loader import get_config
from . import registry


def build_components_from_config(cfg: RootConfig) -> Tuple[Any, Any, Any, Any]:
    """
    Build (card_encoder, action_encoder, model, policy) from a RootConfig.
    """
    # Inject the config into encoders so they can access bet_bins and other settings
    card_encoder = registry.build_card_encoder(
        cfg.card_encoder.name, config=cfg, **cfg.card_encoder.kwargs
    )
    action_encoder = registry.build_action_encoder(
        cfg.action_encoder.name, config=cfg, **cfg.action_encoder.kwargs
    )

    model = registry.build_model(cfg.model.name, **cfg.model.kwargs)

    policy = registry.build_policy(cfg.policy.name, **cfg.policy.kwargs)

    return card_encoder, action_encoder, model, policy
