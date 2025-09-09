from __future__ import annotations

from typing import Any, Tuple

from omegaconf import DictConfig

from . import registry


def build_components_from_config(cfg) -> Tuple[Any, Any, Any, Any]:
    """
    Build (card_encoder, action_encoder, model, policy) from a Config dataclass or DictConfig.
    """
    # Handle both dataclass and DictConfig
    if hasattr(cfg, "env"):
        # New dataclass structure
        card_encoder = registry.build_card_encoder(
            cfg.env.card_encoder.name, config=cfg, **cfg.env.card_encoder.kwargs
        )
        action_encoder = registry.build_action_encoder(
            cfg.env.action_encoder.name, config=cfg, **cfg.env.action_encoder.kwargs
        )

        model = registry.build_model(cfg.model.name, **cfg.model.kwargs)

        policy = registry.build_policy(cfg.model.policy.name, **cfg.model.policy.kwargs)
    else:
        # Legacy DictConfig structure
        card_encoder = registry.build_card_encoder(
            cfg.card_encoder.name, config=cfg, **cfg.card_encoder.kwargs
        )
        action_encoder = registry.build_action_encoder(
            cfg.action_encoder.name, config=cfg, **cfg.action_encoder.kwargs
        )

        model = registry.build_model(cfg.model.name, **cfg.model.kwargs)

        policy = registry.build_policy(cfg.policy.name, **cfg.policy.kwargs)

    return card_encoder, action_encoder, model, policy
