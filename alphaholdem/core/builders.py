from __future__ import annotations

from typing import Any, Tuple

from . import registry


def build_components_from_config(cfg) -> Tuple[Any, Any, Any, Any]:
    """
    Build (card_encoder, action_encoder, model, policy) from a Config dataclass.
    """
    # Build components using the new Config structure
    card_encoder = registry.build_card_encoder(
        cfg.env.card_encoder["name"], config=cfg, **cfg.env.card_encoder["kwargs"]
    )
    action_encoder = registry.build_action_encoder(
        cfg.env.action_encoder["name"],
        config=cfg,
        **cfg.env.action_encoder["kwargs"],
    )

    # Pass gradient checkpointing configuration to model
    model_kwargs = cfg.model.kwargs.copy() if cfg.model.kwargs else {}
    model_kwargs["use_gradient_checkpointing"] = cfg.model.use_gradient_checkpointing
    model = registry.build_model(cfg.model.name, **model_kwargs)

    policy = registry.build_policy(
        cfg.model.policy["name"], **cfg.model.policy["kwargs"]
    )

    return card_encoder, action_encoder, model, policy
