from __future__ import annotations

from typing import Any, Tuple

from .config import RootConfig
from .config_loader import get_config
from . import registry


def build_components_from_config(cfg: RootConfig) -> Tuple[Any, Any, Any, Any, int]:
    """
    Build (card_encoder, action_encoder, model, policy, nb) from a RootConfig.
    """
    card_encoder = registry.build_card_encoder(cfg.card_encoder.name, **cfg.card_encoder.kwargs)
    action_encoder = registry.build_action_encoder(cfg.action_encoder.name, **cfg.action_encoder.kwargs)

    # Ensure model receives num_actions (nb)
    model_kwargs = dict(cfg.model.kwargs)
    if "num_actions" not in model_kwargs:
        model_kwargs["num_actions"] = cfg.nb
    model = registry.build_model(cfg.model.name, **model_kwargs)

    policy = registry.build_policy(cfg.policy.name, **cfg.policy.kwargs)

    return card_encoder, action_encoder, model, policy, cfg.nb

