from __future__ import annotations

from typing import Any, Tuple

# Import modules to trigger registration
from alphaholdem.core import registry
from alphaholdem.models import heads  # Import to trigger policy registration
from alphaholdem.models.cnn import (
    cards_encoder,  # Import to trigger card encoder registration
)
from alphaholdem.models.cnn import (
    siamese_convnet,  # Import to trigger model registration
)
from alphaholdem.models.cnn import (  # Import to trigger action encoder registration
    actions_encoder,
)
from alphaholdem.models.transformer import (  # Import to trigger transformer model registration
    poker_transformer,
)


def build_components_from_config(cfg) -> Tuple[Any, Any, Any, Any]:
    """
    Build (card_encoder, action_encoder, model, policy) from a Config dataclass.
    For transformer models, card_encoder and action_encoder will be None.
    """
    # Check if this is a transformer model
    is_transformer = cfg.model.name.startswith("poker_transformer")

    if is_transformer:
        # Transformer models don't need separate card/action encoders
        card_encoder = None
        action_encoder = None
    else:
        # Build CNN encoders
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
    model = registry.build_model(cfg.model.name, **model_kwargs)

    policy = registry.build_policy(
        cfg.model.policy["name"], **cfg.model.policy["kwargs"]
    )

    return card_encoder, action_encoder, model, policy
