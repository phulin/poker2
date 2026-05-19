#!/usr/bin/env python3

from __future__ import annotations

import hydra

import p2.encoding.actions_encoder  # noqa: F401

# Ensure registries are populated via side-effect imports
import p2.encoding.cards_encoder  # noqa: F401
import p2.models.policy  # noqa: F401
import p2.models.siamese_convnet  # noqa: F401
from p2.core.builders import build_components_from_config
from p2.core.structured_config import Config
from p2.utils.model_utils import count_model_parameters


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Config) -> None:
    """
    Print total and trainable parameter counts for the current model.

    Args:
        cfg: Hydra configuration object
    """
    _, _, model, _, _ = build_components_from_config(cfg)

    # Totals
    parameter_metrics = count_model_parameters(model)
    total_params = parameter_metrics["total_parameters"]
    trainable_params = parameter_metrics["trainable_parameters"]

    # Split: conv trunks vs the rest (fusion + heads)
    conv_modules = []
    if hasattr(model, "cards_trunk"):
        conv_modules.append(model.cards_trunk)
    if hasattr(model, "actions_trunk"):
        conv_modules.append(model.actions_trunk)

    conv_total = sum(p.numel() for m in conv_modules for p in m.parameters())
    conv_trainable = sum(
        p.numel() for m in conv_modules for p in m.parameters() if p.requires_grad
    )

    rest_total = total_params - conv_total
    rest_trainable = trainable_params - conv_trainable

    print(f"Total parameters: {total_params:,}")
    print(f"  ConvNets (trunks): {conv_total:,}")
    print(f"  Fusion+Heads:      {rest_total:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"  ConvNets (trunks): {conv_trainable:,}")
    print(f"  Fusion+Heads:      {rest_trainable:,}")


if __name__ == "__main__":
    main()
