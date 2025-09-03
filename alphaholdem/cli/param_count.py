#!/usr/bin/env python3

from __future__ import annotations

import argparse

from alphaholdem.core.config_loader import get_config
from alphaholdem.core.builders import build_components_from_config

# Ensure registries are populated via side-effect imports
import alphaholdem.encoding.cards_encoder  # noqa: F401
import alphaholdem.encoding.actions_encoder  # noqa: F401
import alphaholdem.models.siamese_convnet  # noqa: F401
import alphaholdem.models.heads  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print total and trainable parameter counts for the current model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to YAML config (defaults to configs/default.yaml)",
    )
    args = parser.parse_args()

    cfg = get_config(args.config)
    _, _, model, _, _ = build_components_from_config(cfg)

    # Totals
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
