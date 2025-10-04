"""
Configuration loading utilities for checkpoint management.

This module provides utilities for loading and merging configurations from checkpoints
with CLI overrides, enabling complete config restoration while allowing parameter overrides.
"""

from typing import Optional

import torch

from alphaholdem.core.structured_config import Config


def load_config_from_checkpoint(
    path: str, cli_config: Optional[Config] = None
) -> Config:
    """
    Load config from checkpoint and merge with CLI overrides.

    This method enables complete config restoration from checkpoints while allowing
    CLI overrides for specific parameters. The checkpoint config serves as the base,
    and CLI overrides take precedence for any fields that differ.

    Args:
        path: Path to checkpoint file
        cli_config: CLI configuration that can override checkpoint config (optional)

    Returns:
        Merged configuration with CLI overrides taking precedence

    Example:
        # Load config from checkpoint without overrides
        config = load_config_from_checkpoint("checkpoint.pt")

        # Load config with CLI overrides
        cli_config = Config(num_steps=2000, device="cuda")
        merged_config = load_config_from_checkpoint("checkpoint.pt", cli_config)
    """
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    # Load full config from checkpoint if available
    if "full_config" in checkpoint:
        checkpoint_config = checkpoint["full_config"]
        print("✅ Loaded full config from checkpoint")
    else:
        # Fallback to legacy config format
        print("⚠️ No full config in checkpoint, using legacy config format")
        checkpoint_config = None

    if checkpoint_config is not None:
        if cli_config is not None:
            # Merge checkpoint config with CLI overrides
            merged_config = _merge_configs(checkpoint_config, cli_config)
            print("✅ Merged checkpoint config with CLI overrides")
            return merged_config
        else:
            # No CLI overrides, return checkpoint config as-is
            print("✅ Using checkpoint config without CLI overrides")
            return checkpoint_config
    else:
        # No checkpoint config available, use CLI config as-is
        if cli_config is not None:
            print("⚠️ Using CLI config only (no checkpoint config found)")
            return cli_config
        else:
            raise ValueError(
                "No config available: neither checkpoint config nor CLI config provided"
            )


def _merge_configs(checkpoint_config: Config, cli_config: Config) -> Config:
    """
    Merge checkpoint config with CLI config, giving precedence to CLI overrides.

    Args:
        checkpoint_config: Config loaded from checkpoint
        cli_config: Config from CLI (overrides)

    Returns:
        Merged config with CLI overrides applied
    """
    import copy

    # Start with a deep copy of the checkpoint config
    merged_config = copy.deepcopy(checkpoint_config)

    # For now, we'll use a simple approach: only override if the CLI value
    # is different from the checkpoint value AND it's not a default value
    # This is a heuristic approach - in practice, CLI overrides should be explicit

    # Apply CLI overrides for top-level fields
    for field_name in cli_config.__dataclass_fields__:
        cli_value = getattr(cli_config, field_name)
        checkpoint_value = getattr(checkpoint_config, field_name)

        # Skip None values (not set)
        if cli_value is None:
            continue

        # Skip MISSING values (default dataclass values)
        from omegaconf import MISSING

        if cli_value is MISSING:
            continue

        # Override if values are different
        if cli_value != checkpoint_value:
            setattr(merged_config, field_name, cli_value)

    # Handle nested configs specially
    if cli_config.train is not None and checkpoint_config.train is not None:
        # Check if both are dataclass instances (not MISSING)
        if hasattr(cli_config.train, "__dataclass_fields__") and hasattr(
            checkpoint_config.train, "__dataclass_fields__"
        ):
            merged_train = copy.deepcopy(merged_config.train)
            for field_name in cli_config.train.__dataclass_fields__:
                cli_value = getattr(cli_config.train, field_name)
                checkpoint_value = getattr(checkpoint_config.train, field_name)

                # Skip None values (not set)
                if cli_value is None:
                    continue

                # Override if values are different
                if cli_value != checkpoint_value:
                    setattr(merged_train, field_name, cli_value)
            merged_config.train = merged_train

    if cli_config.model is not None and checkpoint_config.model is not None:
        # Check if both are dataclass instances (not MISSING)
        if hasattr(cli_config.model, "__dataclass_fields__") and hasattr(
            checkpoint_config.model, "__dataclass_fields__"
        ):
            merged_model = copy.deepcopy(merged_config.model)
            for field_name in cli_config.model.__dataclass_fields__:
                cli_value = getattr(cli_config.model, field_name)
                checkpoint_value = getattr(checkpoint_config.model, field_name)

                # Skip None values (not set)
                if cli_value is None:
                    continue

                # Override if values are different
                if cli_value != checkpoint_value:
                    setattr(merged_model, field_name, cli_value)
            merged_config.model = merged_model

    if cli_config.env is not None and checkpoint_config.env is not None:
        # Check if both are dataclass instances (not MISSING)
        if hasattr(cli_config.env, "__dataclass_fields__") and hasattr(
            checkpoint_config.env, "__dataclass_fields__"
        ):
            merged_env = copy.deepcopy(merged_config.env)
            for field_name in cli_config.env.__dataclass_fields__:
                cli_value = getattr(cli_config.env, field_name)
                checkpoint_value = getattr(checkpoint_config.env, field_name)

                # Skip None values (not set)
                if cli_value is None:
                    continue

                # Override if values are different
                if cli_value != checkpoint_value:
                    setattr(merged_env, field_name, cli_value)
            merged_config.env = merged_env

    return merged_config
