from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from .config import RootConfig, load_config


def default_config_path() -> str:
    """Return absolute path to the default YAML config."""
    return str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")


def get_config(config_or_path: Optional[Union[str, RootConfig]] = None) -> RootConfig:
    """
    Centralized config loader with caching.
    - If None: load default YAML (cached)
    - If str: load YAML at the provided path (cached per path)
    - If RootConfig: return as-is
    """
    if not hasattr(get_config, "_cache"):
        get_config._cache = {}
    cache = get_config._cache

    if config_or_path is None:
        key = default_config_path()
    elif isinstance(config_or_path, str):
        key = config_or_path
    else:
        return config_or_path

    if key not in cache:
        cache[key] = load_config(path=key)
    return cache[key]
