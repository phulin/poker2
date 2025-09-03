from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

from .config import RootConfig, load_config


def default_config_path() -> str:
    """Return absolute path to the default YAML config."""
    return str(Path(__file__).resolve().parents[2] / "configs" / "default.yaml")


def get_config(config_or_path: Optional[Union[str, RootConfig]] = None) -> RootConfig:
    """
    Centralized config loader.
    - If None: load default YAML
    - If str: load YAML at the provided path
    - If RootConfig: return as-is
    """
    if config_or_path is None:
        return load_config(path=default_config_path())
    if isinstance(config_or_path, str):
        return load_config(path=config_or_path)
    return config_or_path


