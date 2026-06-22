from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from p2.config.rebel_schema import RebelExperimentConfig
from p2.core.structured_config import Config


_FORBIDDEN_REBEL_TOP_LEVEL_KEYS = frozenset(
    {
        "opponent_pool_type",
        "k_best_pool_size",
        "min_elo_diff",
        "min_step_diff",
        "k_factor",
        "eval_interval",
        "offload_opponent_models",
        "exploiter",
    }
)


def _container_from_dict_config(dict_config: DictConfig) -> dict[str, Any]:
    return dict(OmegaConf.to_container(dict_config, resolve=True))


def _reject_legacy_rebel_keys(container: Mapping[str, Any]) -> None:
    legacy_keys = sorted(_FORBIDDEN_REBEL_TOP_LEVEL_KEYS.intersection(container))
    if legacy_keys:
        joined = ", ".join(legacy_keys)
        raise ValueError(
            "ReBeL configs no longer accept PPO/K-best top-level fields: "
            f"{joined}"
        )


def _apply_rebel_loader_defaults(container: dict[str, Any]) -> dict[str, Any]:
    clean = dict(container)
    if clean.get("wandb_tags") is None:
        clean["wandb_tags"] = ["rebel", "cfr"]
    return clean


def load_rebel_experiment_config(
    dict_config: DictConfig,
) -> RebelExperimentConfig:
    container = _container_from_dict_config(dict_config)
    _reject_legacy_rebel_keys(container)
    cfg = Config.from_dict(_apply_rebel_loader_defaults(container))
    return RebelExperimentConfig.from_trainer_config(cfg)


def load_rebel_config(dict_config: DictConfig) -> Config:
    return load_rebel_experiment_config(dict_config).to_trainer_config()


__all__ = [
    "load_rebel_config",
    "load_rebel_experiment_config",
]
