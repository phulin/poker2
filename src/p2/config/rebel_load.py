from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from p2.config.rebel_schema import RebelExperimentConfig
from p2.core.structured_config import Config, ModelType, PreflopModelType
from p2.env.card_utils import PREFLOP_HANDS


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
_REBEL_DATA_MODES = frozenset({"live", "pregenerated", "hybrid"})
_LIVE_ROOT_SOURCES = frozenset(
    {
        "self_play",
        "random_flop",
        "random_turn",
        "random_river",
        "random_flop_prefix",
        "random_turn_prefix",
        "random_river_prefix",
    }
)
_PREFLOP_MODEL_TYPES = frozenset(
    {PreflopModelType.ffn, PreflopModelType.transformer}
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


def validate_rebel_config(cfg: Config) -> None:
    if cfg.data.mode not in _REBEL_DATA_MODES:
        raise ValueError(
            "data.mode must be one of live, pregenerated, or hybrid; "
            f"got {cfg.data.mode!r}"
        )
    if cfg.data.live_root_source not in _LIVE_ROOT_SOURCES:
        raise ValueError(
            "data.live_root_source must be self_play, random_flop, random_turn, "
            "random_river, random_flop_prefix, random_turn_prefix, or "
            f"random_river_prefix; got {cfg.data.live_root_source!r}"
        )
    if cfg.curriculum.stages and cfg.data.mode not in {"live", "hybrid"}:
        raise ValueError(
            "curriculum stages require data.mode=live or data.mode=hybrid; "
            f"got {cfg.data.mode!r}"
        )
    if cfg.model.preflop_model_type not in _PREFLOP_MODEL_TYPES:
        raise ValueError(
            "model.preflop_model_type must be one of: ffn, transformer; "
            f"got {cfg.model.preflop_model_type!r}"
        )
    if cfg.env.num_players < 2:
        raise ValueError("env.num_players must be at least 2")
    if cfg.env.num_players != 2 and cfg.data.live_root_source == "self_play":
        if cfg.model.name is not ModelType.better_ffn:
            raise ValueError(
                "Multiway preflop self-play requires model.name=BetterFFN so "
                "the evaluator can use compact 169-hand preflop models."
            )
        cfg.model.preflop_hand_dim = PREFLOP_HANDS
    if (
        cfg.env.num_players != 2
        and cfg.search.sparse_fused
        and cfg.data.live_root_source != "self_play"
    ):
        raise ValueError(
            "Multiway fused sparse CFR is only available for preflop self-play roots."
        )


def load_rebel_experiment_config(
    dict_config: DictConfig,
) -> RebelExperimentConfig:
    container = _container_from_dict_config(dict_config)
    _reject_legacy_rebel_keys(container)
    cfg = Config.from_dict(_apply_rebel_loader_defaults(container))
    validate_rebel_config(cfg)
    return RebelExperimentConfig.from_trainer_config(cfg)


def load_rebel_config(dict_config: DictConfig) -> Config:
    return load_rebel_experiment_config(dict_config).to_trainer_config()


__all__ = [
    "load_rebel_config",
    "load_rebel_experiment_config",
    "validate_rebel_config",
]
