from __future__ import annotations

from typing import Any, Callable, Dict

# Global registries for pluggable components
CARD_ENCODERS: Dict[str, Any] = {}
ACTION_ENCODERS: Dict[str, Any] = {}
MODELS: Dict[str, Any] = {}
POLICIES: Dict[str, Any] = {}


def _register(table: Dict[str, Any], name: str, obj: Any) -> Any:
    if name in table:
        raise ValueError(f"Duplicate registration for '{name}'")
    table[name] = obj
    return obj


# Decorators for registration


def register_card_encoder(name: str) -> Callable[[Any], Any]:
    def deco(cls_or_fn: Any) -> Any:
        return _register(CARD_ENCODERS, name, cls_or_fn)

    return deco


def register_action_encoder(name: str) -> Callable[[Any], Any]:
    def deco(cls_or_fn: Any) -> Any:
        return _register(ACTION_ENCODERS, name, cls_or_fn)

    return deco


def register_model(name: str) -> Callable[[Any], Any]:
    def deco(cls_or_fn: Any) -> Any:
        return _register(MODELS, name, cls_or_fn)

    return deco


def register_policy(name: str) -> Callable[[Any], Any]:
    def deco(cls_or_fn: Any) -> Any:
        return _register(POLICIES, name, cls_or_fn)

    return deco


# Builders


def build_card_encoder(name: str, **kwargs: Any) -> Any:
    if name not in CARD_ENCODERS:
        raise KeyError(f"Unknown card encoder '{name}'")
    return CARD_ENCODERS[name](**kwargs)


def build_action_encoder(name: str, **kwargs: Any) -> Any:
    if name not in ACTION_ENCODERS:
        raise KeyError(f"Unknown action encoder '{name}'")
    return ACTION_ENCODERS[name](**kwargs)


def build_model(name: str, **kwargs: Any) -> Any:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'")
    return MODELS[name](**kwargs)


def build_policy(name: str, **kwargs: Any) -> Any:
    if name not in POLICIES:
        raise KeyError(f"Unknown policy '{name}'")
    return POLICIES[name](**kwargs)
