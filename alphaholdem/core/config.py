from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Lazy dependency; caller can supply dict directly


@dataclass
class ComponentSpec:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RootConfig:
    card_encoder: ComponentSpec
    action_encoder: ComponentSpec
    model: ComponentSpec
    policy: ComponentSpec
    nb: int = 9
    seed: int = 0
    ppo_eps: float = 0.2
    ppo_delta1: float = 3.0
    gamma: float = 0.999
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0


def _to_component_spec(data: Dict[str, Any], key: str) -> ComponentSpec:
    sub = data.get(key, {})
    if isinstance(sub, str):
        return ComponentSpec(name=sub)
    return ComponentSpec(name=sub.get("name"), kwargs=sub.get("kwargs", {}))


def load_config(path: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> RootConfig:
    if data is None:
        if path is None:
            raise ValueError("Either path or data must be provided")
        if yaml is None:
            raise ImportError("PyYAML not installed; pass data= dict instead")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    return RootConfig(
        card_encoder=_to_component_spec(data, "card_encoder"),
        action_encoder=_to_component_spec(data, "action_encoder"),
        model=_to_component_spec(data, "model"),
        policy=_to_component_spec(data, "policy"),
        nb=int(data.get("nb", 9)),
        seed=int(data.get("seed", 0)),
        ppo_eps=float(data.get("ppo_eps", 0.2)),
        ppo_delta1=float(data.get("ppo_delta1", 3.0)),
        gamma=float(data.get("gamma", 0.999)),
        gae_lambda=float(data.get("gae_lambda", 0.95)),
        entropy_coef=float(data.get("entropy_coef", 0.01)),
        value_coef=float(data.get("value_coef", 0.5)),
        grad_clip=float(data.get("grad_clip", 1.0)),
    )
