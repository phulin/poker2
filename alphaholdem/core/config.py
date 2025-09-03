from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

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
    seed: int
    ppo_eps: float
    ppo_delta1: float
    gamma: float
    gae_lambda: float
    entropy_coef: float
    value_coef: float
    grad_clip: float
    # Training scale knobs
    learning_rate: float
    batch_size: int
    mini_batch_size: int
    num_epochs: int
    trajectories_per_step: int
    # Betting bins expressed as fractions of total_committed reference
    bet_bins: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])


def _to_component_spec(data: Dict[str, Any], key: str) -> ComponentSpec:
    sub = data.get(key, {})
    if isinstance(sub, str):
        return ComponentSpec(name=sub)
    return ComponentSpec(name=sub.get("name"), kwargs=sub.get("kwargs", {}))


def load_config(
    path: Optional[str] = None, data: Optional[Dict[str, Any]] = None
) -> RootConfig:
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
        seed=int(data["seed"]),
        ppo_eps=float(data["ppo_eps"]),
        ppo_delta1=float(data["ppo_delta1"]),
        gamma=float(data["gamma"]),
        gae_lambda=float(data["gae_lambda"]),
        entropy_coef=float(data["entropy_coef"]),
        value_coef=float(data["value_coef"]),
        grad_clip=float(data["grad_clip"]),
        learning_rate=float(data["learning_rate"]),
        batch_size=int(data["batch_size"]),
        num_epochs=int(data["num_epochs"]),
        trajectories_per_step=int(data["trajectories_per_step"]),
        bet_bins=list(data["bet_bins"]),
    )
