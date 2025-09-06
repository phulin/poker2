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
    stack: int = 1000
    sb: int = 5
    bb: int = 10
    seed: int = 0
    ppo_eps: float = 0.2
    ppo_delta1: float = 3.0
    gamma: float = 0.999
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    # Training scale knobs
    learning_rate: float = 3e-4
    batch_size: int = 2048
    mini_batch_size: int = 2048
    num_epochs: int = 4
    # Betting bins expressed as fractions of total_committed reference
    bet_bins: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    # Replay buffer batches: how many batches of steps to retain in replay
    replay_buffer_batches: int = 1
    # Value loss configuration
    value_loss_type: str = "mse"  # "mse" or "huber"
    huber_delta: float = 1.0


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
        stack=int(data["stack"]),
        sb=int(data["sb"]),
        bb=int(data["bb"]),
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
        mini_batch_size=int(data["mini_batch_size"]),
        num_epochs=int(data["num_epochs"]),
        bet_bins=list(data["bet_bins"]),
        replay_buffer_batches=int(data["replay_buffer_batches"]),
        value_loss_type=str(data["value_loss_type"]),
        huber_delta=float(data["huber_delta"]),
    )
