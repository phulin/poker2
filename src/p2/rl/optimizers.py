from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from p2.core.structured_config import TrainingConfig


class SplitOptimizer:
    """Small composite optimizer for parameter sets with different algorithms."""

    def __init__(self, optimizers: Iterable[tuple[str, torch.optim.Optimizer]]) -> None:
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("SplitOptimizer requires at least one optimizer")

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [
            param_group
            for _, optimizer in self.optimizers
            for param_group in optimizer.param_groups
        ]

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        for _, optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self, closure: Any | None = None) -> None:
        if closure is not None:
            raise RuntimeError("SplitOptimizer does not support closures")
        for _, optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "split",
            "optimizer_order": [name for name, _ in self.optimizers],
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in self.optimizers
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("type") != "split" or "optimizers" not in state_dict:
            raise ValueError(
                "Cannot load a non-split optimizer state into SplitOptimizer"
            )

        saved_optimizers = state_dict["optimizers"]
        for name, optimizer in self.optimizers:
            if name not in saved_optimizers:
                raise ValueError(f"Missing optimizer state for {name!r}")
            optimizer.load_state_dict(saved_optimizers[name])


TrainOptimizer = torch.optim.Optimizer | SplitOptimizer


def _optimizer_name(train_cfg: TrainingConfig) -> str:
    return str(train_cfg.optimizer).strip().lower()


def _adamw(
    params: Iterable[nn.Parameter],
    train_cfg: TrainingConfig,
    device: torch.device,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        fused=(device.type == "cuda"),
    )


def build_optimizer(
    model: nn.Module,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> TrainOptimizer:
    optimizer_name = _optimizer_name(train_cfg)
    if optimizer_name == "adamw":
        return _adamw(model.parameters(), train_cfg, device)
    if optimizer_name != "muon":
        raise ValueError(
            f"train.optimizer must be one of: adamw, muon; got {train_cfg.optimizer!r}"
        )

    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError(
            "train.optimizer=muon requires a PyTorch build with torch.optim.Muon"
        )

    matrix_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    matrix_param_ids: set[int] = set()
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, nn.Linear) and name == "weight" and param.ndim == 2:
                matrix_params.append(param)
                matrix_param_ids.add(id(param))

    for param in model.parameters():
        if param.requires_grad and id(param) not in matrix_param_ids:
            other_params.append(param)

    optimizers: list[tuple[str, torch.optim.Optimizer]] = []
    if matrix_params:
        optimizers.append(
            (
                "muon",
                muon_cls(
                    matrix_params,
                    lr=train_cfg.learning_rate,
                    weight_decay=train_cfg.weight_decay,
                    momentum=train_cfg.muon_momentum,
                    nesterov=train_cfg.muon_nesterov,
                    eps=train_cfg.muon_eps,
                    ns_steps=train_cfg.muon_ns_steps,
                    adjust_lr_fn=train_cfg.muon_adjust_lr_fn,
                ),
            )
        )
    if other_params:
        optimizers.append(("adamw", _adamw(other_params, train_cfg, device)))

    if not optimizers:
        raise ValueError("No trainable parameters found for optimizer")
    if len(optimizers) == 1:
        return optimizers[0][1]
    return SplitOptimizer(optimizers)
