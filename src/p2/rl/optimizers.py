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
            if name in {"adamw", "policy_head_muon"}:
                lr_role = "adamw" if name == "adamw" else name
                for param_group in optimizer.param_groups:
                    param_group["lr_role"] = lr_role


TrainOptimizer = torch.optim.Optimizer | SplitOptimizer


_NORM_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.RMSNorm,
)


def _optimizer_name(train_cfg: TrainingConfig) -> str:
    return str(train_cfg.optimizer).strip().lower()


def _adamw(
    params: Iterable[nn.Parameter],
    train_cfg: TrainingConfig,
    device: torch.device,
    lr: float | None = None,
    no_decay_param_ids: set[int] | None = None,
) -> torch.optim.AdamW:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    no_decay_param_ids = no_decay_param_ids or set()
    for param in params:
        if id(param) in no_decay_param_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "weight_decay": train_cfg.weight_decay,
            }
        )
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg.learning_rate if lr is None else lr,
        fused=(device.type == "cuda"),
    )
    for param_group in optimizer.param_groups:
        param_group["lr_role"] = "adamw"
    return optimizer


def _is_policy_head_param(name: str) -> bool:
    return (
        name.startswith("policy_")
        or name.startswith("policy_head.")
        or name.startswith("policy_model.policy_")
        or name.startswith("policy_model.policy_head.")
    )


def _no_weight_decay_param_ids(model: nn.Module) -> set[int]:
    """Parameters that should not be pulled toward smaller logit/activation scale."""
    no_decay: set[int] = set()
    for module in model.modules():
        if isinstance(module, _NORM_MODULES):
            no_decay.update(
                id(param)
                for param in module.parameters(recurse=False)
                if param.requires_grad
            )
    return no_decay


def _muon(
    params: Iterable[nn.Parameter],
    train_cfg: TrainingConfig,
    lr: float,
) -> torch.optim.Optimizer:
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError(
            "train.optimizer=muon requires a PyTorch build with torch.optim.Muon"
        )
    return muon_cls(
        params,
        lr=lr,
        weight_decay=train_cfg.weight_decay,
        momentum=train_cfg.muon_momentum,
        nesterov=train_cfg.muon_nesterov,
        eps=train_cfg.muon_eps,
        ns_steps=train_cfg.muon_ns_steps,
        adjust_lr_fn=train_cfg.muon_adjust_lr_fn,
    )


def build_optimizer(
    model: nn.Module,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> TrainOptimizer:
    policy_model = getattr(model, "policy_model", None)
    value_model = getattr(model, "value_model", None)
    if isinstance(policy_model, nn.Module) and isinstance(value_model, nn.Module):
        return SplitOptimizer(
            [
                ("policy", build_optimizer(policy_model, train_cfg, device)),
                ("value", build_optimizer(value_model, train_cfg, device)),
            ]
        )

    optimizer_name = _optimizer_name(train_cfg)
    adamw_lr = (
        train_cfg.learning_rate
        if train_cfg.adamw_learning_rate is None
        else float(train_cfg.adamw_learning_rate)
    )
    if adamw_lr <= 0.0:
        raise ValueError("train.adamw_learning_rate must be positive when set")
    no_decay_param_ids = _no_weight_decay_param_ids(model)
    if optimizer_name == "adamw":
        return _adamw(
            model.parameters(),
            train_cfg,
            device,
            lr=adamw_lr,
            no_decay_param_ids=no_decay_param_ids,
        )
    if optimizer_name != "muon":
        raise ValueError(
            f"train.optimizer must be one of: adamw, muon; got {train_cfg.optimizer!r}"
        )

    policy_head_muon_lr = float(train_cfg.policy_head_muon_learning_rate)
    if policy_head_muon_lr <= 0.0:
        raise ValueError("train.policy_head_muon_learning_rate must be positive")

    matrix_params: list[nn.Parameter] = []
    policy_head_matrix_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    matrix_param_ids: set[int] = set()
    for module_prefix, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, nn.Linear) and name == "weight" and param.ndim == 2:
                full_name = f"{module_prefix}.{name}" if module_prefix else name
                if _is_policy_head_param(full_name):
                    policy_head_matrix_params.append(param)
                else:
                    matrix_params.append(param)
                matrix_param_ids.add(id(param))

    for param in model.parameters():
        if param.requires_grad and id(param) not in matrix_param_ids:
            other_params.append(param)

    optimizers: list[tuple[str, torch.optim.Optimizer]] = []
    if matrix_params:
        optimizers.append(
            ("muon", _muon(matrix_params, train_cfg, train_cfg.learning_rate))
        )
    if policy_head_matrix_params:
        policy_head_optimizer = _muon(
            policy_head_matrix_params,
            train_cfg,
            policy_head_muon_lr,
        )
        for param_group in policy_head_optimizer.param_groups:
            param_group["lr_role"] = "policy_head_muon"
        optimizers.append(("policy_head_muon", policy_head_optimizer))
    if other_params:
        optimizers.append(
            (
                "adamw",
                _adamw(
                    other_params,
                    train_cfg,
                    device,
                    lr=adamw_lr,
                    no_decay_param_ids=no_decay_param_ids,
                ),
            )
        )

    if not optimizers:
        raise ValueError("No trainable parameters found for optimizer")
    if len(optimizers) == 1:
        return optimizers[0][1]
    return SplitOptimizer(optimizers)
