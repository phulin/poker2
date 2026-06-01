from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache
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


def _normuon_ns5(
    matrix: torch.Tensor,
    *,
    eps: torch.Tensor,
    ns_steps: int,
) -> torch.Tensor:
    original_dtype = matrix.dtype
    if matrix.dtype in {torch.float16, torch.bfloat16}:
        matrix = matrix.float()

    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.transpose(0, 1)

    x = matrix / (torch.linalg.vector_norm(matrix) + eps)
    a = 3.4445
    b = -4.7750
    c = 2.0315
    for _ in range(ns_steps):
        xx_t = x @ x.transpose(0, 1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.to(original_dtype)


def _normuon_matrix_update(
    param: torch.Tensor,
    grad: torch.Tensor,
    momentum: torch.Tensor,
    row_second_moment: torch.Tensor,
    lr_tensor: torch.Tensor,
    weight_decay_tensor: torch.Tensor,
    beta1_tensor: torch.Tensor,
    beta2_tensor: torch.Tensor,
    eps_tensor: torch.Tensor,
    ns_eps_tensor: torch.Tensor,
    ns_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    momentum = beta1_tensor * momentum + (1.0 - beta1_tensor) * grad
    orthogonal_update = _normuon_ns5(
        momentum,
        eps=ns_eps_tensor,
        ns_steps=ns_steps,
    )
    row_mean_square = orthogonal_update.square().mean(dim=1)
    row_second_moment = (
        beta2_tensor * row_second_moment + (1.0 - beta2_tensor) * row_mean_square
    )
    normalized_update = orthogonal_update / (
        row_second_moment.sqrt()[:, None] + eps_tensor
    )
    update_norm = torch.linalg.vector_norm(normalized_update).clamp_min(eps_tensor)
    muon_original_lr = lr_tensor * math.sqrt(
        max(1.0, float(param.shape[0]) / float(param.shape[1]))
    )
    target_update_norm = muon_original_lr * torch.linalg.vector_norm(
        orthogonal_update
    )
    scaled_lr = target_update_norm / update_norm
    param = param * (1.0 - lr_tensor * weight_decay_tensor)
    param = param - scaled_lr * normalized_update
    return param, momentum, row_second_moment


@lru_cache(maxsize=2)
def _normuon_update_fn(compile_update: bool) -> Any:
    if not compile_update:
        return _normuon_matrix_update
    return torch.compile(_normuon_matrix_update, fullgraph=True)


class NorMuon(torch.optim.Optimizer):
    """NorMuon optimizer for 2D matrix parameters.

    The per-matrix computation follows the supplied algorithm and is factored
    into a pure tensor function so CUDA training can compile it with
    ``torch.compile``.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        beta1: float = 0.95,
        beta2: float = 0.95,
        eps: float = 1e-8,
        ns_eps: float = 1e-7,
        ns_steps: int = 5,
        compile_update: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("NorMuon lr must be positive")
        if weight_decay < 0.0:
            raise ValueError("NorMuon weight_decay must be non-negative")
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError("NorMuon beta1 must be in [0, 1)")
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("NorMuon beta2 must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("NorMuon eps must be positive")
        if ns_eps <= 0.0:
            raise ValueError("NorMuon ns_eps must be positive")
        if ns_steps <= 0:
            raise ValueError("NorMuon ns_steps must be positive")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "ns_eps": ns_eps,
            "ns_steps": ns_steps,
            "compile_update": compile_update,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            update_fn = _normuon_update_fn(bool(group["compile_update"]))
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1 = float(group["beta1"])
            beta2 = float(group["beta2"])
            eps = float(group["eps"])
            ns_eps = float(group["ns_eps"])
            ns_steps = int(group["ns_steps"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.ndim != 2:
                    raise RuntimeError("NorMuon only supports 2D matrix parameters")
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("NorMuon does not support sparse gradients")
                state = self.state[param]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(param)
                    state["row_second_moment"] = torch.zeros(
                        param.shape[0],
                        dtype=param.dtype,
                        device=param.device,
                    )

                lr_tensor = torch.tensor(lr, dtype=param.dtype, device=param.device)
                weight_decay_tensor = torch.tensor(
                    weight_decay,
                    dtype=param.dtype,
                    device=param.device,
                )
                beta1_tensor = torch.tensor(
                    beta1,
                    dtype=param.dtype,
                    device=param.device,
                )
                beta2_tensor = torch.tensor(
                    beta2,
                    dtype=param.dtype,
                    device=param.device,
                )
                eps_tensor = torch.tensor(eps, dtype=param.dtype, device=param.device)
                ns_eps_tensor = torch.tensor(
                    ns_eps,
                    dtype=param.dtype,
                    device=param.device,
                )
                new_param, new_momentum, new_row_second_moment = update_fn(
                    param,
                    grad,
                    state["momentum"],
                    state["row_second_moment"],
                    lr_tensor,
                    weight_decay_tensor,
                    beta1_tensor,
                    beta2_tensor,
                    eps_tensor,
                    ns_eps_tensor,
                    ns_steps,
                )
                param.copy_(new_param)
                state["momentum"].copy_(new_momentum)
                state["row_second_moment"].copy_(new_row_second_moment)
        return loss


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


def _normuon(
    params: Iterable[nn.Parameter],
    train_cfg: TrainingConfig,
    lr: float,
    *,
    compile_update: bool,
) -> NorMuon:
    return NorMuon(
        params,
        lr=lr,
        weight_decay=train_cfg.weight_decay,
        beta1=getattr(train_cfg, "normuon_beta1", train_cfg.muon_momentum),
        beta2=getattr(train_cfg, "normuon_beta2", 0.95),
        eps=getattr(train_cfg, "normuon_eps", 1e-8),
        ns_eps=train_cfg.muon_eps,
        ns_steps=train_cfg.muon_ns_steps,
        compile_update=compile_update,
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
    if optimizer_name not in {"muon", "normuon"}:
        raise ValueError(
            "train.optimizer must be one of: adamw, muon, normuon; "
            f"got {train_cfg.optimizer!r}"
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
        if optimizer_name == "muon":
            matrix_optimizer = _muon(
                matrix_params,
                train_cfg,
                train_cfg.learning_rate,
            )
        else:
            matrix_optimizer = _normuon(
                matrix_params,
                train_cfg,
                train_cfg.learning_rate,
                compile_update=device.type == "cuda",
            )
        optimizers.append((optimizer_name, matrix_optimizer))
    if policy_head_matrix_params:
        if optimizer_name == "muon":
            policy_head_optimizer = _muon(
                policy_head_matrix_params,
                train_cfg,
                policy_head_muon_lr,
            )
        else:
            policy_head_optimizer = _normuon(
                policy_head_matrix_params,
                train_cfg,
                policy_head_muon_lr,
                compile_update=device.type == "cuda",
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
