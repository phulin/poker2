from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import MutableMapping
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from flash_muon.muon import fast_newtonschulz

from p2.core.structured_config import TrainingConfig
from p2.models.mlp.better_ffn import BetterSplitFFN


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
            if name == "adamw":
                for param_group in optimizer.param_groups:
                    param_group["lr_role"] = "adamw"


TrainOptimizer = torch.optim.Optimizer | SplitOptimizer


_MUON_DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MUON_DEFAULT_EPS = 1e-7


def _adjust_muon_lr(
    lr: float,
    adjust_lr_fn: str | None,
    param_shape: torch.Size,
) -> float:
    rows, cols = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, float(rows) / float(cols)))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(float(rows), float(cols)))
    else:
        ratio = 1.0
    return lr * ratio


class FlashMuon(torch.optim.Optimizer):
    """Muon optimizer using flash-muon's fused Newton-Schulz kernel.

    This mirrors PyTorch's single-tensor Muon update and keeps this codebase's
    existing Muon config surface, while avoiding flash-muon's distributed
    optimizer wrapper.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = _MUON_DEFAULT_NS_COEFFICIENTS,
        eps: float = _MUON_DEFAULT_EPS,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn is not None and adjust_lr_fn not in {
            "original",
            "match_rms_adamw",
        }:
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
        if ns_coefficients != _MUON_DEFAULT_NS_COEFFICIENTS:
            raise ValueError(
                "FlashMuon requires the default Muon Newton-Schulz coefficients"
            )
        if float(eps) != _MUON_DEFAULT_EPS:
            raise ValueError("FlashMuon requires muon_eps=1e-7")
        if ns_steps <= 0:
            raise ValueError("FlashMuon ns_steps must be positive")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": ns_coefficients,
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        "FlashMuon only supports 2D parameters whereas we found "
                        f"a parameter with size: {param.size()}"
                    )

    def _init_group(
        self,
        group: MutableMapping[str, Any],
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        momentum_bufs: list[Tensor],
    ) -> None:
        for param in group["params"]:
            if param.grad is None:
                continue
            if torch.is_complex(param):
                raise RuntimeError("FlashMuon does not support complex parameters")
            if param.grad.is_sparse:
                raise RuntimeError("FlashMuon does not support sparse gradients")

            params_with_grad.append(param)
            grads.append(param.grad)

            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    param.grad,
                    memory_format=torch.preserve_format,
                )
            momentum_bufs.append(state["momentum_buffer"])

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            adjust_lr_fn = group["adjust_lr_fn"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            momentum_bufs: list[Tensor] = []
            self._init_group(group, params_with_grad, grads, momentum_bufs)

            for param, grad, buf in zip(params_with_grad, grads, momentum_bufs):
                if grad.ndim != 2:
                    raise ValueError("Param gradient must be a 2D matrix")
                if param.device.type != "cuda":
                    raise RuntimeError("FlashMuon requires CUDA parameters")

                buf.lerp_(grad, 1.0 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf
                update = fast_newtonschulz(update, steps=ns_steps)
                adjusted_lr = _adjust_muon_lr(lr, adjust_lr_fn, param.shape)

                param.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-adjusted_lr)

        return loss


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


class NorMuon(torch.optim.Optimizer):
    """NorMuon optimizer for 2D matrix parameters.

    The per-matrix computation follows the supplied algorithm and runs eagerly
    inside the optimizer step.
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
                new_param, new_momentum, new_row_second_moment = _normuon_matrix_update(
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
    device: torch.device,
) -> torch.optim.Optimizer:
    if device.type == "cuda":
        return FlashMuon(
            params,
            lr=lr,
            weight_decay=train_cfg.weight_decay,
            momentum=train_cfg.muon_momentum,
            nesterov=train_cfg.muon_nesterov,
            eps=train_cfg.muon_eps,
            ns_steps=train_cfg.muon_ns_steps,
            adjust_lr_fn=train_cfg.muon_adjust_lr_fn,
        )
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
        beta1=train_cfg.normuon_beta1,
        beta2=train_cfg.normuon_beta2,
        eps=train_cfg.normuon_eps,
        ns_eps=train_cfg.muon_eps,
        ns_steps=train_cfg.muon_ns_steps,
        compile_update=compile_update,
    )


def build_optimizer(
    model: nn.Module,
    train_cfg: TrainingConfig,
    device: torch.device,
) -> TrainOptimizer:
    if type(model) is BetterSplitFFN:
        return SplitOptimizer(
            [
                ("policy", build_optimizer(model.policy_model, train_cfg, device)),
                ("value", build_optimizer(model.value_model, train_cfg, device)),
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
        if optimizer_name == "muon":
            matrix_optimizer = _muon(
                matrix_params,
                train_cfg,
                train_cfg.learning_rate,
                device,
            )
        else:
            matrix_optimizer = _normuon(
                matrix_params,
                train_cfg,
                train_cfg.learning_rate,
                compile_update=False,
        )
        optimizers.append((optimizer_name, matrix_optimizer))
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
