from __future__ import annotations

from typing import Any

import torch

from p2.rl.rebel_batch import RebelBatch


def pot_relative_value_error_sums(
    output: Any,
    batch: RebelBatch,
    loss_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if output.hand_values is None or batch.value_targets is None:
        return {}
    pot = batch.statistics.get("pot")
    scale = batch.statistics.get("scale")
    if pot is None or scale is None:
        return {}

    corrected = loss_dict.get("value_predictions")
    predictions = (
        corrected.float()
        if isinstance(corrected, torch.Tensor)
        else output.hand_values.float()
    )
    targets = batch.value_targets.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )
    scale_tensor = scale.to(
        device=predictions.device,
        dtype=predictions.dtype,
    ).clamp_min(1.0)
    pot_scale = pot.to(
        device=predictions.device,
        dtype=predictions.dtype,
    ).clamp_min(1.0)
    while scale_tensor.ndim < predictions.ndim:
        scale_tensor = scale_tensor.unsqueeze(-1)
    while pot_scale.ndim < predictions.ndim:
        pot_scale = pot_scale.unsqueeze(-1)

    relative_abs_error = (predictions - targets).abs() * scale_tensor / pot_scale
    relative_sq_error = relative_abs_error.square()
    weights = loss_dict.get("value_weights")
    if isinstance(weights, torch.Tensor):
        value_weights = weights.to(device=predictions.device, dtype=predictions.dtype)
    else:
        value_weights = torch.ones_like(relative_abs_error)
    return {
        "pot_relative_abs_error_sum": (relative_abs_error * value_weights).sum(),
        "pot_relative_sq_error_sum": (relative_sq_error * value_weights).sum(),
        "pot_relative_weight_sum": value_weights.sum().clamp_min(1.0e-8),
    }


def pot_relative_value_error_metrics(
    output: Any,
    batch: RebelBatch,
    loss_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    sums = pot_relative_value_error_sums(output, batch, loss_dict)
    if not sums:
        return {}
    denom = sums["pot_relative_weight_sum"]
    relative_mse = sums["pot_relative_sq_error_sum"] / denom
    return {
        "pot_relative_mae": sums["pot_relative_abs_error_sum"] / denom,
        "pot_relative_mse": relative_mse,
        "pot_relative_rmse": relative_mse.sqrt(),
    }


__all__ = [
    "pot_relative_value_error_metrics",
    "pot_relative_value_error_sums",
]
