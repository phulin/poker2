from __future__ import annotations

import os
from typing import Any

import torch

from p2.models.mlp.better_ffn import BetterSplitFFN


class CheckpointIO:
    """Shared helpers for ReBeL checkpoint loading and tensor dtype restoration."""

    @staticmethod
    def load(path: str, *, map_location: torch.device) -> dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)

    @staticmethod
    def metadata(path: str, *, map_location: torch.device) -> dict[str, Any]:
        if not path or not os.path.exists(path):
            return {}
        checkpoint = CheckpointIO.load(path, map_location=map_location)
        metadata = checkpoint.get("metadata", {})
        return dict(metadata) if type(metadata) is dict else {}

    @staticmethod
    def cast_floating_state_dict(
        state_dict: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(dtype) if value.dtype.is_floating_point else value
            for key, value in state_dict.items()
        }

    @staticmethod
    def model_state_for_host_dtype(
        checkpoint: dict[str, Any],
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        model_state = checkpoint["model"]
        save_dtype = checkpoint.get("save_dtype")
        if save_dtype is not None and save_dtype != str(dtype):
            return CheckpointIO.cast_floating_state_dict(model_state, dtype)
        return model_state

    @staticmethod
    def load_model_weights(
        trainer: Any,
        checkpoint_path: str,
        *,
        strict: bool | None = None,
        sync_models: bool = True,
    ) -> int:
        checkpoint = CheckpointIO.load(checkpoint_path, map_location=trainer.device)
        model_state = CheckpointIO.model_state_for_host_dtype(
            checkpoint,
            trainer.float_dtype,
        )
        strict_loading = (
            trainer.cfg.strict_model_loading if strict is None else bool(strict)
        )
        if checkpoint.get("model_component") == "value_model":
            if type(trainer.model) is not BetterSplitFFN:
                raise TypeError("value-only checkpoints require a BetterSplitFFN model")
            trainer.model.value_model.load_state_dict(
                model_state,
                strict=strict_loading,
            )
        else:
            trainer.model.load_state_dict(model_state, strict=strict_loading)

        returned_step = int(checkpoint.get("step", -1))
        sync_step = int(checkpoint.get("step", 0))
        if sync_models:
            trainer.sync_inference_model()
            trainer.sync_cfr_target_model(sync_step)
        trainer.model.train()
        return returned_step
