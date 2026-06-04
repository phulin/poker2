#!/usr/bin/env python3
"""Evaluate a ReBeL checkpoint's value loss on a solved dataset value stream."""

from __future__ import annotations

import argparse
import json

import torch

from p2.cli.train_rebel import _device_from_config
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.validation_set import RebelValueValidationSetEvaluator


def _load_model_weights(trainer: RebelCFRTrainer, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model"]
    save_dtype_str = checkpoint.get("save_dtype")
    if save_dtype_str is not None and save_dtype_str != str(trainer.float_dtype):
        model_state = {
            key: (
                value.to(trainer.float_dtype)
                if value.dtype.is_floating_point
                else value
            )
            for key, value in model_state.items()
        }

    if checkpoint.get("model_component") == "value_model":
        value_model = getattr(trainer.model, "value_model", trainer.model)
        value_model.load_state_dict(model_state, strict=trainer.cfg.strict_model_loading)
    else:
        trainer.model.load_state_dict(
            model_state, strict=trainer.cfg.strict_model_loading
        )
    trainer.model.to(trainer.device)
    trainer._sync_inference_model()
    trainer.model.eval()


@torch.no_grad()
def evaluate_value_loss(
    *,
    checkpoint_path: str,
    dataset_path: str,
    batch_size: int,
    device_override: str | None,
) -> dict[str, float | int | str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(dict(checkpoint["config"]))
    if device_override is not None:
        cfg.device = device_override
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.model.compile = "off"
    cfg.data.mode = "live"
    cfg.train.replay_buffer_device = "cpu"
    cfg.train.replay_buffer_batches = 1

    device = _device_from_config(cfg)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    trainer = RebelCFRTrainer(cfg=cfg, device=device, pregeneration_only=True)
    _load_model_weights(trainer, checkpoint_path)
    evaluator = RebelValueValidationSetEvaluator(
        trainer=trainer,
        cfg=cfg,
        dataset_path=dataset_path,
        batch_size=batch_size,
    )
    validation_metrics = evaluator.evaluate()
    return {
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
        "device": str(device),
        "examples": validation_metrics["validation_examples"],
        "batch_size": batch_size,
        "value_loss": validation_metrics["validation_value_loss"],
        "element_mean_weighted_square_error": (
            validation_metrics["validation_element_mean_weighted_square_error"]
        ),
        "batch_loss_mean": validation_metrics["validation_batch_loss_mean"],
        "num_batches": validation_metrics["validation_num_batches"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    result = evaluate_value_loss(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        device_override=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
