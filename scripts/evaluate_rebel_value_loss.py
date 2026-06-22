#!/usr/bin/env python3
"""Evaluate a ReBeL checkpoint's value loss on a solved dataset value stream."""

from __future__ import annotations

import copy
import json

import hydra
import torch
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.validation_set import RebelValueValidationSetEvaluator
from p2.runtime.training_run import device_from_config, setup_torch_runtime


def _checkpoint_path(cfg: Config) -> str:
    if cfg.resume_from is None or cfg.resume_from == "":
        raise ValueError("resume_from must point to the checkpoint to evaluate")
    return cfg.resume_from


def _dataset_path(cfg: Config) -> str:
    if cfg.validation_set.dataset is None or cfg.validation_set.dataset == "":
        raise ValueError("validation_set.dataset must point to a solved dataset")
    return cfg.validation_set.dataset


def _load_model_weights(trainer: RebelCFRTrainer, checkpoint_path: str) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location=trainer.device, weights_only=False
    )
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
        if type(trainer.model) is not BetterSplitFFN:
            raise TypeError("value-only checkpoints require a BetterSplitFFN model")
        trainer.model.value_model.load_state_dict(
            model_state, strict=trainer.cfg.strict_model_loading
        )
    else:
        trainer.model.load_state_dict(
            model_state, strict=trainer.cfg.strict_model_loading
        )
    trainer._sync_inference_model()
    trainer.model.eval()


@torch.no_grad()
def evaluate_value_loss(cfg: Config) -> dict[str, float | int | str]:
    eval_cfg = copy.deepcopy(cfg)
    checkpoint_path = _checkpoint_path(eval_cfg)
    dataset_path = _dataset_path(eval_cfg)
    eval_cfg.use_wandb = False
    eval_cfg.trueskill.enabled = False
    eval_cfg.model.compile = "off"
    eval_cfg.data.mode = "live"
    eval_cfg.train.replay_buffer_device = "cpu"
    eval_cfg.train.replay_buffer_batches = 1

    device = device_from_config(eval_cfg)
    setup_torch_runtime(eval_cfg, device)
    trainer = RebelCFRTrainer(cfg=eval_cfg, device=device, pregeneration_only=True)
    _load_model_weights(trainer, checkpoint_path)
    evaluator = RebelValueValidationSetEvaluator(
        trainer=trainer,
        cfg=eval_cfg,
        dataset_path=dataset_path,
        batch_size=eval_cfg.validation_set.batch_size,
        max_examples=eval_cfg.validation_set.max_examples,
    )
    validation_metrics = evaluator.evaluate()
    return {
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
        "device": str(device),
        "examples": validation_metrics["validation_examples"],
        "batch_size": eval_cfg.validation_set.batch_size,
        "value_loss": validation_metrics["validation_value_loss"],
        "element_mean_weighted_square_error": (
            validation_metrics["validation_element_mean_weighted_square_error"]
        ),
        "batch_loss_mean": validation_metrics["validation_batch_loss_mean"],
        "num_batches": validation_metrics["validation_num_batches"],
    }


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config_rebel_evaluate_value_loss",
)
def main(dict_config: DictConfig) -> None:
    cfg = Config.from_dict_config(dict_config)
    result = evaluate_value_loss(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
