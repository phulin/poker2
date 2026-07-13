#!/usr/bin/env python3
"""Evaluate a ReBeL checkpoint using a saved resolved_config.json."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

import p2.models.mlp.better_ffn as better_ffn_module
import p2.rl.cfr_trainer as cfr_trainer_module
from p2.core.structured_config import Config
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.validation_set import RebelValueValidationSetEvaluator
from p2.runtime.training_run import device_from_config, setup_torch_runtime


def _load_resolved_config(path: str | Path) -> Config:
    container: dict[str, Any] = json.loads(Path(path).read_text())
    return _config_from_container(container)


def _load_checkpoint_config(path: str | Path) -> Config:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    container = checkpoint.get("config")
    if not isinstance(container, dict):
        raise ValueError(f"checkpoint does not contain a config dict: {path}")
    return _config_from_container(container)


def _config_from_container(container: dict[str, Any]) -> Config:
    container = _filter_dataclass_fields(Config, container)
    return Config.from_dict(container)


def _filter_dataclass_fields(dataclass_type: type, container: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for field_info in fields(dataclass_type):
        if field_info.name not in container:
            continue
        value = container[field_info.name]
        default_factory = getattr(field_info, "default_factory", MISSING)
        if isinstance(value, dict) and default_factory is not MISSING:
            try:
                default_value = default_factory()
            except TypeError:
                default_value = None
            if default_value is not None and is_dataclass(default_value):
                value = _filter_dataclass_fields(type(default_value), value)
        clean[field_info.name] = value
    return clean


def _load_model_weights(trainer: RebelCFRTrainer, checkpoint_path: str) -> None:
    checkpoint = torch.load(
        checkpoint_path, map_location=trainer.device, weights_only=False
    )
    model_state = checkpoint["model"]
    save_dtype_str = checkpoint.get("save_dtype")
    if save_dtype_str is not None and save_dtype_str != str(trainer.float_dtype):
        model_state = {
            key: value.to(trainer.float_dtype) if value.dtype.is_floating_point else value
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


def _infer_context_in_dim(checkpoint_path: str) -> int | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model"]
    for key in (
        "value_model.context_encoder.linear_in.weight",
        "policy_model.context_encoder.linear_in.weight",
        "context_encoder.linear_in.weight",
    ):
        tensor = model_state.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:
            return int(tensor.shape[1])
    return None


def _maybe_trim_context(features: MLPFeatures, context_prefix_length: int | None) -> MLPFeatures:
    if context_prefix_length is None:
        return features
    if int(features.context.shape[-1]) == int(context_prefix_length):
        return features
    if int(features.context.shape[-1]) < int(context_prefix_length):
        raise ValueError(
            f"cannot trim context width {features.context.shape[-1]} to "
            f"{context_prefix_length}"
        )
    return MLPFeatures(
        context=features.context[..., : int(context_prefix_length)].contiguous(),
        street=features.street,
        to_act=features.to_act,
        board=features.board,
        beliefs=features.beliefs,
        hand_dim=features.hand_dim,
    )


@torch.no_grad()
def _closing_leaf_stratification(
    *,
    trainer: RebelCFRTrainer,
    cfg: Config,
    evaluator: RebelValueValidationSetEvaluator,
    context_prefix_length: int | None,
) -> dict[str, Any]:
    total_examples = evaluator.total_examples
    if cfg.validation_set.max_examples is not None:
        total_examples = min(total_examples, cfg.validation_set.max_examples)
    loss_sums: list[torch.Tensor] = []
    weight_sums: list[torch.Tensor] = []
    closing_counts: list[torch.Tensor] = []
    closing_fracs: list[torch.Tensor] = []
    leaf_totals: list[torch.Tensor] = []
    pots: list[torch.Tensor] = []
    local_exploitabilities: list[torch.Tensor] = []
    belief_entropies: list[torch.Tensor] = []
    belief_entropies_normalized: list[torch.Tensor] = []

    for start in range(0, total_examples, cfg.validation_set.batch_size):
        count = min(cfg.validation_set.batch_size, total_examples - start)
        batch = evaluator.dataset.get_batch(
            "value",
            start,
            count,
            device=trainer.device,
            float_dtype=trainer.float_dtype,
        )
        batch.features = _maybe_trim_context(batch.features, context_prefix_length)
        with trainer.model_autocast():
            output = trainer.inference_model.repeat(
                batch.features,
                count=cfg.model.num_supervisions,
                include_policy=False,
                apply_zero_sum=False,
            )
        loss_dict = trainer.loss_fn.forward_value(output, batch)
        value_loss_all = loss_dict["value_loss_all"].detach().double()
        value_weights = loss_dict["value_weights"].detach().double()
        per_example_loss_sum = value_loss_all.flatten(1).sum(dim=1)
        per_example_weight_sum = value_weights.flatten(1).sum(dim=1).clamp_min(1e-12)

        closing_count = batch.statistics["leaf_target_source_3_count"].double()
        leaf_total = batch.statistics["leaf_total_count"].double().clamp_min(1.0)
        closing_frac = closing_count / leaf_total
        beliefs = batch.features.beliefs.detach().double()
        hand_dim = int(batch.features.hand_dim)
        beliefs = beliefs.view(beliefs.shape[0], -1, hand_dim).clamp_min(0.0)
        belief_mass = beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        belief_probs = beliefs / belief_mass
        belief_entropy = -(belief_probs.clamp_min(1e-12).log() * belief_probs).sum(dim=-1)
        belief_entropy = belief_entropy.mean(dim=-1)
        loss_sums.append(per_example_loss_sum.cpu())
        weight_sums.append(per_example_weight_sum.cpu())
        closing_counts.append(closing_count.cpu())
        closing_fracs.append(closing_frac.cpu())
        leaf_totals.append(leaf_total.cpu())
        pots.append(batch.statistics["pot"].double().cpu())
        local_exploitabilities.append(
            batch.statistics["local_exploitability"].double().cpu()
        )
        belief_entropies.append(belief_entropy.cpu())
        belief_entropies_normalized.append((belief_entropy / math.log(hand_dim)).cpu())

    loss_sum = torch.cat(loss_sums)
    weight_sum = torch.cat(weight_sums).clamp_min(1e-12)
    per_example_loss = loss_sum / weight_sum
    stats = {
        "closing_count": torch.cat(closing_counts),
        "closing_fraction": torch.cat(closing_fracs),
        "leaf_total": torch.cat(leaf_totals),
        "pot": torch.cat(pots),
        "local_exploitability": torch.cat(local_exploitabilities),
        "belief_entropy": torch.cat(belief_entropies),
        "belief_entropy_normalized": torch.cat(belief_entropies_normalized),
    }

    def bucket_by_quantile(name: str, values: torch.Tensor) -> list[dict[str, float | int | str]]:
        quantiles = torch.quantile(
            values,
            torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0], dtype=values.dtype),
        )
        rows: list[dict[str, float | int | str]] = []
        for idx in range(4):
            lower = quantiles[idx]
            upper = quantiles[idx + 1]
            if idx == 3:
                mask = (values >= lower) & (values <= upper)
            else:
                mask = (values >= lower) & (values < upper)
            if not bool(mask.any().item()):
                continue
            rows.append(
                {
                    "stat": name,
                    "bin": f"q{idx + 1}",
                    "min": float(values[mask].min().item()),
                    "max": float(values[mask].max().item()),
                    "examples": int(mask.sum().item()),
                    "mean_stat": float(values[mask].mean().item()),
                    "value_loss": float(
                        loss_sum[mask].sum().item()
                        / max(weight_sum[mask].sum().item(), 1e-12)
                    ),
                    "mean_example_loss": float(per_example_loss[mask].mean().item()),
                }
            )
        return rows

    correlations: dict[str, float] = {}
    centered_loss = per_example_loss - per_example_loss.mean()
    loss_norm = centered_loss.square().sum().sqrt().clamp_min(1e-12)
    for name, values in stats.items():
        centered_values = values - values.mean()
        denom = centered_values.square().sum().sqrt().clamp_min(1e-12) * loss_norm
        correlations[name] = float((centered_values * centered_loss).sum().item() / denom.item())

    entropy = stats["belief_entropy_normalized"]

    def entropy_slice(name: str, mask: torch.Tensor) -> dict[str, float | int | str]:
        if not bool(mask.any().item()):
            return {"slice": name, "examples": 0}
        return {
            "slice": name,
            "examples": int(mask.sum().item()),
            "fraction": float(mask.double().mean().item()),
            "value_loss": float(
                loss_sum[mask].sum().item()
                / max(weight_sum[mask].sum().item(), 1e-12)
            ),
            "mean_example_loss": float(per_example_loss[mask].mean().item()),
            "mean_belief_entropy_normalized": float(entropy[mask].mean().item()),
            "mean_pot": float(stats["pot"][mask].mean().item()),
            "mean_local_exploitability": float(
                stats["local_exploitability"][mask].mean().item()
            ),
        }

    entropy_threshold_slices = [
        entropy_slice("entropy_ge_old_random_q05_0.727", entropy >= 0.727288),
        entropy_slice("entropy_ge_old_random_q10_0.794", entropy >= 0.794355),
        entropy_slice("entropy_ge_old_random_q25_0.830", entropy >= 0.830319),
        entropy_slice("entropy_ge_old_random_median_0.936", entropy >= 0.935698),
        entropy_slice("entropy_lt_old_random_q10_0.794", entropy < 0.794355),
        entropy_slice("entropy_lt_old_random_q25_0.830", entropy < 0.830319),
    ]

    return {
        "correlation_with_per_example_loss": correlations,
        "entropy_threshold_slices": entropy_threshold_slices,
        "quantile_buckets": {
            name: bucket_by_quantile(name, values) for name, values in stats.items()
        },
        "loss_quantile_buckets": bucket_by_quantile("per_example_loss", per_example_loss),
        "mean_stats_by_loss_quantile": [
            {
                "loss_bin": row["bin"],
                "loss_min": row["min"],
                "loss_max": row["max"],
                "examples": row["examples"],
                **{
                    f"mean_{name}": float(
                        values[
                            (
                                (per_example_loss >= row["min"])
                                & (
                                    (per_example_loss <= row["max"])
                                    if row["bin"] == "q4"
                                    else (per_example_loss < row["max"])
                                )
                            )
                        ]
                        .mean()
                        .item()
                    )
                    for name, values in stats.items()
                },
            }
            for row in bucket_by_quantile("per_example_loss", per_example_loss)
        ],
    }


@torch.no_grad()
def evaluate(
    *,
    resolved_config: str,
    checkpoint: str,
    dataset: str,
    device: str,
    batch_size: int | None,
    max_examples: int | None,
    stratify_closing_leaves: bool,
    context_prefix_length: int | None,
) -> dict[str, Any]:
    if resolved_config:
        cfg = copy.deepcopy(_load_resolved_config(resolved_config))
    else:
        cfg = copy.deepcopy(_load_checkpoint_config(checkpoint))
    cfg.resume_from = checkpoint
    cfg.validation_set.enabled = True
    cfg.validation_set.dataset = dataset
    if batch_size is not None:
        cfg.validation_set.batch_size = batch_size
    if max_examples is not None:
        cfg.validation_set.max_examples = max_examples
    cfg.device = device
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.model.compile = "off"
    cfg.data.mode = "live"
    cfg.search.closing_leaf_checkpoint = None
    cfg.train.replay_buffer_device = "cpu"
    cfg.train.replay_buffer_batches = 1

    inferred_context = _infer_context_in_dim(checkpoint)
    if context_prefix_length is None and inferred_context is not None:
        context_prefix_length = inferred_context
    if context_prefix_length is not None:
        original_context_length = cfr_trainer_module.context_length
        original_better_context_length = better_ffn_module.context_length

        def patched_context_length(num_players: int) -> int:
            return int(context_prefix_length)

        cfr_trainer_module.context_length = patched_context_length
        better_ffn_module.context_length = patched_context_length
    else:
        original_context_length = None
        original_better_context_length = None

    torch_device = device_from_config(cfg)
    setup_torch_runtime(cfg, torch_device)
    try:
        trainer = RebelCFRTrainer(cfg=cfg, device=torch_device, pregeneration_only=True)
        _load_model_weights(trainer, checkpoint)
    finally:
        if original_context_length is not None:
            cfr_trainer_module.context_length = original_context_length
            better_ffn_module.context_length = original_better_context_length
    evaluator = RebelValueValidationSetEvaluator(
        trainer=trainer,
        cfg=cfg,
        dataset_path=dataset,
        batch_size=cfg.validation_set.batch_size,
        max_examples=cfg.validation_set.max_examples,
    )
    if context_prefix_length is None:
        metrics = evaluator.evaluate()
    else:
        metrics = _evaluate_with_context_trim(
            trainer=trainer,
            cfg=cfg,
            evaluator=evaluator,
            context_prefix_length=context_prefix_length,
        )
    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "dataset": dataset,
        "device": str(torch_device),
        "batch_size": cfg.validation_set.batch_size,
        "examples": metrics["validation_examples"],
        "value_loss": metrics["validation_value_loss"],
        "element_mean_weighted_square_error": (
            metrics["validation_element_mean_weighted_square_error"]
        ),
        "batch_loss_mean": metrics["validation_batch_loss_mean"],
        "num_batches": metrics["validation_num_batches"],
        "pot_relative_mae": metrics.get("validation_pot_relative_mae", "n/a"),
        "pot_relative_rmse": metrics.get("validation_pot_relative_rmse", "n/a"),
    }
    if stratify_closing_leaves:
        result["closing_leaf_stratification"] = _closing_leaf_stratification(
            trainer=trainer,
            cfg=cfg,
            evaluator=evaluator,
            context_prefix_length=context_prefix_length,
        )
    return result


@torch.no_grad()
def _evaluate_with_context_trim(
    *,
    trainer: RebelCFRTrainer,
    cfg: Config,
    evaluator: RebelValueValidationSetEvaluator,
    context_prefix_length: int,
) -> dict[str, float | int | str]:
    total_examples = evaluator.total_examples
    if cfg.validation_set.max_examples is not None:
        total_examples = min(total_examples, cfg.validation_set.max_examples)
    weighted_loss_sum = 0.0
    weighted_loss_elements = 0
    weight_normalized_denom = 0.0
    batch_loss_sum = 0.0
    num_batches = 0
    for start in range(0, total_examples, cfg.validation_set.batch_size):
        count = min(cfg.validation_set.batch_size, total_examples - start)
        batch = evaluator.dataset.get_batch(
            "value",
            start,
            count,
            device=trainer.device,
            float_dtype=trainer.float_dtype,
        )
        batch.features = _maybe_trim_context(batch.features, context_prefix_length)
        with trainer.model_autocast():
            output = trainer.inference_model.repeat(
                batch.features,
                count=cfg.model.num_supervisions,
                include_policy=False,
                apply_zero_sum=False,
            )
        loss_dict = trainer.loss_fn.forward_value(output, batch)
        value_loss_all = loss_dict["value_loss_all"]
        value_weights = loss_dict["value_weights"]
        weighted_loss_sum += float(value_loss_all.sum().detach().cpu().item())
        weighted_loss_elements += int(value_loss_all.numel())
        weight_normalized_denom += float(value_weights.sum().detach().cpu().item())
        batch_loss_sum += float(loss_dict["value_loss"].detach().cpu().item())
        num_batches += 1
    return {
        "validation_value_loss": weighted_loss_sum / max(weight_normalized_denom, 1e-12),
        "validation_element_mean_weighted_square_error": (
            weighted_loss_sum / max(weighted_loss_elements, 1)
        ),
        "validation_batch_loss_mean": batch_loss_sum / max(num_batches, 1),
        "validation_examples": total_examples,
        "validation_num_batches": num_batches,
        "validation_dataset": evaluator.dataset_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--stratify-closing-leaves", action="store_true")
    parser.add_argument("--context-prefix-length", type=int, default=None)
    args = parser.parse_args()
    result = evaluate(
        resolved_config=args.resolved_config,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        device=args.device,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        stratify_closing_leaves=args.stratify_closing_leaves,
        context_prefix_length=args.context_prefix_length,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
