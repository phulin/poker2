#!/usr/bin/env python3
"""Run controlled 500-step S_turn experiments on fixed pregenerated data."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from p2.config.rebel_load import load_rebel_config_file
from p2.core.structured_config import PregeneratedDatasetConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_loop import run_training_loop
from p2.rl.validation_set import RebelValueValidationSetEvaluator
from p2.stages.curriculum import _initialize_value_from_checkpoint
from run_value_arch_proposal import (
    GpuValueEpoch,
    _benchmark_no_grad_value_inference,
    _dataset_dir,
    _load_gpu_value_epoch,
    _load_manifest,
    _training_step_timing,
)


DEFAULT_DATASET = Path(
    "outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711"
)
DEFAULT_HARD_VALIDATION = Path(
    "outputs/rebel_postflop/"
    "turn_val_4096_5kit_eturn100k_allincutoff_fp32pair_v2_20260707"
)
DEFAULT_INITIALIZATION = Path(
    "checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-wandb-20260708/"
    "t001_lr0p01_300000st_b1024/promoted/E_turn.pt"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/sturn_pregen_500step_sweep_20260711")


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "belief_rank256": {"model": {"belief_low_rank_dim": 256}},
    "belief_linear": {"model": {"belief_linear_encoder": True}},
    "belief_second_moment": {"model": {"belief_second_moment": True}},
    "range_stats": {"model": {"context_range_stats": True}},
    "board_mass": {"model": {"belief_board_mass_features": True}},
    "board_conditioned": {"model": {"belief_low_rank_board_conditioned": True}},
    "cross_range64": {"model": {"cross_range_rank": 64}},
    "turn_blockers": {"model": {"value_turn_range_equity_blockers": True}},
    "turn_equity_feature_head": {
        "model": {"value_turn_range_equity_feature_head": True}
    },
    "turn_equity_input_blockers": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
        }
    },
    "turn_equity_input_board_film": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_board_film": True,
        }
    },
    "turn_equity_input_hand_board_film": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_hand_board_film": True,
        }
    },
    "turn_equity_input_decomposition": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_decomposition_features": True,
        }
    },
    "turn_equity_input_runout_std": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_runout_std_feature": True,
        }
    },
    "turn_equity_input_decomposition_runout_std": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_decomposition_features": True,
            "value_turn_range_equity_runout_std_feature": True,
        }
    },
    "turn_equity_input_blocker_interactions": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_blocker_interactions": True,
        }
    },
    "turn_equity_input_hidden32": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_feature_hidden_dim": 32,
        }
    },
    "turn_equity_input_blocker_interactions_hidden32": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_blocker_interactions": True,
            "value_turn_range_equity_feature_hidden_dim": 32,
        }
    },
    "turn_equity_input_strength_bucket16_relative": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_strength_bucket_count": 16,
            "value_strength_bucket_relative": True,
        }
    },
    "turn_equity_input_strength_bucket32_relative": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_strength_bucket_count": 32,
            "value_strength_bucket_relative": True,
        }
    },
    "turn_equity_input_strength_bucket16_coarse": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_strength_bucket_count": 16,
            "value_strength_bucket_relative": True,
            "value_strength_bucket_coarse_residual": True,
        }
    },
    "turn_equity_input_lr20_cosine": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
        },
        "train": {
            "learning_rate": 0.02,
            "learning_rate_final": 0.002,
            "adamw_learning_rate": 0.02,
        },
    },
    "turn_equity_input_lr80_cosine": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
        },
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "turn_equity_input_lr40_linear": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
        },
        "train": {"lr_schedule": "linear"},
    },
    "turn_equity_input_lr40_wsd": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
        },
        "train": {"lr_schedule": "wsd", "lr_wsd_decay_fraction": 0.4},
    },
    "turn_equity_input_lr40_cosine_warmup100": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
        },
        "train": {"warmup_steps": 100},
    },
    "cheap_turn_lr20_cosine": {
        "train": {
            "learning_rate": 0.02,
            "learning_rate_final": 0.002,
            "adamw_learning_rate": 0.02,
        },
    },
    "cheap_turn_lr10_cosine": {
        "train": {
            "learning_rate": 0.01,
            "learning_rate_final": 0.001,
            "adamw_learning_rate": 0.01,
        },
    },
    "cheap_turn_lr5_cosine": {
        "train": {
            "learning_rate": 0.005,
            "learning_rate_final": 0.0005,
            "adamw_learning_rate": 0.005,
        },
    },
    "cheap_turn_lr2p5_cosine": {
        "train": {
            "learning_rate": 0.0025,
            "learning_rate_final": 0.00025,
            "adamw_learning_rate": 0.0025,
        },
    },
    "cheap_turn_lr4_adamw100": {
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.004,
        },
    },
    "cheap_turn_lr4_adamw30": {
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0012,
        },
    },
    "cheap_turn_lr4_adamw10": {
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
    },
    "lr4_adamw10_turn_blockers": {
        "model": {"value_turn_range_equity_blockers": True},
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
    },
    "lr4_adamw10_second_moment": {
        "model": {"belief_second_moment": True},
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
    },
    "lr4_adamw10_equity_head_blockers": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
    },
    "cheap_turn_lr80_cosine": {
        "train": {
            "learning_rate": 0.08,
            "learning_rate_final": 0.008,
            "adamw_learning_rate": 0.08,
        },
    },
    "cheap_turn_lr40_linear": {"train": {"lr_schedule": "linear"}},
    "cheap_turn_lr40_wsd": {
        "train": {"lr_schedule": "wsd", "lr_wsd_decay_fraction": 0.4}
    },
    "cheap_turn_lr40_cosine_warmup100": {"train": {"warmup_steps": 100}},
    "no_teb_prod_lr2_cosine": {
        "model": {"value_turn_range_equity_baseline": False},
        "train": {
            "learning_rate": 0.002,
            "learning_rate_final": 0.0002,
            "adamw_learning_rate": 0.0002,
        },
    },
    "no_teb_prod_lr4_cosine": {
        "model": {"value_turn_range_equity_baseline": False},
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
    },
    "no_teb_prod_lr8_cosine": {
        "model": {"value_turn_range_equity_baseline": False},
        "train": {
            "learning_rate": 0.008,
            "learning_rate_final": 0.0008,
            "adamw_learning_rate": 0.0008,
        },
    },
    # Output initialization is applied when the model is constructed. These
    # trials intentionally skip E_turn initialization, which would otherwise
    # overwrite the initialized S_turn value-head tensors.
    "no_teb_cold_out0p00": {
        "model": {
            "value_turn_range_equity_baseline": False,
            "value_output_init_scale": 0.0,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
        "initialize_from_checkpoint": False,
    },
    "no_teb_cold_out0p03": {
        "model": {
            "value_turn_range_equity_baseline": False,
            "value_output_init_scale": 0.03,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
        "initialize_from_checkpoint": False,
    },
    "no_teb_cold_out0p10": {
        "model": {
            "value_turn_range_equity_baseline": False,
            "value_output_init_scale": 0.1,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
        "initialize_from_checkpoint": False,
    },
    "no_teb_cold_out0p30": {
        "model": {
            "value_turn_range_equity_baseline": False,
            "value_output_init_scale": 0.3,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
        "initialize_from_checkpoint": False,
    },
    "no_teb_cold_layers6_out0p30": {
        "model": {
            "num_value_layers": 6,
            "value_turn_range_equity_baseline": False,
            "value_output_init_scale": 0.3,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
        "initialize_from_checkpoint": False,
    },
    "no_teb_cold_layers10_out0p30": {
        "model": {
            "num_value_layers": 10,
            "value_turn_range_equity_baseline": False,
            "value_output_init_scale": 0.3,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
        "initialize_from_checkpoint": False,
    },
    "no_teb_cold_layers14_out0p30": {
        "model": {
            "num_value_layers": 14,
            "value_turn_range_equity_baseline": False,
            "value_output_init_scale": 0.3,
        },
        "train": {
            "learning_rate": 0.004,
            "learning_rate_final": 0.0004,
            "adamw_learning_rate": 0.0004,
        },
        "initialize_from_checkpoint": False,
    },
    "turn_equity_input_blockers_second": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "belief_second_moment": True,
        }
    },
    "turn_equity_input_blockers_refit": {
        "model": {
            "value_turn_range_equity_feature_head": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_pos_scale": 1.0593926733198897,
            "value_turn_range_equity_neg_scale": 0.571760860545504,
            "value_turn_range_equity_intercept": -0.008620252614746398,
        }
    },
    "turn_pair_direct": {"environment": {"P2_TURN_EQUITY_PAIR_DIRECT_APPLY": "1"}},
    "second_moment_blockers": {
        "model": {
            "belief_second_moment": True,
            "value_turn_range_equity_blockers": True,
        }
    },
    "turn_blockers_refit": {
        "model": {
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_pos_scale": 1.0593926733198897,
            "value_turn_range_equity_neg_scale": 0.571760860545504,
            "value_turn_range_equity_intercept": -0.008620252614746398,
        }
    },
    "second_moment_blockers_refit": {
        "model": {
            "belief_second_moment": True,
            "value_turn_range_equity_blockers": True,
            "value_turn_range_equity_pos_scale": 1.0593926733198897,
            "value_turn_range_equity_neg_scale": 0.571760860545504,
            "value_turn_range_equity_intercept": -0.008620252614746398,
        }
    },
    "low_entropy_weight3": {"batch_weighting": "low_entropy_weight3"},
    "pot_relative_cap4": {"batch_weighting": "pot_relative_cap4"},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiments",
        nargs="+",
        choices=sorted(EXPERIMENTS),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hard-validation", type=Path, default=DEFAULT_HARD_VALIDATION)
    parser.add_argument("--initialization", type=Path, default=DEFAULT_INITIALIZATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--validation-batch-size", type=int, default=1024)
    parser.add_argument("--matched-validation-examples", type=int, default=32768)
    parser.add_argument("--train-cycle-examples", type=int, default=0)
    parser.add_argument("--timing-warmup-batches", type=int, default=3)
    parser.add_argument("--timing-batches", type=int, default=20)
    return parser.parse_args()


def _build_config(
    args: argparse.Namespace,
    *,
    experiment: str,
    dataset_dir: Path,
):
    cfg = load_rebel_config_file("conf/config_rebel_curriculum_turn.yaml")
    cfg.num_steps = int(args.steps)
    cfg.seed = int(args.seed)
    cfg.use_wandb = False
    cfg.log_interval = int(args.log_interval)
    cfg.checkpoint_interval = int(args.steps)
    cfg.model.compile = str(args.compile_mode)
    cfg.trueskill.enabled = False
    cfg.train.save_replay_buffers = False
    cfg.data.mode = "pregenerated"
    cfg.data.live_root_source = "random_turn"
    cfg.data.pregenerated.value_batch_size = int(args.batch_size)
    cfg.data.pregenerated.policy_batch_size = 0
    cfg.data.pregenerated.shuffle = False
    cfg.data.pregenerated.direct_sample = False
    cfg.data.pregenerated.datasets = [
        PregeneratedDatasetConfig(
            path=str(dataset_dir),
            value_weight=1.0,
            policy_weight=0.0,
        )
    ]
    cfg.curriculum.stages = []
    cfg.curriculum.substeps = {}
    cfg.validation_set.enabled = False

    for key, value in EXPERIMENTS[experiment].get("model", {}).items():
        setattr(cfg.model, key, value)
    for key, value in EXPERIMENTS[experiment].get("train", {}).items():
        current = getattr(cfg.train, key)
        if hasattr(current, "__class__") and hasattr(current.__class__, "__members__"):
            value = current.__class__(value)
        setattr(cfg.train, key, value)

    run_dir = args.output_root / experiment
    cfg.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.wandb_name = f"sturn_pregen_{experiment}_{args.steps}step"
    return cfg, run_dir


def _normalized_belief_entropy(batch: RebelBatch) -> torch.Tensor:
    beliefs = batch.features.beliefs.view(
        len(batch), batch.features.num_players, batch.features.hand_dim
    ).float()
    probabilities = beliefs / beliefs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
    return entropy.mean(dim=-1) / math.log(batch.features.hand_dim)


def _batch_loss_weight(batch: RebelBatch, mode: str | None) -> torch.Tensor | None:
    if mode is None:
        return None
    if mode == "low_entropy_weight3":
        normalized_entropy = _normalized_belief_entropy(batch)
        weights = 1.0 + 2.0 * (normalized_entropy < 0.54).to(torch.float32)
    elif mode == "pot_relative_cap4":
        pot = batch.statistics["pot"].float().clamp_min(1.0)
        scale = batch.statistics["scale"].float().clamp_min(1.0)
        weights = (scale / pot).clamp(max=4.0).square()
    else:
        raise ValueError(f"unknown batch weighting mode {mode!r}")
    return weights / weights.mean().clamp_min(1.0e-8)


def _with_batch_weight(batch: RebelBatch, mode: str | None) -> RebelBatch:
    weights = _batch_loss_weight(batch, mode)
    if weights is None:
        return batch
    return RebelBatch(
        features=batch.features,
        legal_masks=batch.legal_masks,
        policy_targets=batch.policy_targets,
        value_targets=batch.value_targets,
        statistics={**batch.statistics, "value_loss_weight": weights},
    )


def _evaluate(
    trainer: RebelCFRTrainer,
    cfg,
    *,
    dataset: Path,
    batch_size: int,
    max_examples: int | None,
) -> dict[str, float]:
    evaluator = RebelValueValidationSetEvaluator(
        trainer=trainer,
        cfg=cfg,
        dataset_path=str(dataset),
        batch_size=int(batch_size),
        max_examples=max_examples,
    )
    return evaluator.evaluate()


def _run_experiment(
    args: argparse.Namespace,
    *,
    experiment: str,
    dataset_dir: Path,
    manifest: dict[str, Any],
    gpu_epoch: GpuValueEpoch,
) -> dict[str, Any]:
    os.environ.pop("P2_TURN_EQUITY_PAIR_DIRECT_APPLY", None)
    os.environ.update(EXPERIMENTS[experiment].get("environment", {}))
    cfg, run_dir = _build_config(
        args,
        experiment=experiment,
        dataset_dir=dataset_dir,
    )
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = RebelCFRTrainer(cfg, torch.device(cfg.device))
    initialize_from_checkpoint = bool(
        EXPERIMENTS[experiment].get("initialize_from_checkpoint", True)
    )
    initialization_counts = (
        _initialize_value_from_checkpoint(
            trainer,
            str(args.initialization),
            substep_name=f"sturn_pregen_{experiment}",
        )
        if initialize_from_checkpoint
        else None
    )
    weighting = EXPERIMENTS[experiment].get("batch_weighting")
    metadata = {
        "experiment": experiment,
        "settings": EXPERIMENTS[experiment],
        "dataset": str(dataset_dir),
        "dataset_manifest": manifest,
        "initialization": (
            str(args.initialization) if initialize_from_checkpoint else None
        ),
        "initialization_counts": initialization_counts,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "compile_mode": str(args.compile_mode),
        "train_cycle_examples": int(args.train_cycle_examples),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")
    step_times: list[float] = []

    def step_body(step: int) -> dict[str, Any]:
        started = time.time()
        cycle_steps = (
            int(args.train_cycle_examples) // int(args.batch_size)
            if int(args.train_cycle_examples) > 0
            else int(args.steps)
        )
        batch = _with_batch_weight(gpu_epoch.step_batch(step % cycle_steps), weighting)
        metrics = trainer.train_value_batch(batch, step)
        metrics["step_time_s"] = time.time() - started
        step_times.append(metrics["step_time_s"])
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        return metrics

    started = time.time()
    last_step = run_training_loop(
        trainer,
        cfg,
        run=None,
        start_step=0,
        stop_step=cfg.num_steps,
        stage_tag=f"sturn_pregen_{experiment}",
        step_body=step_body,
        checkpoint_metadata=metadata,
        value_only=True,
        print_preflop_analyzer=False,
        log_interval=int(args.log_interval),
    )
    matched_validation = (
        _evaluate(
            trainer,
            cfg,
            dataset=dataset_dir,
            batch_size=args.validation_batch_size,
            max_examples=int(args.matched_validation_examples),
        )
        if int(args.matched_validation_examples) > 0
        else None
    )
    hard_validation = _evaluate(
        trainer,
        cfg,
        dataset=args.hard_validation,
        batch_size=args.validation_batch_size,
        max_examples=None,
    )
    inference_timing = _benchmark_no_grad_value_inference(
        trainer=trainer,
        gpu_epoch=gpu_epoch,
        batch_size=4096,
        warmup_batches=int(args.timing_warmup_batches),
        timed_batches=int(args.timing_batches),
        value_head="post",
    )
    summary = {
        "experiment": experiment,
        "last_step": int(last_step),
        "elapsed_s": time.time() - started,
        "training_step_timing": _training_step_timing(step_times),
        "inference_timing": inference_timing,
        "matched_300cfr_validation": matched_validation,
        "hard_5kcfr_validation": hard_validation,
        "checkpoint_dir": cfg.checkpoint_dir,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = _parse_args()
    dataset_dir = _dataset_dir(args.dataset)
    manifest = _load_manifest(dataset_dir)
    required_examples = int(args.steps) * int(args.batch_size)
    if int(args.train_cycle_examples) > 0:
        cycle_examples = int(args.train_cycle_examples)
        if cycle_examples % int(args.batch_size) != 0:
            raise ValueError("train cycle examples must be divisible by batch size")
        if cycle_examples > int(manifest["value_examples"]):
            raise ValueError("train cycle exceeds the dataset")
    elif required_examples != int(manifest["value_examples"]):
        raise ValueError(
            "S_turn sweep expects one exact epoch: "
            f"steps*batch={required_examples}, "
            f"dataset={manifest['value_examples']}"
        )
    needs_initialization = any(
        bool(EXPERIMENTS[name].get("initialize_from_checkpoint", True))
        for name in args.experiments
    )
    if needs_initialization and not args.initialization.exists():
        raise FileNotFoundError(args.initialization)
    if not args.hard_validation.exists():
        raise FileNotFoundError(args.hard_validation)

    base_cfg = load_rebel_config_file("conf/config_rebel_curriculum_turn.yaml")
    gpu_epoch = _load_gpu_value_epoch(
        dataset_dir=dataset_dir,
        manifest=manifest,
        device=torch.device(base_cfg.device),
        batch_size=int(args.batch_size),
        steps=(
            int(args.train_cycle_examples) // int(args.batch_size)
            if int(args.train_cycle_examples) > 0
            else int(args.steps)
        ),
        shuffle_seed=int(args.seed),
        shuffle=True,
    )
    summaries = []
    for experiment in args.experiments:
        summaries.append(
            _run_experiment(
                args,
                experiment=experiment,
                dataset_dir=dataset_dir,
                manifest=manifest,
                gpu_epoch=gpu_epoch,
            )
        )
        torch.cuda.empty_cache()

    queue_summary = {
        "experiments": list(args.experiments),
        "dataset": str(dataset_dir),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "gpu_epoch_gib": gpu_epoch.tensor_bytes / (1024**3),
        "summaries": summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "queue_summary.json").write_text(
        json.dumps(queue_summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
