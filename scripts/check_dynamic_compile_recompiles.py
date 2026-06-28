#!/usr/bin/env python3
"""Exercise compiled model/loss forwards over varied batch sizes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2.cli.train_rebel_preflop_buckets import _execution_config_from_config  # noqa: E402
from p2.config.rebel_load import load_rebel_config  # noqa: E402
from p2.rl.cfr_trainer import RebelCFRTrainer  # noqa: E402
from p2.runtime.training_run import setup_torch_runtime  # noqa: E402
from p2.stages.preflop_backward_induction import (  # noqa: E402
    _load_model_weights,
    _train_policy_minibatches,
    _train_value_minibatches,
    _validation_to_device,
)


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/preflop_backward_induction/"
    "gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5"
)
DEFAULT_VALIDATION = (
    REPO_ROOT
    / "outputs/preflop_backward_induction/validation_cache/actions_8_11/"
    "validation_n4096_cfr10000_6bff1842e2a5f7bd.pt"
)
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "actions_8_11/checkpoints/specialist_inprogress.pt"


def _load_cfg(args: argparse.Namespace):
    overrides = [
        "device=cuda",
        "use_wandb=false",
        "preflop_buckets.command=train_specialists",
        f"preflop_buckets.state_dataset={args.state_dataset}",
        f"preflop_buckets.base_checkpoint={args.base_checkpoint}",
        f"preflop_buckets.output_dir={args.run_dir}",
        "preflop_buckets.train_bucket=actions_8_11",
        "preflop_buckets.write_solved_shards=false",
        "preflop_buckets.compile=default",
    ]
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_preflop_buckets",
            overrides=overrides,
        )
    cfg = load_rebel_config(dc)
    execution = _execution_config_from_config(cfg)
    return cfg, execution


def _slice_batch(batch, size: int):
    if size > len(batch):
        raise ValueError(f"requested batch size {size} from only {len(batch)} rows")
    return batch[slice(0, size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument(
        "--state-dataset",
        type=Path,
        default=REPO_ROOT
        / "outputs/preflop_policy_states/"
        "eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622",
    )
    parser.add_argument(
        "--sizes",
        default="256,17,1024,513,2048,128,999,256,2048,31",
    )
    parser.add_argument("--train-updates", action="store_true")
    parser.add_argument("--actual-minibatches", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    cfg, _execution = _load_cfg(args)
    setup_torch_runtime(cfg, device, recompile_limit=16)
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    _load_model_weights(trainer, str(args.checkpoint), strict=False)
    validation = torch.load(args.validation, map_location="cpu", weights_only=False)
    validation = _validation_to_device(validation, device)
    value_batch = validation["value_batch"]
    policy_batch = validation["policy_batch"]
    if value_batch is None or policy_batch is None:
        raise ValueError("validation cache must contain value and policy batches")

    sizes = [int(item) for item in args.sizes.split(",") if item.strip()]
    print(f"compile={cfg.model.compile!r} sizes={sizes}", flush=True)

    # First calls compile the value/policy model and loss functions.
    for size in sizes:
        part = _slice_batch(value_batch, min(size, len(value_batch)))
        with torch.no_grad(), trainer.model_autocast():
            output = trainer.model(
                part.features,
                include_policy=False,
                include_value=True,
                apply_zero_sum=False,
            )
            metrics = trainer.loss_fn._call_forward_value(output, part)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"value size={len(part)} loss={float(metrics['total_loss']):.6f}", flush=True)

    for size in sizes:
        part = _slice_batch(policy_batch, min(size, len(policy_batch)))
        with torch.no_grad(), trainer.model_autocast():
            output = trainer.model(
                part.features,
                include_policy=True,
                include_value=False,
                apply_zero_sum=False,
            )
            metrics = trainer.loss_fn._call_forward_policy(output, part)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"policy size={len(part)} loss={float(metrics['total_loss']):.6f}", flush=True)

    if args.train_updates:
        for i, size in enumerate(sizes):
            part = _slice_batch(value_batch, min(size, len(value_batch)))
            metrics = trainer.train_value_batch(
                part,
                step=i,
                sync_inference_model=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                f"train_value size={len(part)} loss={float(metrics['total_loss']):.6f}",
                flush=True,
            )

        for i, size in enumerate(sizes):
            part = _slice_batch(policy_batch, min(size, len(policy_batch)))
            metrics = trainer.supervise_policy_batch(
                part,
                step=i,
                apply_schedules=True,
                sync_inference_model=False,
                sync_cfr_target_model=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                f"train_policy size={len(part)} loss={float(metrics['total_loss']):.6f}",
                flush=True,
            )

    if args.actual_minibatches:
        for i, size in enumerate(sizes):
            part = _slice_batch(value_batch, min(size, len(value_batch)))
            metrics, minibatches = _train_value_minibatches(
                trainer,
                part,
                step=i,
                batch_size=trainer.batch_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                f"actual_value stream={len(part)} minibatches={minibatches} "
                f"loss={float(metrics['total_loss']):.6f}",
                flush=True,
            )

        for i, size in enumerate(sizes):
            part = _slice_batch(policy_batch, min(size, len(policy_batch)))
            metrics, minibatches = _train_policy_minibatches(
                trainer,
                part,
                step=i,
                batch_size=trainer.batch_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(
                f"actual_policy stream={len(part)} minibatches={minibatches} "
                f"loss={float(metrics['total_loss']):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
