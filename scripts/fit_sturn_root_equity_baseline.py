#!/usr/bin/env python3
"""Fit turn-equity baseline coefficients directly to solved S_turn root targets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from diagnose_turn_value_fidelity import (
    MetricAccumulator,
    RegressionAccumulator,
    _stack_design,
    _turn_sdv,
    _value_model,
    _value_weights,
)
from p2.config.rebel_load import load_rebel_config_file
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.rebel_solved_dataset import RebelSolvedDataset


DEFAULT_DATASET = Path(
    "outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711"
)
DEFAULT_OUTPUT = Path(
    "outputs/experiments/sturn_root_turneq_blockers_fit_32768_20260712.json"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rank-bins", type=int, default=144)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--blockers", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


@torch.no_grad()
def fit(args: argparse.Namespace) -> dict:
    cfg = load_rebel_config_file("conf/config_rebel_curriculum_turn.yaml")
    cfg.device = str(args.device)
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.model.compile = "off"
    cfg.search.closing_leaf_checkpoint = None
    cfg.train.replay_buffer_batches = 1
    cfg.train.replay_buffer_device = "cpu"
    device = torch.device(cfg.device)
    trainer = RebelCFRTrainer(cfg=cfg, device=device, pregeneration_only=True)
    value_model = _value_model(trainer.model)
    dataset = RebelSolvedDataset(args.dataset)
    examples = min(int(args.examples), dataset.stream_len("value"))
    regression = RegressionAccumulator.create(
        ("positive_sdv", "negative_sdv", "intercept"), device
    )
    fitted_metrics = MetricAccumulator.create(device)
    started = time.perf_counter()

    batches = []
    for start in range(0, examples, int(args.batch_size)):
        count = min(int(args.batch_size), examples - start)
        batch = dataset.get_batch(
            "value",
            start,
            count,
            device=device,
            float_dtype=torch.float32,
        )
        beliefs = batch.features.beliefs.view(
            count, trainer.num_players, batch.features.hand_dim
        ).float()
        targets = batch.value_targets.float()
        weights = _value_weights(trainer, batch)
        sdv = _turn_sdv(
            value_model,
            beliefs,
            batch,
            rank_bins=int(args.rank_bins),
            chunk_size=int(args.chunk_size),
            blockers=bool(args.blockers),
        )
        pot = batch.statistics["pot"].float()
        scale = batch.statistics["scale"].float()
        regression.update(
            _stack_design(sdv), targets, weights, pot=pot, scale=scale
        )
        batches.append((sdv.cpu(), targets.cpu(), weights.cpu(), pot.cpu(), scale.cpu()))
        print(f"fit rows={start + count}/{examples}", flush=True)

    beta, regression_result = regression.solve(float(args.ridge))
    for sdv_cpu, targets_cpu, weights_cpu, pot_cpu, scale_cpu in batches:
        sdv = sdv_cpu.to(device)
        prediction = (
            sdv.clamp_min(0.0) * beta[0].float()
            + sdv.clamp_max(0.0) * beta[1].float()
            + beta[2].float()
        )
        fitted_metrics.update(
            prediction,
            targets_cpu.to(device),
            weights_cpu.to(device),
            pot=pot_cpu.to(device),
            scale=scale_cpu.to(device),
        )

    result = {
        "dataset": str(args.dataset),
        "examples": examples,
        "batch_size": int(args.batch_size),
        "rank_bins": int(args.rank_bins),
        "chunk_size": int(args.chunk_size),
        "blockers": bool(args.blockers),
        "elapsed_s": time.perf_counter() - started,
        "regression": regression_result,
        "fitted_metrics": fitted_metrics.result(),
        "config_values": {
            "value_turn_range_equity_pos_scale": float(beta[0].item()),
            "value_turn_range_equity_neg_scale": float(beta[1].item()),
            "value_turn_range_equity_intercept": float(beta[2].item()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    dataset.close()
    return result


def main() -> None:
    print(json.dumps(fit(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
