#!/usr/bin/env python3
"""Fit E_turn range-equity baselines against S_river chance targets."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.search.chance_node_helper import ChanceNodeHelper
from p2.search.end_of_street_distillation import build_end_of_street_value_batch
from p2.search.postflop_spot_sampler import sample_end_of_street_chance_roots


DEFAULT_CHECKPOINT = Path(
    "checkpoints-rebel-curriculum-sapcfr-80-40-300it-8000-val-ctx41-live-board96-"
    "belief128-canonical-k32-nobaseline-out0-lr001-random-wandb/promoted/S_river.pt"
)


@dataclass
class RegressionAccumulator:
    names: tuple[str, ...]
    xtx: torch.Tensor
    xty: torch.Tensor
    yty: torch.Tensor
    weight_sum: torch.Tensor
    abs_sum: torch.Tensor
    relative_abs_sum: torch.Tensor
    relative_sq_sum: torch.Tensor

    @classmethod
    def create(cls, names: tuple[str, ...], device: torch.device) -> RegressionAccumulator:
        dim = len(names)
        return cls(
            names=names,
            xtx=torch.zeros(dim, dim, dtype=torch.float64, device=device),
            xty=torch.zeros(dim, dtype=torch.float64, device=device),
            yty=torch.zeros((), dtype=torch.float64, device=device),
            weight_sum=torch.zeros((), dtype=torch.float64, device=device),
            abs_sum=torch.zeros((), dtype=torch.float64, device=device),
            relative_abs_sum=torch.zeros((), dtype=torch.float64, device=device),
            relative_sq_sum=torch.zeros((), dtype=torch.float64, device=device),
        )

    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor,
        *,
        pot: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        x64 = x.reshape(-1, x.shape[-1]).to(torch.float64)
        y64 = y.reshape(-1).to(torch.float64)
        w64 = weight.reshape(-1).to(torch.float64).clamp_min(0.0)
        pot64 = pot[:, None, None].expand_as(y).reshape(-1).to(torch.float64).clamp_min(1.0)
        scale64 = (
            scale[:, None, None].expand_as(y).reshape(-1).to(torch.float64).clamp_min(1.0)
        )
        valid = w64 > 0.0
        if not bool(valid.any().item()):
            return
        x64 = x64[valid]
        y64 = y64[valid]
        w64 = w64[valid]
        pot64 = pot64[valid]
        scale64 = scale64[valid]
        wx = x64 * w64[:, None]
        self.xtx += x64.T @ wx
        self.xty += x64.T @ (y64 * w64)
        self.yty += (y64.square() * w64).sum()
        self.weight_sum += w64.sum()
        self.abs_sum += (y64.abs() * w64).sum()
        relative = y64.abs() * scale64 / pot64
        self.relative_abs_sum += (relative * w64).sum()
        self.relative_sq_sum += (relative.square() * w64).sum()

    def solve(self, ridge: float) -> tuple[torch.Tensor, dict[str, Any]]:
        eye = torch.eye(self.xtx.shape[0], dtype=self.xtx.dtype, device=self.xtx.device)
        beta = torch.linalg.solve(self.xtx + float(ridge) * eye, self.xty)
        sse = (self.yty - beta.dot(self.xty)).clamp_min(0.0)
        mse = sse / self.weight_sum.clamp_min(1e-12)
        return beta, {
            "features": list(self.names),
            "mse": float(mse.cpu().item()),
            "rmse": float(mse.sqrt().cpu().item()),
            "sse": float(sse.cpu().item()),
            "weight_sum": float(self.weight_sum.cpu().item()),
            "target_abs_mean": float(
                (self.abs_sum / self.weight_sum.clamp_min(1e-12)).cpu().item()
            ),
            "target_pot_relative_mae": float(
                (self.relative_abs_sum / self.weight_sum.clamp_min(1e-12)).cpu().item()
            ),
            "target_pot_relative_rmse": float(
                (
                    self.relative_sq_sum / self.weight_sum.clamp_min(1e-12)
                ).sqrt().cpu().item()
            ),
            "coefficients": {
                name: float(value)
                for name, value in zip(self.names, beta.detach().cpu().tolist(), strict=True)
            },
        }


@dataclass
class MetricAccumulator:
    sse: torch.Tensor
    abs_sum: torch.Tensor
    relative_abs_sum: torch.Tensor
    relative_sq_sum: torch.Tensor
    weight_sum: torch.Tensor

    @classmethod
    def create(cls, device: torch.device) -> MetricAccumulator:
        zero = torch.zeros((), dtype=torch.float64, device=device)
        return cls(
            sse=zero.clone(),
            abs_sum=zero.clone(),
            relative_abs_sum=zero.clone(),
            relative_sq_sum=zero.clone(),
            weight_sum=zero.clone(),
        )

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        *,
        pot: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        err = (pred - target).to(torch.float64)
        w64 = weight.to(torch.float64).clamp_min(0.0)
        pot64 = pot[:, None, None].expand_as(target).to(torch.float64).clamp_min(1.0)
        scale64 = scale[:, None, None].expand_as(target).to(torch.float64).clamp_min(1.0)
        self.sse += (err.square() * w64).sum()
        self.abs_sum += (err.abs() * w64).sum()
        rel = err.abs() * scale64 / pot64
        self.relative_abs_sum += (rel * w64).sum()
        self.relative_sq_sum += (rel.square() * w64).sum()
        self.weight_sum += w64.sum()

    def result(self) -> dict[str, float]:
        denom = self.weight_sum.clamp_min(1e-12)
        mse = self.sse / denom
        rel_mse = self.relative_sq_sum / denom
        return {
            "mse": float(mse.cpu().item()),
            "rmse": float(mse.sqrt().cpu().item()),
            "mae": float((self.abs_sum / denom).cpu().item()),
            "pot_relative_mae": float((self.relative_abs_sum / denom).cpu().item()),
            "pot_relative_rmse": float(rel_mse.sqrt().cpu().item()),
            "weight_sum": float(self.weight_sum.cpu().item()),
        }


def _value_model(model: torch.nn.Module) -> torch.nn.Module:
    if type(model) is BetterSplitFFN:
        return model.value_model
    return model


def _load_cfg(args: argparse.Namespace) -> Config:
    conf_dir = Path(__file__).resolve().parents[1] / "conf"
    overrides = [
        "num_steps=1",
        "use_wandb=false",
        "wandb_project=disabled",
        "model.compile=off",
        "data.belief_mode=mixed",
        "data.belief_profile=actions_12_end",
        "model.board_interaction_dim=96",
        "++model.belief_low_rank_dim=128",
        "model.street_value_heads=pre",
        "curriculum.stages=[distill_E_turn]",
        "curriculum.substeps.distill_E_turn.train_overrides.batch_size="
        + str(int(args.batch_size)),
    ]
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        dict_cfg = compose(config_name="config_rebel_curriculum_turn", overrides=overrides)
    cfg = load_rebel_config(dict_cfg)
    cfg.train.batch_size = int(args.batch_size)
    cfg.checkpoint_dir = str(args.scratch_dir)
    return cfg


def _value_weights(trainer: RebelCFRTrainer, batch: RebelBatch) -> torch.Tensor:
    _, allowed_hands_float, unblocked_mass = trainer.loss_fn._base_weights(batch)
    return trainer.loss_fn._value_weights(
        unblocked_mass,
        allowed_hands_float,
        live_mask=trainer.loss_fn._live_player_mask(batch),
    )


def _turn_sdv(
    value_model: torch.nn.Module,
    beliefs: torch.Tensor,
    batch: RebelBatch,
    *,
    rank_bins: int,
    chunk_size: int,
    blockers: bool,
) -> torch.Tensor:
    if not hasattr(value_model, "_turn_range_equity_features"):
        raise TypeError("value model does not expose _turn_range_equity_features")
    previous = {
        "value_turn_range_equity_rank_bins": value_model.value_turn_range_equity_rank_bins,
        "value_turn_range_equity_chunk_size": value_model.value_turn_range_equity_chunk_size,
        "value_turn_range_equity_blockers": value_model.value_turn_range_equity_blockers,
        "value_turn_range_equity_baseline_scale": value_model.value_turn_range_equity_baseline_scale,
        "value_turn_range_equity_pos_scale": value_model.value_turn_range_equity_pos_scale,
        "value_turn_range_equity_neg_scale": value_model.value_turn_range_equity_neg_scale,
        "value_turn_range_equity_intercept": value_model.value_turn_range_equity_intercept,
        "value_turn_range_equity_pot_power": value_model.value_turn_range_equity_pot_power,
    }
    try:
        value_model.value_turn_range_equity_rank_bins = int(rank_bins)
        value_model.value_turn_range_equity_chunk_size = int(chunk_size)
        value_model.value_turn_range_equity_blockers = bool(blockers)
        value_model.value_turn_range_equity_baseline_scale = 1.0
        value_model.value_turn_range_equity_pos_scale = -1.0
        value_model.value_turn_range_equity_neg_scale = -1.0
        value_model.value_turn_range_equity_intercept = 0.0
        value_model.value_turn_range_equity_pot_power = 1.0
        _, feature_values = value_model._turn_range_equity_features(
            beliefs,
            batch.features,
            torch.float32,
        )
    finally:
        for name, value in previous.items():
            setattr(value_model, name, value)
    return feature_values[..., 0].float()


def _stack_design(sdv: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            sdv.clamp_min(0.0),
            sdv.clamp_max(0.0),
            torch.ones_like(sdv),
        ),
        dim=-1,
    )


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    cfg = _load_cfg(args)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"S_river checkpoint does not exist: {checkpoint}")

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    source_model = trainer.load_closing_leaf_model(str(checkpoint))
    target_value_model = _value_model(trainer.model)
    source_model.eval()
    target_value_model.eval()

    chance_helper = ChanceNodeHelper(
        device=device,
        float_dtype=trainer.float_dtype,
        num_players=trainer.num_players,
        model=source_model,
        generator=trainer.rng,
    )

    regression = RegressionAccumulator.create(
        ("positive_sdv", "negative_sdv", "intercept"),
        device,
    )
    metrics = {
        "zero": MetricAccumulator.create(device),
        "sdv_1p0": MetricAccumulator.create(device),
        "sdv_0p65": MetricAccumulator.create(device),
    }
    examples_seen = 0
    t0 = time.perf_counter()

    for start in range(0, int(args.examples), int(args.batch_size)):
        count = min(int(args.batch_size), int(args.examples) - start)
        if count <= 0:
            break
        sample = sample_end_of_street_chance_roots(
            trainer.env,
            batch_size=count,
            closed_street=2,
            generator=trainer.rng,
            compact_preflop_beliefs=False,
            belief_mode=str(args.belief_mode),
            belief_profile=str(args.belief_profile),
        )
        encoder = target_value_model.create_feature_encoder(
            env=sample.pbs.env,
            device=device,
            dtype=trainer.float_dtype,
        )
        batch = build_end_of_street_value_batch(
            sample,
            value_encoder=encoder,
            target_model=source_model,
            chance_helper=chance_helper,
            chance="single_card",
            float_dtype=trainer.float_dtype,
            generator=trainer.rng,
        )
        if batch.value_targets is None:
            raise RuntimeError("distillation batch unexpectedly lacks value targets")
        if int(batch.features.hand_dim) != NUM_HANDS:
            raise ValueError(
                f"expected combo hand_dim={NUM_HANDS}, got {batch.features.hand_dim}"
            )
        if not bool((batch.features.street == 2).all().item()):
            raise ValueError("diagnostic expects turn pre-chance features")

        beliefs = batch.features.beliefs.view(count, trainer.num_players, NUM_HANDS).float()
        targets = batch.value_targets.float()
        weights = _value_weights(trainer, batch)
        sdv = _turn_sdv(
            target_value_model,
            beliefs,
            batch,
            rank_bins=int(args.rank_bins),
            chunk_size=int(args.chunk_size),
            blockers=bool(args.blockers),
        )
        x = _stack_design(sdv)
        pot = batch.statistics["pot"].float()
        scale = batch.statistics["scale"].float()
        regression.update(x, targets, weights, pot=pot, scale=scale)
        metrics["zero"].update(torch.zeros_like(targets), targets, weights, pot=pot, scale=scale)
        metrics["sdv_1p0"].update(sdv, targets, weights, pot=pot, scale=scale)
        metrics["sdv_0p65"].update(sdv * 0.65, targets, weights, pot=pot, scale=scale)
        examples_seen += count
        if args.progress:
            elapsed = time.perf_counter() - t0
            print(
                f"processed {examples_seen}/{args.examples} examples "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    beta, regression_result = regression.solve(float(args.ridge))
    regression_result["config_values"] = {
        "value_turn_range_equity_pos_scale": regression_result["coefficients"][
            "positive_sdv"
        ],
        "value_turn_range_equity_neg_scale": regression_result["coefficients"][
            "negative_sdv"
        ],
        "value_turn_range_equity_intercept": regression_result["coefficients"][
            "intercept"
        ],
    }
    return {
        "checkpoint": str(checkpoint),
        "examples": examples_seen,
        "batch_size": int(args.batch_size),
        "device": str(device),
        "rank_bins": int(args.rank_bins),
        "chunk_size": int(args.chunk_size),
        "blockers": bool(args.blockers),
        "belief_mode": str(args.belief_mode),
        "belief_profile": str(args.belief_profile),
        "elapsed_s": time.perf_counter() - t0,
        "regression": regression_result,
        "fixed_baselines": {name: acc.result() for name, acc in metrics.items()},
        "raw_coefficients": beta.detach().cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--examples", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--rank-bins", type=int, default=144)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--belief-mode", default="mixed")
    parser.add_argument("--belief-profile", default="actions_12_end")
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp/p2_turn_regression"))
    parser.add_argument("--no-blockers", dest="blockers", action="store_false")
    parser.set_defaults(blockers=True)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = diagnose(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
