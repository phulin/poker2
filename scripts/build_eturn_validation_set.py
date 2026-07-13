#!/usr/bin/env python3
"""Build a fixed E_turn validation set from S_river chance expectations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.chance_node_helper import ChanceNodeHelper
from p2.search.end_of_street_distillation import build_end_of_street_value_batch
from p2.search.postflop_spot_sampler import sample_end_of_street_chance_roots
from p2.search.rebel_solved_dataset import RebelSolvedDatasetWriter


DEFAULT_SOURCE_CHECKPOINT = Path(
    "checkpoints-rebel-curriculum-sapcfr-80-40-300it-8000-val-ctx41-live-board96-"
    "belief128-canonical-k32-nobaseline-out0-lr001-random-wandb/promoted/S_river.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/rebel_postflop/eturn_val_16384_current_teb_sriver8000_20260708"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "trueskill.enabled=false",
        "model.compile=off",
        "data.belief_mode=" + str(args.belief_mode),
        "data.belief_profile=" + str(args.belief_profile),
        "data.live_root_source=random_turn",
        "model.street_value_heads=pre",
        "curriculum.stages=[distill_E_turn]",
        "curriculum.substeps.distill_E_turn.train_overrides.batch_size="
        + str(int(args.batch_size)),
    ]
    if args.value_output_init_scale is not None:
        overrides.append(f"model.value_output_init_scale={args.value_output_init_scale}")
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        dict_cfg = compose(config_name="config_rebel_curriculum_turn", overrides=overrides)
    cfg = load_rebel_config(dict_cfg)
    cfg.seed = int(args.seed)
    cfg.train.batch_size = int(args.batch_size)
    cfg.checkpoint_dir = str(args.scratch_dir)
    return cfg


def _target_model_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "role": "chance_expectation_source",
        "net": "S_river",
        "checkpoint": str(path),
        "sha256": _sha256_file(path),
        "chance": "single_card",
        "chance_outputs_per_example": 48,
    }
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_metadata = checkpoint.get("metadata")
    if isinstance(ckpt_metadata, dict):
        metadata["checkpoint_metadata"] = {
            key: value
            for key, value in ckpt_metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    return metadata


@torch.no_grad()
def build_validation_set(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.source_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"S_river checkpoint does not exist: {checkpoint}")
    output_dir = Path(args.output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"{output_dir / 'manifest.json'} already exists")

    device = torch.device(args.device)
    cfg = _load_cfg(args)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(args.seed))

    print(f"Using device: {device}", flush=True)
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    source_model = trainer.load_closing_leaf_model(str(checkpoint))
    value_model = _value_model(trainer.model)
    source_model.eval()
    value_model.eval()

    chance_helper = ChanceNodeHelper(
        device=device,
        float_dtype=trainer.float_dtype,
        num_players=trainer.num_players,
        model=source_model,
        generator=trainer.rng,
    )
    writer = RebelSolvedDatasetWriter(
        output_dir,
        storage_float_dtype=args.storage_dtype,
    )

    examples_seen = 0
    t0 = time.perf_counter()
    target_examples = int(args.examples)
    batch_size = int(args.batch_size)
    while examples_seen < target_examples:
        count = min(batch_size, target_examples - examples_seen)
        sample = sample_end_of_street_chance_roots(
            trainer.env,
            batch_size=count,
            closed_street=2,
            generator=trainer.rng,
            compact_preflop_beliefs=False,
            belief_mode=str(args.belief_mode),
            belief_profile=str(args.belief_profile),
        )
        encoder = value_model.create_feature_encoder(
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
            raise RuntimeError("validation batch unexpectedly lacks value targets")
        if int(batch.features.hand_dim) != NUM_HANDS:
            raise ValueError(
                f"expected combo hand_dim={NUM_HANDS}, got {batch.features.hand_dim}"
            )
        if not bool((batch.features.street == 2).all().item()):
            raise ValueError("E_turn validation features must be turn pre-chance rows")
        writer.append("value", batch)
        examples_seen += len(batch)
        if args.progress:
            elapsed = time.perf_counter() - t0
            print(
                f"Generated value={examples_seen}/{target_examples} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    manifest = writer.finalize(
        metadata={
            "stage": "E_turn_validation",
            "root_source": "end_of_turn_chance_roots",
            "root_streets": ["turn"],
            "model_family": cfg.model.name.value,
            "action_schedule": {
                "bet_bins": list(cfg.env.bet_bins),
                "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
                "allin_by_depth": cfg.search.allin_by_depth,
            },
            "generator": {
                "seed": int(args.seed),
                "device": str(device),
                "examples": examples_seen,
                "batch_size": batch_size,
                "elapsed_s": time.perf_counter() - t0,
            },
            "target_model": _target_model_metadata(checkpoint),
            "quality": {
                "target_construction": "mean over legal single-card river chance outcomes",
                "chance_outputs_per_example": 48,
            },
            "model_config": asdict(cfg.model),
            "env_config": asdict(cfg.env),
            "search_config": asdict(cfg.search),
            "spot_sampler_config": {
                "closed_street": 2,
                "chance": "single_card",
                "belief_mode": str(args.belief_mode),
                "belief_profile": str(args.belief_profile),
            },
        },
    )
    print(
        f"wrote {manifest['value_examples']} E_turn validation examples to {output_dir}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--examples", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--belief-mode", default="mixed")
    parser.add_argument("--belief-profile", default="actions_12_end")
    parser.add_argument("--storage-dtype", default=None)
    parser.add_argument("--value-output-init-scale", type=float, default=0.1)
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp/p2_eturn_validation"))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    manifest = build_validation_set(args)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
