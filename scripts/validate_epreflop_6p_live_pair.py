#!/usr/bin/env python3
"""Build and evaluate fixed 6-player E_preflop live-pair validation sets."""

from __future__ import annotations

import argparse
import json
import time
import copy
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config, ModelType, PreflopModelType
from p2.env.card_utils import PREFLOP_HANDS
from p2.env.pbs_env import PBSEnv
from p2.rl.checkpoint_io import CheckpointIO
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.search.preflop_live_pair_distillation import (
    uniform_live_pair_value_targets,
)
from p2.stages.preflop_backward_induction import (
    STATE_FIELDS,
    _copy_public_states_to_env,
    _random_beliefs,
)
from distill_epreflop_6p_live_pair import (
    DEFAULT_TARGET_CHECKPOINT,
    StreetClosedStateReader,
    _make_env_from_manifest,
    _value_model,
)


DEFAULT_STATE_DATASET = (
    "outputs/preflop_street_closed_states/live_pair_all_shuffled_seed20260628"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--state-dataset", default=DEFAULT_STATE_DATASET)
    build.add_argument("--target-checkpoint", default=DEFAULT_TARGET_CHECKPOINT)
    build.add_argument(
        "--output",
        default="outputs/epreflop_6p_live_pair/validation/live_pair_val_32768.pt",
    )
    build.add_argument("--examples", type=int, default=32_768)
    build.add_argument("--batch-size", type=int, default=1024)
    build.add_argument("--device", default="cuda")
    build.add_argument("--seed", type=int, default=2026062701)
    build.add_argument(
        "--belief-mode",
        choices=("random", "uniform", "histogram", "mixed", "coverage", "topk"),
        default="mixed",
    )
    build.add_argument("--belief-profile", default="actions_12_end")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument(
        "--validation",
        default="outputs/epreflop_6p_live_pair/validation/live_pair_val_32768.pt",
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--batch-size", type=int, default=1024)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--compile", default="off")
    evaluate.add_argument("--output-json", default=None)
    evaluate.add_argument("--wandb", action="store_true")
    evaluate.add_argument("--wandb-project", default="poker-rebel-epreflop-6p-distill")
    evaluate.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def _make_env_for_validation(
    manifest: dict[str, Any],
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
) -> PBSEnv:
    return _make_env_from_manifest(
        manifest,
        num_envs=num_envs,
        device=device,
        seed=seed,
    )


def _cat_states(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        name: torch.cat([chunk[name] for chunk in chunks], dim=0)
        for name in STATE_FIELDS
    }


@torch.no_grad()
def build_validation(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    reader = StreetClosedStateReader(args.state_dataset, seed=int(args.seed))
    target_model = _load_target_model_for_build(args, reader, device)
    env = _make_env_for_validation(
        reader.manifest,
        num_envs=int(args.batch_size),
        device=device,
        seed=int(args.seed) + 17,
    )
    belief_rng = torch.Generator(device=device)
    belief_rng.manual_seed(int(args.seed) + 31)

    state_chunks: list[dict[str, torch.Tensor]] = []
    belief_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    rows_seen = 0
    for states in reader.iter_batches(
        batch_size=int(args.batch_size),
        max_rows=int(args.examples),
        epoch=0,
    ):
        rows = _copy_public_states_to_env(env, states)
        beliefs = _random_beliefs(
            rows,
            6,
            device=device,
            rng=belief_rng,
            mode=str(args.belief_mode),
            profile=str(args.belief_profile),
        )
        targets = uniform_live_pair_value_targets(
            env,
            beliefs,
            target_model=target_model,
            target_dtype=torch.float32,
        )
        state_chunks.append({name: value[:rows].cpu() for name, value in states.items()})
        belief_chunks.append(beliefs[:rows].cpu())
        target_chunks.append(targets[:rows].cpu())
        rows_seen += rows
        if rows_seen >= int(args.examples):
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "epreflop_6p_live_pair_validation",
        "version": 1,
        "state_dataset": str(Path(args.state_dataset).resolve()),
        "target_checkpoint": str(Path(args.target_checkpoint).resolve()),
        "examples": rows_seen,
        "seed": int(args.seed),
        "belief_mode": str(args.belief_mode),
        "belief_profile": str(args.belief_profile),
        "source_manifest": reader.manifest,
        "states": _cat_states(state_chunks),
        "beliefs": torch.cat(belief_chunks, dim=0)[:rows_seen],
        "value_targets": torch.cat(target_chunks, dim=0)[:rows_seen],
    }
    torch.save(payload, output)
    print(f"wrote validation set: {output} rows={rows_seen:,}", flush=True)
    return output


def _load_target_model_for_build(
    args: argparse.Namespace,
    reader: StreetClosedStateReader,
    device: torch.device,
) -> torch.nn.Module:
    cfg = Config()
    cfg.device = str(device)
    cfg.num_envs = int(args.batch_size)
    cfg.use_wandb = False
    cfg.model.name = ModelType.better_ffn
    cfg.model.preflop_hand_dim = PREFLOP_HANDS
    cfg.model.preflop_model_type = PreflopModelType.gated_token_mixer
    cfg.model.enforce_zero_sum = False
    cfg.env.num_players = 6
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.train.replay_buffer_batches = 1
    cfg.train.replay_buffer_device = "cpu"
    trainer = RebelCFRTrainer(
        cfg=cfg,
        device=device,
        pregeneration_only=True,
    )
    return trainer.load_closing_leaf_model(args.target_checkpoint)


def _config_from_checkpoint(path: str, *, device: torch.device, batch_size: int) -> Config:
    checkpoint = CheckpointIO.load(path, map_location=torch.device("cpu"))
    raw_cfg = checkpoint.get("config")
    if isinstance(raw_cfg, Config):
        cfg = copy.deepcopy(raw_cfg)
    elif isinstance(raw_cfg, dict):
        cfg = Config.from_dict(copy.deepcopy(raw_cfg))
    else:
        raise ValueError(f"{path} does not contain a trainer config")
    cfg.device = str(device)
    cfg.num_envs = int(batch_size)
    cfg.train.batch_size = int(batch_size)
    cfg.use_wandb = False
    cfg.model.compile = "off"
    cfg.data.warmup_self_play_roots = False
    cfg.train.save_replay_buffers = False
    cfg.train.replay_buffer_batches = 1
    cfg.train.replay_buffer_device = "cpu"
    return cfg


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    payload = torch.load(args.validation, map_location="cpu", weights_only=False)
    rows = int(payload["examples"])
    cfg = _config_from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=int(args.batch_size),
    )
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    CheckpointIO.load_model_weights(
        trainer,
        args.checkpoint,
        strict=cfg.strict_model_loading,
        sync_models=False,
    )
    trainer.model.eval()
    value_model = _value_model(trainer.model)

    env = _make_env_for_validation(
        payload["source_manifest"],
        num_envs=int(args.batch_size),
        device=device,
        seed=2026062702,
    )
    total_loss = 0.0
    total_weight = 0.0
    start_time = time.time()
    for start in range(0, rows, int(args.batch_size)):
        end = min(start + int(args.batch_size), rows)
        states = {
            name: payload["states"][name][start:end]
            for name in STATE_FIELDS
        }
        count = _copy_public_states_to_env(env, states)
        beliefs = payload["beliefs"][start:end].to(device=device)
        targets = payload["value_targets"][start:end].to(device=device)
        encoder = value_model.create_feature_encoder(
            env=env,
            device=device,
            dtype=trainer.float_dtype,
        )
        features = encoder.encode(beliefs, pre_chance_node=True)
        stats = {
            "street": torch.zeros(count, dtype=torch.long, device=device),
            "has_folded": env.has_folded[:count].clone(),
            "is_allin": env.is_allin[:count].clone(),
        }
        batch = RebelBatch(
            features=features,
            legal_masks=env.legal_bins_mask()[:count],
            value_targets=targets,
            statistics=stats,
        )
        with trainer.model_autocast():
            output = trainer.model(
                batch.features,
                include_policy=False,
                apply_zero_sum=False,
            )
        loss_dict = trainer.loss_fn._call_forward_value(output, batch)
        value_weights = loss_dict["value_weights"]
        value_loss_all = loss_dict["value_loss_all"]
        total_loss += float(value_loss_all.sum().detach().cpu().item())
        total_weight += float(value_weights.sum().detach().cpu().item())

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "validation": str(Path(args.validation).resolve()),
        "examples": rows,
        "batch_size": int(args.batch_size),
        "validation_value_loss": total_loss / max(total_weight, 1.0e-12),
        "elapsed_s": time.time() - start_time,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    if bool(getattr(args, "wandb", False)):
        import wandb

        run = wandb.init(
            project=str(args.wandb_project),
            name=args.wandb_name,
            config=result,
        )
        run.log(
            {
                "validation/value_loss": result["validation_value_loss"],
                "validation/examples": result["examples"],
                "validation/elapsed_s": result["elapsed_s"],
            },
            step=0,
        )
        run.summary.update(
            {
                "validation_value_loss": result["validation_value_loss"],
                "examples": result["examples"],
            }
        )
        run.finish()
    return result


def main() -> None:
    args = _parse_args()
    if args.command == "build":
        build_validation(args)
    elif args.command == "evaluate":
        evaluate_checkpoint(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
