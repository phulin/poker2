#!/usr/bin/env python3
"""Distill a 6-player E_preflop value model from uniform live-pair projections."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import torch

from p2.config.rebel_schema import RebelExperimentConfig
from p2.core.structured_config import Config, LrSchedule, ModelType, PreflopModelType
from p2.env.card_utils import PREFLOP_HANDS
from p2.env.pbs_env import PBSEnv
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.rl.checkpoint_io import CheckpointIO
from p2.rl.cfr_trainer import (
    RebelCFRTrainer,
    _minimal_frozen_model_config_dict,
    _pot_relative_value_error_metrics,
)
from p2.rl.rebel_batch import RebelBatch
from p2.search.preflop_live_pair_distillation import (
    build_preflop_live_pair_value_batch,
)
from p2.runtime.training_run import (
    setup_torch_runtime,
    wandb_run,
    write_resolved_config,
)
from p2.stages.preflop_backward_induction import (
    STATE_FIELDS,
    _checkpoint_shapes_match_model,
    _copy_public_states_to_env,
    _load_model_weights,
    _random_beliefs,
)


DEFAULT_STATE_DATASET = "outputs/preflop_street_closed_states/live_pair_5m"
DEFAULT_TARGET_CHECKPOINT = (
    "checkpoints-epreflop-distill-100k-lr2e4-sample256-from-sflop2000-169/"
    "promoted/E_preflop.pt"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 6-player compact E_preflop value model on preflop "
            "street-closed states. Targets enumerate uniform live heads-up "
            "pairs and evaluate each pair with a frozen 2-player E_preflop."
        )
    )
    parser.add_argument("--state-dataset", default=DEFAULT_STATE_DATASET)
    parser.add_argument("--target-checkpoint", default=DEFAULT_TARGET_CHECKPOINT)
    parser.add_argument("--output-dir", default="outputs/epreflop_6p_live_pair")
    parser.add_argument("--student-init", default=None)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=0,
        help=(
            "Optimizer schedule horizon. Defaults to --steps; use this to stop "
            "early while preserving the schedule from a longer reference run."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument(
        "--belief-mode",
        choices=("random", "uniform", "histogram", "mixed", "coverage", "topk"),
        default="mixed",
    )
    parser.add_argument("--belief-profile", default="actions_12_end")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--validation-output-json", default=None)
    parser.add_argument(
        "--throughput-warmup-steps",
        type=int,
        default=0,
        help=(
            "Exclude this many initial optimizer steps from throughput timing. "
            "When positive, timed steps synchronize CUDA before and after each step."
        ),
    )
    parser.add_argument("--compile", default="off")
    parser.add_argument("--recompile-limit", type=int, default=16)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="poker-rebel-epreflop-6p-distill")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--preflop-model-type",
        choices=("ffn", "transformer", "gated_token_mixer"),
        default="gated_token_mixer",
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--range-hidden-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=0)
    parser.add_argument("--num-policy-layers", type=int, default=4)
    parser.add_argument("--num-value-layers", type=int, default=5)
    parser.add_argument("--policy-rank", type=int, default=96)
    parser.add_argument("--policy-hand-bias-rank", type=int, default=24)
    parser.add_argument("--preflop-transformer-heads", type=int, default=8)
    parser.add_argument("--preflop-range-slot-moment-slots", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--learning-rate-final", type=float, default=2.0e-5)
    parser.add_argument("--lr-schedule", choices=("cosine", "linear", "wsd"), default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-wsd-decay-fraction", type=float, default=0.2)
    parser.add_argument("--optimizer", choices=("adamw", "muon", "normuon"), default="adamw")
    parser.add_argument(
        "--adamw-learning-rate",
        type=float,
        default=None,
        help=(
            "Learning rate for AdamW parameter groups when using muon/normuon. "
            "Defaults to --learning-rate."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--permutation-coef", type=float, default=0.1)
    return parser.parse_args()


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"No manifest.json found under {root}")
    import json

    return json.loads(path.read_text())


def _state_bucket(manifest: dict[str, Any]) -> dict[str, Any]:
    for item in manifest.get("buckets", []):
        if item.get("label") == "preflop_street_closed":
            return dict(item)
    buckets = list(manifest.get("buckets", []))
    if len(buckets) == 1:
        return dict(buckets[0])
    raise ValueError("state dataset must contain a preflop_street_closed bucket")


class StreetClosedStateReader:
    def __init__(self, root: str | Path, *, seed: int) -> None:
        self.root = Path(root)
        self.manifest = _load_manifest(self.root)
        self.bucket = _state_bucket(self.manifest)
        self.rows = int(self.bucket.get("num_rows", self.manifest.get("num_rows", 0)))
        self.shards = list(self.bucket.get("shards", []))
        if not self.shards:
            raise ValueError(f"No state shards listed in {self.root / 'manifest.json'}")
        self.seed = int(seed)

    def iter_batches(
        self,
        *,
        batch_size: int,
        max_rows: int,
        epoch: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        target_rows = self.rows if max_rows <= 0 else min(int(max_rows), self.rows)
        if target_rows <= 0:
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 1009 * int(epoch))
        shard_order = torch.randperm(len(self.shards), generator=generator).tolist()
        consumed = 0
        pending: list[dict[str, torch.Tensor]] = []
        pending_rows = 0
        pad_pool: dict[str, torch.Tensor] | None = None

        def rows_of(batch: dict[str, torch.Tensor]) -> int:
            return int(next(iter(batch.values())).shape[0])

        def concat_batches(
            batches: list[dict[str, torch.Tensor]],
        ) -> dict[str, torch.Tensor]:
            if len(batches) == 1:
                return batches[0]
            return {
                name: torch.cat([batch[name] for batch in batches], dim=0)
                for name in batches[0]
            }

        def take_pending(take_rows: int) -> dict[str, torch.Tensor]:
            nonlocal pending, pending_rows
            combined = concat_batches(pending)
            out = {name: tensor[:take_rows] for name, tensor in combined.items()}
            tail_rows = rows_of(combined) - take_rows
            pending = (
                [{name: tensor[take_rows:] for name, tensor in combined.items()}]
                if tail_rows > 0
                else []
            )
            pending_rows = tail_rows
            return out

        def pad_to_full(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            nonlocal pad_pool
            rows = rows_of(batch)
            if rows == int(batch_size):
                return batch
            source = pad_pool if pad_pool is not None else batch
            source_rows = rows_of(source)
            pad_rows = int(batch_size) - rows
            pad_indices = torch.arange(pad_rows, dtype=torch.long) % source_rows
            padded = {
                name: torch.cat(
                    [tensor, source[name].index_select(0, pad_indices)],
                    dim=0,
                )
                for name, tensor in batch.items()
            }
            return padded

        for shard_idx in shard_order:
            if consumed >= target_rows:
                break
            shard = self.shards[int(shard_idx)]
            path = self.root / "preflop_street_closed" / str(shard["path"])
            payload = torch.load(path, map_location="cpu", weights_only=True)
            states = payload["states"]
            rows = int(payload.get("num_rows", next(iter(states.values())).shape[0]))
            perm = torch.randperm(rows, generator=generator)
            for start in range(0, rows, int(batch_size)):
                if consumed >= target_rows:
                    break
                take = min(int(batch_size), rows - start, target_rows - consumed)
                indices = perm[start : start + take]
                batch = {
                    name: tensor.index_select(0, indices)
                    for name, tensor in states.items()
                }
                consumed += take
                if pending_rows == 0 and take == int(batch_size):
                    if pad_pool is None:
                        pad_pool = batch
                    yield batch
                    continue
                pending.append(batch)
                pending_rows += take
                while pending_rows >= int(batch_size):
                    out = take_pending(int(batch_size))
                    if pad_pool is None:
                        pad_pool = out
                    yield out
        if pending_rows > 0:
            yield pad_to_full(take_pending(pending_rows))


def _make_env_from_manifest(
    manifest: dict[str, Any],
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
) -> PBSEnv:
    env_cfg = dict(manifest.get("env_config", {}))
    model = dict(manifest.get("model", {}))
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    env = PBSEnv(
        num_envs=num_envs,
        num_players=int(model.get("num_players", env_cfg.get("num_players", 6))),
        mean_stack=int(env_cfg.get("stack", 10000)),
        sb=int(env_cfg.get("sb", 50)),
        bb=int(env_cfg.get("bb", 100)),
        default_bet_bins=list(env_cfg.get("bet_bins", [0.5, 0.75, 1.0, 1.5, 2.0])),
        device=device,
        rng=rng,
        float_dtype=torch.float32,
        stack_mode=str(env_cfg.get("stack_mode", "fixed")),
        min_stack_bb=int(env_cfg.get("min_stack_bb", 10)),
        mid_stack_bb=int(env_cfg.get("mid_stack_bb", 200)),
        max_stack_bb=int(env_cfg.get("max_stack_bb", 400)),
        high_stack_mass_ratio=float(env_cfg.get("high_stack_mass_ratio", 1.0 / 3.0)),
        force_heads_up_preflop_flop=False,
    )
    env.reset()
    return env


def _cfg_from_checkpoint(path: str) -> Config:
    checkpoint = CheckpointIO.load(path, map_location=torch.device("cpu"))
    raw_cfg = checkpoint.get("config")
    if isinstance(raw_cfg, Config):
        return copy.deepcopy(raw_cfg)
    if isinstance(raw_cfg, dict):
        return Config.from_dict(_minimal_frozen_model_config_dict(copy.deepcopy(raw_cfg)))
    return Config()


def _student_config(args: argparse.Namespace, reader: StreetClosedStateReader) -> Config:
    cfg = _cfg_from_checkpoint(args.target_checkpoint)
    env_cfg = dict(reader.manifest.get("env_config", {}))
    model = dict(reader.manifest.get("model", {}))
    cfg.device = str(args.device)
    cfg.seed = int(args.seed)
    cfg.num_envs = int(args.batch_size)
    train_steps = int(args.steps) if int(args.steps) > 0 else max(
        1,
        math.ceil(
            (reader.rows if int(args.max_rows) <= 0 else min(reader.rows, int(args.max_rows)))
            / max(1, int(args.batch_size))
        ),
    )
    cfg.num_steps = int(args.schedule_steps) if int(args.schedule_steps) > 0 else train_steps
    cfg.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    cfg.checkpoint_interval = int(args.checkpoint_interval)
    cfg.use_wandb = bool(args.wandb)
    cfg.wandb_project = str(args.wandb_project)
    cfg.wandb_name = args.wandb_name
    cfg.train.batch_size = int(args.batch_size)
    cfg.train.save_replay_buffers = False
    cfg.train.replay_buffer_batches = 1
    cfg.train.replay_buffer_device = "cpu"
    cfg.train.learning_rate = float(args.learning_rate)
    cfg.train.learning_rate_final = float(args.learning_rate_final)
    cfg.train.lr_schedule = LrSchedule(str(args.lr_schedule))
    cfg.train.warmup_steps = int(args.warmup_steps)
    cfg.train.lr_wsd_decay_fraction = float(args.lr_wsd_decay_fraction)
    cfg.train.optimizer = str(args.optimizer)
    cfg.train.adamw_learning_rate = (
        float(args.learning_rate)
        if args.adamw_learning_rate is None
        else float(args.adamw_learning_rate)
    )
    cfg.train.value_head_learning_rate = float(args.learning_rate)
    cfg.train.value_head_learning_rate_final = float(args.learning_rate_final)
    cfg.train.policy_trunk_learning_rate = float(args.learning_rate)
    cfg.train.policy_trunk_learning_rate_final = float(args.learning_rate_final)
    cfg.train.weight_decay = float(args.weight_decay)
    cfg.train.grad_clip = float(args.grad_clip)
    cfg.train.permutation_coef = float(args.permutation_coef)
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.model.name = ModelType.better_ffn
    cfg.model.compile = str(args.compile)
    cfg.model.preflop_hand_dim = PREFLOP_HANDS
    cfg.model.preflop_model_type = PreflopModelType(str(args.preflop_model_type))
    cfg.model.hidden_dim = int(args.hidden_dim)
    cfg.model.range_hidden_dim = int(args.range_hidden_dim)
    cfg.model.ffn_dim = int(args.ffn_dim)
    cfg.model.num_hidden_layers = int(args.num_hidden_layers)
    cfg.model.num_policy_layers = int(args.num_policy_layers)
    cfg.model.num_value_layers = int(args.num_value_layers)
    cfg.model.policy_rank = int(args.policy_rank)
    cfg.model.policy_hand_bias_rank = int(args.policy_hand_bias_rank)
    cfg.model.preflop_transformer_heads = int(args.preflop_transformer_heads)
    cfg.model.preflop_range_slot_moment_slots = int(
        args.preflop_range_slot_moment_slots
    )
    cfg.model.enforce_zero_sum = False
    if str(args.device) != "cuda":
        cfg.search.sparse_fused = False
    cfg.env.num_players = int(model.get("num_players", env_cfg.get("num_players", 6)))
    cfg.env.stack = int(env_cfg.get("stack", cfg.env.stack))
    cfg.env.sb = int(env_cfg.get("sb", cfg.env.sb))
    cfg.env.bb = int(env_cfg.get("bb", cfg.env.bb))
    cfg.env.bet_bins = list(env_cfg.get("bet_bins", cfg.env.bet_bins))
    cfg.env.randomize_stacks = bool(env_cfg.get("randomize_stacks", False))
    cfg.env.stack_mode = str(env_cfg.get("stack_mode", cfg.env.stack_mode))
    cfg.env.min_stack_bb = int(env_cfg.get("min_stack_bb", cfg.env.min_stack_bb))
    cfg.env.mid_stack_bb = int(env_cfg.get("mid_stack_bb", cfg.env.mid_stack_bb))
    cfg.env.max_stack_bb = int(env_cfg.get("max_stack_bb", cfg.env.max_stack_bb))
    cfg.env.high_stack_mass_ratio = float(
        env_cfg.get("high_stack_mass_ratio", cfg.env.high_stack_mass_ratio)
    )
    return cfg


def _value_model(model: torch.nn.Module) -> torch.nn.Module:
    if type(model) is BetterSplitFFN:
        return model.value_model
    return model


def _maybe_load_student_init(trainer: RebelCFRTrainer, path: str | None) -> bool:
    if not path:
        return False
    if not _checkpoint_shapes_match_model(trainer, path):
        print(f"student init is shape-incompatible; skipping: {path}", flush=True)
        return False
    _load_model_weights(trainer, path, strict=False)
    print(f"loaded student init: {path}", flush=True)
    return True


def _save_checkpoint(
    trainer: RebelCFRTrainer,
    path: Path,
    *,
    step: int,
    run_id: str | None,
    metadata: dict[str, Any],
) -> None:
    trainer.save_checkpoint(
        str(path),
        step=step,
        wandb_run_id=run_id,
        save_optimizer=True,
        save_dtype=torch.bfloat16,
        metadata=metadata,
    )


@torch.no_grad()
def _evaluate_validation_in_memory(
    *,
    trainer: RebelCFRTrainer,
    value_model: BetterSplitFFN,
    validation_path: str | Path,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(validation_path, map_location="cpu", weights_only=False)
    rows = int(payload["examples"])
    if rows % int(batch_size) != 0:
        raise ValueError(
            "in-memory validation requires the fixed validation set to be divisible "
            f"by the native batch size; examples={rows} batch_size={batch_size}"
        )
    env = _make_env_from_manifest(
        payload["source_manifest"],
        num_envs=int(batch_size),
        device=device,
        seed=2026062702,
    )
    was_training = trainer.model.training
    trainer.model.eval()
    total_loss = 0.0
    total_weight = 0.0
    metric_totals: dict[str, float] = {}
    start_time = time.time()
    try:
        for start in range(0, rows, int(batch_size)):
            end = start + int(batch_size)
            states = {name: payload["states"][name][start:end] for name in STATE_FIELDS}
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
                "pot": env.pot[:count].clone(),
                "scale": env.scale[:count].clone(),
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
            pot_metrics = _pot_relative_value_error_metrics(output, batch, loss_dict)
            value_weights = loss_dict["value_weights"]
            value_loss_all = loss_dict["value_loss_all"]
            total_loss += float(value_loss_all.sum().detach().cpu().item())
            total_weight += float(value_weights.sum().detach().cpu().item())
            for name, value in pot_metrics.items():
                metric_totals[name] = metric_totals.get(name, 0.0) + float(
                    value.detach().cpu().item()
                ) * count
    finally:
        if was_training:
            trainer.model.train()
    result = {
        "validation": str(Path(validation_path).resolve()),
        "examples": rows,
        "batch_size": int(batch_size),
        "validation_value_loss": total_loss / max(total_weight, 1.0e-12),
        "elapsed_s": time.time() - start_time,
    }
    result.update(
        {
            f"validation_{name}": value / max(1, rows)
            for name, value in metric_totals.items()
        }
    )
    return result


def run(args: argparse.Namespace) -> Path:
    reader = StreetClosedStateReader(args.state_dataset, seed=int(args.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    cfg = _student_config(args, reader)
    setup_torch_runtime(cfg, device, recompile_limit=int(args.recompile_limit))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(
        cfg,
        output_dir,
        resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
    )

    trainer = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device)
    _maybe_load_student_init(trainer, args.student_init)
    target_model = trainer.load_closing_leaf_model(args.target_checkpoint)
    value_model = _value_model(trainer.model)
    env = _make_env_from_manifest(
        reader.manifest,
        num_envs=int(args.batch_size),
        device=device,
        seed=int(args.seed) + 17,
    )
    belief_rng = torch.Generator(device=device)
    belief_rng.manual_seed(int(args.seed) + 31)
    total_steps = int(args.steps) if int(args.steps) > 0 else int(cfg.num_steps)
    max_rows = reader.rows if int(args.max_rows) <= 0 else int(args.max_rows)
    global_step = 0
    roots_seen = 0
    throughput_warmup_steps = max(0, int(args.throughput_warmup_steps))
    post_warmup_elapsed_s = 0.0
    post_warmup_roots = 0
    post_warmup_steps = 0
    metadata = {
        "kind": "epreflop_6p_uniform_live_pair_distilled_model",
        "state_dataset": str(Path(args.state_dataset).resolve()),
        "target_checkpoint": str(Path(args.target_checkpoint).resolve()),
        "belief_mode": str(args.belief_mode),
        "belief_profile": str(args.belief_profile),
        "target_projection": "uniform unordered live pairs with stack-baseline nonparticipants",
        "preflop_model_type": str(args.preflop_model_type),
        "learning_rate": float(args.learning_rate),
        "adamw_learning_rate": cfg.train.adamw_learning_rate,
        "batch_size": int(args.batch_size),
        "train_steps": int(total_steps),
        "schedule_steps": int(cfg.num_steps),
        "throughput_warmup_steps": throughput_warmup_steps,
    }

    with wandb_run(
        cfg,
        name=args.wandb_name,
        stage="epreflop_6p_live_pair_distill",
        resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
        extra_config={"live_pair_distill": metadata},
    ) as run:
        if run is not None:
            run.summary.update({"state_rows": reader.rows})
        epoch = 0
        while global_step < total_steps:
            made_progress = False
            for states in reader.iter_batches(
                batch_size=int(args.batch_size),
                max_rows=max_rows,
                epoch=epoch,
            ):
                if global_step >= total_steps:
                    break
                rows_in_batch = int(next(iter(states.values())).shape[0])
                if rows_in_batch != int(args.batch_size):
                    continue
                timing_step = global_step >= throughput_warmup_steps
                if timing_step and device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_start_s = time.perf_counter() if timing_step else 0.0
                rows = _copy_public_states_to_env(env, states)
                beliefs = _random_beliefs(
                    rows,
                    trainer.num_players,
                    device=device,
                    rng=belief_rng,
                    mode=str(args.belief_mode),
                    profile=str(args.belief_profile),
                )
                value_encoder = value_model.create_feature_encoder(
                    env=env,
                    device=device,
                    dtype=trainer.float_dtype,
                )
                batch = build_preflop_live_pair_value_batch(
                    env,
                    beliefs,
                    value_encoder=value_encoder,
                    target_model=target_model,
                    float_dtype=trainer.float_dtype,
                )
                metrics = trainer.train_value_batch(
                    batch,
                    global_step,
                    sync_inference_model=True,
                )
                if timing_step:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    step_elapsed_s = time.perf_counter() - step_start_s
                    post_warmup_elapsed_s += step_elapsed_s
                    post_warmup_roots += rows
                    post_warmup_steps += 1
                    metrics["post_warmup_step_s"] = step_elapsed_s
                    metrics["post_warmup_examples_per_s"] = (
                        rows / step_elapsed_s if step_elapsed_s > 0.0 else float("inf")
                    )
                    metrics["post_warmup_cumulative_examples_per_s"] = (
                        post_warmup_roots / post_warmup_elapsed_s
                        if post_warmup_elapsed_s > 0.0
                        else float("inf")
                    )
                roots_seen += rows
                global_step += 1
                made_progress = True
                payload = {
                    "global_step": global_step,
                    "distill/roots_seen": roots_seen,
                    "distill/epoch": epoch,
                }
                payload.update({f"value/{key}": value for key, value in metrics.items()})
                if run is not None:
                    run.log(payload, step=global_step)
                if (
                    int(args.log_interval) > 0
                    and global_step % int(args.log_interval) == 0
                ):
                    loss = metrics.get("value_loss", metrics.get("loss"))
                    print(
                        f"step={global_step:,}/{total_steps:,} "
                        f"roots={roots_seen:,} value_loss={loss} "
                        f"post_warmup_ex_s="
                        f"{metrics.get('post_warmup_cumulative_examples_per_s')}",
                        flush=True,
                    )
                if (
                    int(args.checkpoint_interval) > 0
                    and global_step % int(args.checkpoint_interval) == 0
                ):
                    inprogress = output_dir / "checkpoints" / "distilled_inprogress.pt"
                    _save_checkpoint(
                        trainer,
                        inprogress,
                        step=global_step,
                        run_id=None if run is None else run.id,
                        metadata={**metadata, "snapshot": "in_progress"},
                    )
                    print(f"saved checkpoint: {inprogress}", flush=True)
            if not made_progress:
                raise RuntimeError("state reader yielded no batches")
            epoch += 1

        final = output_dir / "checkpoints" / "distilled_final.pt"
        _save_checkpoint(
            trainer,
            final,
            step=global_step,
            run_id=None if run is None else run.id,
            metadata={**metadata, "snapshot": "final"},
        )
        print(f"saved final checkpoint: {final}", flush=True)
        if args.validation:
            validation_result = _evaluate_validation_in_memory(
                trainer=trainer,
                value_model=value_model,
                validation_path=args.validation,
                batch_size=int(args.batch_size),
                device=device,
            )
            validation_text = json.dumps(validation_result, indent=2, sort_keys=True)
            print(validation_text, flush=True)
            validation_json = (
                Path(args.validation_output_json)
                if args.validation_output_json
                else output_dir / "validation_final.json"
            )
            validation_json.parent.mkdir(parents=True, exist_ok=True)
            validation_json.write_text(validation_text + "\n")
            if run is not None:
                run.log(
                    {
                        "validation/value_loss": validation_result[
                            "validation_value_loss"
                        ],
                        "validation/examples": validation_result["examples"],
                        "validation/elapsed_s": validation_result["elapsed_s"],
                    },
                    step=global_step,
                )
                run.summary.update(
                    {
                        "validation_value_loss": validation_result[
                            "validation_value_loss"
                        ],
                        "validation_examples": validation_result["examples"],
                    }
                )
    return final


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
