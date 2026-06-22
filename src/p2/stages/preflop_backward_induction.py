#!/usr/bin/env python3
"""Backward-induction training and distillation for preflop depth buckets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset

from p2.config.rebel_schema import RebelExperimentConfig
from p2.core.structured_config import Config
from p2.env.card_utils import PREFLOP_HANDS, preflop_class_multiplicity_tensor
from p2.env.pbs_env import PBSEnv
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.search.rebel_solved_dataset import RebelSolvedDatasetWriter
from p2.stages.preflop_buckets import (
    PreflopBucketExecutionConfig,
    build_run_config,
    load_base_config,
)
from p2.runtime.training_run import wandb_run, write_resolved_config
from p2.utils.model_utils import compute_masked_logits, count_model_parameters


DEFAULT_CHECKPOINT = (
    "/home/user/poker2/checkpoints-rebel-curriculum-preflop_2000_p6_lr0p01_"
    "backupcons_actor_lam01_rb32_from2p_norb/preflop/rebel_latest.pt"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_FIELDS = (
    "button",
    "street",
    "to_act",
    "last_to_act",
    "pot",
    "min_raise",
    "last_aggressive_amount",
    "actions_this_round",
    "actions_last_round",
    "stacks",
    "starting_stacks",
    "scale",
    "committed",
    "chips_placed",
    "has_folded",
    "is_allin",
    "acted_this_round",
    "done",
    "winner",
    "winners",
    "board_indices",
    "last_board_indices",
    "deck_pos",
)


@dataclass(frozen=True)
class BucketSpec:
    label: str
    low: int
    high: int


BUCKET_SPECS = (
    BucketSpec("actions_0_3", 0, 3),
    BucketSpec("actions_4_7", 4, 7),
    BucketSpec("actions_8_11", 8, 11),
    BucketSpec("actions_12_15", 12, 15),
)


BUCKET_ORDER_DEEP_TO_SHALLOW = (
    "actions_12_15",
    "actions_8_11",
    "actions_4_7",
    "actions_0_3",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _init_wandb(args: PreflopBucketExecutionConfig, cfg: Config, *, name: str):
    run_name: str = str(args.wandb_name or name)
    group: str | None = None if args.wandb_group is None else str(args.wandb_group)
    return wandb_run(
        cfg,
        group=group,
        name=run_name,
        resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
    )


@torch.no_grad()
def _load_model_weights(
    trainer: RebelCFRTrainer,
    checkpoint_path: str,
    *,
    strict: bool | None = None,
) -> int:
    checkpoint = torch.load(
        checkpoint_path, map_location=trainer.device, weights_only=False
    )
    model_state = checkpoint["model"]
    save_dtype = checkpoint.get("save_dtype")
    if save_dtype is not None and save_dtype != str(trainer.float_dtype):
        model_state = {
            key: value.to(trainer.float_dtype)
            if value.dtype.is_floating_point
            else value
            for key, value in model_state.items()
        }
    if checkpoint.get("model_component") == "value_model":
        if type(trainer.model) is not BetterSplitFFN:
            raise TypeError("value-only checkpoints require a BetterSplitFFN model")
        value_model = trainer.model.value_model
        value_model.load_state_dict(
            model_state,
            strict=trainer.cfg.strict_model_loading if strict is None else strict,
        )
    else:
        trainer.model.load_state_dict(
            model_state,
            strict=trainer.cfg.strict_model_loading if strict is None else strict,
        )
    trainer._sync_inference_model()
    trainer._sync_cfr_target_model(int(checkpoint.get("step", 0)))
    trainer.model.train()
    return int(checkpoint.get("step", -1))


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
    rng.manual_seed(seed)
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
        force_heads_up_preflop_flop=True,
    )
    env.reset()
    return env


def _load_state_manifest(root: Path, *, allow_partial: bool) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() and allow_partial:
        manifest_path = root / "manifest.partial.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found under {root}; use --allow-partial for active runs"
        )
    return json.loads(manifest_path.read_text())


def _bucket_shards(
    manifest: dict[str, Any],
    root: Path,
    bucket_label: str,
) -> tuple[Path, int, list[dict[str, Any]]]:
    for key in ("buckets", "frontiers"):
        for item in manifest.get(key, []):
            label = item.get("label")
            if label is None and "action_count" in item:
                label = f"frontier_{int(item['action_count'])}"
            if label != bucket_label:
                continue
            bucket_dir = root / str(label)
            rows = int(item.get("num_rows", 0))
            shards = list(item.get("shards", []))
            return bucket_dir, rows, shards
    raise KeyError(f"Bucket {bucket_label!r} not found in state manifest")


def _seed_for_label(seed: int, label: str, *, salt: int = 0) -> int:
    mixed = (int(seed) + 0x9E3779B97F4A7C15 + salt) & 0x7FFF_FFFF_FFFF_FFFF
    for index, char in enumerate(label):
        mixed ^= (ord(char) + index + 1) * 0xBF58476D1CE4E5B9
        mixed = (mixed * 0x94D049BB133111EB) & 0x7FFF_FFFF_FFFF_FFFF
    return mixed


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


class PublicStateBucketReader:
    def __init__(
        self,
        root: str | Path,
        bucket_label: str,
        *,
        allow_partial: bool,
        seed: int,
    ):
        self.root = Path(root)
        self.manifest = _load_state_manifest(self.root, allow_partial=allow_partial)
        self.bucket_label = bucket_label
        self.seed = int(seed)
        self.bucket_dir, self.rows, self.shards = _bucket_shards(
            self.manifest, self.root, bucket_label
        )
        if len(self.shards) != 1:
            raise ValueError(
                f"{bucket_label} has {len(self.shards)} shards; pack the state "
                "dataset to one shard per bucket before training"
            )
        shard = self.shards[0]
        shard_path = self.bucket_dir / str(shard.get("path", shard.get("file")))
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        self.states = payload["states"]
        self.rows = int(
            payload.get("num_rows", next(iter(self.states.values())).shape[0])
        )

    def iter_state_batches(
        self,
        *,
        batch_size: int,
        max_rows: int,
        seed: int | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        target_rows = min(int(max_rows), int(self.rows))
        if target_rows <= 0:
            return
        dataset = TensorDataset(torch.arange(self.rows, dtype=torch.long))
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=True,
            generator=_cpu_generator(
                _seed_for_label(
                    self.seed if seed is None else seed,
                    self.bucket_label,
                    salt=31,
                )
            ),
            num_workers=0,
        )
        yielded = 0
        for (batch_indices,) in loader:
            if yielded >= target_rows:
                break
            take = min(int(batch_indices.numel()), target_rows - yielded)
            if take < int(batch_indices.numel()):
                batch_indices = batch_indices[:take]
            yield {
                name: tensor.index_select(0, batch_indices)
                for name, tensor in self.states.items()
            }
            yielded += take


def _copy_public_states_to_env(env: PBSEnv, states: dict[str, torch.Tensor]) -> int:
    count = int(next(iter(states.values())).shape[0])
    rows = torch.arange(count, device=env.device)
    env.reset(rows)
    for name in STATE_FIELDS:
        value = states[name].to(device=env.device)
        target = getattr(env, name)
        if target.dtype.is_floating_point:
            value = value.to(dtype=target.dtype)
        else:
            value = value.to(dtype=target.dtype)
        target[rows] = value
    env.board_onehot[rows] = False
    board = env.board_indices[rows]
    valid = board >= 0
    if bool(valid.any().item()):
        safe_board = board.clamp(0, 51)
        onehot = env.card_onehot_cache[safe_board]
        env.board_onehot[rows] = onehot & valid[:, :, None, None]
    if count < env.N:
        env.done[count:] = True
    return count


def _random_beliefs(
    rows: int,
    num_players: int,
    *,
    device: torch.device,
    rng: torch.Generator,
    mode: str,
) -> torch.Tensor:
    prior = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    prior = prior / prior.sum().clamp_min(1.0)
    if mode == "uniform":
        return (
            prior.view(1, 1, PREFLOP_HANDS)
            .expand(rows, num_players, PREFLOP_HANDS)
            .clone()
        )
    if mode == "random":
        weights = torch.empty(
            rows,
            num_players,
            PREFLOP_HANDS,
            device=device,
            dtype=torch.float32,
        )
        weights.exponential_(1.0, generator=rng)
        weights *= prior.view(1, 1, PREFLOP_HANDS)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    raise ValueError(f"unsupported belief mode: {mode}")


def _filter_batch_by_action_bucket(
    batch: RebelBatch | None,
    *,
    low: int,
    high: int,
) -> RebelBatch | None:
    if batch is None or len(batch) == 0:
        return None
    actions = batch.statistics.get("actions_this_round")
    if actions is None:
        return batch
    mask = (actions >= low) & (actions <= high)
    if not bool(mask.any().item()):
        return None
    return batch[mask]


def _value_only(batch: RebelBatch | None) -> RebelBatch | None:
    if batch is None or len(batch) == 0 or batch.value_targets is None:
        return None
    return RebelBatch(
        features=batch.features,
        legal_masks=batch.legal_masks,
        value_targets=batch.value_targets,
        statistics=batch.statistics,
    )


def _policy_only(batch: RebelBatch | None) -> RebelBatch | None:
    if batch is None or len(batch) == 0 or batch.policy_targets is None:
        return None
    return RebelBatch(
        features=batch.features,
        legal_masks=batch.legal_masks,
        policy_targets=batch.policy_targets,
        statistics=batch.statistics,
    )


def _float_metrics(stats: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in stats.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            out[key] = float(value.detach().item())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _prefixed_metrics(
    scope: str,
    prefix: str,
    stats: dict[str, Any],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in _float_metrics(stats).items():
        metric_key = "train_step" if key == "step" else key
        out[f"{scope}/{prefix}_{metric_key}"] = value
    return out


def _policy_update(
    trainer: RebelCFRTrainer,
    batch: RebelBatch,
    *,
    step: int,
) -> dict[str, float]:
    trainer._apply_schedules(step)
    trainer.model.train()
    stats = trainer._supervise_policy_only(batch.to(trainer.device))
    trainer._sync_inference_model()
    trainer._sync_cfr_target_model(step + 1)
    return _float_metrics(stats)


def _evaluator_tree_stats(evaluator: Any) -> dict[str, float]:
    root_nodes = int(getattr(evaluator, "root_nodes", 0))
    total_nodes = int(getattr(evaluator, "total_nodes", 0))
    return {
        "evaluator_total_nodes": float(total_nodes),
        "evaluator_root_nodes": float(root_nodes),
        "evaluator_nodes_per_root": total_nodes / max(1, root_nodes),
    }


def _solve_public_state_batch(
    solver: RebelCFRTrainer,
    env: PBSEnv,
    beliefs: torch.Tensor,
    *,
    include_policy: bool,
) -> tuple[RebelBatch | None, RebelBatch | None, dict[str, float]]:
    root_indices = torch.arange(beliefs.shape[0], device=solver.device)
    evaluator = solver.cfr_evaluator
    evaluator.initialize_subgame(env, root_indices, beliefs)
    tree_stats = _evaluator_tree_stats(evaluator)
    evaluator.evaluate_cfr(training_mode=True, sample_continuation=False)
    value_batch, _, policy_batch = evaluator.training_data(
        include_pre_chance_value_batch=False,
        include_policy_batch=include_policy,
    )
    return value_batch, policy_batch, tree_stats


def _save_trainer_checkpoint(
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
        save_optimizer=False,
        save_dtype=torch.bfloat16,
        metadata=metadata,
    )


def _bucket_spec(label: str):
    for spec in BUCKET_SPECS:
        if spec.label == label:
            return spec
    raise KeyError(label)


def _checkpoint_signature(checkpoint_path: str) -> dict[str, Any]:
    path = Path(checkpoint_path)
    stat = path.stat()
    return {
        "path": os.path.realpath(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _bucket_epochs(args: PreflopBucketExecutionConfig, bucket_label: str) -> int:
    if bucket_label == "actions_12_15":
        return max(1, int(args.actions_12_15_epochs))
    return 1


def _bucket_cfr_batch_size(
    args: PreflopBucketExecutionConfig, bucket_label: str
) -> int:
    overrides = {
        "actions_12_15": args.actions_12_15_cfr_batch_size,
        "actions_8_11": args.actions_8_11_cfr_batch_size,
        "actions_4_7": None,
        "actions_0_3": None,
    }
    override = overrides[bucket_label]
    if override is not None:
        return max(1, int(override))
    return max(1, int(args.cfr_batch_size))


def _max_cfr_batch_size(args: PreflopBucketExecutionConfig) -> int:
    return max(
        _bucket_cfr_batch_size(args, label) for label in BUCKET_ORDER_DEEP_TO_SHALLOW
    )


def _estimate_train_updates(args: PreflopBucketExecutionConfig) -> int:
    manifest = _load_state_manifest(
        Path(args.state_dataset), allow_partial=args.allow_partial
    )
    total_updates = 0
    for bucket_label in BUCKET_ORDER_DEEP_TO_SHALLOW:
        _, rows, _ = _bucket_shards(manifest, Path(args.state_dataset), bucket_label)
        rows = min(int(args.states_per_bucket), int(rows))
        batches = math.ceil(rows / _bucket_cfr_batch_size(args, bucket_label))
        updates_per_batch = 1 if bucket_label == "actions_0_3" else 2
        total_updates += (
            batches * updates_per_batch * _bucket_epochs(args, bucket_label)
        )
    return max(1, total_updates)


def _validation_cache_metadata(
    args: PreflopBucketExecutionConfig,
    *,
    bucket_label: str,
    cutoff_checkpoint: str,
) -> dict[str, Any]:
    return {
        "kind": "preflop_backward_induction_validation_cache",
        "bucket_label": bucket_label,
        "state_dataset": os.path.realpath(args.state_dataset),
        "cutoff_checkpoint": _checkpoint_signature(cutoff_checkpoint),
        "validation_items": int(args.validation_items),
        "validation_cfr_iterations": int(args.validation_cfr_iterations),
        "cfr_batch_size": _bucket_cfr_batch_size(args, bucket_label),
        "validation_seed": int(args.seed) + 900_000,
        "depth": int(args.depth),
        "warm_start_iterations": int(args.warm_start_iterations),
        "sparse_fused": bool(args.sparse_fused),
        "belief_mode": str(args.belief_mode),
    }


def _validation_cache_path(
    bucket_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    encoded = json.dumps(_jsonable(metadata), sort_keys=True).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()[:16]
    return (
        bucket_dir
        / "validation"
        / (
            f"validation_n{metadata['validation_items']}_"
            f"cfr{metadata['validation_cfr_iterations']}_{cache_key}.pt"
        )
    )


def _slice_batch(batch: RebelBatch, start: int, end: int) -> RebelBatch:
    return batch[slice(start, end)]


def _evaluate_validation_split(
    trainer: RebelCFRTrainer,
    batch: RebelBatch | None,
    *,
    include_value: bool,
    eval_batch_size: int,
) -> dict[str, float]:
    if batch is None or len(batch) == 0:
        return {}
    trainer.model.eval()
    totals: dict[str, float] = {}
    total_rows = 0
    with torch.no_grad():
        for start in range(0, len(batch), eval_batch_size):
            part = _slice_batch(
                batch, start, min(start + eval_batch_size, len(batch))
            ).to(trainer.device)
            rows = len(part)
            with trainer._model_autocast():
                output = trainer.model(
                    part.features,
                    include_policy=not include_value,
                    include_value=include_value,
                )
            loss_dict = (
                trainer.loss_fn._call_forward_value(output, part)
                if include_value
                else trainer.loss_fn._call_forward_policy(output, part)
            )
            metrics = _float_metrics(loss_dict)
            for key, value in metrics.items():
                if key.endswith("_all") or key.endswith("_weights"):
                    continue
                totals[key] = totals.get(key, 0.0) + value * rows
            total_rows += rows
    trainer.model.train()
    return {key: value / max(1, total_rows) for key, value in totals.items()}


def _evaluate_validation_set(
    trainer: RebelCFRTrainer,
    validation: dict[str, Any],
    *,
    eval_batch_size: int,
) -> dict[str, float]:
    def metric_name(prefix: str, key: str) -> str:
        if key.startswith(f"{prefix}_"):
            return f"validation_{key}"
        return f"validation_{prefix}_{key}"

    metrics: dict[str, float] = {}
    value_metrics = _evaluate_validation_split(
        trainer,
        validation.get("value_batch"),
        include_value=True,
        eval_batch_size=eval_batch_size,
    )
    policy_metrics = _evaluate_validation_split(
        trainer,
        validation.get("policy_batch"),
        include_value=False,
        eval_batch_size=eval_batch_size,
    )
    metrics.update(
        {metric_name("value", key): value for key, value in value_metrics.items()}
    )
    metrics.update(
        {metric_name("policy", key): value for key, value in policy_metrics.items()}
    )
    return metrics


def _build_validation_cache(
    *,
    args: PreflopBucketExecutionConfig,
    bucket_label: str,
    spec: BucketSpec,
    bucket_dir: Path,
    cutoff_checkpoint: str,
    cfg: Config,
    reader: PublicStateBucketReader,
    device: torch.device,
    bucket_index: int,
) -> dict[str, Any]:
    metadata = _validation_cache_metadata(
        args,
        bucket_label=bucket_label,
        cutoff_checkpoint=cutoff_checkpoint,
    )
    cache_path = _validation_cache_path(bucket_dir, metadata)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("metadata") == metadata:
            print(f"loaded validation cache {cache_path}", flush=True)
            return cached

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cfr_batch_size = _bucket_cfr_batch_size(args, bucket_label)
    validation_cfg = copy.deepcopy(cfg)
    validation_cfg.search.iterations = int(args.validation_cfr_iterations)
    validation_cfg.num_envs = cfr_batch_size
    validation_solver = RebelCFRTrainer(
        cfg=validation_cfg,
        device=device,
        pregeneration_only=True,
    )
    _load_model_weights(validation_solver, cutoff_checkpoint)
    env = _make_env_from_manifest(
        reader.manifest,
        num_envs=cfr_batch_size,
        device=device,
        seed=args.seed + bucket_index + 10_000,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(_seed_for_label(args.seed, bucket_label, salt=900_000))

    value_parts: list[RebelBatch] = []
    policy_parts: list[RebelBatch] = []
    roots_solved = 0
    validation_tree_batches = 0
    validation_total_nodes_sum = 0.0
    validation_root_nodes_sum = 0.0
    validation_max_total_nodes = 0.0
    for states in reader.iter_state_batches(
        batch_size=cfr_batch_size,
        max_rows=args.validation_items,
        seed=_seed_for_label(args.seed, bucket_label, salt=800_000),
    ):
        rows = _copy_public_states_to_env(env, states)
        beliefs = _random_beliefs(
            rows,
            validation_solver.num_players,
            device=device,
            rng=rng,
            mode=args.belief_mode,
        )
        value_batch, policy_batch, tree_stats = _solve_public_state_batch(
            validation_solver,
            env,
            beliefs,
            include_policy=True,
        )
        validation_tree_batches += 1
        validation_total_nodes_sum += tree_stats["evaluator_total_nodes"]
        validation_root_nodes_sum += tree_stats["evaluator_root_nodes"]
        validation_max_total_nodes = max(
            validation_max_total_nodes,
            tree_stats["evaluator_total_nodes"],
        )
        value_batch = _filter_batch_by_action_bucket(
            value_batch,
            low=spec.low,
            high=spec.high,
        )
        policy_batch = _filter_batch_by_action_bucket(
            policy_batch,
            low=spec.low,
            high=spec.high,
        )
        value_stream = _value_only(value_batch)
        policy_stream = _policy_only(policy_batch)
        if value_stream is not None:
            value_parts.append(value_stream.to(torch.device("cpu")))
        if policy_stream is not None:
            policy_parts.append(policy_stream.to(torch.device("cpu")))
        roots_solved += rows
        if roots_solved >= args.validation_items:
            break

    value_cache = RebelBatch.cat(value_parts) if value_parts else None
    policy_cache = RebelBatch.cat(policy_parts) if policy_parts else None
    validation = {
        "metadata": metadata,
        "roots_solved": roots_solved,
        "value_batch": value_cache,
        "policy_batch": policy_cache,
        "solve_stats": {
            "evaluator_total_nodes_mean": validation_total_nodes_sum
            / max(1, validation_tree_batches),
            "evaluator_total_nodes_max": validation_max_total_nodes,
            "evaluator_root_nodes_mean": validation_root_nodes_sum
            / max(1, validation_tree_batches),
            "evaluator_nodes_per_root_mean": validation_total_nodes_sum
            / max(1.0, validation_root_nodes_sum),
            "evaluator_solve_batches": float(validation_tree_batches),
        },
    }
    torch.save(validation, cache_path)
    print(
        f"created validation cache {cache_path}: roots={roots_solved:,} "
        f"value={0 if value_cache is None else len(value_cache):,} "
        f"policy={0 if policy_cache is None else len(policy_cache):,}",
        flush=True,
    )
    del validation_solver
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return validation


def run_train_specialists(
    args: PreflopBucketExecutionConfig,
    *,
    base_template: Config | None = None,
) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_updates_guess = _estimate_train_updates(args)
    if base_template is None:
        base_template = load_base_config(
            repo_root=REPO_ROOT,
            config_name=args.config_name,
            overrides=args.config_overrides,
        )
    base_cfg = build_run_config(
        base_template,
        args,
        checkpoint_dir=output_dir / "checkpoints",
        num_steps=total_updates_guess,
        num_envs=_max_cfr_batch_size(args),
    )
    write_resolved_config(
        base_cfg,
        output_dir,
        resolved_config=RebelExperimentConfig.from_trainer_config(base_cfg),
    )
    run_cm = _init_wandb(
        args,
        base_cfg,
        name=f"preflop-bi-specialists-{_now_slug()}",
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))

    with run_cm as run:
        previous_value_checkpoint = args.base_checkpoint
        global_step = 0
        specialist_paths: dict[str, str] = {}
        for bucket_index, bucket_label in enumerate(BUCKET_ORDER_DEEP_TO_SHALLOW):
            spec = _bucket_spec(bucket_label)
            train_value = bucket_label != "actions_0_3"
            cfr_batch_size = _bucket_cfr_batch_size(args, bucket_label)
            bucket_dir = output_dir / bucket_label
            bucket_dir.mkdir(parents=True, exist_ok=True)
            solved_dir = bucket_dir / "solved"
            if args.write_solved_shards:
                if (
                    solved_dir.exists()
                    and any(solved_dir.iterdir())
                    and not args.overwrite
                ):
                    raise FileExistsError(f"{solved_dir} exists; pass --overwrite")
                solved_dir.mkdir(parents=True, exist_ok=True)

            cfg = build_run_config(
                base_template,
                args,
                checkpoint_dir=bucket_dir / "checkpoints",
                num_steps=total_updates_guess,
                num_envs=cfr_batch_size,
            )
            write_resolved_config(
                cfg,
                resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
            )
            reader = PublicStateBucketReader(
                args.state_dataset,
                bucket_label,
                allow_partial=args.allow_partial,
                seed=args.seed + bucket_index * 10_000,
            )
            validation = _build_validation_cache(
                args=args,
                bucket_label=bucket_label,
                spec=spec,
                bucket_dir=bucket_dir,
                cutoff_checkpoint=previous_value_checkpoint,
                cfg=cfg,
                reader=reader,
                device=device,
                bucket_index=bucket_index,
            )
            solver = RebelCFRTrainer(
                cfg=copy.deepcopy(cfg), device=device, pregeneration_only=True
            )
            _load_model_weights(solver, previous_value_checkpoint)
            trainer = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device)
            _load_model_weights(trainer, previous_value_checkpoint)
            if bucket_index == 0 and run is not None:
                run.summary.update(count_model_parameters(trainer.model))

            env = _make_env_from_manifest(
                reader.manifest,
                num_envs=cfr_batch_size,
                device=device,
                seed=args.seed + bucket_index + 100,
            )
            writer = (
                RebelSolvedDatasetWriter(
                    solved_dir,
                    storage_float_dtype=args.storage_dtype,
                )
                if args.write_solved_shards
                else None
            )
            roots_solved = 0
            value_examples = 0
            value_step = 0
            policy_examples = 0
            bucket_step = 0
            bucket_epochs = _bucket_epochs(args, bucket_label)
            bucket_start = time.time()
            progress_interval = max(int(args.progress_roots), cfr_batch_size)
            next_progress_roots = progress_interval
            last_tree_stats: dict[str, float] = {}

            def log_validation(epoch: int, epoch_roots: int) -> None:
                if run is None:
                    return
                metrics = _evaluate_validation_set(
                    trainer,
                    validation,
                    eval_batch_size=args.validation_eval_batch_size,
                )
                payload = {
                    f"{bucket_label}/validation_step": bucket_step,
                    f"{bucket_label}/epoch": epoch,
                    f"{bucket_label}/epoch_roots": epoch_roots,
                    f"{bucket_label}/roots_solved": roots_solved,
                    f"{bucket_label}/global_step": global_step,
                    f"{bucket_label}/value_step": value_step,
                    "global_step": global_step,
                }
                payload.update(
                    {f"{bucket_label}/{key}": value for key, value in metrics.items()}
                )
                payload.update(
                    {
                        f"{bucket_label}/validation_{key}": value
                        for key, value in validation.get("solve_stats", {}).items()
                    }
                )
                run.log(payload, step=global_step)

            def print_progress(
                reason: str,
                *,
                epoch: int,
                epoch_roots: int,
                elapsed: float,
            ) -> None:
                roots_per_s = roots_solved / max(elapsed, 1.0e-9)
                print(
                    f"{bucket_label}: {reason} epoch={epoch + 1}/{bucket_epochs} "
                    f"roots={roots_solved:,} epoch_roots={epoch_roots:,} "
                    f"value={value_examples:,} policy={policy_examples:,} "
                    f"nodes={int(last_tree_stats.get('evaluator_total_nodes', 0)):,} "
                    f"cfr_batch={cfr_batch_size} step={global_step} "
                    f"bucket_step={bucket_step} "
                    f"roots/s={roots_per_s:.2f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if args.validation_interval_steps > 0:
                log_validation(epoch=0, epoch_roots=0)

            for epoch in range(bucket_epochs):
                epoch_roots = 0
                epoch_seed = args.seed + bucket_index * 10_000 + epoch * 1_000_000
                for states in reader.iter_state_batches(
                    batch_size=cfr_batch_size,
                    max_rows=args.states_per_bucket,
                    seed=epoch_seed,
                ):
                    rows = _copy_public_states_to_env(env, states)
                    beliefs = _random_beliefs(
                        rows,
                        solver.num_players,
                        device=device,
                        rng=rng,
                        mode=args.belief_mode,
                    )
                    value_batch, policy_batch, tree_stats = _solve_public_state_batch(
                        solver,
                        env,
                        beliefs,
                        include_policy=True,
                    )
                    last_tree_stats = tree_stats
                    value_batch = _filter_batch_by_action_bucket(
                        value_batch,
                        low=spec.low,
                        high=spec.high,
                    )
                    policy_batch = _filter_batch_by_action_bucket(
                        policy_batch,
                        low=spec.low,
                        high=spec.high,
                    )
                    value_stream = _value_only(value_batch) if train_value else None
                    policy_stream = _policy_only(policy_batch)
                    if value_stream is not None:
                        if writer is not None:
                            writer.append("value", value_stream)
                        value_examples += len(value_stream)
                        value_stats = trainer.train_value_batch(
                            value_stream,
                            global_step,
                            sync_inference_model=True,
                        )
                        value_step += 1
                        global_step += 1
                    else:
                        value_stats = {}
                    if policy_stream is not None:
                        if writer is not None:
                            writer.append("policy", policy_stream)
                        policy_examples += len(policy_stream)
                        policy_stats = _policy_update(
                            trainer,
                            policy_stream,
                            step=global_step,
                        )
                        global_step += 1
                    else:
                        policy_stats = {}

                    roots_solved += rows
                    epoch_roots += rows
                    bucket_step += 1
                    elapsed = time.time() - bucket_start
                    log_payload = {
                        f"{bucket_label}/roots_solved": roots_solved,
                        f"{bucket_label}/epoch": epoch,
                        f"{bucket_label}/epoch_roots": epoch_roots,
                        f"{bucket_label}/bucket_step": bucket_step,
                        f"{bucket_label}/cfr_batch_size": cfr_batch_size,
                        f"{bucket_label}/global_step": global_step,
                        f"{bucket_label}/value_step": value_step,
                        f"{bucket_label}/value_examples": value_examples,
                        f"{bucket_label}/policy_examples": policy_examples,
                        f"{bucket_label}/elapsed_s": elapsed,
                        f"{bucket_label}/roots_per_s": roots_solved
                        / max(elapsed, 1.0e-9),
                        "global_step": global_step,
                    }
                    log_payload.update(
                        _prefixed_metrics(bucket_label, "value", value_stats)
                    )
                    log_payload.update(
                        _prefixed_metrics(bucket_label, "policy", policy_stats)
                    )
                    log_payload.update(
                        {
                            f"{bucket_label}/{key}": value
                            for key, value in tree_stats.items()
                        }
                    )
                    if run is not None:
                        run.log(log_payload, step=global_step)
                    if (
                        args.validation_interval_steps > 0
                        and bucket_step % int(args.validation_interval_steps) == 0
                    ):
                        log_validation(epoch=epoch, epoch_roots=epoch_roots)
                    if roots_solved >= next_progress_roots:
                        print_progress(
                            "progress",
                            epoch=epoch,
                            epoch_roots=epoch_roots,
                            elapsed=elapsed,
                        )
                        while next_progress_roots <= roots_solved:
                            next_progress_roots += progress_interval
                    if epoch_roots >= args.states_per_bucket:
                        break
                if epoch_roots > 0:
                    print_progress(
                        "epoch_complete",
                        epoch=epoch,
                        epoch_roots=epoch_roots,
                        elapsed=time.time() - bucket_start,
                    )

            if value_examples == 0 and policy_examples == 0:
                raise RuntimeError(f"No solved examples produced for {bucket_label}")
            bucket_summary = {
                "format_note": "preflop backward-induction solved bucket",
                "bucket_label": bucket_label,
                "bucket_low": spec.low,
                "bucket_high": spec.high,
                "root_states_solved": roots_solved,
                "value_examples": value_examples,
                "policy_examples": policy_examples,
                "train_value": train_value,
                "write_solved_shards": bool(args.write_solved_shards),
                "source_state_dataset": os.path.realpath(args.state_dataset),
                "solver_checkpoint": os.path.realpath(previous_value_checkpoint),
                "depth": args.depth,
                "cfr_iterations": args.cfr_iterations,
                "cfr_batch_size": cfr_batch_size,
                "belief_mode": args.belief_mode,
            }
            if writer is not None:
                bucket_summary["solved_dataset"] = writer.finalize(bucket_summary)
                bucket_summary["solved_manifest"] = str(solved_dir / "manifest.json")
            else:
                summary_path = bucket_dir / "training_summary.json"
                summary_path.write_text(
                    json.dumps(_jsonable(bucket_summary), indent=2, sort_keys=True)
                    + "\n"
                )
            checkpoint_path = bucket_dir / "checkpoints" / "rebel_latest.pt"
            _save_trainer_checkpoint(
                trainer,
                checkpoint_path,
                step=global_step,
                run_id=None if run is None else run.id,
                metadata={
                    "kind": "preflop_backward_induction_specialist",
                    "bucket_label": bucket_label,
                    "train_value": train_value,
                    "training_summary": bucket_summary,
                    "solved_manifest": bucket_summary.get("solved_manifest"),
                    "solver_checkpoint": os.path.realpath(previous_value_checkpoint),
                },
            )
            specialist_paths[bucket_label] = str(checkpoint_path)
            if train_value:
                previous_value_checkpoint = str(checkpoint_path)
            if run is not None:
                run.summary[f"{bucket_label}/checkpoint"] = str(checkpoint_path)
                run.summary[f"{bucket_label}/roots_solved"] = roots_solved
                run.summary[f"{bucket_label}/value_examples"] = value_examples
                run.summary[f"{bucket_label}/policy_examples"] = policy_examples
            print(
                f"completed {bucket_label}: checkpoint={checkpoint_path} "
                f"roots={roots_solved:,} value={value_examples:,} "
                f"policy={policy_examples:,}",
                flush=True,
            )

        summary = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state_dataset": os.path.realpath(args.state_dataset),
            "base_checkpoint": os.path.realpath(args.base_checkpoint),
            "specialists": specialist_paths,
            "global_updates": global_step,
        }
        (output_dir / "specialists_summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
        )
        if run is not None:
            run.summary.update(_jsonable(summary))


def _distill_batch_from_teacher(
    teacher: RebelCFRTrainer,
    env: PBSEnv,
    beliefs: torch.Tensor,
    *,
    include_value: bool,
) -> tuple[RebelBatch | None, RebelBatch]:
    encoder = teacher.model.create_feature_encoder(
        env,
        device=teacher.device,
        dtype=torch.bfloat16 if teacher.device.type == "cuda" else torch.float32,
    )
    features = encoder.encode(
        beliefs, indices=torch.arange(beliefs.shape[0], device=teacher.device)
    )
    _, legal = env.legal_bins_amounts_and_mask()
    legal = legal[: beliefs.shape[0]]
    teacher.model.eval()
    with torch.no_grad():
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if teacher.device.type == "cuda"
            else nullcontext()
        ):
            output = teacher.model(
                features,
                include_policy=True,
                include_value=include_value,
            )
        logits = output.policy_logits.float()
        policy_probs = torch.softmax(
            compute_masked_logits(logits, legal[:, None, :]),
            dim=-1,
        )
        statistics = {
            "to_act": env.to_act[: beliefs.shape[0]].clone(),
            "street": env.street[: beliefs.shape[0]].clone(),
            "stage": 2 * env.street[: beliefs.shape[0]].clone(),
            "board": env.board_indices[: beliefs.shape[0]].clone(),
            "pot": env.pot[: beliefs.shape[0]].clone(),
            "actions_this_round": env.actions_this_round[: beliefs.shape[0]].clone(),
            "node_depth": torch.zeros(
                beliefs.shape[0],
                dtype=torch.long,
                device=teacher.device,
            ),
            "bet_amounts": env.legal_bins_amounts_and_mask()[0][: beliefs.shape[0]],
        }
        if hasattr(env, "has_folded"):
            statistics["has_folded"] = env.has_folded[: beliefs.shape[0]].clone()
        if hasattr(env, "is_allin"):
            statistics["is_allin"] = env.is_allin[: beliefs.shape[0]].clone()
        policy_batch = RebelBatch(
            features=features,
            legal_masks=legal,
            policy_targets=policy_probs.detach().clone(),
            statistics=statistics,
        )
        value_batch = None
        if include_value:
            if output.hand_values is None:
                raise RuntimeError("teacher did not produce hand_values")
            value_batch = RebelBatch(
                features=features,
                legal_masks=legal,
                value_targets=output.hand_values.float().detach().clone(),
                statistics=statistics,
            )
    return value_batch, policy_batch


def run_distill(
    args: PreflopBucketExecutionConfig,
    *,
    base_template: Config | None = None,
) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_updates = max(
        1,
        math.ceil(args.states_per_bucket / max(1, args.distill_batch_size))
        * len(BUCKET_ORDER_DEEP_TO_SHALLOW)
        * 2,
    )
    if base_template is None:
        base_template = load_base_config(
            repo_root=REPO_ROOT,
            config_name=args.config_name,
            overrides=args.config_overrides,
        )
    cfg = build_run_config(
        base_template,
        args,
        checkpoint_dir=output_dir / "checkpoints",
        num_steps=total_updates,
    )
    write_resolved_config(
        cfg,
        output_dir,
        resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
    )
    student = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device)
    _load_model_weights(student, args.student_init or args.base_checkpoint)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))
    run_cm = _init_wandb(args, cfg, name=f"preflop-bi-distill-{_now_slug()}")

    with run_cm as run:
        if run is not None:
            run.summary.update(count_model_parameters(student.model))
        global_step = 0
        for bucket_label in BUCKET_ORDER_DEEP_TO_SHALLOW:
            checkpoints = {
                "actions_12_15": args.checkpoint_12_15,
                "actions_8_11": args.checkpoint_8_11,
                "actions_4_7": args.checkpoint_4_7,
                "actions_0_3": args.checkpoint_0_3,
            }
            checkpoint = checkpoints[bucket_label]
            if checkpoint is None:
                raise ValueError(f"missing specialist checkpoint for {bucket_label}")
            include_value = bucket_label != "actions_0_3"
            teacher_cfg = build_run_config(
                base_template,
                args,
                checkpoint_dir=output_dir / "teacher_tmp",
                num_steps=total_updates,
            )
            teacher = RebelCFRTrainer(
                cfg=copy.deepcopy(teacher_cfg),
                device=device,
                pregeneration_only=True,
            )
            _load_model_weights(teacher, checkpoint)
            reader = PublicStateBucketReader(
                args.state_dataset,
                bucket_label,
                allow_partial=args.allow_partial,
                seed=args.seed
                + 100_000
                + BUCKET_ORDER_DEEP_TO_SHALLOW.index(bucket_label),
            )
            env = _make_env_from_manifest(
                reader.manifest,
                num_envs=args.distill_batch_size,
                device=device,
                seed=args.seed + 1000,
            )
            roots = 0
            value_step = 0
            for states in reader.iter_state_batches(
                batch_size=args.distill_batch_size,
                max_rows=args.states_per_bucket,
            ):
                rows = _copy_public_states_to_env(env, states)
                beliefs = _random_beliefs(
                    rows,
                    teacher.num_players,
                    device=device,
                    rng=rng,
                    mode=args.belief_mode,
                )
                value_batch, policy_batch = _distill_batch_from_teacher(
                    teacher,
                    env,
                    beliefs,
                    include_value=include_value,
                )
                if value_batch is not None:
                    value_stats = student.train_value_batch(
                        value_batch,
                        global_step,
                        sync_inference_model=True,
                    )
                    value_step += 1
                    global_step += 1
                else:
                    value_stats = {}
                policy_stats = _policy_update(student, policy_batch, step=global_step)
                global_step += 1
                roots += rows
                payload = {
                    f"distill/{bucket_label}/roots": roots,
                    f"distill/{bucket_label}/global_step": global_step,
                    f"distill/{bucket_label}/value_step": value_step,
                    "global_step": global_step,
                }
                payload.update(
                    _prefixed_metrics(f"distill/{bucket_label}", "value", value_stats)
                )
                payload.update(
                    _prefixed_metrics(f"distill/{bucket_label}", "policy", policy_stats)
                )
                if run is not None:
                    run.log(payload, step=global_step)
                if roots >= args.states_per_bucket:
                    break
            print(f"distilled {bucket_label}: roots={roots:,}", flush=True)

        checkpoint_path = output_dir / "checkpoints" / "rebel_latest.pt"
        _save_trainer_checkpoint(
            student,
            checkpoint_path,
            step=global_step,
            run_id=None if run is None else run.id,
            metadata={
                "kind": "preflop_backward_induction_distilled_model",
                "state_dataset": os.path.realpath(args.state_dataset),
                "specialist_checkpoints": {
                    "actions_12_15": args.checkpoint_12_15,
                    "actions_8_11": args.checkpoint_8_11,
                    "actions_4_7": args.checkpoint_4_7,
                    "actions_0_3": args.checkpoint_0_3,
                },
            },
        )
        print(f"saved distilled model: {checkpoint_path}", flush=True)
