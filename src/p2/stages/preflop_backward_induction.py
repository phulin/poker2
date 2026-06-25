#!/usr/bin/env python3
"""Backward-induction training and distillation for preflop depth buckets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset

from p2.config.rebel_schema import RebelExperimentConfig
from p2.core.structured_config import Config, PreflopModelType
from p2.env.card_utils import PREFLOP_HANDS, preflop_class_multiplicity_tensor
from p2.env.pbs_env import PBSEnv
from p2.rl.checkpoint_io import CheckpointIO
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.search.cfr_evaluator import CFREvaluator
from p2.search.rebel_solved_dataset import RebelSolvedDatasetWriter
from p2.stages.preflop_buckets import (
    PreflopBucketExecutionConfig,
    build_run_config,
)
from p2.runtime.training_run import wandb_run, write_resolved_config
from p2.utils.model_utils import compute_masked_logits, count_model_parameters


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

LEGACY_LATEST_CHECKPOINT = "rebel_latest.pt"
SPECIALIST_FINAL_CHECKPOINT = "specialist_final.pt"
SPECIALIST_INPROGRESS_CHECKPOINT = "specialist_inprogress.pt"
BOOTSTRAP_DISTILLED_CHECKPOINT = "bootstrap_distilled.pt"
DISTILLED_FINAL_CHECKPOINT = "distilled_final.pt"
DISTILLED_INPROGRESS_CHECKPOINT = "distilled_inprogress.pt"
BOOTSTRAP_DISTILL_VALUE_SEMANTICS = "folded_baseline_live_zerosum_v1"
SHARED_VALIDATION_CACHE_DIR = (
    Path("outputs") / "preflop_backward_induction" / "validation_cache"
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


def _init_wandb(
    args: PreflopBucketExecutionConfig,
    cfg: Config,
    *,
    name: str,
    stage: str,
):
    run_name: str = str(args.wandb_name or name)
    group: str | None = None if args.wandb_group is None else str(args.wandb_group)
    return wandb_run(
        cfg,
        group=group,
        name=run_name,
        stage=stage,
        resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
    )


@torch.no_grad()
def _load_model_weights(
    trainer: RebelCFRTrainer,
    checkpoint_path: str,
    *,
    strict: bool | None = None,
) -> int:
    return CheckpointIO.load_model_weights(trainer, checkpoint_path, strict=strict)


def _checkpoint_model_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise ValueError(f"{checkpoint_path} does not contain a model state dict")
    return model_state


def _checkpoint_model_config(base_cfg: Config, checkpoint_path: str) -> Config:
    """Build a model config that matches a preflop checkpoint's tensor shapes."""

    state = _checkpoint_model_state(checkpoint_path)
    cfg = copy.deepcopy(base_cfg)
    street_embedding = state.get("policy_model.street_embedding.weight")
    if street_embedding is None:
        raise ValueError(
            f"Cannot infer preflop model dimensions from {checkpoint_path}"
        )

    cfg.model.hidden_dim = int(street_embedding.shape[1])
    is_transformer = any(key.endswith(".qkv.weight") for key in state)
    is_gated_token_mixer = any(".token_mixer." in key for key in state)
    if is_transformer or is_gated_token_mixer:
        hand_encoder = state.get("policy_model.hand_encoder.0.weight")
        ffn_in = state.get("policy_model.policy_encoder.0.ffn.linear_in.weight")
        if hand_encoder is None or ffn_in is None:
            raise ValueError(
                f"Cannot infer token preflop dimensions from {checkpoint_path}"
            )
        cfg.model.range_hidden_dim = int(hand_encoder.shape[0])
        cfg.model.ffn_dim = int(ffn_in.shape[0])
        cfg.model.num_hidden_layers = 0
        cfg.model.num_policy_layers = _checkpoint_encoder_depth(
            state, "policy_model.policy_encoder"
        )
        cfg.model.num_value_layers = _checkpoint_encoder_depth(
            state, "value_model.value_encoder"
        )
        cfg.model.preflop_model_type = (
            PreflopModelType.transformer
            if is_transformer
            else PreflopModelType.gated_token_mixer
        )
    else:
        class_embedding = state.get("policy_model.class_hi_embedding.weight")
        ffn_in = _first_checkpoint_tensor(
            state,
            (
                "policy_model.policy_tower.0.inner.linear_in.weight",
                "value_model.value_head.0.inner.linear_in.weight",
                "policy_model.trunk.0.inner.linear_in.weight",
            ),
        )
        if class_embedding is None or ffn_in is None:
            raise ValueError(
                f"Cannot infer compact FFN preflop dimensions from {checkpoint_path}"
            )
        cfg.model.range_hidden_dim = int(class_embedding.shape[1])
        cfg.model.ffn_dim = int(ffn_in.shape[0])
        cfg.model.num_hidden_layers = _checkpoint_sequential_depth(
            state, "policy_model.trunk"
        )
        cfg.model.num_policy_layers = _checkpoint_sequential_depth(
            state, "policy_model.policy_tower"
        )
        cfg.model.num_value_layers = _checkpoint_sequential_depth(
            state, "value_model.value_head"
        )
        cfg.model.preflop_model_type = PreflopModelType.ffn
    cfg.model.preflop_hand_dim = PREFLOP_HANDS
    cfg.model.board_interaction_dim = 0
    cfg.model.enforce_zero_sum = False
    cfg.model.preflop_range_slot_moment_slots = _checkpoint_range_slot_moment_slots(
        state
    )
    return cfg


def _checkpoint_range_slot_moment_slots(state: dict[str, torch.Tensor]) -> int:
    slots: list[int] = []
    for prefix in ("policy_model", "value_model"):
        slot_logits = state.get(f"{prefix}.range_slot_moment_pool.slot_logits")
        if slot_logits is not None:
            slots.append(int(slot_logits.shape[1]))
    if not slots:
        return 0
    first = slots[0]
    if any(value != first for value in slots):
        raise ValueError(
            "Checkpoint policy/value range pool slot counts do not match: "
            f"{slots}"
        )
    return first


def _first_checkpoint_tensor(
    state: dict[str, torch.Tensor],
    keys: tuple[str, ...],
) -> torch.Tensor | None:
    for key in keys:
        value = state.get(key)
        if value is not None:
            return value
    return None


def _checkpoint_encoder_depth(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> int:
    depth = 0
    marker = ".ffn.linear_in.weight"
    for key in state:
        if not key.startswith(prefix) or not key.endswith(marker):
            continue
        suffix = key[len(prefix) + 1 : -len(marker)]
        if suffix.isdigit():
            depth = max(depth, int(suffix) + 1)
    if depth == 0:
        raise ValueError(f"Cannot infer encoder depth for {prefix}")
    return depth


def _checkpoint_sequential_depth(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> int:
    depth = 0
    marker = ".inner.linear_in.weight"
    for key in state:
        if not key.startswith(prefix) or not key.endswith(marker):
            continue
        suffix = key[len(prefix) + 1 : -len(marker)]
        if suffix.isdigit():
            depth = max(depth, int(suffix) + 1)
    return depth


def _checkpoint_shapes_match_model(
    trainer: RebelCFRTrainer,
    checkpoint_path: str,
) -> bool:
    checkpoint_state = _checkpoint_model_state(checkpoint_path)
    model_state = trainer.model.state_dict()
    for key, checkpoint_value in checkpoint_state.items():
        model_value = model_state.get(key)
        if model_value is None:
            continue
        if tuple(checkpoint_value.shape) != tuple(model_value.shape):
            return False
    return True


def _maybe_load_student_weights(
    trainer: RebelCFRTrainer,
    checkpoint_path: str | None,
    *,
    required: bool,
) -> bool:
    if checkpoint_path is None:
        return False
    if not _checkpoint_shapes_match_model(trainer, checkpoint_path):
        if required:
            raise RuntimeError(
                f"student_init checkpoint is not shape-compatible with the student "
                f"model: {checkpoint_path}"
            )
        print(
            f"skipping student warm-start from incompatible checkpoint "
            f"{checkpoint_path}",
            flush=True,
        )
        return False
    _load_model_weights(trainer, checkpoint_path)
    return True


def _student_checkpoint_for_bucket(
    args: PreflopBucketExecutionConfig,
    *,
    bucket_index: int,
    previous_value_checkpoint: str,
) -> str | None:
    if bucket_index != 0:
        return previous_value_checkpoint
    if args.student_init is not None:
        return args.student_init
    if bool(args.student_init_from_base):
        return previous_value_checkpoint
    return None


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
    stats = trainer.supervise_policy_batch(
        batch,
        step=step,
        apply_schedules=True,
        sync_inference_model=True,
        sync_cfr_target_model=True,
    )
    return _float_metrics(stats)


def _evaluator_tree_stats(evaluator: CFREvaluator) -> dict[str, float]:
    root_nodes = int(evaluator.root_nodes)
    total_nodes = int(evaluator.total_nodes)
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


def _copy_checkpoint_alias(source: Path, alias: Path) -> None:
    if source == alias:
        return
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() or alias.is_symlink():
        alias.unlink()
    shutil.copy2(source, alias)


def _bucket_spec(label: str):
    for spec in BUCKET_SPECS:
        if spec.label == label:
            return spec
    raise KeyError(label)


def _train_bucket_labels(args: PreflopBucketExecutionConfig) -> tuple[str, ...]:
    if args.train_bucket is None:
        return BUCKET_ORDER_DEEP_TO_SHALLOW
    label = str(args.train_bucket)
    _bucket_spec(label)
    return (label,)


def _distill_bucket_labels(args: PreflopBucketExecutionConfig) -> tuple[str, ...]:
    labels = args.distill_buckets
    if labels is None:
        return BUCKET_ORDER_DEEP_TO_SHALLOW
    parsed = tuple(str(label) for label in labels)
    if not parsed:
        raise ValueError("preflop_buckets.distill_buckets must not be empty")
    for label in parsed:
        _bucket_spec(label)
    return parsed


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
    return max(_bucket_cfr_batch_size(args, label) for label in _train_bucket_labels(args))


def _estimate_train_updates(args: PreflopBucketExecutionConfig) -> int:
    return max(
        1,
        sum(
            _estimate_bucket_train_updates(args, bucket_label)
            for bucket_label in _train_bucket_labels(args)
        ),
    )


def _estimate_bucket_train_updates(
    args: PreflopBucketExecutionConfig,
    bucket_label: str,
) -> int:
    manifest = _load_state_manifest(
        Path(args.state_dataset), allow_partial=args.allow_partial
    )
    _, rows, _ = _bucket_shards(manifest, Path(args.state_dataset), bucket_label)
    rows = min(int(args.states_per_bucket), int(rows))
    cfr_batches = math.ceil(rows / _bucket_cfr_batch_size(args, bucket_label))
    return max(
        1,
        cfr_batches * _bucket_epochs(args, bucket_label)
        + _estimate_bootstrap_distill_updates(args, bucket_label, rows),
    )


def _bootstrap_distill_enabled(
    args: PreflopBucketExecutionConfig,
    bucket_label: str,
) -> bool:
    labels = _train_bucket_labels(args)
    return (
        args.bootstrap_distill_checkpoint is not None
        and int(args.bootstrap_distill_epochs) > 0
        and bool(labels)
        and bucket_label == labels[0]
    )


def _bootstrap_distill_batch_size(args: PreflopBucketExecutionConfig) -> int:
    if args.bootstrap_distill_batch_size is not None:
        return max(1, int(args.bootstrap_distill_batch_size))
    return max(1, int(args.distill_batch_size))


def _bootstrap_distill_schedule_roots(
    args: PreflopBucketExecutionConfig,
    bucket_label: str,
) -> int:
    return max(1, _bucket_cfr_batch_size(args, bucket_label))


def _estimate_bootstrap_distill_updates(
    args: PreflopBucketExecutionConfig,
    bucket_label: str,
    bucket_rows: int,
) -> int:
    if not _bootstrap_distill_enabled(args, bucket_label):
        return 0
    rows = int(bucket_rows)
    if args.bootstrap_distill_rows is not None:
        rows = min(rows, max(0, int(args.bootstrap_distill_rows)))
    if rows <= 0:
        return 0
    return (
        math.ceil(rows / _bootstrap_distill_schedule_roots(args, bucket_label))
        * max(1, int(args.bootstrap_distill_epochs))
    )


def _bootstrap_distill_student_metadata(cfg: Config) -> dict[str, Any]:
    fields = (
        "preflop_model_type",
        "preflop_hand_dim",
        "hidden_dim",
        "range_hidden_dim",
        "ffn_dim",
        "preflop_transformer_heads",
        "preflop_range_slot_moment_slots",
        "num_hidden_layers",
        "num_value_layers",
        "num_policy_layers",
        "board_interaction_dim",
        "street_value_heads",
        "enforce_zero_sum",
    )
    return {field: _jsonable(getattr(cfg.model, field, None)) for field in fields}


def _bootstrap_distill_checkpoint_path(bucket_dir: Path) -> Path:
    return bucket_dir / "checkpoints" / BOOTSTRAP_DISTILLED_CHECKPOINT


def _bootstrap_distill_expected_metadata(
    args: PreflopBucketExecutionConfig,
    *,
    bucket_label: str,
    cfg: Config,
    teacher_checkpoint: str,
    batch_size: int,
    max_rows: int,
    include_value: bool,
) -> dict[str, Any]:
    return {
        "kind": "preflop_backward_induction_bootstrap_distilled_model",
        "format_version": 1,
        "bucket_label": bucket_label,
        "state_dataset": os.path.realpath(args.state_dataset),
        "teacher_checkpoint": _checkpoint_signature(teacher_checkpoint),
        "bootstrap_distill_epochs": int(args.bootstrap_distill_epochs),
        "bootstrap_distill_rows": args.bootstrap_distill_rows,
        "bootstrap_distill_max_rows": int(max_rows),
        "bootstrap_distill_batch_size": int(batch_size),
        "bootstrap_distill_train_value": bool(args.bootstrap_distill_train_value),
        "include_value": bool(include_value),
        "belief_mode": str(args.belief_mode),
        "seed": int(args.seed),
        "student_num_steps": int(cfg.num_steps),
        "student_model": _bootstrap_distill_student_metadata(cfg),
        "value_target_semantics": BOOTSTRAP_DISTILL_VALUE_SEMANTICS,
        "policy_target_semantics": "teacher_masked_softmax_v1",
    }


def _metadata_matches_expected(
    metadata: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return all(metadata.get(key) == value for key, value in expected.items())


def _try_load_bootstrap_distill_checkpoint(
    trainer: RebelCFRTrainer,
    checkpoint_path: Path,
    expected_metadata: dict[str, Any],
) -> tuple[int, int, int] | None:
    if not checkpoint_path.exists():
        return None
    metadata = CheckpointIO.metadata(
        str(checkpoint_path),
        map_location=torch.device("cpu"),
    )
    if not _metadata_matches_expected(metadata, expected_metadata):
        print(
            f"ignoring stale bootstrap distill checkpoint {checkpoint_path}",
            flush=True,
        )
        return None
    step = _load_model_weights(trainer, str(checkpoint_path))
    fallback_step = max(0, int(step))
    global_step = int(metadata.get("global_step", fallback_step))
    bucket_train_step = int(metadata.get("bucket_train_step", fallback_step))
    roots = int(metadata.get("bootstrap_distill_roots", 0))
    print(
        f"loaded bootstrap distill checkpoint {checkpoint_path} "
        f"step={bucket_train_step} roots={roots:,}",
        flush=True,
    )
    return global_step, bucket_train_step, roots


def _validation_cache_metadata(
    args: PreflopBucketExecutionConfig,
    *,
    bucket_label: str,
    cutoff_checkpoint: str,
) -> dict[str, Any]:
    return {
        "kind": "preflop_backward_induction_validation_cache",
        "format_version": 2,
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


def _validation_cache_key(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(metadata), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validation_cache_filename(metadata: dict[str, Any]) -> str:
    cache_key = _validation_cache_key(metadata)
    return (
        f"validation_n{metadata['validation_items']}_"
        f"cfr{metadata['validation_cfr_iterations']}_{cache_key}.pt"
    )


def _validation_cache_path(metadata: dict[str, Any]) -> Path:
    return (
        SHARED_VALIDATION_CACHE_DIR
        / str(metadata["bucket_label"])
        / _validation_cache_filename(metadata)
    )


def _legacy_validation_cache_path(
    bucket_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    return bucket_dir / "validation" / _validation_cache_filename(metadata)


def _load_validation_cache(
    cache_path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cached.get("metadata") != metadata:
        return None
    print(f"loaded validation cache {cache_path}", flush=True)
    return cached


def _promote_validation_cache(source: Path, destination: Path) -> None:
    if source == destination or destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    print(f"promoted validation cache {source} -> {destination}", flush=True)


def _slice_batch(batch: RebelBatch, start: int, end: int) -> RebelBatch:
    return batch[slice(start, end)]


def _aggregate_minibatch_stats(
    weighted_stats: list[tuple[int, dict[str, Any]]],
) -> dict[str, float]:
    if not weighted_stats:
        return {}
    total_rows = sum(rows for rows, _ in weighted_stats)
    out: dict[str, float] = {
        "minibatch_updates": float(len(weighted_stats)),
        "minibatch_examples": float(total_rows),
    }
    keys = set().union(*(stats.keys() for _, stats in weighted_stats))
    for key in keys:
        values: list[tuple[int, float]] = []
        for rows, stats in weighted_stats:
            value = _float_metrics(stats).get(key)
            if value is not None:
                values.append((rows, value))
        if not values:
            continue
        if key in {"learning_rate", "step"}:
            out[key] = values[-1][1]
            continue
        denom = sum(rows for rows, _ in values)
        out[key] = sum(rows * value for rows, value in values) / max(1, denom)
    return out


def _train_value_minibatches(
    trainer: RebelCFRTrainer,
    batch: RebelBatch,
    *,
    step: int,
    batch_size: int,
) -> tuple[dict[str, float], int]:
    weighted_stats: list[tuple[int, dict[str, Any]]] = []
    schedule_step = int(step)
    for start in range(0, len(batch), batch_size):
        part = _slice_batch(batch, start, min(start + batch_size, len(batch)))
        stats = trainer.train_value_batch(
            part,
            schedule_step,
            sync_inference_model=True,
        )
        weighted_stats.append((len(part), stats))
    return _aggregate_minibatch_stats(weighted_stats), len(weighted_stats)


def _train_policy_minibatches(
    trainer: RebelCFRTrainer,
    batch: RebelBatch,
    *,
    step: int,
    batch_size: int,
) -> tuple[dict[str, float], int]:
    weighted_stats: list[tuple[int, dict[str, Any]]] = []
    schedule_step = int(step)
    for start in range(0, len(batch), batch_size):
        part = _slice_batch(batch, start, min(start + batch_size, len(batch)))
        stats = _policy_update(
            trainer,
            part,
            step=schedule_step,
        )
        weighted_stats.append((len(part), stats))
    return _aggregate_minibatch_stats(weighted_stats), len(weighted_stats)


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
            with trainer.model_autocast():
                output = trainer.model(
                    part.features,
                    include_policy=not include_value,
                    include_value=include_value,
                    apply_zero_sum=False,
                )
            loss_dict = (
                trainer.loss_fn._call_forward_value(output, part)
                if include_value
                else trainer.loss_fn._call_forward_policy(output, part)
            )
            if include_value:
                loss_dict = {
                    **loss_dict,
                    **_pot_relative_value_error_metrics(output, part, loss_dict),
                }
            metrics = _float_metrics(loss_dict)
            for key, value in metrics.items():
                if key.endswith("_all") or key.endswith("_weights"):
                    continue
                totals[key] = totals.get(key, 0.0) + value * rows
            total_rows += rows
    trainer.model.train()
    return {key: value / max(1, total_rows) for key, value in totals.items()}


def _pot_relative_value_error_metrics(
    output: Any,
    batch: RebelBatch,
    loss_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if output.hand_values is None or batch.value_targets is None:
        return {}
    pot = batch.statistics.get("pot")
    scale = batch.statistics.get("scale")
    if pot is None or scale is None:
        return {}

    corrected = loss_dict.get("value_predictions")
    predictions = (
        corrected.float()
        if isinstance(corrected, torch.Tensor)
        else output.hand_values.float()
    )
    targets = batch.value_targets.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )
    scale_tensor = scale.to(
        device=predictions.device,
        dtype=predictions.dtype,
    ).clamp_min(1.0)
    pot_scale = pot.to(
        device=predictions.device,
        dtype=predictions.dtype,
    ).clamp_min(1.0)
    while scale_tensor.ndim < predictions.ndim:
        scale_tensor = scale_tensor.unsqueeze(-1)
    while pot_scale.ndim < predictions.ndim:
        pot_scale = pot_scale.unsqueeze(-1)

    relative_abs_error = (predictions - targets).abs() * scale_tensor / pot_scale
    relative_sq_error = relative_abs_error.square()
    weights = loss_dict.get("value_weights")
    if isinstance(weights, torch.Tensor):
        value_weights = weights.to(device=predictions.device, dtype=predictions.dtype)
    else:
        value_weights = torch.ones_like(relative_abs_error)
    denom = value_weights.sum().clamp_min(1.0e-8)
    relative_mse = (relative_sq_error * value_weights).sum() / denom
    return {
        "pot_relative_mae": (relative_abs_error * value_weights).sum() / denom,
        "pot_relative_mse": relative_mse,
        "pot_relative_rmse": relative_mse.sqrt(),
    }


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
    cache_path = _validation_cache_path(metadata)
    legacy_cache_path = _legacy_validation_cache_path(bucket_dir, metadata)
    cached = _load_validation_cache(cache_path, metadata)
    if cached is not None:
        return cached
    cached = _load_validation_cache(legacy_cache_path, metadata)
    if cached is not None:
        _promote_validation_cache(legacy_cache_path, cache_path)
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


def _run_bootstrap_distill_for_bucket(
    *,
    args: PreflopBucketExecutionConfig,
    bucket_label: str,
    bucket_dir: Path,
    trainer: RebelCFRTrainer,
    cfg: Config,
    reader: PublicStateBucketReader,
    device: torch.device,
    rng: torch.Generator,
    run: Any,
    global_step: int,
    bucket_train_step: int,
    train_value: bool,
    bucket_index: int,
) -> tuple[int, int, int]:
    if not _bootstrap_distill_enabled(args, bucket_label):
        return global_step, bucket_train_step, 0
    checkpoint = args.bootstrap_distill_checkpoint
    if checkpoint is None:
        return global_step, bucket_train_step, 0

    batch_size = _bootstrap_distill_batch_size(args)
    max_rows = int(args.states_per_bucket)
    if args.bootstrap_distill_rows is not None:
        max_rows = min(max_rows, max(0, int(args.bootstrap_distill_rows)))
    if max_rows <= 0:
        return global_step, bucket_train_step, 0

    schedule_roots = _bootstrap_distill_schedule_roots(args, bucket_label)
    include_value = bool(args.bootstrap_distill_train_value) and train_value
    checkpoint_path = _bootstrap_distill_checkpoint_path(bucket_dir)
    expected_metadata = _bootstrap_distill_expected_metadata(
        args,
        bucket_label=bucket_label,
        cfg=cfg,
        teacher_checkpoint=checkpoint,
        batch_size=batch_size,
        max_rows=max_rows,
        include_value=include_value,
    )
    cached = _try_load_bootstrap_distill_checkpoint(
        trainer,
        checkpoint_path,
        expected_metadata,
    )
    if cached is not None:
        return cached

    teacher_cfg = _checkpoint_model_config(cfg, checkpoint)
    teacher_cfg.num_envs = batch_size
    teacher_cfg.num_steps = max(
        1,
        _estimate_bootstrap_distill_updates(args, bucket_label, reader.rows),
    )
    teacher = RebelCFRTrainer(
        cfg=copy.deepcopy(teacher_cfg),
        device=device,
        pregeneration_only=True,
    )
    _load_model_weights(teacher, checkpoint)
    env = _make_env_from_manifest(
        reader.manifest,
        num_envs=batch_size,
        device=device,
        seed=args.seed + bucket_index + 50_000,
    )
    epochs = max(1, int(args.bootstrap_distill_epochs))
    roots_total = 0
    print(
        f"{bucket_label}: bootstrap_distill start checkpoint={checkpoint} "
        f"epochs={epochs} max_rows={max_rows:,} batch={batch_size} "
        f"schedule_roots={schedule_roots:,} "
        f"include_value={include_value}",
        flush=True,
    )
    for epoch in range(epochs):
        epoch_roots = 0
        step_roots = 0
        step_microbatches = 0
        step_value_minibatches = 0
        step_policy_minibatches = 0
        step_value_stats: list[tuple[int, dict[str, Any]]] = []
        step_policy_stats: list[tuple[int, dict[str, Any]]] = []

        def flush_step() -> None:
            nonlocal global_step
            nonlocal bucket_train_step
            nonlocal step_roots
            nonlocal step_microbatches
            nonlocal step_value_minibatches
            nonlocal step_policy_minibatches
            nonlocal step_value_stats
            nonlocal step_policy_stats
            if step_roots <= 0:
                return
            value_stats = _aggregate_minibatch_stats(step_value_stats)
            policy_stats = _aggregate_minibatch_stats(step_policy_stats)
            global_step += 1
            bucket_train_step += 1
            payload = {
                f"{bucket_label}/bootstrap_distill_epoch": epoch + 1,
                f"{bucket_label}/bootstrap_distill_roots": roots_total,
                f"{bucket_label}/bootstrap_distill_epoch_roots": epoch_roots,
                f"{bucket_label}/bootstrap_distill_step_roots": step_roots,
                f"{bucket_label}/bootstrap_distill_microbatches": step_microbatches,
                f"{bucket_label}/bootstrap_distill_value_minibatches": step_value_minibatches,
                f"{bucket_label}/bootstrap_distill_policy_minibatches": step_policy_minibatches,
                f"{bucket_label}/bucket_train_step": bucket_train_step,
                f"{bucket_label}/global_step": global_step,
                "global_step": global_step,
            }
            payload.update(
                _prefixed_metrics(
                    f"{bucket_label}/bootstrap_distill", "value", value_stats
                )
            )
            payload.update(
                _prefixed_metrics(
                    f"{bucket_label}/bootstrap_distill", "policy", policy_stats
                )
            )
            if run is not None:
                run.log(payload, step=global_step)
            step_roots = 0
            step_microbatches = 0
            step_value_minibatches = 0
            step_policy_minibatches = 0
            step_value_stats = []
            step_policy_stats = []

        for states in reader.iter_state_batches(
            batch_size=batch_size,
            max_rows=max_rows,
            seed=_seed_for_label(
                args.seed,
                bucket_label,
                salt=500_000 + epoch,
            ),
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
            schedule_step = bucket_train_step
            if value_batch is not None:
                value_stats, value_minibatches = _train_value_minibatches(
                    trainer,
                    value_batch,
                    step=schedule_step,
                    batch_size=max(1, int(args.train_batch_size)),
                )
            else:
                value_stats = {}
                value_minibatches = 0
            policy_stats, policy_minibatches = _train_policy_minibatches(
                trainer,
                policy_batch,
                step=schedule_step,
                batch_size=max(1, int(args.train_batch_size)),
            )
            roots_total += rows
            epoch_roots += rows
            step_roots += rows
            step_microbatches += 1
            step_value_minibatches += value_minibatches
            step_policy_minibatches += policy_minibatches
            if value_batch is not None:
                step_value_stats.append((rows, value_stats))
            step_policy_stats.append((rows, policy_stats))
            if step_roots >= schedule_roots:
                flush_step()
        flush_step()
        print(
            f"{bucket_label}: bootstrap_distill epoch_complete "
            f"epoch={epoch + 1}/{epochs} roots={roots_total:,} "
            f"bucket_train_step={bucket_train_step}",
            flush=True,
        )

    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _save_trainer_checkpoint(
        trainer,
        checkpoint_path,
        step=bucket_train_step,
        run_id=None if run is None else run.id,
        metadata={
            **expected_metadata,
            "snapshot": "after_bootstrap_distill",
            "global_step": int(global_step),
            "bucket_train_step": int(bucket_train_step),
            "bootstrap_distill_roots": int(roots_total),
        },
    )
    if run is not None:
        run.summary[f"{bucket_label}/bootstrap_distill_checkpoint"] = str(
            checkpoint_path
        )
    print(
        f"{bucket_label}: saved bootstrap distill checkpoint {checkpoint_path} "
        f"step={bucket_train_step} roots={roots_total:,}",
        flush=True,
    )
    return global_step, bucket_train_step, roots_total


def run_presolve_values(
    args: PreflopBucketExecutionConfig,
    *,
    base_template: Config,
) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    bucket_label = str(args.presolve_bucket)
    spec = _bucket_spec(bucket_label)
    cfr_batch_size = _bucket_cfr_batch_size(args, bucket_label)
    output_dir = Path(args.output_dir)
    bucket_dir = output_dir / bucket_label
    solved_dir = bucket_dir / "solved"
    if solved_dir.exists() and any(solved_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{solved_dir} exists; pass overwrite=true")
        shutil.rmtree(solved_dir)
    solved_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_run_config(
        base_template,
        args,
        checkpoint_dir=bucket_dir / "checkpoints",
        num_steps=1,
        num_envs=cfr_batch_size,
    )
    write_resolved_config(
        cfg,
        output_dir,
        resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
    )
    run_cm = _init_wandb(
        args,
        cfg,
        name=f"preflop-bi-presolve-values-{bucket_label}-{_now_slug()}",
        stage="preflop_bucket_presolve_values",
    )
    reader = PublicStateBucketReader(
        args.state_dataset,
        bucket_label,
        allow_partial=args.allow_partial,
        seed=args.seed,
    )
    solver = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device, pregeneration_only=True)
    _load_model_weights(solver, args.base_checkpoint)
    env = _make_env_from_manifest(
        reader.manifest,
        num_envs=cfr_batch_size,
        device=device,
        seed=args.seed + 100,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(_seed_for_label(args.seed, bucket_label, salt=500_000))
    writer = RebelSolvedDatasetWriter(
        solved_dir,
        storage_float_dtype=args.storage_dtype,
    )
    roots_solved = 0
    value_examples = 0
    solve_batches = 0
    total_nodes_sum = 0.0
    root_nodes_sum = 0.0
    max_total_nodes = 0.0
    progress_interval = max(int(args.progress_roots), cfr_batch_size)
    next_progress_roots = progress_interval
    start_time = time.time()

    with run_cm as run:
        for states in reader.iter_state_batches(
            batch_size=cfr_batch_size,
            max_rows=args.states_per_bucket,
            seed=_seed_for_label(args.seed, bucket_label, salt=600_000),
        ):
            rows = _copy_public_states_to_env(env, states)
            beliefs = _random_beliefs(
                rows,
                solver.num_players,
                device=device,
                rng=rng,
                mode=args.belief_mode,
            )
            value_batch, _, tree_stats = _solve_public_state_batch(
                solver,
                env,
                beliefs,
                include_policy=False,
            )
            value_batch = _filter_batch_by_action_bucket(
                value_batch,
                low=spec.low,
                high=spec.high,
            )
            value_stream = _value_only(value_batch)
            if value_stream is not None:
                writer.append("value", value_stream)
                value_examples += len(value_stream)
            roots_solved += rows
            solve_batches += 1
            total_nodes = float(tree_stats["evaluator_total_nodes"])
            root_nodes = float(tree_stats["evaluator_root_nodes"])
            total_nodes_sum += total_nodes
            root_nodes_sum += root_nodes
            max_total_nodes = max(max_total_nodes, total_nodes)
            elapsed = time.time() - start_time
            if run is not None:
                run.log(
                    {
                        f"{bucket_label}/roots_solved": roots_solved,
                        f"{bucket_label}/value_examples": value_examples,
                        f"{bucket_label}/solve_batches": solve_batches,
                        f"{bucket_label}/cfr_batch_size": cfr_batch_size,
                        f"{bucket_label}/elapsed_s": elapsed,
                        f"{bucket_label}/roots_per_s": roots_solved
                        / max(elapsed, 1.0e-9),
                        f"{bucket_label}/evaluator_total_nodes": total_nodes,
                        f"{bucket_label}/evaluator_root_nodes": root_nodes,
                        "roots_solved": roots_solved,
                    },
                    step=roots_solved,
                )
            if roots_solved >= next_progress_roots:
                print(
                    f"{bucket_label}: presolve progress roots={roots_solved:,} "
                    f"value={value_examples:,} nodes={int(total_nodes):,} "
                    f"cfr_batch={cfr_batch_size} roots/s={roots_solved / max(elapsed, 1.0e-9):.2f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                while next_progress_roots <= roots_solved:
                    next_progress_roots += progress_interval
            if roots_solved >= args.states_per_bucket:
                break

        if value_examples == 0:
            raise RuntimeError(f"No value examples produced for {bucket_label}")
        summary = {
            "format_note": "preflop backward-induction value-only presolve bucket",
            "bucket_label": bucket_label,
            "bucket_low": spec.low,
            "bucket_high": spec.high,
            "root_states_solved": roots_solved,
            "value_examples": value_examples,
            "policy_examples": 0,
            "source_state_dataset": os.path.realpath(args.state_dataset),
            "solver_checkpoint": os.path.realpath(args.base_checkpoint),
            "depth": args.depth,
            "cfr_iterations": args.cfr_iterations,
            "cfr_batch_size": cfr_batch_size,
            "belief_mode": args.belief_mode,
            "storage_dtype": args.storage_dtype,
            "solve_batches": solve_batches,
            "evaluator_total_nodes_mean": total_nodes_sum / max(1, solve_batches),
            "evaluator_total_nodes_max": max_total_nodes,
            "evaluator_root_nodes_mean": root_nodes_sum / max(1, solve_batches),
            "evaluator_nodes_per_root_mean": total_nodes_sum / max(1.0, root_nodes_sum),
        }
        manifest = writer.finalize(summary)
        summary["solved_dataset"] = manifest
        summary["solved_manifest"] = str(solved_dir / "manifest.json")
        summary_path = bucket_dir / "presolve_summary.json"
        summary_path.write_text(
            json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
        )
        if run is not None:
            run.summary[f"{bucket_label}/roots_solved"] = roots_solved
            run.summary[f"{bucket_label}/value_examples"] = value_examples
            run.summary[f"{bucket_label}/solved_manifest"] = str(
                solved_dir / "manifest.json"
            )
    print(
        f"completed {bucket_label} value presolve: manifest={solved_dir / 'manifest.json'} "
        f"roots={roots_solved:,} value={value_examples:,}",
        flush=True,
    )


def run_train_specialists(
    args: PreflopBucketExecutionConfig,
    *,
    base_template: Config,
) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_updates_guess = _estimate_train_updates(args)
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
        stage="preflop_bucket_specialists",
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))

    with run_cm as run:
        previous_value_checkpoint = args.base_checkpoint
        global_step = 0
        specialist_paths: dict[str, str] = {}
        for bucket_index, bucket_label in enumerate(_train_bucket_labels(args)):
            spec = _bucket_spec(bucket_label)
            train_value = bucket_label != "actions_0_3"
            cfr_batch_size = _bucket_cfr_batch_size(args, bucket_label)
            bucket_updates_guess = _estimate_bucket_train_updates(args, bucket_label)
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
                num_steps=bucket_updates_guess,
                num_envs=cfr_batch_size,
            )
            write_resolved_config(
                cfg,
                resolved_config=RebelExperimentConfig.from_trainer_config(cfg),
            )
            cutoff_cfg = _checkpoint_model_config(cfg, previous_value_checkpoint)
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
                cfg=cutoff_cfg,
                reader=reader,
                device=device,
                bucket_index=bucket_index,
            )
            solver = RebelCFRTrainer(
                cfg=copy.deepcopy(cutoff_cfg), device=device, pregeneration_only=True
            )
            _load_model_weights(solver, previous_value_checkpoint)
            trainer = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device)
            student_checkpoint = _student_checkpoint_for_bucket(
                args,
                bucket_index=bucket_index,
                previous_value_checkpoint=previous_value_checkpoint,
            )
            _maybe_load_student_weights(
                trainer,
                student_checkpoint,
                required=args.student_init is not None and bucket_index == 0,
            )
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
            policy_examples = 0
            bucket_step = 0
            bucket_train_step = 0
            bucket_epochs = _bucket_epochs(args, bucket_label)
            bucket_start = time.time()
            train_batch_size = max(1, int(args.train_batch_size))
            progress_interval = max(int(args.progress_roots), cfr_batch_size)
            next_progress_roots = progress_interval
            last_tree_stats: dict[str, float] = {}
            (
                global_step,
                bucket_train_step,
                bootstrap_distill_roots,
            ) = _run_bootstrap_distill_for_bucket(
                args=args,
                bucket_label=bucket_label,
                bucket_dir=bucket_dir,
                trainer=trainer,
                cfg=cfg,
                reader=reader,
                device=device,
                rng=rng,
                run=run,
                global_step=global_step,
                bucket_train_step=bucket_train_step,
                train_value=train_value,
                bucket_index=bucket_index,
            )
            bucket_start = time.time()

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
                    f"{bucket_label}/bucket_train_step": bucket_train_step,
                    f"{bucket_label}/epoch": epoch,
                    f"{bucket_label}/epoch_roots": epoch_roots,
                    f"{bucket_label}/roots_solved": roots_solved,
                    f"{bucket_label}/global_step": global_step,
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
                    f"bucket_train_step={bucket_train_step} "
                    f"train_batch={train_batch_size} "
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
                    schedule_step = bucket_train_step
                    trained_this_step = False
                    if value_stream is not None:
                        if writer is not None:
                            writer.append("value", value_stream)
                        value_examples += len(value_stream)
                        value_stats, value_minibatches = _train_value_minibatches(
                            trainer,
                            value_stream,
                            step=schedule_step,
                            batch_size=train_batch_size,
                        )
                        trained_this_step = True
                    else:
                        value_stats = {}
                        value_minibatches = 0
                    if policy_stream is not None:
                        if writer is not None:
                            writer.append("policy", policy_stream)
                        policy_examples += len(policy_stream)
                        policy_stats, policy_minibatches = _train_policy_minibatches(
                            trainer,
                            policy_stream,
                            step=schedule_step,
                            batch_size=train_batch_size,
                        )
                        trained_this_step = True
                    else:
                        policy_stats = {}
                        policy_minibatches = 0
                    if trained_this_step:
                        global_step += 1
                        bucket_train_step += 1

                    roots_solved += rows
                    epoch_roots += rows
                    bucket_step += 1
                    elapsed = time.time() - bucket_start
                    log_payload = {
                        f"{bucket_label}/roots_solved": roots_solved,
                        f"{bucket_label}/epoch": epoch,
                        f"{bucket_label}/epoch_roots": epoch_roots,
                        f"{bucket_label}/bucket_step": bucket_step,
                        f"{bucket_label}/bucket_train_step": bucket_train_step,
                        f"{bucket_label}/bucket_schedule_steps": bucket_updates_guess,
                        f"{bucket_label}/cfr_batch_size": cfr_batch_size,
                        f"{bucket_label}/train_batch_size": train_batch_size,
                        f"{bucket_label}/global_step": global_step,
                        f"{bucket_label}/value_examples": value_examples,
                        f"{bucket_label}/policy_examples": policy_examples,
                        f"{bucket_label}/value_minibatches": value_minibatches,
                        f"{bucket_label}/policy_minibatches": policy_minibatches,
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
                    if (
                        args.snapshot_interval_steps > 0
                        and trained_this_step
                        and bucket_step % int(args.snapshot_interval_steps) == 0
                    ):
                        snapshot_path = (
                            bucket_dir
                            / "checkpoints"
                            / SPECIALIST_INPROGRESS_CHECKPOINT
                        )
                        _save_trainer_checkpoint(
                            trainer,
                            snapshot_path,
                            step=bucket_train_step,
                            run_id=None if run is None else run.id,
                            metadata={
                                "kind": "preflop_backward_induction_specialist",
                                "snapshot": "in_progress",
                                "bucket_label": bucket_label,
                                "bucket_step": bucket_step,
                                "bucket_train_step": bucket_train_step,
                                "global_step": global_step,
                                "epoch": epoch,
                                "epoch_roots": epoch_roots,
                                "roots_solved": roots_solved,
                                "value_examples": value_examples,
                                "policy_examples": policy_examples,
                                "train_value": train_value,
                                "solver_checkpoint": os.path.realpath(
                                    previous_value_checkpoint
                                ),
                            },
                        )
                        if run is not None:
                            run.summary[
                                f"{bucket_label}/inprogress_checkpoint"
                            ] = str(snapshot_path)
                        print(
                            f"{bucket_label}: saved in-progress checkpoint "
                            f"{snapshot_path} at bucket_step={bucket_step} "
                            f"roots={roots_solved:,}",
                            flush=True,
                        )
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
                "bootstrap_distill_checkpoint": args.bootstrap_distill_checkpoint,
                "bootstrap_distill_epochs": int(args.bootstrap_distill_epochs),
                "bootstrap_distill_roots": bootstrap_distill_roots,
                "bootstrap_distilled_checkpoint": str(
                    _bootstrap_distill_checkpoint_path(bucket_dir)
                ),
                "bucket_train_steps": bucket_train_step,
                "bucket_schedule_steps": bucket_updates_guess,
                "global_updates_at_finish": global_step,
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
            checkpoint_path = (
                bucket_dir / "checkpoints" / SPECIALIST_FINAL_CHECKPOINT
            )
            _save_trainer_checkpoint(
                trainer,
                checkpoint_path,
                step=bucket_train_step,
                run_id=None if run is None else run.id,
                metadata={
                    "kind": "preflop_backward_induction_specialist",
                    "bucket_label": bucket_label,
                    "train_value": train_value,
                    "training_summary": bucket_summary,
                    "bucket_train_step": bucket_train_step,
                    "global_step": global_step,
                    "solved_manifest": bucket_summary.get("solved_manifest"),
                    "solver_checkpoint": os.path.realpath(previous_value_checkpoint),
                },
            )
            _copy_checkpoint_alias(
                checkpoint_path,
                bucket_dir / "checkpoints" / LEGACY_LATEST_CHECKPOINT,
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
            value_targets = _postprocess_distilled_value_targets(
                output.hand_values,
                beliefs,
                env,
            )
            value_batch = RebelBatch(
                features=features,
                legal_masks=legal,
                value_targets=value_targets.detach().clone(),
                statistics=statistics,
            )
    return value_batch, policy_batch


def _postprocess_distilled_value_targets(
    hand_values: torch.Tensor,
    beliefs: torch.Tensor,
    env: PBSEnv,
) -> torch.Tensor:
    values = hand_values.float()
    rows = values.shape[0]
    folded = getattr(env, "has_folded", None)
    if folded is None:
        live_mask = torch.ones(
            values.shape[:2],
            dtype=torch.bool,
            device=values.device,
        )
    else:
        folded_mask = folded[:rows].to(device=values.device, dtype=torch.bool)
        live_mask = ~folded_mask
        scale = env.scale[:rows].to(device=values.device, dtype=torch.float32)
        scale = scale.clamp_min(1.0)
        folded_values = (
            env.stacks[:rows].to(device=values.device, dtype=torch.float32)
            - env.starting_stacks[:rows].to(device=values.device, dtype=torch.float32)
        ) / scale[:, None]
        folded_values = folded_values[:, :, None].expand_as(values)
        values = torch.where(folded_mask[:, :, None], folded_values, values)

    belief_values = beliefs[:rows].to(device=values.device, dtype=values.dtype)
    live_weight = live_mask[:, :, None].to(dtype=values.dtype)
    denom = (belief_values * live_weight).sum(dim=(1, 2), keepdim=True)
    expected_sum = (values * belief_values).sum(dim=(1, 2), keepdim=True)
    correction = expected_sum / denom.clamp_min(1.0e-12)
    return torch.where(live_mask[:, :, None], values - correction, values)


def run_distill(
    args: PreflopBucketExecutionConfig,
    *,
    base_template: Config,
) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_updates = max(
        1,
        math.ceil(args.states_per_bucket / max(1, args.distill_batch_size))
        * len(_distill_bucket_labels(args)),
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
    _maybe_load_student_weights(
        student,
        args.student_init or args.base_checkpoint,
        required=args.student_init is not None,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))
    run_cm = _init_wandb(
        args,
        cfg,
        name=f"preflop-bi-distill-{_now_slug()}",
        stage="preflop_bucket_distill",
    )

    with run_cm as run:
        if run is not None:
            run.summary.update(count_model_parameters(student.model))
        global_step = 0
        for bucket_label in _distill_bucket_labels(args):
            checkpoints = {
                "actions_12_15": args.checkpoint_12_15,
                "actions_8_11": args.checkpoint_8_11,
                "actions_4_7": args.checkpoint_4_7,
                "actions_0_3": args.checkpoint_0_3,
            }
            checkpoint = checkpoints[bucket_label]
            if checkpoint is None:
                raise ValueError(f"missing specialist checkpoint for {bucket_label}")
            include_value = (
                bool(args.distill_train_value) and bucket_label != "actions_0_3"
            )
            teacher_cfg = _checkpoint_model_config(cfg, checkpoint)
            teacher_cfg.checkpoint_dir = str(output_dir / "teacher_tmp")
            teacher_cfg.num_steps = total_updates
            teacher_cfg.num_envs = int(args.distill_batch_size)
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
                step_before_updates = global_step
                trained_this_step = False
                if value_batch is not None:
                    value_stats = student.train_value_batch(
                        value_batch,
                        step_before_updates,
                        sync_inference_model=True,
                    )
                    trained_this_step = True
                else:
                    value_stats = {}
                policy_stats = _policy_update(
                    student, policy_batch, step=step_before_updates
                )
                trained_this_step = True
                if trained_this_step:
                    global_step += 1
                roots += rows
                payload = {
                    f"distill/{bucket_label}/roots": roots,
                    f"distill/{bucket_label}/global_step": global_step,
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
                if (
                    args.snapshot_interval_steps > 0
                    and trained_this_step
                    and global_step % int(args.snapshot_interval_steps) == 0
                ):
                    snapshot_path = (
                        output_dir
                        / "checkpoints"
                        / DISTILLED_INPROGRESS_CHECKPOINT
                    )
                    _save_trainer_checkpoint(
                        student,
                        snapshot_path,
                        step=global_step,
                        run_id=None if run is None else run.id,
                        metadata={
                            "kind": "preflop_backward_induction_distilled_model",
                            "snapshot": "in_progress",
                            "current_bucket": bucket_label,
                            "current_bucket_roots": roots,
                            "global_step": global_step,
                            "state_dataset": os.path.realpath(args.state_dataset),
                            "distill_buckets": list(_distill_bucket_labels(args)),
                            "distill_train_value": bool(args.distill_train_value),
                        },
                    )
                    if run is not None:
                        run.summary["distill/inprogress_checkpoint"] = str(
                            snapshot_path
                        )
                    print(
                        f"distill: saved in-progress checkpoint {snapshot_path} "
                        f"at global_step={global_step} bucket={bucket_label} "
                        f"roots={roots:,}",
                        flush=True,
                    )
                if roots >= args.states_per_bucket:
                    break
            print(f"distilled {bucket_label}: roots={roots:,}", flush=True)

        checkpoint_path = output_dir / "checkpoints" / DISTILLED_FINAL_CHECKPOINT
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
                "distill_buckets": list(_distill_bucket_labels(args)),
                "distill_train_value": bool(args.distill_train_value),
            },
        )
        _copy_checkpoint_alias(
            checkpoint_path,
            output_dir / "checkpoints" / LEGACY_LATEST_CHECKPOINT,
        )
        print(f"saved distilled model: {checkpoint_path}", flush=True)
