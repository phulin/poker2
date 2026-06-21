#!/usr/bin/env python3
"""Backward-induction training and distillation for preflop depth buckets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
import wandb

from p2.core.structured_config import Config
from p2.env.card_utils import PREFLOP_HANDS, preflop_class_multiplicity_tensor
from p2.env.pbs_env import PBSEnv
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.search.rebel_solved_dataset import RebelSolvedDatasetWriter
from p2.utils.model_utils import compute_masked_logits, count_model_parameters


DEFAULT_CHECKPOINT = (
    "/home/user/poker2/checkpoints-rebel-curriculum-preflop_2000_p6_lr0p01_"
    "backupcons_actor_lam01_rb32_from2p_norb/preflop/rebel_latest.pt"
)
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


def _init_wandb(args: argparse.Namespace, cfg: Config, *, name: str):
    if not args.use_wandb:
        return nullcontext()
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name or name,
        "group": args.wandb_group,
        "tags": list(args.wandb_tags),
        "config": {
            "args": vars(args),
            "trainer_config": _jsonable(asdict(cfg)),
        },
    }
    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"W&B init failed ({exc}); continuing without W&B.", flush=True)
        return nullcontext()


def _load_checkpoint_config(
    checkpoint_path: str,
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    num_steps: int,
) -> Config:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(copy.deepcopy(checkpoint["config"]))
    cfg.device = args.device
    cfg.num_envs = int(args.cfr_batch_size)
    cfg.num_steps = max(1, int(num_steps))
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.use_wandb = bool(args.use_wandb)
    cfg.wandb_project = args.wandb_project
    cfg.wandb_name = args.wandb_name
    cfg.wandb_tags = list(args.wandb_tags)
    cfg.resume_from = None
    cfg.data.mode = "live"
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.data.include_pre_chance_value_batches = False
    cfg.train.batch_size = int(args.train_batch_size)
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = max(1, int(args.replay_buffer_batches))
    cfg.train.save_replay_buffers = False
    cfg.search.depth = int(args.depth)
    cfg.search.iterations = int(args.cfr_iterations)
    cfg.search.iterations_final = None
    cfg.search.warm_start_iterations = int(args.warm_start_iterations)
    cfg.search.sparse = True
    cfg.search.sparse_fused = bool(args.sparse_fused)
    if args.compile is not None:
        cfg.model.compile = args.compile
    return cfg


@torch.no_grad()
def _load_model_weights(
    trainer: RebelCFRTrainer,
    checkpoint_path: str,
    *,
    strict: bool | None = None,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
    model_state = checkpoint["model"]
    save_dtype = checkpoint.get("save_dtype")
    if save_dtype is not None and save_dtype != str(trainer.float_dtype):
        model_state = {
            key: value.to(trainer.float_dtype) if value.dtype.is_floating_point else value
            for key, value in model_state.items()
        }
    if checkpoint.get("model_component") == "value_model":
        value_model = getattr(trainer.model, "value_model", trainer.model)
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


class PublicStateBucketReader:
    def __init__(self, root: str | Path, bucket_label: str, *, allow_partial: bool):
        self.root = Path(root)
        self.manifest = _load_state_manifest(self.root, allow_partial=allow_partial)
        self.bucket_label = bucket_label
        self.bucket_dir, self.rows, self.shards = _bucket_shards(
            self.manifest, self.root, bucket_label
        )

    def iter_state_batches(
        self,
        *,
        batch_size: int,
        max_rows: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        remaining = min(int(max_rows), int(self.rows))
        if remaining <= 0:
            return
        for shard in self.shards:
            if remaining <= 0:
                break
            shard_path = self.bucket_dir / str(shard.get("path", shard.get("file")))
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            states = payload["states"]
            shard_rows = int(payload.get("num_rows", next(iter(states.values())).shape[0]))
            cursor = 0
            while cursor < shard_rows and remaining > 0:
                take = min(int(batch_size), shard_rows - cursor, remaining)
                yield {
                    name: tensor[cursor : cursor + take]
                    for name, tensor in states.items()
                }
                cursor += take
                remaining -= take


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
        return prior.view(1, 1, PREFLOP_HANDS).expand(
            rows, num_players, PREFLOP_HANDS
        ).clone()
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


def _solve_public_state_batch(
    solver: RebelCFRTrainer,
    env: PBSEnv,
    beliefs: torch.Tensor,
    *,
    include_policy: bool,
) -> tuple[RebelBatch | None, RebelBatch | None]:
    root_indices = torch.arange(beliefs.shape[0], device=solver.device)
    evaluator = solver.cfr_evaluator
    evaluator.initialize_subgame(env, root_indices, beliefs)
    evaluator.evaluate_cfr(training_mode=True, sample_continuation=False)
    value_batch, _, policy_batch = evaluator.training_data(
        include_pre_chance_value_batch=False,
        include_policy_batch=include_policy,
    )
    return value_batch, policy_batch


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


def run_train_specialists(args: argparse.Namespace) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_updates_guess = max(
        1,
        math.ceil(args.states_per_bucket / max(1, args.cfr_batch_size))
        * len(BUCKET_ORDER_DEEP_TO_SHALLOW)
        * 2,
    )
    base_cfg = _load_checkpoint_config(
        args.base_checkpoint,
        args=args,
        checkpoint_dir=output_dir / "checkpoints",
        num_steps=total_updates_guess,
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

            cfg = _load_checkpoint_config(
                previous_value_checkpoint,
                args=args,
                checkpoint_dir=bucket_dir / "checkpoints",
                num_steps=total_updates_guess,
            )
            solver = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device, pregeneration_only=True)
            _load_model_weights(solver, previous_value_checkpoint)
            trainer = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device)
            _load_model_weights(trainer, previous_value_checkpoint)
            if bucket_index == 0 and isinstance(run, wandb.Run):
                run.summary.update(count_model_parameters(trainer.model))

            reader = PublicStateBucketReader(
                args.state_dataset,
                bucket_label,
                allow_partial=args.allow_partial,
            )
            env = _make_env_from_manifest(
                reader.manifest,
                num_envs=args.cfr_batch_size,
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
            bucket_start = time.time()

            for states in reader.iter_state_batches(
                batch_size=args.cfr_batch_size,
                max_rows=args.states_per_bucket,
            ):
                rows = _copy_public_states_to_env(env, states)
                beliefs = _random_beliefs(
                    rows,
                    solver.num_players,
                    device=device,
                    rng=rng,
                    mode=args.belief_mode,
                )
                value_batch, policy_batch = _solve_public_state_batch(
                    solver,
                    env,
                    beliefs,
                    include_policy=True,
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
                elapsed = time.time() - bucket_start
                log_payload = {
                    f"{bucket_label}/roots_solved": roots_solved,
                    f"{bucket_label}/value_examples": value_examples,
                    f"{bucket_label}/policy_examples": policy_examples,
                    f"{bucket_label}/elapsed_s": elapsed,
                    f"{bucket_label}/roots_per_s": roots_solved / max(elapsed, 1.0e-9),
                    "global_step": global_step,
                }
                log_payload.update(
                    {
                        f"{bucket_label}/value_{key}": value
                        for key, value in _float_metrics(value_stats).items()
                    }
                )
                log_payload.update(
                    {
                        f"{bucket_label}/policy_{key}": value
                        for key, value in policy_stats.items()
                    }
                )
                if isinstance(run, wandb.Run):
                    run.log(log_payload, step=global_step)
                if roots_solved % max(args.progress_roots, args.cfr_batch_size) == 0:
                    print(
                        f"{bucket_label}: roots={roots_solved:,} "
                        f"value={value_examples:,} policy={policy_examples:,} "
                        f"step={global_step} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                if roots_solved >= args.states_per_bucket:
                    break

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
                run_id=run.id if isinstance(run, wandb.Run) else None,
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
            if isinstance(run, wandb.Run):
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
        if isinstance(run, wandb.Run):
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
    features = encoder.encode(beliefs, indices=torch.arange(beliefs.shape[0], device=teacher.device))
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


def run_distill(args: argparse.Namespace) -> None:
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
    cfg = _load_checkpoint_config(
        args.base_checkpoint,
        args=args,
        checkpoint_dir=output_dir / "checkpoints",
        num_steps=total_updates,
    )
    student = RebelCFRTrainer(cfg=copy.deepcopy(cfg), device=device)
    _load_model_weights(student, args.student_init or args.base_checkpoint)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))
    run_cm = _init_wandb(args, cfg, name=f"preflop-bi-distill-{_now_slug()}")

    with run_cm as run:
        if isinstance(run, wandb.Run):
            run.summary.update(count_model_parameters(student.model))
        global_step = 0
        for bucket_label in BUCKET_ORDER_DEEP_TO_SHALLOW:
            checkpoint = getattr(args, bucket_label.replace("actions_", "checkpoint_"))
            if checkpoint is None:
                raise ValueError(f"missing specialist checkpoint for {bucket_label}")
            include_value = bucket_label != "actions_0_3"
            teacher_cfg = _load_checkpoint_config(
                checkpoint,
                args=args,
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
                if value_batch is not None:
                    value_stats = student.train_value_batch(
                        value_batch,
                        global_step,
                        sync_inference_model=True,
                    )
                    global_step += 1
                else:
                    value_stats = {}
                policy_stats = _policy_update(student, policy_batch, step=global_step)
                global_step += 1
                roots += rows
                payload = {
                    f"distill/{bucket_label}/roots": roots,
                    "global_step": global_step,
                }
                payload.update(
                    {
                        f"distill/{bucket_label}/value_{key}": value
                        for key, value in _float_metrics(value_stats).items()
                    }
                )
                payload.update(
                    {
                        f"distill/{bucket_label}/policy_{key}": value
                        for key, value in policy_stats.items()
                    }
                )
                if isinstance(run, wandb.Run):
                    run.log(payload, step=global_step)
                if roots >= args.states_per_bucket:
                    break
            print(f"distilled {bucket_label}: roots={roots:,}", flush=True)

        checkpoint_path = output_dir / "checkpoints" / "rebel_latest.pt"
        _save_trainer_checkpoint(
            student,
            checkpoint_path,
            step=global_step,
            run_id=run.id if isinstance(run, wandb.Run) else None,
            metadata={
                "kind": "preflop_backward_induction_distilled_model",
                "state_dataset": os.path.realpath(args.state_dataset),
                "specialist_checkpoints": {
                    label: getattr(args, label.replace("actions_", "checkpoint_"))
                    for label in BUCKET_ORDER_DEEP_TO_SHALLOW
                },
            },
        )
        print(f"saved distilled model: {checkpoint_path}", flush=True)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-dataset", required=True)
    parser.add_argument("--base-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cfr-iterations", type=int, default=400)
    parser.add_argument("--warm-start-iterations", type=int, default=0)
    parser.add_argument("--sparse-fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", default=None, choices=["off", "default", "max-autotune"])
    parser.add_argument("--belief-mode", default="random", choices=["random", "uniform"])
    parser.add_argument("--states-per-bucket", type=int, default=100_000)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--cfr-batch-size", type=int, default=512)
    parser.add_argument("--replay-buffer-batches", type=int, default=1)
    parser.add_argument("--storage-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--write-solved-shards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write solved RebelBatch value/policy shards while training.",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-roots", type=int, default=10_000)
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project", default="poker-rebel-preflop-backward-induction")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=["preflop", "backward-induction"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-specialists")
    _add_common_args(train_parser)

    distill_parser = subparsers.add_parser("distill")
    _add_common_args(distill_parser)
    distill_parser.add_argument("--student-init", default=None)
    distill_parser.add_argument("--distill-batch-size", type=int, default=1024)
    distill_parser.add_argument("--checkpoint_12_15", required=True)
    distill_parser.add_argument("--checkpoint_8_11", required=True)
    distill_parser.add_argument("--checkpoint_4_7", required=True)
    distill_parser.add_argument("--checkpoint_0_3", required=True)

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    if args.command == "train-specialists":
        run_train_specialists(args)
    elif args.command == "distill":
        run_distill(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
