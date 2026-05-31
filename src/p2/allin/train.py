from __future__ import annotations

import itertools
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from p2.allin.data import PreflopAllInBatch
from p2.allin.model import PreflopAllInEquityModel, PreflopAllInTransformerModel
from p2.allin.training_data import (
    AllInDataGenConfig,
    PregeneratedAllInDataset,
    PreflopAllInEstimatorWorkspace,
    generate_allin_training_chunk,
    permute_allin_batch_by_players,
    permute_allin_batch_by_suit,
    tensors_to_batch,
)
from p2.rl.optimizers import TrainOptimizer, build_optimizer


@dataclass
class AllInTrainConfig:
    steps: int = 10_000
    batch_size: int = 64
    batch_size_schedule: str = ""
    players: int = 4
    optimizer: str = "muon"
    learning_rate: float = 2.5e-3
    adamw_learning_rate: float = 4.0e-3
    weight_decay: float = 1.0e-4
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1.0e-7
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = None
    policy_head_muon_learning_rate: float = 3.0e-4
    lr_decay: str = "cosine"
    lr_warmdown_start_step: int = 1
    cosine_lr_decay_ratio: float = 1.0
    cosine_lr_decay_steps: int = 0
    model_type: str = "mlp"
    hidden_dim: int = 512
    hand_dim: int = 128
    layers: int = 4
    film_rank: int = 64
    transformer_heads: int = 8
    dense_belief_residual: bool = False
    dense_output_residual: bool = False
    compile_model: bool = False
    compile_dynamic: bool = True
    compile_mode: str = ""
    sample_count: int = 50_000
    board_samples: int = 256
    tuple_samples: int = 0
    tuple_tries: int = 4
    board_chunk: int = 8
    hand_chunk: int = 128
    bb: int = 100
    min_stack_bb: int = 10
    mid_stack_bb: int = 200
    max_stack_bb: int = 400
    high_stack_mass_ratio: float = 1.0 / 3.0
    concentration: float = 1.0
    folded_commit_max_frac: float = 0.35
    seed: int = 0
    device: str = "cuda"
    wandb_project: str = "p2-allin-equity"
    wandb_name: str | None = None
    no_wandb: bool = False
    log_interval: int = 10
    eval_interval: int = 0
    eval_batch_size: int = 0
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "outputs/allin_equity"
    resume_checkpoint: str | None = None
    pregenerated_data: str | None = None
    validation_data: str | None = None

    @classmethod
    def from_dict_config(cls, dict_config: DictConfig) -> "AllInTrainConfig":
        container = OmegaConf.to_container(dict_config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Hydra config must resolve to a mapping")
        return cls(**container)


TrainConfig = AllInTrainConfig


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _init_wandb(cfg: TrainConfig) -> Any:
    if cfg.no_wandb:
        return nullcontext()
    try:
        return wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            config=asdict(cfg),
        )
    except Exception as exc:
        print(f"wandb init failed ({exc}); continuing without logging")
        return nullcontext()


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: TrainOptimizer,
    generator: torch.Generator,
    cfg: TrainConfig,
    step: int,
    examples_seen: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "examples_seen": int(examples_seen),
        },
        path,
    )


def _build_model(cfg: TrainConfig) -> torch.nn.Module:
    model_type = str(cfg.model_type).strip().lower()
    if model_type in {"mlp", "ffn"}:
        return PreflopAllInEquityModel(
            players=cfg.players,
            hidden_dim=cfg.hidden_dim,
            hand_dim=cfg.hand_dim,
            num_layers=cfg.layers,
            film_rank=cfg.film_rank,
            dense_belief_residual=cfg.dense_belief_residual,
            dense_output_residual=cfg.dense_output_residual,
        )
    if model_type in {"player_transformer", "transformer", "tokens"}:
        return PreflopAllInTransformerModel(
            players=cfg.players,
            hidden_dim=cfg.hidden_dim,
            hand_dim=cfg.hand_dim,
            num_layers=cfg.layers,
            film_rank=cfg.film_rank,
            transformer_heads=cfg.transformer_heads,
            dense_belief_residual=cfg.dense_belief_residual,
            dense_output_residual=cfg.dense_output_residual,
        )
    raise ValueError(
        "model_type must be one of: mlp, player_transformer; "
        f"got {cfg.model_type!r}"
    )


def _parse_batch_size_schedule(
    spec: str,
    *,
    default_batch_size: int,
    total_steps: int,
) -> list[tuple[int, int]]:
    if not spec:
        return [(1, int(default_batch_size))]

    phases = [part.strip() for part in spec.split(",") if part.strip()]
    if not phases:
        return [(1, int(default_batch_size))]

    if all("x" in phase for phase in phases):
        schedule = []
        start_step = 1
        for phase in phases:
            batch_text, duration_text = phase.split("x", 1)
            batch_size = int(batch_text)
            duration = int(duration_text)
            if batch_size <= 0 or duration <= 0:
                raise ValueError("batch-size schedule entries must be positive")
            schedule.append((start_step, batch_size))
            start_step += duration
        if start_step != total_steps + 1:
            raise ValueError(
                "duration batch-size schedule must sum to --steps "
                f"({start_step - 1} != {total_steps})"
            )
        return schedule

    if all(":" in phase for phase in phases):
        schedule = []
        for phase in phases:
            step_text, batch_text = phase.split(":", 1)
            start_step = int(step_text)
            batch_size = int(batch_text)
            if start_step <= 0 or batch_size <= 0:
                raise ValueError("batch-size schedule entries must be positive")
            schedule.append((start_step, batch_size))
        schedule.sort()
        if len({start for start, _ in schedule}) != len(schedule):
            raise ValueError("batch-size schedule contains duplicate start steps")
        if schedule[0][0] != 1:
            schedule.insert(0, (1, int(default_batch_size)))
        return schedule

    raise ValueError(
        "batch-size schedule must use either duration phases like "
        "'64x100,128x50' or step phases like '1:64,101:128'"
    )


def _batch_size_for_step(schedule: list[tuple[int, int]], step: int) -> int:
    batch_size = schedule[0][1]
    for start_step, scheduled_batch_size in schedule:
        if step < start_step:
            break
        batch_size = scheduled_batch_size
    return batch_size


def _required_examples(
    schedule: list[tuple[int, int]],
    *,
    start_step: int,
    end_step: int,
) -> int:
    return sum(
        _batch_size_for_step(schedule, step) for step in range(start_step, end_step + 1)
    )


def _data_config_from_train(cfg: TrainConfig) -> AllInDataGenConfig:
    return AllInDataGenConfig(
        players=cfg.players,
        sample_count=cfg.sample_count,
        board_samples=cfg.board_samples,
        tuple_samples=cfg.tuple_samples,
        tuple_tries=cfg.tuple_tries,
        board_chunk=cfg.board_chunk,
        hand_chunk=cfg.hand_chunk,
        bb=cfg.bb,
        min_stack_bb=cfg.min_stack_bb,
        mid_stack_bb=cfg.mid_stack_bb,
        max_stack_bb=cfg.max_stack_bb,
        high_stack_mass_ratio=cfg.high_stack_mass_ratio,
        concentration=cfg.concentration,
        folded_commit_max_frac=cfg.folded_commit_max_frac,
    )


def _pregenerated_target_diag(
    targets: torch.Tensor,
    *,
    elapsed: float,
) -> dict[str, float]:
    return {
        "target_seconds": elapsed,
        "target_boards_per_second": 0.0,
        "target_samples_per_second": 0.0,
        "target_zero_denom_frac": 0.0,
        "target_value_mean": float(targets.mean().item()),
        "target_value_std": float(targets.std().item()),
        "target_board_samples": 0.0,
        "target_tuple_samples": 0.0,
    }


def _pregenerated_suit_permutation_idxs(
    *,
    global_start: int,
    batch_size: int,
    dataset_examples: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if dataset_examples <= 0:
        raise ValueError("dataset_examples must be positive")
    global_indices = torch.arange(
        global_start,
        global_start + batch_size,
        device=device,
        dtype=torch.long,
    )
    row_indices = global_indices.remainder(dataset_examples)
    epoch_indices = torch.div(
        global_indices,
        dataset_examples,
        rounding_mode="floor",
    )

    seed_tensor = torch.full(
        (), int(seed) & 0x7FFFFFFF, device=device, dtype=torch.long
    )
    hashed = row_indices + seed_tensor + 0x9E3779B9
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 16))
    hashed = hashed * 0x45D9F3B
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 16))
    hashed = hashed * 0x45D9F3B
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 16))
    base_perm_idxs = hashed.remainder(24)
    return (base_perm_idxs + epoch_indices.remainder(24)).remainder(24)


@lru_cache(maxsize=None)
def _player_permutation_table(
    players: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.tensor(
        list(itertools.permutations(range(players))),
        device=device,
        dtype=torch.long,
    )


def _pregenerated_player_permutations(
    *,
    global_start: int,
    batch_size: int,
    dataset_examples: int,
    players: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if dataset_examples <= 0:
        raise ValueError("dataset_examples must be positive")
    if players <= 0:
        raise ValueError("players must be positive")
    permutation_table = _player_permutation_table(
        players,
        device.type,
        device.index,
    )
    global_indices = torch.arange(
        global_start,
        global_start + batch_size,
        device=device,
        dtype=torch.long,
    )
    row_indices = global_indices.remainder(dataset_examples)
    epoch_indices = torch.div(
        global_indices,
        dataset_examples,
        rounding_mode="floor",
    )

    seed_tensor = torch.full(
        (), int(seed) & 0x7FFFFFFF, device=device, dtype=torch.long
    )
    hashed = row_indices + seed_tensor + 0x9E3779B9
    hashed = hashed + epoch_indices * 0x85EBCA6B
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 16))
    hashed = hashed * 0x45D9F3B
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 16))
    hashed = hashed * 0x45D9F3B
    hashed = torch.bitwise_xor(hashed, torch.bitwise_right_shift(hashed, 16))
    permutation_idxs = hashed.remainder(permutation_table.shape[0])
    return permutation_table.index_select(0, permutation_idxs)


@torch.no_grad()
def _evaluate_pregenerated_dataset(
    model: torch.nn.Module,
    dataset: PregeneratedAllInDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError("validation batch size must be positive")
    model_was_training = model.training
    model.eval()
    start = time.perf_counter()
    players = int(dataset.players)
    squared_by_live = torch.zeros(players + 1, dtype=torch.float64, device=device)
    abs_by_live = torch.zeros(players + 1, dtype=torch.float64, device=device)
    examples_by_live = torch.zeros(players + 1, dtype=torch.float64, device=device)
    max_abs = torch.zeros((), dtype=torch.float32, device=device)
    row_start = 0
    while row_start < len(dataset):
        cur = min(batch_size, len(dataset) - row_start)
        batch, targets = dataset.get_batch(row_start, cur, device=device)
        pred = model(
            batch.beliefs,
            batch.starting_stacks,
            batch.committed,
            batch.stacks_after,
            batch.allin_mask,
            batch.folded_mask,
        )
        err = (pred - targets).detach()
        live_counts = (~batch.folded_mask).sum(dim=1).long()
        squared_by_live += torch.bincount(
            live_counts,
            weights=err.square().sum(dim=(1, 2)).to(torch.float64),
            minlength=players + 1,
        )
        abs_err = err.abs()
        abs_by_live += torch.bincount(
            live_counts,
            weights=abs_err.sum(dim=(1, 2)).to(torch.float64),
            minlength=players + 1,
        )
        examples_by_live += torch.bincount(
            live_counts,
            minlength=players + 1,
        ).to(torch.float64)
        max_abs = torch.maximum(max_abs, abs_err.amax())
        row_start += cur
    if model_was_training:
        model.train()
    values_per_example = players * int(dataset.hands)
    value_count_by_live = examples_by_live * values_per_example
    squared_by_live_cpu = squared_by_live.cpu()
    abs_by_live_cpu = abs_by_live.cpu()
    examples_by_live_cpu = examples_by_live.cpu()
    value_count_by_live_cpu = value_count_by_live.cpu()
    total_value_count = max(float(value_count_by_live_cpu.sum().item()), 1.0)
    metrics = {
        "eval/mse": float(squared_by_live_cpu.sum().item()) / total_value_count,
        "eval/mae": float(abs_by_live_cpu.sum().item()) / total_value_count,
        "eval/max_abs": float(max_abs.cpu().item()),
        "eval/examples": float(len(dataset)),
        "eval/seconds": time.perf_counter() - start,
    }
    for live_count in range(players + 1):
        examples = float(examples_by_live_cpu[live_count].item())
        if examples <= 0.0:
            continue
        value_count = float(value_count_by_live_cpu[live_count].item())
        metrics[f"eval/live_players/{live_count}/mse"] = (
            float(squared_by_live_cpu[live_count].item()) / value_count
        )
        metrics[f"eval/live_players/{live_count}/mae"] = (
            float(abs_by_live_cpu[live_count].item()) / value_count
        )
        metrics[f"eval/live_players/{live_count}/examples"] = examples
    return metrics


def _base_lr_for_group(param_group: dict[str, Any], cfg: TrainConfig) -> float:
    role = param_group.get("lr_role")
    if role == "adamw":
        return float(cfg.adamw_learning_rate)
    if role == "policy_head_muon":
        return float(cfg.policy_head_muon_learning_rate)
    return float(cfg.learning_rate)


def _cosine_lr_scale(step: int, *, total_steps: int, ratio: float) -> float:
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("cosine_lr_decay_ratio must be in [0, 1]")
    if ratio == 1.0:
        return 1.0
    if total_steps <= 1:
        return ratio
    progress = min(max((step - 1) / float(total_steps - 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return ratio + (1.0 - ratio) * cosine


def _linear_lr_scale(step: int, *, total_steps: int, ratio: float) -> float:
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("cosine_lr_decay_ratio must be in [0, 1]")
    if ratio == 1.0:
        return 1.0
    if total_steps <= 1:
        return ratio
    progress = min(max((step - 1) / float(total_steps - 1), 0.0), 1.0)
    return ratio + (1.0 - ratio) * (1.0 - progress)


def _warmdown_lr_scale(
    step: int,
    *,
    total_steps: int,
    ratio: float,
    start_step: int,
    shape: str,
) -> float:
    if start_step < 1:
        raise ValueError("lr_warmdown_start_step must be positive")
    if start_step > total_steps:
        raise ValueError("lr_warmdown_start_step cannot exceed decay steps")
    if step < start_step:
        return 1.0
    warmdown_steps = total_steps - start_step + 1
    warmdown_step = step - start_step + 1
    if shape == "cosine":
        return _cosine_lr_scale(warmdown_step, total_steps=warmdown_steps, ratio=ratio)
    if shape == "linear":
        return _linear_lr_scale(warmdown_step, total_steps=warmdown_steps, ratio=ratio)
    raise ValueError(f"unsupported warmdown shape {shape!r}")


def _lr_scale(
    step: int,
    *,
    total_steps: int,
    ratio: float,
    decay: str,
    warmdown_start_step: int,
) -> float:
    decay = decay.strip().lower()
    if decay == "cosine":
        return _cosine_lr_scale(step, total_steps=total_steps, ratio=ratio)
    if decay == "linear":
        return _linear_lr_scale(step, total_steps=total_steps, ratio=ratio)
    if decay in {"stable_warmdown", "cosine_warmdown"}:
        return _warmdown_lr_scale(
            step,
            total_steps=total_steps,
            ratio=ratio,
            start_step=warmdown_start_step,
            shape="cosine",
        )
    if decay == "linear_warmdown":
        return _warmdown_lr_scale(
            step,
            total_steps=total_steps,
            ratio=ratio,
            start_step=warmdown_start_step,
            shape="linear",
        )
    raise ValueError(
        "lr_decay must be one of: cosine, linear, stable_warmdown, "
        f"cosine_warmdown, linear_warmdown; got {decay!r}"
    )


def _set_optimizer_lrs(
    optimizer: TrainOptimizer,
    cfg: TrainConfig,
    *,
    step: int,
) -> float:
    decay_steps = (
        cfg.cosine_lr_decay_steps if cfg.cosine_lr_decay_steps > 0 else cfg.steps
    )
    scale = _lr_scale(
        step,
        total_steps=decay_steps,
        ratio=float(cfg.cosine_lr_decay_ratio),
        decay=cfg.lr_decay,
        warmdown_start_step=cfg.lr_warmdown_start_step,
    )
    for param_group in optimizer.param_groups:
        base_lr = _base_lr_for_group(param_group, cfg)
        param_group["initial_lr"] = base_lr
        param_group["lr"] = base_lr * scale
    return scale


@dataclass
class _PrefetchedPregeneratedBatch:
    start: int
    count: int
    batch: PreflopAllInBatch
    targets: torch.Tensor
    stream: torch.cuda.Stream | None


def _record_pregenerated_batch_stream(
    batch: PreflopAllInBatch,
    targets: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    for field in fields(PreflopAllInBatch):
        tensor = getattr(batch, field.name)
        if tensor.device.type == "cuda":
            tensor.record_stream(stream)
    if targets.device.type == "cuda":
        targets.record_stream(stream)


class _PregeneratedBatchPrefetcher:
    def __init__(
        self,
        dataset: PregeneratedAllInDataset,
        *,
        device: torch.device,
        enabled: bool,
    ) -> None:
        self.dataset = dataset
        self.device = device
        self._stream = torch.cuda.Stream(device=device) if enabled else None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="allin-batch-prefetch")
            if enabled
            else None
        )
        self._future: Future[_PrefetchedPregeneratedBatch] | None = None
        self._future_key: tuple[int, int] | None = None

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.dataset.close()

    def _load(
        self,
        start: int,
        count: int,
    ) -> _PrefetchedPregeneratedBatch:
        if self._stream is None:
            batch, targets = self.dataset.get_wrapped_batch(
                start,
                count,
                device=self.device,
            )
            return _PrefetchedPregeneratedBatch(start, count, batch, targets, None)

        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            batch, targets = self.dataset.get_wrapped_batch(
                start,
                count,
                device=self.device,
            )
        self.dataset.prefetch_shard_for_row(start + count)
        return _PrefetchedPregeneratedBatch(
            start,
            count,
            batch,
            targets,
            self._stream,
        )

    def prefetch(self, start: int, count: int) -> None:
        if self._executor is None:
            return
        key = (int(start), int(count))
        if self._future_key == key:
            return
        if self._future is not None and not self._future.done():
            self._future.result()
        self._future_key = key
        self._future = self._executor.submit(self._load, key[0], key[1])

    def get(
        self,
        start: int,
        count: int,
    ) -> tuple[PreflopAllInBatch, torch.Tensor]:
        key = (int(start), int(count))
        if self._executor is None:
            prefetched = self._load(key[0], key[1])
        else:
            if self._future_key != key or self._future is None:
                self.prefetch(key[0], key[1])
            assert self._future is not None
            prefetched = self._future.result()
            self._future = None
            self._future_key = None

        if prefetched.stream is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(prefetched.stream)
            _record_pregenerated_batch_stream(
                prefetched.batch,
                prefetched.targets,
                current_stream,
            )
        return prefetched.batch, prefetched.targets


def train(cfg: TrainConfig) -> None:
    device = _device(cfg.device)
    batch_size_schedule = _parse_batch_size_schedule(
        cfg.batch_size_schedule,
        default_batch_size=cfg.batch_size,
        total_steps=cfg.steps,
    )
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    init_generator = torch.Generator(device=device).manual_seed(cfg.seed)
    model = _build_model(cfg).to(device)
    model.init_weights(init_generator)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    optimizer = build_optimizer(model, cfg, device)
    compiled_model = model
    if cfg.compile_model:
        compile_kwargs: dict[str, Any] = {"dynamic": cfg.compile_dynamic}
        if cfg.compile_mode:
            compile_kwargs["mode"] = cfg.compile_mode
        compiled_model = torch.compile(model, **compile_kwargs)
    start_step = 0
    examples_seen = 0
    if cfg.resume_checkpoint is not None:
        checkpoint = torch.load(cfg.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "generator_state" in checkpoint:
            generator.set_state(checkpoint["generator_state"].cpu())
        start_step = int(checkpoint.get("step", 0))
        examples_seen = int(checkpoint.get("examples_seen", 0))

    pregenerated_dataset = (
        PregeneratedAllInDataset(
            cfg.pregenerated_data,
            pin_memory=device.type == "cuda",
            async_shard_prefetch=device.type == "cuda",
        )
        if cfg.pregenerated_data is not None
        else None
    )
    validation_dataset = (
        PregeneratedAllInDataset(cfg.validation_data)
        if cfg.validation_data is not None
        else None
    )
    if pregenerated_dataset is not None:
        if pregenerated_dataset.players != cfg.players:
            raise ValueError(
                f"pregenerated data has {pregenerated_dataset.players} players, "
                f"trainer expects {cfg.players}"
            )
        remaining_examples = _required_examples(
            batch_size_schedule,
            start_step=start_step + 1,
            end_step=cfg.steps,
        )
        planned_examples_seen = examples_seen + remaining_examples
        planned_pregenerated_epochs = planned_examples_seen / len(pregenerated_dataset)
        starting_pregenerated_epoch = examples_seen / len(pregenerated_dataset)
    if validation_dataset is not None and validation_dataset.players != cfg.players:
        raise ValueError(
            f"validation data has {validation_dataset.players} players, "
            f"trainer expects {cfg.players}"
        )
    pregenerated_prefetcher = (
        _PregeneratedBatchPrefetcher(
            pregenerated_dataset,
            device=device,
            enabled=device.type == "cuda",
        )
        if pregenerated_dataset is not None
        else None
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Training all-in model on {device}: "
        f"params={total_params:,} trainable={trainable_params:,}",
        flush=True,
    )
    if pregenerated_dataset is not None:
        print(
            "Pregenerated training data: "
            f"rows={len(pregenerated_dataset):,} "
            f"start_epoch={starting_pregenerated_epoch:.3f} "
            f"planned_epochs={planned_pregenerated_epochs:.3f}",
            flush=True,
        )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    data_cfg = _data_config_from_train(cfg)
    online_target_workspace = (
        PreflopAllInEstimatorWorkspace() if pregenerated_dataset is None else None
    )
    started = time.perf_counter()
    try:
        if pregenerated_prefetcher is not None and start_step < cfg.steps:
            first_step = start_step + 1
            pregenerated_prefetcher.prefetch(
                examples_seen,
                _batch_size_for_step(batch_size_schedule, first_step),
            )

        with _init_wandb(cfg) as run:
            if isinstance(run, wandb.Run):
                run.summary["total_parameters"] = total_params
                run.summary["trainable_parameters"] = trainable_params

            for step in range(start_step + 1, cfg.steps + 1):
                step_start = time.perf_counter()
                is_log_step = step % cfg.log_interval == 0 or step == 1
                lr_scale = _set_optimizer_lrs(optimizer, cfg, step=step)
                batch_size = _batch_size_for_step(batch_size_schedule, step)
                if pregenerated_dataset is None:
                    generated, target_diag = generate_allin_training_chunk(
                        batch_size,
                        data_cfg,
                        device=device,
                        generator=generator,
                        compute_stats=is_log_step,
                        workspace=online_target_workspace,
                    )
                    batch, targets = tensors_to_batch(generated, device=device)
                else:
                    target_start = time.perf_counter()
                    assert pregenerated_prefetcher is not None
                    batch, targets = pregenerated_prefetcher.get(
                        examples_seen,
                        batch_size,
                    )
                    next_examples_seen = examples_seen + batch_size
                    if step < cfg.steps:
                        pregenerated_prefetcher.prefetch(
                            next_examples_seen,
                            _batch_size_for_step(batch_size_schedule, step + 1),
                        )
                    suit_permutation_idxs = _pregenerated_suit_permutation_idxs(
                        global_start=examples_seen,
                        batch_size=batch_size,
                        dataset_examples=len(pregenerated_dataset),
                        seed=cfg.seed,
                        device=device,
                    )
                    batch, targets = permute_allin_batch_by_suit(
                        batch,
                        targets,
                        suit_permutation_idxs=suit_permutation_idxs,
                    )
                    player_permutations = _pregenerated_player_permutations(
                        global_start=examples_seen,
                        batch_size=batch_size,
                        dataset_examples=len(pregenerated_dataset),
                        players=cfg.players,
                        seed=cfg.seed,
                        device=device,
                    )
                    batch, targets = permute_allin_batch_by_players(
                        batch,
                        targets,
                        player_permutations=player_permutations,
                    )
                    target_diag = (
                        _pregenerated_target_diag(
                            targets,
                            elapsed=time.perf_counter() - target_start,
                        )
                        if is_log_step
                        else {}
                    )

                pred = compiled_model(
                    batch.beliefs,
                    batch.starting_stacks,
                    batch.committed,
                    batch.stacks_after,
                    batch.allin_mask,
                    batch.folded_mask,
                )
                loss = F.mse_loss(pred, targets)
                mae = (pred - targets).abs().mean()
                max_abs = (pred - targets).abs().amax()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                examples_seen += batch_size

                elapsed = time.perf_counter() - step_start
                if is_log_step:
                    muon_lrs = [
                        group["lr"]
                        for group in optimizer.param_groups
                        if group.get("lr_role") != "adamw"
                    ]
                    adamw_lrs = [
                        group["lr"]
                        for group in optimizer.param_groups
                        if group.get("lr_role") == "adamw"
                    ]
                    metrics = {
                        "step": step,
                        "loss/mse": float(loss.detach().item()),
                        "loss/mae": float(mae.detach().item()),
                        "loss/max_abs": float(max_abs.detach().item()),
                        "optim/grad_norm": float(grad_norm.detach().item()),
                        "optim/learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "optim/muon_learning_rate": float(muon_lrs[0])
                        if muon_lrs
                        else 0.0,
                        "optim/adamw_learning_rate": float(adamw_lrs[0])
                        if adamw_lrs
                        else 0.0,
                        "optim/lr_scale": lr_scale,
                        "data/batch_size": batch_size,
                        "data/examples_seen": examples_seen,
                        "data/live_players_mean": float(
                            (~batch.folded_mask).float().sum(dim=1).mean().item()
                        ),
                        "data/allin_players_mean": float(
                            batch.allin_mask.float().sum(dim=1).mean().item()
                        ),
                        "data/pregenerated_epoch": (
                            examples_seen / len(pregenerated_dataset)
                            if pregenerated_dataset is not None
                            else 0.0
                        ),
                        "data/stack_mean": float(batch.starting_stacks.mean().item()),
                        "data/committed_mean": float(batch.committed.mean().item()),
                        "target/value_mean": target_diag["target_value_mean"],
                        "target/value_std": target_diag["target_value_std"],
                        "target/zero_denom_frac": target_diag["target_zero_denom_frac"],
                        "perf/step_seconds": elapsed,
                        "perf/target_seconds": target_diag["target_seconds"],
                        "perf/target_boards_per_second": target_diag[
                            "target_boards_per_second"
                        ],
                        "perf/target_samples_per_second": target_diag[
                            "target_samples_per_second"
                        ],
                        "target/board_samples": target_diag["target_board_samples"],
                        "target/tuple_samples": target_diag["target_tuple_samples"],
                        "perf/total_minutes": (time.perf_counter() - started) / 60.0,
                    }
                    if "target_kernel_launch_seconds" in target_diag:
                        metrics["perf/target_kernel_launch_seconds"] = target_diag[
                            "target_kernel_launch_seconds"
                        ]
                    print(
                        f"[{step:06d}/{cfg.steps}] "
                        f"bs={batch_size} "
                        f"mse={metrics['loss/mse']:.6f} "
                        f"mae={metrics['loss/mae']:.5f} "
                        f"target={metrics['perf/target_seconds']:.2f}s "
                        f"step={elapsed:.2f}s",
                        flush=True,
                    )
                    if isinstance(run, wandb.Run):
                        run.log(metrics, step=step)

                if (
                    validation_dataset is not None
                    and cfg.eval_interval > 0
                    and (step % cfg.eval_interval == 0 or step == cfg.steps)
                ):
                    eval_batch_size = (
                        cfg.eval_batch_size if cfg.eval_batch_size > 0 else batch_size
                    )
                    eval_metrics = _evaluate_pregenerated_dataset(
                        compiled_model,
                        validation_dataset,
                        batch_size=eval_batch_size,
                        device=device,
                    )
                    print(
                        f"[eval {step:06d}/{cfg.steps}] "
                        f"mse={eval_metrics['eval/mse']:.6f} "
                        f"mae={eval_metrics['eval/mae']:.5f} "
                        f"max_abs={eval_metrics['eval/max_abs']:.5f} "
                        f"seconds={eval_metrics['eval/seconds']:.2f}s",
                        flush=True,
                    )
                    if isinstance(run, wandb.Run):
                        run.log(eval_metrics, step=step)

                if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
                    _save_checkpoint(
                        checkpoint_dir / f"allin_equity_step_{step}.pt",
                        model=model,
                        optimizer=optimizer,
                        generator=generator,
                        cfg=cfg,
                        step=step,
                        examples_seen=examples_seen,
                    )
                    _save_checkpoint(
                        checkpoint_dir / "latest.pt",
                        model=model,
                        optimizer=optimizer,
                        generator=generator,
                        cfg=cfg,
                        step=step,
                        examples_seen=examples_seen,
                    )
    finally:
        if pregenerated_prefetcher is not None:
            pregenerated_prefetcher.close()
        if validation_dataset is not None:
            validation_dataset.close()


cs = ConfigStore.instance()
cs.store(group="allin", name="config_schema", node=AllInTrainConfig)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="allin/config",
)
def main(dict_config: DictConfig) -> None:
    train(AllInTrainConfig.from_dict_config(dict_config))


if __name__ == "__main__":
    main()
