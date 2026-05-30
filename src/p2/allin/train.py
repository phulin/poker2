from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb

from p2.allin.model import PreflopAllInEquityModel
from p2.allin.training_data import (
    AllInDataGenConfig,
    PregeneratedAllInDataset,
    generate_allin_training_chunk,
    permute_allin_batch_by_suit,
    tensors_to_batch,
)
from p2.rl.optimizers import TrainOptimizer, build_optimizer


@dataclass
class TrainConfig:
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
    cosine_lr_decay_ratio: float = 1.0
    cosine_lr_decay_steps: int = 0
    hidden_dim: int = 512
    hand_dim: int = 128
    layers: int = 4
    film_rank: int = 64
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
    model: PreflopAllInEquityModel,
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
    squared_sum = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    value_count = 0
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
        squared_sum += float(err.square().sum().item())
        abs_sum += float(err.abs().sum().item())
        max_abs = max(max_abs, float(err.abs().amax().item()))
        value_count += err.numel()
        row_start += cur
    if model_was_training:
        model.train()
    return {
        "eval/mse": squared_sum / max(value_count, 1),
        "eval/mae": abs_sum / max(value_count, 1),
        "eval/max_abs": max_abs,
        "eval/examples": float(len(dataset)),
        "eval/seconds": time.perf_counter() - start,
    }


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


def _set_optimizer_lrs(
    optimizer: TrainOptimizer,
    cfg: TrainConfig,
    *,
    step: int,
) -> float:
    decay_steps = (
        cfg.cosine_lr_decay_steps if cfg.cosine_lr_decay_steps > 0 else cfg.steps
    )
    scale = _cosine_lr_scale(
        step,
        total_steps=decay_steps,
        ratio=float(cfg.cosine_lr_decay_ratio),
    )
    for param_group in optimizer.param_groups:
        base_lr = _base_lr_for_group(param_group, cfg)
        param_group["initial_lr"] = base_lr
        param_group["lr"] = base_lr * scale
    return scale


def train(cfg: TrainConfig) -> None:
    device = _device(cfg.device)
    batch_size_schedule = _parse_batch_size_schedule(
        cfg.batch_size_schedule,
        default_batch_size=cfg.batch_size,
        total_steps=cfg.steps,
    )
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    init_generator = torch.Generator(device=device).manual_seed(cfg.seed)
    model = PreflopAllInEquityModel(
        players=cfg.players,
        hidden_dim=cfg.hidden_dim,
        hand_dim=cfg.hand_dim,
        num_layers=cfg.layers,
        film_rank=cfg.film_rank,
    ).to(device)
    model.init_weights(init_generator)
    if device == "cuda":
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
        PregeneratedAllInDataset(cfg.pregenerated_data)
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
    started = time.perf_counter()
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
                )
                batch, targets = tensors_to_batch(generated, device=device)
            else:
                target_start = time.perf_counter()
                batch, targets = pregenerated_dataset.get_wrapped_batch(
                    examples_seen,
                    batch_size,
                    device=device,
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
                    "optim/muon_learning_rate": float(muon_lrs[0]) if muon_lrs else 0.0,
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


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_def in TrainConfig.__dataclass_fields__.items():
        default = field_def.default
        arg = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            if field_name.startswith("no_"):
                parser.add_argument(arg, action="store_true", default=default)
            else:
                parser.add_argument(
                    arg,
                    action=argparse.BooleanOptionalAction,
                    default=default,
                )
        else:
            parser.add_argument(
                arg, type=type(default) if default is not None else str, default=default
            )
    ns = parser.parse_args()
    return TrainConfig(**vars(ns))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
