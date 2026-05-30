from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb

from p2.allin.data import make_random_preflop_allin_batch
from p2.allin.model import PreflopAllInEquityModel
from p2.allin.sampler import estimate_preflop_allin_values


@dataclass
class TrainConfig:
    steps: int = 10_000
    batch_size: int = 64
    players: int = 4
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 512
    hand_dim: int = 128
    layers: int = 4
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
    checkpoint_interval: int = 1000
    checkpoint_dir: str = "outputs/allin_equity"


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
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train(cfg: TrainConfig) -> None:
    device = _device(cfg.device)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    init_generator = torch.Generator(device=device).manual_seed(cfg.seed)
    model = PreflopAllInEquityModel(
        players=cfg.players,
        hidden_dim=cfg.hidden_dim,
        hand_dim=cfg.hand_dim,
        num_layers=cfg.layers,
    ).to(device)
    model.init_weights(init_generator)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Training all-in model on {device}: "
        f"params={total_params:,} trainable={trainable_params:,}"
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    started = time.perf_counter()
    with _init_wandb(cfg) as run:
        if isinstance(run, wandb.Run):
            run.summary["total_parameters"] = total_params
            run.summary["trainable_parameters"] = trainable_params

        for step in range(1, cfg.steps + 1):
            step_start = time.perf_counter()
            batch = make_random_preflop_allin_batch(
                cfg.batch_size,
                cfg.players,
                bb=cfg.bb,
                min_stack_bb=cfg.min_stack_bb,
                mid_stack_bb=cfg.mid_stack_bb,
                max_stack_bb=cfg.max_stack_bb,
                high_stack_mass_ratio=cfg.high_stack_mass_ratio,
                concentration=cfg.concentration,
                folded_commit_max_frac=cfg.folded_commit_max_frac,
                device=device,
                generator=generator,
            )
            targets, target_diag = estimate_preflop_allin_values(
                batch,
                sample_count=cfg.sample_count,
                board_samples=cfg.board_samples,
                tuple_samples=cfg.tuple_samples if cfg.tuple_samples > 0 else None,
                tuple_tries=cfg.tuple_tries,
                board_chunk=cfg.board_chunk,
                hand_chunk=cfg.hand_chunk,
                generator=generator,
            )

            pred = model(
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

            elapsed = time.perf_counter() - step_start
            if step % cfg.log_interval == 0 or step == 1:
                metrics = {
                    "step": step,
                    "loss/mse": float(loss.detach().item()),
                    "loss/mae": float(mae.detach().item()),
                    "loss/max_abs": float(max_abs.detach().item()),
                    "optim/grad_norm": float(grad_norm.detach().item()),
                    "data/live_players_mean": float(batch.allin_mask.float().sum(dim=1).mean().item()),
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
                    f"mse={metrics['loss/mse']:.6f} "
                    f"mae={metrics['loss/mae']:.5f} "
                    f"target={metrics['perf/target_seconds']:.2f}s "
                    f"step={elapsed:.2f}s"
                )
                if isinstance(run, wandb.Run):
                    run.log(metrics, step=step)

            if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
                _save_checkpoint(
                    checkpoint_dir / f"allin_equity_step_{step}.pt",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    step=step,
                )
                _save_checkpoint(
                    checkpoint_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    step=step,
                )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_def in TrainConfig.__dataclass_fields__.items():
        default = field_def.default
        arg = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(arg, action="store_true", default=default)
        else:
            parser.add_argument(arg, type=type(default) if default is not None else str, default=default)
    ns = parser.parse_args()
    return TrainConfig(**vars(ns))


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
