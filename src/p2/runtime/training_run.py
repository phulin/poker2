from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
import wandb

from p2.core.structured_config import Config
from p2.utils.model_utils import count_model_parameters


def device_from_config(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_torch_runtime(
    cfg: Config,
    device: torch.device,
    *,
    recompile_limit: int = 16,
) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = int(recompile_limit)
    torch.manual_seed(cfg.seed)


def wandb_run_id_from_checkpoint(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    run_id = checkpoint.get("wandb_run_id")
    return str(run_id) if run_id else None


@dataclass
class TrainingRunContext:
    cfg: Config
    device: torch.device
    run: Any

    @property
    def wandb_run_id(self) -> str | None:
        return self.run.id if isinstance(self.run, wandb.Run) else None

    def log_model_parameter_summary(self, model: torch.nn.Module) -> None:
        metrics = count_model_parameters(model)
        print(
            "Model parameters: "
            f"total={metrics['total_parameters']:,}; "
            f"trainable={metrics['trainable_parameters']:,}"
        )
        if isinstance(self.run, wandb.Run):
            self.run.summary.update(metrics)

    def watch_model(self, model: torch.nn.Module, *, log_freq: int = 100) -> None:
        if isinstance(self.run, wandb.Run):
            self.run.watch(model, log_freq=log_freq)


def _wandb_init_kwargs(
    cfg: Config,
    *,
    group: str | None,
    name: str | None,
    stage: str | None,
) -> dict[str, Any]:
    config_payload: dict[str, Any] = {"resolved_config": asdict(cfg)}
    if stage is not None:
        config_payload["stage"] = {"name": stage}

    init_kwargs: dict[str, Any] = {
        "project": cfg.wandb_project,
        "name": cfg.wandb_name if name is None else name,
        "tags": cfg.wandb_tags or [],
        "config": config_payload,
    }
    if group is not None:
        init_kwargs["group"] = group

    checkpoint_run_id = wandb_run_id_from_checkpoint(cfg.resume_from)
    run_id = cfg.wandb_run_id or checkpoint_run_id
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "must"
        if checkpoint_run_id:
            print(f"Found wandb run ID in checkpoint: {checkpoint_run_id}")
    return init_kwargs


@contextmanager
def training_run(
    cfg: Config,
    *,
    group: str | None = None,
    name: str | None = None,
    stage: str | None = None,
    create_checkpoint_dir: bool = True,
    configure_torch: bool = True,
    recompile_limit: int = 16,
) -> Iterator[TrainingRunContext]:
    if create_checkpoint_dir:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = device_from_config(cfg)
    print(f"Using device: {device}")
    if configure_torch:
        setup_torch_runtime(cfg, device, recompile_limit=recompile_limit)

    run_cm: Any = nullcontext()
    if cfg.use_wandb:
        try:
            run_cm = wandb.init(
                **_wandb_init_kwargs(cfg, group=group, name=name, stage=stage)
            )
        except Exception as exc:
            print(f"Wandb init failed ({exc}); continuing without logging.")
            cfg.use_wandb = False
            run_cm = nullcontext()

    with run_cm as run:
        yield TrainingRunContext(cfg=cfg, device=device, run=run)


__all__ = [
    "TrainingRunContext",
    "device_from_config",
    "setup_torch_runtime",
    "training_run",
    "wandb_run_id_from_checkpoint",
]
