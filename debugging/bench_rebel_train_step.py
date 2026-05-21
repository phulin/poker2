from __future__ import annotations

import argparse
import json
import os
import time

import torch
from omegaconf import OmegaConf

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


DEFAULT_RUN_OVERRIDES = [
    "train.optimizer=muon",
    "train.episodes_per_step=60",
    "train.batch_size=2048",
    "train.value_reuse_goal=2",
    "num_envs=1024",
    "train.learning_rate=17e-3",
    "train.learning_rate_final=1.7e-3",
    "num_steps=2000",
    "trueskill.snapshot_frac=0.025",
]

BENCH_OVERRIDES = [
    "use_wandb=false",
    "trueskill.enabled=false",
    "checkpoint_interval=1000000000",
    "economize_checkpoints=false",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_cfg(overrides: list[str], include_default_run_overrides: bool) -> Config:
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "conf",
        "config_rebel_cfr.yaml",
    )
    base = OmegaConf.load(cfg_path)
    dotlist = [*BENCH_OVERRIDES]
    if include_default_run_overrides:
        dotlist.extend(DEFAULT_RUN_OVERRIDES)
    dotlist.extend(overrides)
    merged = OmegaConf.merge(base, OmegaConf.from_dotlist(dotlist))
    OmegaConf.set_struct(merged, False)
    merged.pop("defaults", None)
    return Config.from_dict_config(merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=1)
    parser.add_argument(
        "--no-default-run-overrides",
        action="store_true",
        help="Use config_rebel_cfr.yaml values except benchmark safety overrides.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    cfg = _load_cfg(
        args.overrides,
        include_default_run_overrides=not args.no_default_run_overrides,
    )
    device = torch.device(
        "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    _sync(device)

    warmup_times = []
    for step in range(args.warmup_steps):
        start = time.perf_counter()
        trainer.train_step(step)
        _sync(device)
        warmup_times.append(time.perf_counter() - start)

    timed_times = []
    for i in range(args.timed_steps):
        step = args.warmup_steps + i
        start = time.perf_counter()
        trainer.train_step(step)
        _sync(device)
        timed_times.append(time.perf_counter() - start)

    print(
        json.dumps(
            {
                "device": str(device),
                "compiled": str(cfg.model.compile),
                "batch_size": cfg.train.batch_size,
                "episodes_per_step": cfg.train.episodes_per_step,
                "value_reuse_goal": cfg.train.value_reuse_goal,
                "num_envs": cfg.num_envs,
                "search_depth": cfg.search.depth,
                "search_iterations": cfg.search.iterations,
                "warmup_step_s": warmup_times,
                "timed_step_s": timed_times,
                "timed_mean_s": sum(timed_times) / max(1, len(timed_times)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
