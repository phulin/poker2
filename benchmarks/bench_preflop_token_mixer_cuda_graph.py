from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from p2.core.structured_config import NonlinearityType
from p2.models.mlp.better_ffn import _PreflopGatedTokenMixerBlock
from p2.models.mlp.preflop_token_mixer_graph import (
    PreflopGatedTokenMixerCudaGraphRunner,
    preflop_gated_token_mixer_token_path,
)


DEFAULT_CONFIG = "conf/config_rebel_preflop_buckets.yaml"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / iters)

    start_time = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start_time) * 1000.0 / iters


def load_config(path: str) -> DictConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return OmegaConf.load(config_path)


def config_int(cfg: DictConfig, dotted_key: str, default: int) -> int:
    value = OmegaConf.select(cfg, dotted_key, default=default)
    return int(value)


def parse_batch_sizes(value: str) -> list[int]:
    batch_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not batch_sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return batch_sizes


def auto_batch_sizes(cfg: DictConfig) -> list[int]:
    seed_keys = (
        "train.batch_size",
        "preflop_buckets.train_batch_size",
        "preflop_buckets.policy_train_batch_size",
        "preflop_buckets.cfr_batch_size",
        "preflop_buckets.validation_eval_batch_size",
        "preflop_buckets.distill_batch_size",
        "preflop_buckets.actions_8_11_cfr_batch_size",
        "preflop_buckets.actions_12_15_cfr_batch_size",
    )
    seeds = {
        int(value)
        for key in seed_keys
        if (value := OmegaConf.select(cfg, key, default=None)) is not None
    }
    if not seeds:
        seeds = {512, 2048, 8192}
    max_seed = max(seeds)
    expanded = set(seeds)
    expanded.update(max_seed * multiplier for multiplier in (2, 4, 8))
    expanded.update(batch for batch in (512, 1024, 2048, 4096, 8192) if batch <= max_seed)
    return sorted(expanded)


def make_block(
    *,
    dim: int,
    ffn_dim: int,
    token_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _PreflopGatedTokenMixerBlock:
    block = _PreflopGatedTokenMixerBlock(
        dim,
        token_count=token_count,
        ffn_dim=ffn_dim,
        nonlinearity=NonlinearityType.leaky_relu,
    )
    block.eval()
    return block.to(device=device, dtype=dtype)


def current_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    return preflop_gated_token_mixer_token_path(block, x)


def eager_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    y = block.token_norm(x)
    mixed = block.token_mixer(y.transpose(1, 2)).transpose(1, 2)
    gate = block.token_gate(y)
    return x + mixed * torch.sigmoid(gate) / math.sqrt(2.0)


def benchmark_batch(
    *,
    batch_size: int,
    dim: int,
    ffn_dim: int,
    token_count: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    block = make_block(
        dim=dim,
        ffn_dim=ffn_dim,
        token_count=token_count,
        device=device,
        dtype=dtype,
    )
    x = torch.randn(batch_size, token_count, dim, device=device, dtype=dtype)

    with torch.no_grad():
        expected = eager_token_path(block, x)
        current = current_token_path(block, x)
        runner = PreflopGatedTokenMixerCudaGraphRunner(block, x)
        graph_static = runner.replay_static()
        graph_copy = runner.copy_replay(x)
        sync(device)

        errors = {
            "current_token_path": (current.float() - expected.float()).abs().max().item(),
            "cuda_graph_replay_static": (
                graph_static.float() - expected.float()
            ).abs().max().item(),
            "cuda_graph_copy_replay": (
                graph_copy.float() - expected.float()
            ).abs().max().item(),
        }
        variants: list[tuple[str, Callable[[], torch.Tensor]]] = [
            ("eager_token_path", lambda: eager_token_path(block, x)),
            ("current_token_path", lambda: current_token_path(block, x)),
            ("cuda_graph_replay_static", runner.replay_static),
            ("cuda_graph_copy_replay", lambda: runner.copy_replay(x)),
        ]
        timings = [
            {
                "name": name,
                "ms": time_call(fn, device=device, warmup=warmup, iters=iters),
            }
            for name, fn in variants
        ]

    timing_by_name = {row["name"]: float(row["ms"]) for row in timings}
    current_ms = timing_by_name["current_token_path"]
    return {
        "batch_size": batch_size,
        "max_abs_error": errors,
        "timings": timings,
        "speedups_vs_current": {
            name: current_ms / ms for name, ms in timing_by_name.items() if ms > 0.0
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--token-count", type=int, default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dim = args.dim if args.dim is not None else config_int(cfg, "model.hidden_dim", 192)
    ffn_dim = (
        args.ffn_dim if args.ffn_dim is not None else config_int(cfg, "model.ffn_dim", 256)
    )
    token_count = (
        args.token_count
        if args.token_count is not None
        else config_int(cfg, "env.num_players", 6) + 1
    )
    batch_sizes = args.batch_sizes if args.batch_sizes is not None else auto_batch_sizes(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA")
    if token_count != 7:
        raise ValueError("the current Triton candidate is specialized to 7 tokens")
    torch.set_float32_matmul_precision("high")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    results = [
        benchmark_batch(
            batch_size=batch_size,
            dim=dim,
            ffn_dim=ffn_dim,
            token_count=token_count,
            device=device,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
        )
        for batch_size in batch_sizes
    ]
    payload = {
        "device": torch.cuda.get_device_name(device),
        "config": args.config,
        "dtype": str(dtype).replace("torch.", ""),
        "dim": dim,
        "ffn_dim": ffn_dim,
        "token_count": token_count,
        "batch_sizes": batch_sizes,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        return

    print()
    print("batch,variant,ms,speedup_vs_current,max_abs_error")
    for row in results:
        speedups = row["speedups_vs_current"]
        errors = row["max_abs_error"]
        for timing in row["timings"]:
            name = str(timing["name"])
            err = 0.0 if name == "eager_token_path" else float(errors[name])
            print(
                f"{row['batch_size']},"
                f"{name},"
                f"{float(timing['ms']):.6f},"
                f"{speedups[name]:.3f},"
                f"{err:.6g}"
            )


if __name__ == "__main__":
    main()
