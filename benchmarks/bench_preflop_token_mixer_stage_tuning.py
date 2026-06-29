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
from p2.models.mlp.better_ffn import (
    _PreflopGatedTokenMixerBlock,
    _preflop_token_mixer_gate_residual_persistent_triton,
    _preflop_token_mixer_gate_residual_triton,
)


DEFAULT_CONFIG = "conf/config_rebel_preflop_buckets.yaml"
DEFAULT_VARIANTS = (
    "grid_b8d32",
    "grid_b8d32w1",
    "grid_b8d32w2",
    "grid_b4d32",
    "grid_b8d16",
    "grid_b8d64",
    "grid_b16d16",
    "grid_b16d32",
    "grid_b16d64",
    "persistent_b8d32x8",
    "persistent_b8d32x8w1",
    "persistent_b8d32x8w2",
    "persistent_b8d64x8",
    "persistent_b16d32x8",
    "persistent_b16d64x8",
)


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


def parse_variants(value: str) -> list[str]:
    variants = [part.strip() for part in value.split(",") if part.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("at least one variant is required")
    return variants


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


def eager_stage(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    mixed = block.token_mixer(y.transpose(1, 2)).transpose(1, 2)
    return x + mixed * torch.sigmoid(gate) / math.sqrt(2.0)


def split_warps(text: str) -> tuple[str, int]:
    if "w" not in text:
        return text, 4
    body, warps_text = text.rsplit("w", maxsplit=1)
    return body, int(warps_text)


def parse_variant_name(name: str) -> tuple[str, int, int, int, int]:
    if name.startswith("grid_b"):
        shape, num_warps = split_warps(name.removeprefix("grid_b"))
        block_b_text, block_d_text = shape.split("d", maxsplit=1)
        return "grid", int(block_b_text), int(block_d_text), 0, num_warps
    if name.startswith("persistent_b"):
        shape, num_warps = split_warps(name.removeprefix("persistent_b"))
        block_b_text, tail = shape.split("d", maxsplit=1)
        block_d_text, programs_text = tail.split("x", maxsplit=1)
        return (
            "persistent",
            int(block_b_text),
            int(block_d_text),
            int(programs_text),
            num_warps,
        )
    raise ValueError(f"unknown variant {name!r}")


def variant_call(
    *,
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
) -> torch.Tensor:
    kind, block_b, block_d, programs_per_sm, num_warps = parse_variant_name(name)
    if kind == "grid":
        return _preflop_token_mixer_gate_residual_triton(
            x,
            y,
            gate,
            w_in,
            w_out,
            block_b=block_b,
            block_d=block_d,
            num_warps=num_warps,
        )
    return _preflop_token_mixer_gate_residual_persistent_triton(
        x,
        y,
        gate,
        w_in,
        w_out,
        programs_per_sm=programs_per_sm,
        block_b=block_b,
        block_d=block_d,
        num_warps=num_warps,
    )


def benchmark_batch(
    *,
    batch_size: int,
    dim: int,
    ffn_dim: int,
    token_count: int,
    variants: list[str],
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
        y = block.token_norm(x)
        gate = block.token_gate(y)
        expected = eager_stage(block, x, y, gate)
        sync(device)

        rows: list[dict[str, float | str]] = []
        for name in variants:
            actual = variant_call(
                name=name,
                x=x,
                y=y,
                gate=gate,
                w_in=block.token_mixer.linear_in.weight,
                w_out=block.token_mixer.linear_out.weight,
            )
            sync(device)
            max_abs_error = (actual.float() - expected.float()).abs().max().item()
            ms = time_call(
                lambda name=name: variant_call(
                    name=name,
                    x=x,
                    y=y,
                    gate=gate,
                    w_in=block.token_mixer.linear_in.weight,
                    w_out=block.token_mixer.linear_out.weight,
                ),
                device=device,
                warmup=warmup,
                iters=iters,
            )
            rows.append({"name": name, "ms": ms, "max_abs_error": max_abs_error})

    baseline = next(row for row in rows if row["name"] == "grid_b8d32")
    baseline_ms = float(baseline["ms"])
    return {
        "batch_size": batch_size,
        "timings": rows,
        "speedups_vs_grid_b8d32": {
            str(row["name"]): baseline_ms / float(row["ms"])
            for row in rows
            if float(row["ms"]) > 0.0
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=None)
    parser.add_argument("--variants", type=parse_variants, default=list(DEFAULT_VARIANTS))
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
        raise RuntimeError("This benchmark requires CUDA because the candidates are Triton-only")
    if token_count != 7:
        raise ValueError("the current Triton candidates are specialized to 7 tokens")
    if "grid_b8d32" not in args.variants:
        raise ValueError("--variants must include grid_b8d32 as the baseline")

    torch.set_float32_matmul_precision("high")
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    results = [
        benchmark_batch(
            batch_size=batch_size,
            dim=dim,
            ffn_dim=ffn_dim,
            token_count=token_count,
            variants=args.variants,
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
        "variants": args.variants,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        return

    print()
    print("batch,variant,ms,speedup_vs_grid_b8d32,max_abs_error")
    for row in results:
        speedups = row["speedups_vs_grid_b8d32"]
        for timing in row["timings"]:
            name = str(timing["name"])
            print(
                f"{row['batch_size']},"
                f"{name},"
                f"{float(timing['ms']):.6f},"
                f"{speedups[name]:.3f},"
                f"{float(timing['max_abs_error']):.6g}"
            )


if __name__ == "__main__":
    main()
