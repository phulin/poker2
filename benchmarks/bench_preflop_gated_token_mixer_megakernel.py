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
    _preflop_token_mixer_norm_gate_residual_bf16_gate_triton,
    _preflop_token_mixer_norm_gate_residual_triton,
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


def eager_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    y = block.token_norm(x)
    mixed = block.token_mixer(y.transpose(1, 2)).transpose(1, 2)
    gate = block.token_gate(y)
    return x + mixed * torch.sigmoid(gate) / math.sqrt(2.0)


def current_triton_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    y = block.token_norm(x)
    gate = block.token_gate(y)
    return _preflop_token_mixer_gate_residual_triton(
        x,
        y,
        gate,
        block.token_mixer.linear_in.weight,
        block.token_mixer.linear_out.weight,
    )


def persistent_staged_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
    *,
    programs_per_sm: int,
) -> torch.Tensor:
    y = block.token_norm(x)
    gate = block.token_gate(y)
    return _preflop_token_mixer_gate_residual_persistent_triton(
        x,
        y,
        gate,
        block.token_mixer.linear_in.weight,
        block.token_mixer.linear_out.weight,
        programs_per_sm=programs_per_sm,
    )


def parse_programs_per_sm(value: str) -> list[int]:
    programs = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not programs:
        raise argparse.ArgumentTypeError("at least one program count is required")
    if any(program <= 0 for program in programs):
        raise argparse.ArgumentTypeError("program counts must be positive")
    return programs


def megakernel_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    return _preflop_token_mixer_norm_gate_residual_triton(
        x,
        block.token_norm.weight,
        block.token_gate.weight,
        block.token_gate.bias,
        block.token_mixer.linear_in.weight,
        block.token_mixer.linear_out.weight,
        eps=block.token_norm.eps,
    )


def megakernel_bf16_gate_token_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    return _preflop_token_mixer_norm_gate_residual_bf16_gate_triton(
        x,
        block.token_norm.weight,
        block.token_gate.weight,
        block.token_gate.bias,
        block.token_mixer.linear_in.weight,
        block.token_mixer.linear_out.weight,
        eps=block.token_norm.eps,
    )


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
    full_block: bool,
    persistent_programs: list[int],
    include_megakernel: bool,
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
        eager_out = eager_token_path(block, x)
        current_out = current_triton_token_path(block, x)
        persistent_outputs = {
            program: persistent_staged_token_path(block, x, programs_per_sm=program)
            for program in persistent_programs
        }
        if include_megakernel:
            mega_out = megakernel_token_path(block, x)
            mega_bf16_gate_out = megakernel_bf16_gate_token_path(block, x)
        sync(device)

        current_err = (current_out.float() - eager_out.float()).abs().max().item()
        persistent_errors = {
            f"persistent_staged_x{program}_token_path": (
                output.float() - eager_out.float()
            ).abs().max().item()
            for program, output in persistent_outputs.items()
        }
        megakernel_errors = {}
        if include_megakernel:
            megakernel_errors = {
                "megakernel_token_path": (
                    mega_out.float() - eager_out.float()
                ).abs().max().item(),
                "megakernel_bf16_gate_token_path": (
                    mega_bf16_gate_out.float() - eager_out.float()
                ).abs().max().item(),
            }

        variants: list[tuple[str, Callable[[], torch.Tensor]]] = [
            ("eager_token_path", lambda: eager_token_path(block, x)),
            ("current_triton_token_path", lambda: current_triton_token_path(block, x)),
        ]
        variants.extend(
            (
                f"persistent_staged_x{program}_token_path",
                lambda program=program: persistent_staged_token_path(
                    block,
                    x,
                    programs_per_sm=program,
                ),
            )
            for program in persistent_programs
        )
        if include_megakernel:
            variants.extend(
                (
                    ("megakernel_token_path", lambda: megakernel_token_path(block, x)),
                    (
                        "megakernel_bf16_gate_token_path",
                        lambda: megakernel_bf16_gate_token_path(block, x),
                    ),
                )
            )
        if full_block:
            variants.append(("block_forward_current", lambda: block(x)))

        timings = [
            {
                "name": name,
                "ms": time_call(fn, device=device, warmup=warmup, iters=iters),
            }
            for name, fn in variants
        ]

    timing_by_name = {row["name"]: float(row["ms"]) for row in timings}
    eager_ms = timing_by_name["eager_token_path"]
    return {
        "batch_size": batch_size,
        "max_abs_error": {
            "current_triton_token_path": current_err,
            **persistent_errors,
            **megakernel_errors,
        },
        "timings": timings,
        "speedups_vs_eager": {
            name: eager_ms / ms for name, ms in timing_by_name.items() if ms > 0.0
        },
    }


def parse_batch_sizes(value: str) -> list[int]:
    batch_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not batch_sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return batch_sizes


def load_config(path: str) -> DictConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return OmegaConf.load(config_path)


def config_int(cfg: DictConfig, dotted_key: str, default: int) -> int:
    value = OmegaConf.select(cfg, dotted_key, default=default)
    return int(value)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=None,
        help="Comma-separated batch sizes. Defaults to large sizes from the buckets Hydra config.",
    )
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--token-count", type=int, default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--full-block", action="store_true")
    parser.add_argument(
        "--persistent-programs",
        type=parse_programs_per_sm,
        default=[1, 2, 4, 8],
        help="Comma-separated persistent programs-per-SM variants to benchmark.",
    )
    parser.add_argument(
        "--skip-megakernel",
        action="store_true",
        help="Skip the known-slow full RMSNorm+gate megakernel variants.",
    )
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
        raise RuntimeError("This benchmark requires CUDA because the candidate is Triton-only")
    if dim <= 0:
        raise ValueError("--dim must be positive")
    if ffn_dim <= 0:
        raise ValueError("--ffn-dim must be positive")
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
            full_block=args.full_block,
            persistent_programs=args.persistent_programs,
            include_megakernel=not args.skip_megakernel,
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
        "persistent_programs": args.persistent_programs,
        "include_megakernel": not args.skip_megakernel,
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(payload, indent=2))
    print()
    variant_names = [item["name"] for item in results[0]["timings"]]
    metric_names = [name.removesuffix("_token_path") for name in variant_names]
    header = ["batch"]
    header.extend(f"{name}_ms" for name in metric_names)
    header.extend(f"{name}_speedup" for name in metric_names)
    header.extend(f"{name}_max_abs_err" for name in metric_names[1:])
    print(",".join(header))
    for row in results:
        timings = {item["name"]: float(item["ms"]) for item in row["timings"]}
        speedups = row["speedups_vs_eager"]
        errors = row["max_abs_error"]
        fields = [str(row["batch_size"])]
        fields.extend(f"{timings[name]:.6f}" for name in variant_names)
        fields.extend(f"{speedups[name]:.3f}" for name in variant_names)
        fields.extend(f"{errors[name]:.6g}" for name in variant_names[1:])
        print(",".join(fields))


if __name__ == "__main__":
    main()
