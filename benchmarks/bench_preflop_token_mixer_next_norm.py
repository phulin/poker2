from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from p2.core.structured_config import NonlinearityType
from p2.models.mlp.better_ffn import (
    _PreflopGatedTokenMixerBlock,
    _preflop_token_mixer_gate_residual_next_norm_triton,
    _preflop_token_mixer_gate_residual_triton,
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
    timing_mode: str,
) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    if device.type == "cuda":
        if timing_mode == "cuda_graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = fn()
            for _ in range(warmup):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize(device)
            # Keep graph-owned outputs alive until after replay timing.
            _ = static_out
            return float(start.elapsed_time(end) / iters)

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
    weight_dtype: torch.dtype,
) -> _PreflopGatedTokenMixerBlock:
    block = _PreflopGatedTokenMixerBlock(
        dim,
        token_count=token_count,
        ffn_dim=ffn_dim,
        nonlinearity=NonlinearityType.leaky_relu,
    )
    block.eval()
    return block.to(device=device, dtype=weight_dtype)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def old_fastpath_block_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    y = block.token_norm(x)
    gate = block.token_gate(y)
    token_out = _preflop_token_mixer_gate_residual_triton(
        x,
        y,
        gate,
        block.token_mixer.linear_in.weight,
        block.token_mixer.linear_out.weight,
    )
    return token_out + block.ffn(token_out) / math.sqrt(2.0)


def naive_block_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
) -> torch.Tensor:
    y = block.token_norm(x)
    gate = block.token_gate(y)
    mixed = block.token_mixer(y.transpose(1, 2)).transpose(1, 2)
    token_out = x + mixed * torch.sigmoid(gate) / math.sqrt(2.0)
    return token_out + block.ffn(token_out) / math.sqrt(2.0)


def fused_next_norm_block_path(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
    *,
    block_b: int,
    num_warps: int,
) -> torch.Tensor:
    y = block.token_norm(x)
    gate = block.token_gate(y)
    token_out, ffn_in = _preflop_token_mixer_gate_residual_next_norm_triton(
        x,
        y,
        gate,
        block.token_mixer.linear_in.weight,
        block.token_mixer.linear_out.weight,
        block.ffn.norm.weight,
        eps=block.ffn.norm.eps,
        block_b=block_b,
        num_warps=num_warps,
    )
    h = block.ffn.linear_in(ffn_in)
    h = block.ffn.activation(h)
    h = block.ffn.linear_out(h)
    return token_out + h / math.sqrt(2.0)


def benchmark_batch(
    *,
    batch_size: int,
    dim: int,
    ffn_dim: int,
    token_count: int,
    device: torch.device,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    use_autocast: bool,
    include_compiled_naive: bool,
    timing_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    block = make_block(
        dim=dim,
        ffn_dim=ffn_dim,
        token_count=token_count,
        device=device,
        weight_dtype=weight_dtype,
    )
    x = torch.randn(batch_size, token_count, dim, device=device, dtype=input_dtype)

    def with_autocast(fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        with autocast_context(device, use_autocast):
            return fn()

    compiled_naive = None
    if include_compiled_naive:
        compiled_naive = torch.compile(
            lambda inp: naive_block_path(block, inp),
            dynamic=False,
        )

    with torch.no_grad():
        expected = with_autocast(lambda: naive_block_path(block, x))
        old_fastpath = with_autocast(lambda: old_fastpath_block_path(block, x))
        module_forward = with_autocast(lambda: block(x))
        compiled_naive_out = (
            None
            if compiled_naive is None
            else with_autocast(lambda: compiled_naive(x))
        )
        fused_b1 = with_autocast(
            lambda: fused_next_norm_block_path(block, x, block_b=1, num_warps=8)
        )
        fused_b2 = with_autocast(
            lambda: fused_next_norm_block_path(block, x, block_b=2, num_warps=8)
        )
        sync(device)

        errors = {
            "old_token_mixer_fastpath": (
                old_fastpath.float() - expected.float()
            ).abs().max().item(),
            "block_forward_module": (
                module_forward.float() - expected.float()
            ).abs().max().item(),
            "fused_next_norm_b1": (
                fused_b1.float() - expected.float()
            ).abs().max().item(),
            "fused_next_norm_b2": (
                fused_b2.float() - expected.float()
            ).abs().max().item(),
        }
        if compiled_naive_out is not None:
            errors["compiled_naive_block"] = (
                compiled_naive_out.float() - expected.float()
            ).abs().max().item()
        variants: list[tuple[str, Callable[[], torch.Tensor]]] = [
            (
                "old_token_mixer_fastpath",
                lambda: with_autocast(lambda: old_fastpath_block_path(block, x)),
            ),
            ("block_forward_module", lambda: with_autocast(lambda: block(x))),
        ]
        if compiled_naive is not None:
            variants.append(
                (
                    "compiled_naive_block",
                    lambda: with_autocast(lambda: compiled_naive(x)),
                )
            )
        variants.extend(
            [
                (
                    "fused_next_norm_b1",
                    lambda: with_autocast(
                        lambda: fused_next_norm_block_path(
                            block,
                            x,
                            block_b=1,
                            num_warps=8,
                        )
                    ),
                ),
                (
                    "fused_next_norm_b2",
                    lambda: with_autocast(
                        lambda: fused_next_norm_block_path(
                            block,
                            x,
                            block_b=2,
                            num_warps=8,
                        )
                    ),
                ),
            ]
        )
        timings = [
            {
                "name": name,
                "ms": time_call(
                    fn,
                    device=device,
                    warmup=warmup,
                    iters=iters,
                    timing_mode=timing_mode,
                ),
            }
            for name, fn in variants
        ]

    timing_by_name = {row["name"]: float(row["ms"]) for row in timings}
    current_ms = timing_by_name["old_token_mixer_fastpath"]
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
    parser.add_argument("--weight-dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--include-compiled-naive", action="store_true")
    parser.add_argument(
        "--timing-mode",
        choices=("launches", "cuda_graph"),
        default="launches",
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
        raise RuntimeError("This benchmark requires CUDA")
    if token_count != 7:
        raise ValueError("the current Triton candidate is specialized to 7 tokens")
    torch.set_float32_matmul_precision("high")
    input_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    weight_dtype_arg = args.weight_dtype if args.weight_dtype is not None else args.dtype
    weight_dtype = torch.float32 if weight_dtype_arg == "float32" else torch.bfloat16

    results = [
        benchmark_batch(
            batch_size=batch_size,
            dim=dim,
            ffn_dim=ffn_dim,
            token_count=token_count,
            device=device,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            use_autocast=args.autocast,
            include_compiled_naive=args.include_compiled_naive,
            timing_mode=args.timing_mode,
            warmup=args.warmup,
            iters=args.iters,
        )
        for batch_size in batch_sizes
    ]
    payload = {
        "device": torch.cuda.get_device_name(device),
        "config": args.config,
        "input_dtype": str(input_dtype).replace("torch.", ""),
        "weight_dtype": str(weight_dtype).replace("torch.", ""),
        "autocast": args.autocast,
        "include_compiled_naive": args.include_compiled_naive,
        "timing_mode": args.timing_mode,
        "dim": dim,
        "ffn_dim": ffn_dim,
        "token_count": token_count,
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
            err = float(errors[name])
            print(
                f"{row['batch_size']},"
                f"{name},"
                f"{float(timing['ms']):.6f},"
                f"{speedups[name]:.3f},"
                f"{err:.6g}"
            )


if __name__ == "__main__":
    main()
