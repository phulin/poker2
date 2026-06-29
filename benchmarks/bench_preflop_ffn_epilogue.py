from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - benchmark requires CUDA/Triton
    triton = None
    tl = None

from p2.core.structured_config import NonlinearityType
from p2.models.mlp.better_ffn import (
    _PreflopGatedTokenMixerBlock,
    _preflop_ffn_residual_next_token_norm_triton,
)


DEFAULT_CONFIG = "conf/config_rebel_preflop_buckets.yaml"


if triton is not None:

    @triton.jit
    def _ffn_linear_out_residual_next_norm_kernel(
        h_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        norm_weight_ptr,
        out_ptr,
        normed_out_ptr,
        rows: tl.constexpr,
        hidden_dim: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_mn = (offs_m[:, None] < rows) & (offs_n[None, :] < dim)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, hidden_dim, BLOCK_K):
            k = k0 + offs_k
            h = tl.load(
                h_ptr + offs_m[:, None] * hidden_dim + k[None, :],
                mask=(offs_m[:, None] < rows) & (k[None, :] < hidden_dim),
                other=0.0,
            )
            weight = tl.load(
                weight_ptr + offs_n[None, :] * hidden_dim + k[:, None],
                mask=(offs_n[None, :] < dim) & (k[:, None] < hidden_dim),
                other=0.0,
            )
            acc += tl.dot(h, weight, input_precision="tf32")

        bias = tl.load(bias_ptr + offs_n, mask=offs_n < dim, other=0.0).to(tl.float32)
        residual = tl.load(
            residual_ptr + offs_m[:, None] * dim + offs_n[None, :],
            mask=mask_mn,
            other=0.0,
        ).to(tl.float32)
        out = residual + (acc + bias[None, :]) * scale
        ss = tl.sum(tl.where(mask_mn, out * out, 0.0), axis=1)
        norm_weight = tl.load(
            norm_weight_ptr + offs_n,
            mask=offs_n < dim,
            other=0.0,
        ).to(tl.float32)
        normed = out * tl.rsqrt(ss[:, None] / dim + eps) * norm_weight[None, :]

        tl.store(out_ptr + offs_m[:, None] * dim + offs_n[None, :], out, mask=mask_mn)
        tl.store(
            normed_out_ptr + offs_m[:, None] * dim + offs_n[None, :],
            normed,
            mask=mask_mn,
        )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
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


def current_boundary(
    linear_out: nn.Linear,
    residual: torch.Tensor,
    h: torch.Tensor,
    next_norm: nn.RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    ffn_out = linear_out(h)
    return _preflop_ffn_residual_next_token_norm_triton(
        residual,
        ffn_out,
        next_norm.weight,
        eps=next_norm.eps,
        block_b=2,
        num_warps=8,
    )


def torch_boundary(
    linear_out: nn.Linear,
    residual: torch.Tensor,
    h: torch.Tensor,
    next_norm: nn.RMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = residual + linear_out(h) / math.sqrt(2.0)
    return out, next_norm(out)


def triton_linear_epilogue(
    linear_out: nn.Linear,
    residual: torch.Tensor,
    h: torch.Tensor,
    next_norm: nn.RMSNorm,
    *,
    linear_weight: torch.Tensor | None = None,
    block_m: int,
    block_n: int = 256,
    block_k: int = 64,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not h.is_contiguous():
        h = h.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if linear_out.bias is None:
        raise ValueError("linear_out bias is required")
    weight = linear_out.weight if linear_weight is None else linear_weight
    batch_size, token_count, dim = residual.shape
    rows = batch_size * token_count
    hidden_dim = h.shape[-1]
    if h.shape != (batch_size, token_count, hidden_dim):
        raise ValueError("h must be a [batch, token, hidden] tensor")
    if weight.shape != (dim, hidden_dim):
        raise ValueError("linear_out weight shape does not match residual/h")
    if weight.dtype != h.dtype:
        raise ValueError("Triton dot operands must have matching dtypes")
    if next_norm.weight is None or next_norm.weight.shape != (dim,):
        raise ValueError("next_norm weight must match residual dim")
    if dim > block_n:
        raise ValueError("block_n must cover the full output dimension")
    out = torch.empty_like(residual)
    normed_out = torch.empty_like(residual)
    grid = (triton.cdiv(rows, block_m),)
    _ffn_linear_out_residual_next_norm_kernel[grid](
        h,
        weight,
        linear_out.bias,
        residual,
        next_norm.weight,
        out,
        normed_out,
        rows,
        hidden_dim,
        dim,
        next_norm.eps,
        1.0 / math.sqrt(2.0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out, normed_out


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
    include_compiled: bool,
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
    next_norm = nn.RMSNorm(dim, eps=1e-5).to(device=device, dtype=weight_dtype)
    residual = torch.randn(batch_size, token_count, dim, device=device, dtype=input_dtype)
    ffn_in = torch.randn(batch_size, token_count, dim, device=device, dtype=input_dtype)
    with autocast_context(device, use_autocast):
        h = block.ffn.linear_in(ffn_in)
        h = block.ffn.activation(h)
    h = h.contiguous()
    triton_linear_out_weight = block.ffn.linear_out.weight.detach()
    if triton_linear_out_weight.dtype != h.dtype:
        triton_linear_out_weight = triton_linear_out_weight.to(dtype=h.dtype)
    triton_linear_out_weight = triton_linear_out_weight.contiguous()

    compiled_boundary = None
    if include_compiled:
        compiled_boundary = torch.compile(
            lambda residual_arg, h_arg: torch_boundary(
                block.ffn.linear_out,
                residual_arg,
                h_arg,
                next_norm,
            ),
            dynamic=False,
        )

    def with_autocast(fn: Callable[[], tuple[torch.Tensor, torch.Tensor]]):
        with autocast_context(device, use_autocast):
            return fn()

    with torch.no_grad():
        expected = with_autocast(
            lambda: current_boundary(block.ffn.linear_out, residual, h, next_norm)
        )
        variants: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = [
            (
                "cublas_linear_plus_triton_epilogue",
                lambda: with_autocast(
                    lambda: current_boundary(block.ffn.linear_out, residual, h, next_norm)
                ),
            ),
            (
                "torch_naive_boundary",
                lambda: with_autocast(
                    lambda: torch_boundary(block.ffn.linear_out, residual, h, next_norm)
                ),
            ),
            (
                "triton_fulln_bm4",
                lambda: with_autocast(
                    lambda: triton_linear_epilogue(
                        block.ffn.linear_out,
                        residual,
                        h,
                        next_norm,
                        linear_weight=triton_linear_out_weight,
                        block_m=4,
                    )
                ),
            ),
            (
                "triton_fulln_bm8",
                lambda: with_autocast(
                    lambda: triton_linear_epilogue(
                        block.ffn.linear_out,
                        residual,
                        h,
                        next_norm,
                        linear_weight=triton_linear_out_weight,
                        block_m=8,
                    )
                ),
            ),
            (
                "triton_fulln_bm16",
                lambda: with_autocast(
                    lambda: triton_linear_epilogue(
                        block.ffn.linear_out,
                        residual,
                        h,
                        next_norm,
                        linear_weight=triton_linear_out_weight,
                        block_m=16,
                    )
                ),
            ),
        ]
        if compiled_boundary is not None:
            variants.insert(
                2,
                (
                    "compiled_torch_boundary",
                    lambda: with_autocast(lambda: compiled_boundary(residual, h)),
                ),
            )

        errors = {}
        for name, fn in variants:
            if name == "cublas_linear_plus_triton_epilogue":
                continue
            actual = fn()
            errors[name] = {
                "out": (actual[0].float() - expected[0].float()).abs().max().item(),
                "normed": (actual[1].float() - expected[1].float()).abs().max().item(),
            }
        sync(device)

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
    current_ms = timing_by_name["cublas_linear_plus_triton_epilogue"]
    return {
        "batch_size": batch_size,
        "rows": batch_size * token_count,
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
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--weight-dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--include-compiled", action="store_true")
    parser.add_argument(
        "--timing-mode",
        choices=("launches", "cuda_graph"),
        default="cuda_graph",
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
            include_compiled=args.include_compiled,
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
        "include_compiled": args.include_compiled,
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
    print("batch,rows,variant,ms,speedup_vs_current,max_abs_error_out,max_abs_error_normed")
    for row in results:
        speedups = row["speedups_vs_current"]
        errors = row["max_abs_error"]
        for timing in row["timings"]:
            name = str(timing["name"])
            error = errors.get(name, {"out": 0.0, "normed": 0.0})
            print(
                f"{row['batch_size']},"
                f"{row['rows']},"
                f"{name},"
                f"{float(timing['ms']):.6f},"
                f"{speedups[name]:.3f},"
                f"{float(error['out']):.6g},"
                f"{float(error['normed']):.6g}"
            )


if __name__ == "__main__":
    main()
