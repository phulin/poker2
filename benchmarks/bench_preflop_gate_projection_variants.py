"""Benchmark gated-token-mixer gate projection layouts.

The first megakernel computes the gate projection as seven separate
`[BLOCK_B, K] @ [K, BLOCK_D]` dots per K tile. This benchmark isolates that
projection and compares it with a token-combined layout that treats the 7
tokens as part of the M dimension.

Example:

    uv run python benchmarks/bench_preflop_gate_projection_variants.py --batch-sizes 8192,65536 --json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - benchmark-only optional dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _gate_separate_fp32_kernel(
        y_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_bd = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)
        bias = tl.load(bias_ptr + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        g0 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g1 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g2 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g3 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g4 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g5 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g6 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        for k0 in tl.range(0, dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_bk = (offs_b[:, None] < batch_size) & (offs_k[None, :] < dim)
            w = tl.load(
                w_ptr + offs_d[None, :] * dim + offs_k[:, None],
                mask=(offs_k[:, None] < dim) & (offs_d[None, :] < dim),
                other=0.0,
            ).to(tl.float32)
            y0 = tl.load(y_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_k[None, :], mask=mask_bk, other=0.0).to(tl.float32)
            y1 = tl.load(y_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_k[None, :], mask=mask_bk, other=0.0).to(tl.float32)
            y2 = tl.load(y_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_k[None, :], mask=mask_bk, other=0.0).to(tl.float32)
            y3 = tl.load(y_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_k[None, :], mask=mask_bk, other=0.0).to(tl.float32)
            y4 = tl.load(y_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_k[None, :], mask=mask_bk, other=0.0).to(tl.float32)
            y5 = tl.load(y_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_k[None, :], mask=mask_bk, other=0.0).to(tl.float32)
            y6 = tl.load(y_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_k[None, :], mask=mask_bk, other=0.0).to(tl.float32)
            g0 += tl.dot(y0, w, input_precision="tf32")
            g1 += tl.dot(y1, w, input_precision="tf32")
            g2 += tl.dot(y2, w, input_precision="tf32")
            g3 += tl.dot(y3, w, input_precision="tf32")
            g4 += tl.dot(y4, w, input_precision="tf32")
            g5 += tl.dot(y5, w, input_precision="tf32")
            g6 += tl.dot(y6, w, input_precision="tf32")
        tl.store(out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], g0, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], g1, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], g2, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], g3, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], g4, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], g5, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], g6, mask=mask_bd)

    @triton.jit
    def _gate_separate_bf16_kernel(
        y_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_b = tl.program_id(0)
        pid_d = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_bd = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)
        bias = tl.load(bias_ptr + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)
        g0 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g1 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g2 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g3 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g4 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g5 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        g6 = tl.broadcast_to(bias[None, :], (BLOCK_B, BLOCK_D))
        for k0 in tl.range(0, dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_bk = (offs_b[:, None] < batch_size) & (offs_k[None, :] < dim)
            w = tl.load(
                w_ptr + offs_d[None, :] * dim + offs_k[:, None],
                mask=(offs_k[:, None] < dim) & (offs_d[None, :] < dim),
                other=0.0,
            )
            y0 = tl.load(y_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_k[None, :], mask=mask_bk, other=0.0)
            y1 = tl.load(y_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_k[None, :], mask=mask_bk, other=0.0)
            y2 = tl.load(y_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_k[None, :], mask=mask_bk, other=0.0)
            y3 = tl.load(y_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_k[None, :], mask=mask_bk, other=0.0)
            y4 = tl.load(y_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_k[None, :], mask=mask_bk, other=0.0)
            y5 = tl.load(y_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_k[None, :], mask=mask_bk, other=0.0)
            y6 = tl.load(y_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_k[None, :], mask=mask_bk, other=0.0)
            g0 += tl.dot(y0, w)
            g1 += tl.dot(y1, w)
            g2 += tl.dot(y2, w)
            g3 += tl.dot(y3, w)
            g4 += tl.dot(y4, w)
            g5 += tl.dot(y5, w)
            g6 += tl.dot(y6, w)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 0) * dim + offs_d[None, :], g0, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 1) * dim + offs_d[None, :], g1, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 2) * dim + offs_d[None, :], g2, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 3) * dim + offs_d[None, :], g3, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 4) * dim + offs_d[None, :], g4, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 5) * dim + offs_d[None, :], g5, mask=mask_bd)
        tl.store(out_ptr + (offs_b[:, None] * 7 + 6) * dim + offs_d[None, :], g6, mask=mask_bd)

    @triton.jit
    def _gate_combined_bf16_kernel(
        y_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        rows: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        token_rows = rows * 7
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        batch = offs_m // 7
        token = offs_m - batch * 7
        valid_m = offs_m < token_rows
        acc = tl.broadcast_to(
            tl.load(bias_ptr + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)[None, :],
            (BLOCK_M, BLOCK_D),
        )
        for k0 in tl.range(0, dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            y = tl.load(
                y_ptr + (batch[:, None] * 7 + token[:, None]) * dim + offs_k[None, :],
                mask=valid_m[:, None] & (offs_k[None, :] < dim),
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_d[None, :] * dim + offs_k[:, None],
                mask=(offs_k[:, None] < dim) & (offs_d[None, :] < dim),
                other=0.0,
            )
            acc += tl.dot(y, w)
        tl.store(
            out_ptr + (batch[:, None] * 7 + token[:, None]) * dim + offs_d[None, :],
            acc,
            mask=valid_m[:, None] & (offs_d[None, :] < dim),
        )

    @triton.jit
    def _gate_combined_fp32_kernel(
        y_ptr,
        w_ptr,
        bias_ptr,
        out_ptr,
        rows: tl.constexpr,
        dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)
        token_rows = rows * 7
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        batch = offs_m // 7
        token = offs_m - batch * 7
        valid_m = offs_m < token_rows
        acc = tl.broadcast_to(
            tl.load(bias_ptr + offs_d, mask=offs_d < dim, other=0.0).to(tl.float32)[None, :],
            (BLOCK_M, BLOCK_D),
        )
        for k0 in tl.range(0, dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            y = tl.load(
                y_ptr + (batch[:, None] * 7 + token[:, None]) * dim + offs_k[None, :],
                mask=valid_m[:, None] & (offs_k[None, :] < dim),
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                w_ptr + offs_d[None, :] * dim + offs_k[:, None],
                mask=(offs_k[:, None] < dim) & (offs_d[None, :] < dim),
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(y, w, input_precision="tf32")
        tl.store(
            out_ptr + (batch[:, None] * 7 + token[:, None]) * dim + offs_d[None, :],
            acc,
            mask=valid_m[:, None] & (offs_d[None, :] < dim),
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
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


def launch_separate_fp32(
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    *,
    block_b: int,
    block_d: int,
    block_k: int,
    combined_block_m: int,
) -> torch.Tensor:
    del combined_block_m
    batch_size, _, dim = y.shape
    grid = (triton.cdiv(batch_size, block_b), triton.cdiv(dim, block_d))
    _gate_separate_fp32_kernel[grid](
        y,
        weight,
        bias,
        out,
        batch_size,
        dim,
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def launch_separate_bf16(
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    *,
    block_b: int,
    block_d: int,
    block_k: int,
    combined_block_m: int,
) -> torch.Tensor:
    del combined_block_m
    batch_size, _, dim = y.shape
    grid = (triton.cdiv(batch_size, block_b), triton.cdiv(dim, block_d))
    _gate_separate_bf16_kernel[grid](
        y,
        weight,
        bias,
        out,
        batch_size,
        dim,
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def launch_combined_bf16(
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    *,
    block_b: int,
    block_d: int,
    block_k: int,
    combined_block_m: int,
) -> torch.Tensor:
    del block_b
    batch_size, _, dim = y.shape
    grid = (triton.cdiv(batch_size * 7, combined_block_m), triton.cdiv(dim, block_d))
    _gate_combined_bf16_kernel[grid](
        y,
        weight,
        bias,
        out,
        batch_size,
        dim,
        BLOCK_M=combined_block_m,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def launch_combined_fp32(
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    *,
    block_b: int,
    block_d: int,
    block_k: int,
    combined_block_m: int,
) -> torch.Tensor:
    del block_b
    batch_size, _, dim = y.shape
    grid = (triton.cdiv(batch_size * 7, combined_block_m), triton.cdiv(dim, block_d))
    _gate_combined_fp32_kernel[grid](
        y,
        weight,
        bias,
        out,
        batch_size,
        dim,
        BLOCK_M=combined_block_m,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


def torch_linear(y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.linear(y, weight, bias)


def benchmark_batch(
    *,
    batch_size: int,
    dim: int,
    device: torch.device,
    warmup: int,
    iters: int,
    block_b: int,
    block_d: int,
    block_k: int,
    combined_block_m: int,
) -> dict[str, object]:
    y = torch.randn(batch_size, 7, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, dim, device=device, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=device, dtype=torch.bfloat16)
    out = torch.empty(batch_size, 7, dim, device=device, dtype=torch.float32)
    ref = torch_linear(y, weight, bias).float()
    sync(device)
    variants: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("torch_linear", lambda: torch_linear(y, weight, bias)),
        (
            "separate_token_fp32_tf32",
            lambda: launch_separate_fp32(
                y,
                weight,
                bias,
                out,
                block_b=block_b,
                block_d=block_d,
                block_k=block_k,
                combined_block_m=combined_block_m,
            ),
        ),
        (
            "separate_token_bf16",
            lambda: launch_separate_bf16(
                y,
                weight,
                bias,
                out,
                block_b=block_b,
                block_d=block_d,
                block_k=block_k,
                combined_block_m=combined_block_m,
            ),
        ),
        (
            "combined_token_fp32_tf32",
            lambda: launch_combined_fp32(
                y,
                weight,
                bias,
                out,
                block_b=block_b,
                block_d=block_d,
                block_k=block_k,
                combined_block_m=combined_block_m,
            ),
        ),
        (
            "combined_token_bf16",
            lambda: launch_combined_bf16(
                y,
                weight,
                bias,
                out,
                block_b=block_b,
                block_d=block_d,
                block_k=block_k,
                combined_block_m=combined_block_m,
            ),
        ),
    ]
    timings = []
    errors = {}
    for name, fn in variants:
        result = fn()
        sync(device)
        errors[name] = float((result.float() - ref).abs().max().item())
        timings.append(
            {
                "name": name,
                "ms": time_call(fn, device=device, warmup=warmup, iters=iters),
            }
        )
    best = min(float(row["ms"]) for row in timings)
    return {
        "batch_size": batch_size,
        "timings": timings,
        "relative_to_best": {row["name"]: float(row["ms"]) / best for row in timings},
        "max_abs_error": errors,
    }


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[8192, 65536])
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--block-b", type=int, default=8)
    parser.add_argument("--block-d", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--combined-block-m", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if triton is None:
        raise RuntimeError("Triton is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA")
    if args.dim <= 0:
        raise ValueError("--dim must be positive")
    results = [
        benchmark_batch(
            batch_size=batch_size,
            dim=args.dim,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            block_b=args.block_b,
            block_d=args.block_d,
            block_k=args.block_k,
            combined_block_m=args.combined_block_m,
        )
        for batch_size in args.batch_sizes
    ]
    payload = {
        "device": torch.cuda.get_device_name(device),
        "dim": args.dim,
        "block_b": args.block_b,
        "block_d": args.block_d,
        "block_k": args.block_k,
        "combined_block_m": args.combined_block_m,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(json.dumps(payload, indent=2))
    print()
    print("batch,variant,ms,relative_to_best,max_abs_error")
    for result in results:
        rel = result["relative_to_best"]
        err = result["max_abs_error"]
        for timing in result["timings"]:
            name = timing["name"]
            print(
                f"{result['batch_size']},{name},"
                f"{float(timing['ms']):.6f},{float(rel[name]):.3f},"
                f"{float(err[name]):.6g}"
            )


if __name__ == "__main__":
    main()
