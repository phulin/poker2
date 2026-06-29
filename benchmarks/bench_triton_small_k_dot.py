"""Microbenchmark Triton `tl.dot` versus manual accumulation for small K.

The gated token mixer has tiny token-axis products (K=7 and K=28) and the
experimental megakernel currently computes gate-projection tiles with
`tl.dot` over K=32 chunks. This benchmark isolates those shapes so we can
decide whether `tl.dot` is a good primitive on the local GPU.

Example:

    uv run python benchmarks/bench_triton_small_k_dot.py --rows 57344 --json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - benchmark-only optional dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _small_k_dot_bf16_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        rows: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < rows) & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k[:, None] * N + offs_n[None, :],
                mask=(k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc)
        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _small_k_dot_fp32_tf32_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        rows: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < rows) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + k[:, None] * N + offs_n[None, :],
                mask=(k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            acc = tl.dot(a, b, acc, input_precision="tf32")
        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _small_k_dot_fp32_ieee_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        rows: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < rows) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + k[:, None] * N + offs_n[None, :],
                mask=(k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            acc = tl.dot(a, b, acc, input_precision="ieee")
        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < N),
        )

    @triton.jit
    def _small_k_manual_fp32_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        rows: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.static_range(0, K):
            a = tl.load(
                a_ptr + offs_m * K + k,
                mask=offs_m < rows,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + k * N + offs_n,
                mask=offs_n < N,
                other=0.0,
            ).to(tl.float32)
            acc += a[:, None] * b[None, :]
        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < rows) & (offs_n[None, :] < N),
        )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    fn: Callable[[], None],
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


def launch_dot_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    rows, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    _small_k_dot_bf16_kernel[grid](
        a,
        b,
        c,
        rows,
        k,
        n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


def launch_dot_fp32_tf32(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    rows, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    _small_k_dot_fp32_tf32_kernel[grid](
        a,
        b,
        c,
        rows,
        k,
        n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


def launch_dot_fp32_ieee(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    rows, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    _small_k_dot_fp32_ieee_kernel[grid](
        a,
        b,
        c,
        rows,
        k,
        n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )


def launch_manual_fp32(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    del block_k
    rows, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(rows, block_m), triton.cdiv(n, block_n))
    _small_k_manual_fp32_kernel[grid](
        a,
        b,
        c,
        rows,
        k,
        n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )


def benchmark_case(
    *,
    rows: int,
    k: int,
    n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    a = torch.randn(rows, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
    c = torch.empty(rows, n, device=device, dtype=torch.float32)
    ref = torch.matmul(a.float(), b.float())
    sync(device)
    variants: list[tuple[str, Callable[[], None]]] = [
        (
            "tl_dot_bf16",
            lambda: launch_dot_bf16(
                a, b, c, block_m=block_m, block_n=block_n, block_k=block_k
            ),
        ),
        (
            "tl_dot_fp32_tf32",
            lambda: launch_dot_fp32_tf32(
                a, b, c, block_m=block_m, block_n=block_n, block_k=block_k
            ),
        ),
        (
            "tl_dot_fp32_ieee",
            lambda: launch_dot_fp32_ieee(
                a, b, c, block_m=block_m, block_n=block_n, block_k=block_k
            ),
        ),
        (
            "manual_fp32",
            lambda: launch_manual_fp32(
                a, b, c, block_m=block_m, block_n=block_n, block_k=block_k
            ),
        ),
    ]

    timings = []
    max_abs_error = {}
    for name, fn in variants:
        fn()
        sync(device)
        max_abs_error[name] = float((c - ref).abs().max().item())
        timings.append(
            {
                "name": name,
                "ms": time_call(fn, device=device, warmup=warmup, iters=iters),
            }
        )
    best_ms = min(float(row["ms"]) for row in timings)
    return {
        "rows": rows,
        "k": k,
        "n": n,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "timings": timings,
        "relative_to_best": {
            row["name"]: float(row["ms"]) / best_ms for row in timings
        },
        "max_abs_error": max_abs_error,
    }


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases = []
    for item in value.split(","):
        if not item.strip():
            continue
        k_str, n_str = item.lower().split("x", 1)
        cases.append((int(k_str), int(n_str)))
    if not cases:
        raise argparse.ArgumentTypeError("at least one KxN case is required")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192 * 7)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("7x28,28x7,32x32,64x32"))
    parser.add_argument("--block-m", type=int, default=8)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if triton is None:
        raise RuntimeError("Triton is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA")
    if args.rows <= 0 or args.block_m <= 0 or args.block_n <= 0 or args.block_k <= 0:
        raise ValueError("rows and block sizes must be positive")

    results = [
        benchmark_case(
            rows=args.rows,
            k=k,
            n=n,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        for k, n in args.cases
    ]
    payload = {
        "device": torch.cuda.get_device_name(device),
        "rows": args.rows,
        "cases": [{"k": k, "n": n} for k, n in args.cases],
        "block_m": args.block_m,
        "block_n": args.block_n,
        "block_k": args.block_k,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(payload, indent=2))
    print()
    print("k,n,variant,ms,relative_to_best,max_abs_error")
    for result in results:
        rel = result["relative_to_best"]
        err = result["max_abs_error"]
        for timing in result["timings"]:
            name = timing["name"]
            print(
                f"{result['k']},{result['n']},{name},"
                f"{float(timing['ms']):.6f},{float(rel[name]):.3f},"
                f"{float(err[name]):.6g}"
            )


if __name__ == "__main__":
    main()
