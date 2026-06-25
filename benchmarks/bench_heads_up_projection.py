"""Benchmark 6-way-to-heads-up closing-belief projection."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from p2.search.fused_cfr_triton import select_heads_up_beliefs_triton


def _parse_rows(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype {name!r}")


def _cuda_time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iters)


def _make_live_players(rows: int, players: int, device: torch.device) -> torch.Tensor:
    p0 = torch.randint(0, players, (rows,), device=device)
    delta = torch.randint(1, players, (rows,), device=device)
    p1 = (p0 + delta) % players
    return torch.stack((p0, p1), dim=1).contiguous()


def benchmark_rows(
    *,
    rows: int,
    feature_rows: int,
    players: int,
    hand_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    block_h: int,
    num_warps: int | None,
) -> dict[str, float]:
    device = torch.device("cuda")
    beliefs = torch.rand(
        feature_rows,
        players * hand_dim,
        device=device,
        dtype=dtype,
    ).contiguous()
    positions = torch.randint(0, feature_rows, (rows,), device=device).contiguous()
    live_players = _make_live_players(rows, players, device)
    gather_index = live_players[:, :, None].expand(-1, 2, hand_dim)
    out = torch.empty((rows, 2, hand_dim), device=device, dtype=dtype)

    def torch_current() -> torch.Tensor:
        selected = beliefs[positions].view(rows, players, hand_dim)
        return selected.gather(1, gather_index)

    def triton_alloc() -> torch.Tensor:
        return select_heads_up_beliefs_triton(
            beliefs,
            positions,
            live_players,
            num_players=players,
            hand_dim=hand_dim,
            block_h=block_h,
            num_warps=num_warps,
        )

    def triton_reuse() -> torch.Tensor:
        return select_heads_up_beliefs_triton(
            beliefs,
            positions,
            live_players,
            num_players=players,
            hand_dim=hand_dim,
            out=out,
            block_h=block_h,
            num_warps=num_warps,
        )

    reference = torch_current()
    candidate = triton_reuse()
    torch.cuda.synchronize()
    max_diff = float((reference.float() - candidate.float()).abs().max().item())

    current_ms = _cuda_time_ms(torch_current, warmup=warmup, iters=iters)
    triton_alloc_ms = _cuda_time_ms(triton_alloc, warmup=warmup, iters=iters)
    triton_reuse_ms = _cuda_time_ms(triton_reuse, warmup=warmup, iters=iters)

    return {
        "rows": float(rows),
        "feature_rows": float(feature_rows),
        "torch_current_ms": current_ms,
        "triton_alloc_ms": triton_alloc_ms,
        "triton_reuse_ms": triton_reuse_ms,
        "triton_alloc_speedup": current_ms / triton_alloc_ms,
        "triton_reuse_speedup": current_ms / triton_reuse_ms,
        "max_diff": max_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="512,1536,4096,8192,32768")
    parser.add_argument("--feature-row-multiplier", type=float, default=1.0)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--hand-dim", type=int, default=169)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--block-h", type=int, default=256)
    parser.add_argument(
        "--num-warps",
        type=int,
        default=0,
        help="Triton num_warps; 0 uses the production auto-selection.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    rows_list = _parse_rows(args.rows)
    dtype = _dtype(args.dtype)
    num_warps = None if args.num_warps <= 0 else args.num_warps
    print(
        "heads_up_projection_bench "
        f"players={args.players} hand_dim={args.hand_dim} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters} "
        f"block_h={args.block_h} num_warps={num_warps or 'auto'}"
    )
    for rows in rows_list:
        feature_rows = max(rows, int(round(rows * args.feature_row_multiplier)))
        result = benchmark_rows(
            rows=rows,
            feature_rows=feature_rows,
            players=args.players,
            hand_dim=args.hand_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            block_h=args.block_h,
            num_warps=num_warps,
        )
        print(
            "rows={rows:.0f} feature_rows={feature_rows:.0f} "
            "torch={torch_current_ms:.6f}ms "
            "triton_alloc={triton_alloc_ms:.6f}ms "
            "triton_reuse={triton_reuse_ms:.6f}ms "
            "speedup_alloc={triton_alloc_speedup:.2f}x "
            "speedup_reuse={triton_reuse_speedup:.2f}x "
            "max_diff={max_diff:.3g}".format(**result)
        )


if __name__ == "__main__":
    main()
