from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
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
    _run_preflop_gated_token_mixer_blocks,
    _preflop_token_mixer_gate_residual_next_norm_triton,
    _preflop_token_mixer_gate_residual_triton,
)


DEFAULT_CONFIG = "conf/config_rebel_preflop_buckets.yaml"
NEXT_NORM_CUTOFF = 16_384


if triton is not None:

    @triton.jit
    def _ffn_residual_next_token_norm_kernel(
        residual_ptr,
        ffn_out_ptr,
        norm_weight_ptr,
        out_ptr,
        normed_out_ptr,
        batch_size: tl.constexpr,
        token_count: tl.constexpr,
        dim: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_b[:, None] < batch_size) & (offs_d[None, :] < dim)
        base = (offs_b[:, None] * token_count + pid_t) * dim + offs_d[None, :]

        residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
        ffn_out = tl.load(ffn_out_ptr + base, mask=mask, other=0.0).to(tl.float32)
        out = residual + ffn_out * scale
        ss = tl.sum(tl.where(mask, out * out, 0.0), axis=1)
        norm_weight = tl.load(norm_weight_ptr + offs_d, mask=offs_d < dim, other=0.0).to(
            tl.float32
        )
        normed = out * tl.rsqrt(ss[:, None] / dim + eps) * norm_weight[None, :]

        tl.store(out_ptr + base, out, mask=mask)
        tl.store(normed_out_ptr + base, normed, mask=mask)


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


def make_blocks(
    *,
    depth: int,
    dim: int,
    ffn_dim: int,
    token_count: int,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> nn.ModuleList:
    blocks = nn.ModuleList(
        [
            _PreflopGatedTokenMixerBlock(
                dim,
                token_count=token_count,
                ffn_dim=ffn_dim,
                nonlinearity=NonlinearityType.leaky_relu,
            )
            for _ in range(depth)
        ]
    )
    blocks.eval()
    return blocks.to(device=device, dtype=weight_dtype)


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def ffn_residual_next_token_norm_triton(
    residual: torch.Tensor,
    ffn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float,
    block_b: int,
    block_d: int = 256,
    num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not ffn_out.is_contiguous():
        ffn_out = ffn_out.contiguous()
    if residual.shape != ffn_out.shape:
        raise ValueError("residual and ffn_out must have matching shapes")
    out = torch.empty_like(residual)
    normed_out = torch.empty_like(residual)
    batch_size, token_count, dim = residual.shape
    if norm_weight.shape != (dim,):
        raise ValueError("norm_weight must match the hidden dimension")
    if dim > block_d:
        raise ValueError("block_d must cover the full hidden dimension")
    grid = (triton.cdiv(batch_size, block_b), token_count)
    _ffn_residual_next_token_norm_kernel[grid](
        residual,
        ffn_out,
        norm_weight,
        out,
        normed_out,
        batch_size,
        token_count,
        dim,
        eps,
        1.0 / math.sqrt(2.0),
        BLOCK_B=block_b,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return out, normed_out


def block_from_token_norm(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    gate = block.token_gate(y)
    if x.shape[0] <= NEXT_NORM_CUTOFF:
        token_out, ffn_in = _preflop_token_mixer_gate_residual_next_norm_triton(
            x,
            y,
            gate,
            block.token_mixer.linear_in.weight,
            block.token_mixer.linear_out.weight,
            block.ffn.norm.weight,
            eps=block.ffn.norm.eps,
            block_b=2,
            num_warps=8,
        )
    else:
        token_out = _preflop_token_mixer_gate_residual_triton(
            x,
            y,
            gate,
            block.token_mixer.linear_in.weight,
            block.token_mixer.linear_out.weight,
        )
        ffn_in = block.ffn.norm(token_out)
    h = block.ffn.linear_in(ffn_in)
    h = block.ffn.activation(h)
    h = block.ffn.linear_out(h)
    return token_out + h / math.sqrt(2.0)


def block_with_next_token_norm(
    block: _PreflopGatedTokenMixerBlock,
    x: torch.Tensor,
    y: torch.Tensor,
    next_norm: nn.RMSNorm,
    *,
    block_b: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = block.token_gate(y)
    if x.shape[0] <= NEXT_NORM_CUTOFF:
        token_out, ffn_in = _preflop_token_mixer_gate_residual_next_norm_triton(
            x,
            y,
            gate,
            block.token_mixer.linear_in.weight,
            block.token_mixer.linear_out.weight,
            block.ffn.norm.weight,
            eps=block.ffn.norm.eps,
            block_b=2,
            num_warps=8,
        )
    else:
        token_out = _preflop_token_mixer_gate_residual_triton(
            x,
            y,
            gate,
            block.token_mixer.linear_in.weight,
            block.token_mixer.linear_out.weight,
        )
        ffn_in = block.ffn.norm(token_out)
    h = block.ffn.linear_in(ffn_in)
    h = block.ffn.activation(h)
    ffn_out = block.ffn.linear_out(h)
    return ffn_residual_next_token_norm_triton(
        token_out,
        ffn_out,
        next_norm.weight,
        eps=next_norm.eps,
        block_b=block_b,
        num_warps=8,
    )


def current_stack_path(blocks: Sequence[_PreflopGatedTokenMixerBlock], x: torch.Tensor) -> torch.Tensor:
    for block in blocks:
        x = block(x)
    return x


def naive_block_path(block: _PreflopGatedTokenMixerBlock, x: torch.Tensor) -> torch.Tensor:
    y = block.token_norm(x)
    gate = block.token_gate(y)
    mixed = block.token_mixer(y.transpose(1, 2)).transpose(1, 2)
    token_out = x + mixed * torch.sigmoid(gate) / math.sqrt(2.0)
    return token_out + block.ffn(token_out) / math.sqrt(2.0)


def naive_stack_path(blocks: Sequence[_PreflopGatedTokenMixerBlock], x: torch.Tensor) -> torch.Tensor:
    for block in blocks:
        x = naive_block_path(block, x)
    return x


def wired_stack_path(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    return _run_preflop_gated_token_mixer_blocks(blocks, x)


_dynamo_disabled_wired_stack_path = torch.compiler.disable(wired_stack_path)


def cross_norm_stack_path(
    blocks: Sequence[_PreflopGatedTokenMixerBlock],
    x: torch.Tensor,
    *,
    block_b: int,
) -> torch.Tensor:
    precomputed_y: torch.Tensor | None = None
    for index, block in enumerate(blocks):
        y = block.token_norm(x) if precomputed_y is None else precomputed_y
        if index + 1 < len(blocks):
            next_block = blocks[index + 1]
            x, precomputed_y = block_with_next_token_norm(
                block,
                x,
                y,
                next_block.token_norm,
                block_b=block_b,
            )
        else:
            x = block_from_token_norm(block, x, y)
            precomputed_y = None
    return x


def benchmark_batch(
    *,
    batch_size: int,
    depth: int,
    dim: int,
    ffn_dim: int,
    token_count: int,
    device: torch.device,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    use_autocast: bool,
    include_compiled_naive: bool,
    include_compiled_wired: bool,
    compile_dynamic: bool,
    timing_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    blocks = make_blocks(
        depth=depth,
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
            lambda inp: naive_stack_path(blocks, inp),
            dynamic=compile_dynamic,
        )
    compiled_wired = None
    compiled_disabled_wired = None
    compiled_cross_b2 = None
    if include_compiled_wired:
        compiled_wired = torch.compile(
            lambda inp: wired_stack_path(blocks, inp),
            dynamic=compile_dynamic,
        )
        compiled_disabled_wired = torch.compile(
            lambda inp: _dynamo_disabled_wired_stack_path(blocks, inp),
            dynamic=compile_dynamic,
        )
        compiled_cross_b2 = torch.compile(
            lambda inp: cross_norm_stack_path(blocks, inp, block_b=2),
            dynamic=compile_dynamic,
        )

    with torch.no_grad():
        expected = with_autocast(lambda: current_stack_path(blocks, x))
        wired_stack = with_autocast(lambda: wired_stack_path(blocks, x))
        compiled_naive_out = (
            None
            if compiled_naive is None
            else with_autocast(lambda: compiled_naive(x))
        )
        compiled_wired_out = (
            None
            if compiled_wired is None
            else with_autocast(lambda: compiled_wired(x))
        )
        compiled_disabled_wired_out = (
            None
            if compiled_disabled_wired is None
            else with_autocast(lambda: compiled_disabled_wired(x))
        )
        compiled_cross_b2_out = (
            None
            if compiled_cross_b2 is None
            else with_autocast(lambda: compiled_cross_b2(x))
        )
        cross_b1 = with_autocast(lambda: cross_norm_stack_path(blocks, x, block_b=1))
        cross_b2 = with_autocast(lambda: cross_norm_stack_path(blocks, x, block_b=2))
        sync(device)

        errors = {
            "wired_stack_runner": (wired_stack.float() - expected.float()).abs().max().item(),
            "cross_boundary_norm_b1": (cross_b1.float() - expected.float()).abs().max().item(),
            "cross_boundary_norm_b2": (cross_b2.float() - expected.float()).abs().max().item(),
        }
        if compiled_naive_out is not None:
            errors["compiled_naive_stack"] = (
                compiled_naive_out.float() - expected.float()
            ).abs().max().item()
        if compiled_wired_out is not None:
            errors["compiled_wired_stack"] = (
                compiled_wired_out.float() - expected.float()
            ).abs().max().item()
        if compiled_disabled_wired_out is not None:
            errors["compiled_disabled_wired_stack"] = (
                compiled_disabled_wired_out.float() - expected.float()
            ).abs().max().item()
        if compiled_cross_b2_out is not None:
            errors["compiled_cross_boundary_norm_b2"] = (
                compiled_cross_b2_out.float() - expected.float()
            ).abs().max().item()
        variants: list[tuple[str, Callable[[], torch.Tensor]]] = [
            ("old_module_loop", lambda: with_autocast(lambda: current_stack_path(blocks, x))),
            ("wired_stack_runner", lambda: with_autocast(lambda: wired_stack_path(blocks, x))),
        ]
        if compiled_naive is not None:
            variants.append(
                (
                    "compiled_naive_stack",
                    lambda: with_autocast(lambda: compiled_naive(x)),
                )
            )
        if compiled_wired is not None:
            variants.append(
                (
                    "compiled_wired_stack",
                    lambda: with_autocast(lambda: compiled_wired(x)),
                )
            )
        if compiled_disabled_wired is not None:
            variants.append(
                (
                    "compiled_disabled_wired_stack",
                    lambda: with_autocast(lambda: compiled_disabled_wired(x)),
                )
            )
        if compiled_cross_b2 is not None:
            variants.append(
                (
                    "compiled_cross_boundary_norm_b2",
                    lambda: with_autocast(lambda: compiled_cross_b2(x)),
                )
            )
        variants.extend(
            [
                (
                    "cross_boundary_norm_b1",
                    lambda: with_autocast(lambda: cross_norm_stack_path(blocks, x, block_b=1)),
                ),
                (
                    "cross_boundary_norm_b2",
                    lambda: with_autocast(lambda: cross_norm_stack_path(blocks, x, block_b=2)),
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
    current_ms = timing_by_name["old_module_loop"]
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
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--token-count", type=int, default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--weight-dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--include-compiled-naive", action="store_true")
    parser.add_argument("--include-compiled-wired", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
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
    if token_count != 7:
        raise ValueError("the current Triton candidates are specialized to 7 tokens")
    if args.depth <= 1:
        raise ValueError("depth must be greater than one for cross-boundary fusion")
    torch.set_float32_matmul_precision("high")
    input_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    weight_dtype_arg = args.weight_dtype if args.weight_dtype is not None else args.dtype
    weight_dtype = torch.float32 if weight_dtype_arg == "float32" else torch.bfloat16

    results = [
        benchmark_batch(
            batch_size=batch_size,
            depth=args.depth,
            dim=dim,
            ffn_dim=ffn_dim,
            token_count=token_count,
            device=device,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            use_autocast=args.autocast,
            include_compiled_naive=args.include_compiled_naive,
            include_compiled_wired=args.include_compiled_wired,
            compile_dynamic=args.compile_dynamic,
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
        "include_compiled_wired": args.include_compiled_wired,
        "compile_dynamic": args.compile_dynamic,
        "timing_mode": args.timing_mode,
        "depth": args.depth,
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
            err = 0.0 if name == "old_module_loop" else float(errors[name])
            print(
                f"{row['batch_size']},"
                f"{name},"
                f"{float(timing['ms']):.6f},"
                f"{speedups[name]:.3f},"
                f"{err:.6g}"
            )


if __name__ == "__main__":
    main()
