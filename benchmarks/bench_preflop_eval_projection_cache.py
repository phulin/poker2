from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from p2.core.structured_config import NonlinearityType
from p2.env.card_utils import PREFLOP_HANDS
from p2.models.mlp.better_features import context_length
from p2.models.mlp.better_ffn import BetterPreflopGatedTokenMixerValueFFN
from p2.models.mlp.mlp_features import MLPFeatures


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


def make_model(
    *,
    dim: int,
    range_dim: int,
    ffn_dim: int,
    num_players: int,
    device: torch.device,
    weight_dtype: torch.dtype,
) -> BetterPreflopGatedTokenMixerValueFFN:
    model = BetterPreflopGatedTokenMixerValueFFN(
        num_actions=1,
        hidden_dim=dim,
        range_hidden_dim=range_dim,
        ffn_dim=ffn_dim,
        num_hidden_layers=0,
        num_value_layers=5,
        num_policy_layers=4,
        num_players=num_players,
        nonlinearity=NonlinearityType.leaky_relu,
        enforce_zero_sum=False,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(123))
    model.eval()
    return model.to(device=device, dtype=weight_dtype)


def make_features(
    *,
    batch_size: int,
    num_players: int,
    device: torch.device,
    dtype: torch.dtype,
) -> MLPFeatures:
    beliefs = torch.rand(
        batch_size,
        num_players,
        PREFLOP_HANDS,
        device=device,
        dtype=dtype,
    )
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
    context = torch.randn(
        batch_size,
        context_length(num_players),
        device=device,
        dtype=dtype,
    )
    return MLPFeatures(
        context=context,
        street=torch.zeros(batch_size, device=device, dtype=torch.long),
        to_act=torch.zeros(batch_size, device=device, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, device=device, dtype=torch.long),
        beliefs=beliefs.flatten(1),
        hand_dim=PREFLOP_HANDS,
    )


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def benchmark_batch(
    *,
    batch_size: int,
    dim: int,
    range_dim: int,
    ffn_dim: int,
    num_players: int,
    device: torch.device,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    use_autocast: bool,
    compile_dynamic: bool,
    timing_mode: str,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    cold_model = make_model(
        dim=dim,
        range_dim=range_dim,
        ffn_dim=ffn_dim,
        num_players=num_players,
        device=device,
        weight_dtype=weight_dtype,
    )
    warm_model = make_model(
        dim=dim,
        range_dim=range_dim,
        ffn_dim=ffn_dim,
        num_players=num_players,
        device=device,
        weight_dtype=weight_dtype,
    )
    warm_model.load_state_dict(cold_model.state_dict())
    cold_model.eval()
    warm_model.eval()

    features = make_features(
        batch_size=batch_size,
        num_players=num_players,
        device=device,
        dtype=input_dtype,
    )

    with torch.no_grad(), autocast_context(device, use_autocast):
        cold_static_base = cold_model.static_feature_base(features).clone()
        warm_static_base = warm_model.static_feature_base(features).clone()
        warm_model.prepare_preflop_eval_cache()

    cold_fn = torch.compile(
        lambda f, s: cold_model.forward_hand_values_static_base(
            f,
            s,
            apply_zero_sum=False,
        ),
        dynamic=compile_dynamic,
    )
    warm_fn = torch.compile(
        lambda f, s: warm_model.forward_hand_values_static_base(
            f,
            s,
            apply_zero_sum=False,
        ),
        dynamic=compile_dynamic,
    )

    def with_autocast(fn: Callable[[], torch.Tensor]) -> torch.Tensor:
        with autocast_context(device, use_autocast):
            return fn()

    with torch.no_grad():
        cold_out = with_autocast(lambda: cold_fn(features, cold_static_base))
        warm_out = with_autocast(lambda: warm_fn(features, warm_static_base))
        errors = {
            "compiled_warm_eval_projection_cache": (
                warm_out.float() - cold_out.float()
            ).abs().max().item()
        }
        variants: list[tuple[str, Callable[[], torch.Tensor]]] = [
            (
                "compiled_cold_live_projection",
                lambda: with_autocast(lambda: cold_fn(features, cold_static_base)),
            ),
            (
                "compiled_warm_eval_projection_cache",
                lambda: with_autocast(lambda: warm_fn(features, warm_static_base)),
            ),
        ]
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
    current_ms = timing_by_name["compiled_cold_live_projection"]
    return {
        "batch_size": batch_size,
        "rows": batch_size * num_players,
        "max_abs_error": errors,
        "timings": timings,
        "speedups_vs_cold": {
            name: current_ms / ms for name, ms in timing_by_name.items() if ms > 0.0
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--range-dim", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=None)
    parser.add_argument("--num-players", type=int, default=None)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--weight-dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--autocast", action="store_true")
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
    range_dim = (
        args.range_dim
        if args.range_dim is not None
        else config_int(cfg, "model.range_hidden_dim", 256)
    )
    ffn_dim = (
        args.ffn_dim if args.ffn_dim is not None else config_int(cfg, "model.ffn_dim", 256)
    )
    num_players = (
        args.num_players
        if args.num_players is not None
        else config_int(cfg, "env.num_players", 6)
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
            range_dim=range_dim,
            ffn_dim=ffn_dim,
            num_players=num_players,
            device=device,
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            use_autocast=args.autocast,
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
        "compile_dynamic": args.compile_dynamic,
        "timing_mode": args.timing_mode,
        "dim": dim,
        "range_dim": range_dim,
        "ffn_dim": ffn_dim,
        "num_players": num_players,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        return

    print()
    print("batch,rows,variant,ms,speedup_vs_cold,max_abs_error")
    for row in results:
        speedups = row["speedups_vs_cold"]
        errors = row["max_abs_error"]
        for timing in row["timings"]:
            name = str(timing["name"])
            err = 0.0 if name == "compiled_cold_live_projection" else float(errors[name])
            print(
                f"{row['batch_size']},"
                f"{row['rows']},"
                f"{name},"
                f"{float(timing['ms']):.6f},"
                f"{speedups[name]:.3f},"
                f"{err:.6g}"
            )


if __name__ == "__main__":
    main()
