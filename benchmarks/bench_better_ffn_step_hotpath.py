from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable

import torch

from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.better_features import context_length
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.mlp_features import MLPFeatures
from p2.search.chance_node_helper import ChanceNodeHelper


def make_model(device: torch.device, compile_model: bool) -> BetterFFN:
    model = BetterFFN(
        num_actions=5,
        hidden_dim=1024,
        range_hidden_dim=256,
        ffn_dim=1024,
        num_hidden_layers=4,
        num_policy_layers=2,
        num_value_layers=1,
        num_players=2,
        board_interaction_dim=64,
        policy_rank=128,
        policy_hand_bias_rank=32,
        value_rank=128,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(123))
    model = model.to(device)
    model.eval()
    if compile_model:
        model.compile_forward_modes(dynamic=True)
    return model


def make_features(batch_size: int, device: torch.device) -> MLPFeatures:
    beliefs = torch.full(
        (batch_size, 2, NUM_HANDS),
        1.0 / NUM_HANDS,
        dtype=torch.float32,
        device=device,
    )
    board = torch.full((batch_size, 5), -1, dtype=torch.long, device=device)
    if batch_size >= 2:
        board[1::2, :3] = torch.tensor([12, 25, 38], device=device)
    return MLPFeatures(
        context=torch.zeros(batch_size, context_length(2), device=device),
        street=torch.zeros(batch_size, dtype=torch.long, device=device),
        to_act=torch.zeros(batch_size, dtype=torch.long, device=device),
        board=board,
        beliefs=beliefs.view(batch_size, -1),
    )


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    name: str,
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float | str]:
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
        elapsed_ms = start.elapsed_time(end)
    else:
        start_time = time.perf_counter()
        for _ in range(iters):
            fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return {"name": name, "ms": elapsed_ms / iters}


def bench_forward(
    model: BetterFFN,
    features: MLPFeatures,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[dict[str, float | str]]:
    autocast_enabled = device.type == "cuda"

    def value_only() -> object:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
            ),
        ):
            return model(features, include_policy=False)

    def both() -> object:
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
            ),
        ):
            return model(features)

    return [
        time_call("forward_value", value_only, device, warmup, iters),
        time_call("forward_both", both, device, warmup, iters),
    ]


def bench_chance(
    model: BetterFFN,
    batch_size: int,
    sample_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float | str]:
    helper = ChanceNodeHelper(
        device=device,
        float_dtype=torch.float32,
        num_players=2,
        model=model,
    )
    helper.FLOP_SAMPLE_SIZE = sample_size
    root_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    root_features = make_features(batch_size, device)
    pre_chance_beliefs = torch.full(
        (batch_size, 2, NUM_HANDS),
        1.0 / NUM_HANDS,
        dtype=torch.float32,
        device=device,
    )

    def fn() -> object:
        return helper.flop_chance_values(
            root_indices,
            root_features,
            pre_chance_beliefs,
        )

    return time_call("flop_chance_values", fn, device, warmup, iters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chance-batch-size", type=int, default=8)
    parser.add_argument("--flop-sample-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("all", "forward", "chance"),
        default="all",
    )
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = make_model(device, compile_model=not args.no_compile)
    results = []
    if args.mode in {"all", "forward"}:
        features = make_features(args.batch_size, device)
        results.extend(bench_forward(model, features, device, args.warmup, args.iters))
    if args.mode in {"all", "chance"}:
        results.append(
            bench_chance(
                model,
                args.chance_batch_size,
                args.flop_sample_size,
                device,
                max(1, args.warmup // 2),
                max(3, args.iters // 2),
            )
        )
    print(
        json.dumps(
            {
                "device": str(device),
                "batch_size": args.batch_size,
                "chance_batch_size": args.chance_batch_size,
                "flop_sample_size": args.flop_sample_size,
                "compiled": not args.no_compile,
                "mode": args.mode,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
