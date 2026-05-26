from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from p2.env.pbs_env import PBSEnv
from p2.env.triton_pbs_env import (
    TritonPBSEnv,
    TritonPBSEnvBuffer,
    triton_is_available,
)


def synchronize() -> None:
    torch.cuda.synchronize()


def cuda_time_ms(fn: Callable[[], None], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    synchronize()
    return start.elapsed_time(end) / repeats


def make_env_pair(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    seed: int,
) -> tuple[PBSEnv, TritonPBSEnv, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    button = torch.arange(num_envs, device=device, dtype=torch.long) % num_players
    base_deck = torch.arange(5, device=device, dtype=torch.long)
    deck = (base_deck[None, :] + torch.arange(num_envs, device=device)[:, None] * 5) % 52
    torch_rng = torch.Generator(device=device).manual_seed(seed)
    triton_rng = torch.Generator(device=device).manual_seed(seed)
    base = PBSEnv(
        num_envs=num_envs,
        num_players=num_players,
        starting_stack=stack,
        sb=sb,
        bb=bb,
        device=device,
        rng=torch_rng,
    )
    triton_env = TritonPBSEnv(
        num_envs=num_envs,
        num_players=num_players,
        starting_stack=stack,
        sb=sb,
        bb=bb,
        device=device,
        rng=triton_rng,
    )
    base.reset(force_button=button, force_deck=deck)
    triton_env.reset(force_button=button, force_deck=deck)
    return base, triton_env, button, deck


def assert_same_fields(base: PBSEnv, triton_env: TritonPBSEnv) -> None:
    fields = (
        "deck",
        "deck_pos",
        "button",
        "street",
        "to_act",
        "last_to_act",
        "pot",
        "min_raise",
        "last_aggressive_amount",
        "actions_this_round",
        "actions_last_round",
        "stacks",
        "starting_stacks",
        "scale",
        "committed",
        "chips_placed",
        "has_folded",
        "is_allin",
        "acted_this_round",
        "done",
        "winner",
        "winners",
        "board_indices",
        "last_board_indices",
        "board_onehot",
    )
    for field in fields:
        if not torch.equal(getattr(base, field), getattr(triton_env, field)):
            raise AssertionError(f"field mismatch: {field}")


def print_result(name: str, torch_ms: float, triton_ms: float, rows: int) -> None:
    speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
    print(
        f"{name:<18} rows={rows:>7,d} | "
        f"torch={torch_ms:>8.4f} ms | triton={triton_ms:>8.4f} ms | "
        f"speedup={speedup:>6.2f}x"
    )


def bench_reset(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats: int,
) -> None:
    base, triton_env, button, deck = make_env_pair(num_envs, num_players, stack, sb, bb, 10)
    base.reset(force_button=button, force_deck=deck)
    triton_env.reset(force_button=button, force_deck=deck)
    assert_same_fields(base, triton_env)
    torch_ms = cuda_time_ms(lambda: base.reset(force_button=button, force_deck=deck), warmup, repeats)
    triton_ms = cuda_time_ms(lambda: triton_env.reset(force_button=button, force_deck=deck), warmup, repeats)
    assert_same_fields(base, triton_env)
    print_result("reset", torch_ms, triton_ms, num_envs)


def bench_legal(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats: int,
) -> None:
    base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 20)
    expected = base.legal_bins_amounts_and_mask()
    actual = triton_env.legal_bins_amounts_and_mask()
    if not torch.equal(expected[0], actual[0]) or not torch.equal(expected[1], actual[1]):
        raise AssertionError("legal output mismatch")
    torch_ms = cuda_time_ms(lambda: base.legal_bins_amounts_and_mask(), warmup, repeats)
    triton_ms = cuda_time_ms(lambda: triton_env.legal_bins_amounts_and_mask(), warmup, repeats)
    print_result("legal", torch_ms, triton_ms, num_envs)


def bench_step_bins(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats: int,
) -> None:
    actions = torch.ones(num_envs, dtype=torch.long, device="cuda")

    def make() -> tuple[PBSEnv, TritonPBSEnv]:
        base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 30)
        return base, triton_env

    base, triton_env = make()
    expected = base.step_bins(actions)
    actual = triton_env.step_bins(actions)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        if not torch.equal(expected_tensor, actual_tensor):
            raise AssertionError("step_bins output mismatch")
    assert_same_fields(base, triton_env)

    def torch_fn() -> None:
        env, _ = make()
        env.step_bins(actions)

    def triton_fn() -> None:
        _, env = make()
        env.step_bins(actions)

    torch_ms = cuda_time_ms(torch_fn, warmup, repeats)
    triton_ms = cuda_time_ms(triton_fn, warmup, repeats)
    print_result("step_bins", torch_ms, triton_ms, num_envs)


def bench_step_direct(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats: int,
) -> None:
    action_indices = torch.ones(num_envs, dtype=torch.long, device="cuda")
    bet_amounts = torch.zeros(num_envs, dtype=torch.long, device="cuda")

    def make() -> tuple[PBSEnv, TritonPBSEnv]:
        base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 40)
        return base, triton_env

    base, triton_env = make()
    expected = base.step(action_indices, bet_amounts)
    actual = triton_env.step(action_indices, bet_amounts)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        if not torch.equal(expected_tensor, actual_tensor):
            raise AssertionError("step output mismatch")
    assert_same_fields(base, triton_env)

    def torch_fn() -> None:
        env, _ = make()
        env.step(action_indices, bet_amounts)

    def triton_fn() -> None:
        _, env = make()
        env.step(action_indices, bet_amounts)

    torch_ms = cuda_time_ms(torch_fn, warmup, repeats)
    triton_ms = cuda_time_ms(triton_fn, warmup, repeats)
    print_result("step", torch_ms, triton_ms, num_envs)


def bench_copy(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats: int,
) -> None:
    base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 50)
    dst_base = PBSEnv(num_envs=num_envs, num_players=num_players, starting_stack=stack, sb=sb, bb=bb, device="cuda")
    dst_triton = TritonPBSEnv(num_envs=num_envs, num_players=num_players, starting_stack=stack, sb=sb, bb=bb, device="cuda")
    src = torch.arange(num_envs, device="cuda")
    dst = torch.arange(num_envs, device="cuda")
    dst_base.copy_state_from(base, src, dst)
    dst_triton.copy_state_from(triton_env, src, dst)
    assert_same_fields(dst_base, dst_triton)
    torch_ms = cuda_time_ms(lambda: dst_base.copy_state_from(base, src, dst), warmup, repeats)
    triton_ms = cuda_time_ms(lambda: dst_triton.copy_state_from(triton_env, src, dst), warmup, repeats)
    print_result("copy", torch_ms, triton_ms, num_envs)


def bench_gather(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats: int,
) -> None:
    base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 60)
    indices = torch.arange(num_envs, device="cuda").flip(0)
    expected = base.gather_rows(indices)
    actual = triton_env.gather_rows(indices)
    assert_same_fields(expected, actual)
    torch_ms = cuda_time_ms(lambda: base.gather_rows(indices), warmup, repeats)
    triton_ms = cuda_time_ms(lambda: triton_env.gather_rows(indices), warmup, repeats)
    print_result("gather", torch_ms, triton_ms, num_envs)


def bench_gather_into(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats: int,
) -> None:
    base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 65)
    indices = torch.arange(num_envs, device="cuda").flip(0)
    dst_ids = torch.arange(num_envs, device="cuda")
    base_dst = PBSEnv(
        num_envs=num_envs,
        num_players=num_players,
        starting_stack=stack,
        sb=sb,
        bb=bb,
        device="cuda",
    )
    buffer = TritonPBSEnvBuffer(triton_env, initial_capacity=num_envs)
    dst_view = buffer.view(num_envs)

    base_dst.copy_state_from(base, indices, dst_ids)
    triton_env.gather_rows_into(indices, dst_view)
    assert_same_fields(base_dst, dst_view)
    torch_ms = cuda_time_ms(lambda: base_dst.copy_state_from(base, indices, dst_ids), warmup, repeats)
    triton_ms = cuda_time_ms(lambda: triton_env.gather_rows_into(indices, dst_view), warmup, repeats)
    print_result("gather_into", torch_ms, triton_ms, num_envs)


def bench_repeat(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats_count: int,
) -> None:
    base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 70)
    repeats = torch.full((num_envs,), 2, dtype=torch.long, device="cuda")
    output_size = num_envs * 2
    expected = base.repeat_interleave(repeats, output_size=output_size)
    actual = triton_env.repeat_interleave(repeats, output_size=output_size)
    assert_same_fields(expected, actual)
    torch_ms = cuda_time_ms(
        lambda: base.repeat_interleave(repeats, output_size=output_size),
        warmup,
        repeats_count,
    )
    triton_ms = cuda_time_ms(
        lambda: triton_env.repeat_interleave(repeats, output_size=output_size),
        warmup,
        repeats_count,
    )
    print_result("repeat", torch_ms, triton_ms, output_size)


def bench_repeat_into(
    num_envs: int,
    num_players: int,
    stack: int,
    sb: int,
    bb: int,
    warmup: int,
    repeats_count: int,
) -> None:
    base, triton_env, _, _ = make_env_pair(num_envs, num_players, stack, sb, bb, 75)
    repeat_counts = torch.full((num_envs,), 2, dtype=torch.long, device="cuda")
    output_size = num_envs * 2
    indices = torch.arange(num_envs, device="cuda").repeat_interleave(
        repeat_counts, output_size=output_size
    )
    dst_ids = torch.arange(output_size, device="cuda")
    base_dst = PBSEnv(
        num_envs=output_size,
        num_players=num_players,
        starting_stack=stack,
        sb=sb,
        bb=bb,
        device="cuda",
    )
    buffer = TritonPBSEnvBuffer(triton_env, initial_capacity=output_size)
    dst_view = buffer.view(output_size)
    offsets = buffer.long_workspace(num_envs)

    base_dst.copy_state_from(base, indices, dst_ids)
    triton_env.repeat_interleave_into(
        repeat_counts,
        dst_view,
        output_size=output_size,
        max_repeat=2,
        offsets_out=offsets,
    )
    assert_same_fields(base_dst, dst_view)
    torch_ms = cuda_time_ms(lambda: base_dst.copy_state_from(base, indices, dst_ids), warmup, repeats_count)
    triton_ms = cuda_time_ms(
        lambda: triton_env.repeat_interleave_into(
            repeat_counts,
            dst_view,
            output_size=output_size,
            max_repeat=2,
            offsets_out=offsets,
        ),
        warmup,
        repeats_count,
    )
    print_result("repeat_into", torch_ms, triton_ms, output_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PBSEnv vs TritonPBSEnv kernels")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--num-players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=1000)
    parser.add_argument("--sb", type=int, default=5)
    parser.add_argument("--bb", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    if not triton_is_available():
        raise SystemExit("Triton is not installed.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    print(f"Device: {torch.cuda.get_device_name(torch.device('cuda'))}")
    print(
        f"N={args.num_envs:,}, P={args.num_players}, "
        f"warmup={args.warmup}, repeats={args.repeats}"
    )
    bench_reset(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_legal(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_step_bins(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_step_direct(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_copy(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_gather(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_gather_into(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_repeat(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)
    bench_repeat_into(args.num_envs, args.num_players, args.stack, args.sb, args.bb, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
