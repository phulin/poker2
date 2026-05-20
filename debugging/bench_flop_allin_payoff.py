from __future__ import annotations

import itertools
import math

import torch
import triton
import triton.language as tl

from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.env.rules_triton import rank_7_cards_single_batch_triton


@triton.jit
def _payoff_tile_kernel(
    ranks,
    out,
    T: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BI: tl.constexpr,
    BJ: tl.constexpr,
):
    pi = tl.program_id(0)
    pj = tl.program_id(1)
    pt = tl.program_id(2)
    oi = pi * BI + tl.arange(0, BI)
    oj = pj * BJ + tl.arange(0, BJ)
    ot = pt * BT + tl.arange(0, BT)
    mi = oi < H
    mj = oj < H
    mt = ot < T

    ri = tl.load(
        ranks + ot[:, None] * H + oi[None, :],
        mask=mt[:, None] & mi[None, :],
        other=0,
    )
    rj = tl.load(
        ranks + ot[:, None] * H + oj[None, :],
        mask=mt[:, None] & mj[None, :],
        other=0,
    )
    wins = ri[:, :, None] > rj[:, None, :]
    losses = ri[:, :, None] < rj[:, None, :]
    valid = mt[:, None, None] & mi[None, :, None] & mj[None, None, :]
    contrib = tl.sum(
        tl.where(valid & wins, 1, 0) - tl.where(valid & losses, 1, 0),
        axis=0,
    )
    tl.atomic_add(
        out + oi[:, None] * H + oj[None, :],
        contrib,
        sem="relaxed",
        mask=mi[:, None] & mj[None, :],
    )


def _sync() -> None:
    torch.cuda.synchronize()


def _event_time(fn, *, repeats: int = 5, warmup: int = 2) -> tuple[list[float], torch.Tensor]:
    for _ in range(warmup):
        out = fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    out = None
    for _ in range(repeats):
        start.record()
        out = fn()
        end.record()
        _sync()
        times.append(float(start.elapsed_time(end)))
    assert out is not None
    return times, out


def _summarize(name: str, times: list[float]) -> None:
    ts = torch.tensor(times, dtype=torch.float64)
    print(
        f"{name}: median={ts.median().item():.3f} ms "
        f"mean={ts.mean().item():.3f} ms min={ts.min().item():.3f} ms "
        f"max={ts.max().item():.3f} ms"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"mem free={free / 1024**3:.2f} GiB total={total / 1024**3:.2f} GiB")

    flop_cards = [0, 1, 2]
    flop = torch.tensor(flop_cards, device=device, dtype=torch.int16)
    remaining = [c for c in range(52) if c not in set(flop_cards)]
    runouts = torch.tensor(
        list(itertools.combinations(remaining, 2)),
        dtype=torch.int16,
        device=device,
    )
    assert runouts.shape[0] == math.comb(49, 2)

    full_boards = torch.cat(
        [flop.view(1, 3).expand(runouts.shape[0], 3), runouts],
        dim=1,
    ).contiguous()
    combos = hand_combos_tensor(device=device).to(torch.int16)
    cards = torch.cat(
        [
            combos.view(1, NUM_HANDS, 2).expand(runouts.shape[0], NUM_HANDS, 2),
            full_boards.view(runouts.shape[0], 1, 5).expand(
                runouts.shape[0],
                NUM_HANDS,
                5,
            ),
        ],
        dim=-1,
    ).reshape(-1, 7).contiguous()
    print("runouts:", runouts.shape[0], "hands:", NUM_HANDS, "rank rows:", cards.shape[0])

    rank_times, ranks = _event_time(
        lambda: rank_7_cards_single_batch_triton(cards).view(runouts.shape[0], NUM_HANDS),
        repeats=10,
        warmup=3,
    )
    ranks = ranks.contiguous()
    _summarize("rank all runout x hand scores", rank_times)

    def torch_payoff() -> torch.Tensor:
        return (
            (ranks[:, :, None] > ranks[:, None, :]).sum(dim=0, dtype=torch.int32)
            - (ranks[:, :, None] < ranks[:, None, :]).sum(dim=0, dtype=torch.int32)
        )

    torch_times, ref = _event_time(torch_payoff, repeats=10, warmup=3)
    _summarize("payoff matrix from existing ranks, torch broadcast", torch_times)

    def triton_payoff(*, bt: int, bi: int, bj: int) -> torch.Tensor:
        out = torch.empty((NUM_HANDS, NUM_HANDS), device=device, dtype=torch.int32)
        out.zero_()
        grid = (
            triton.cdiv(NUM_HANDS, bi),
            triton.cdiv(NUM_HANDS, bj),
            triton.cdiv(runouts.shape[0], bt),
        )
        _payoff_tile_kernel[grid](
            ranks,
            out,
            runouts.shape[0],
            NUM_HANDS,
            BT=bt,
            BI=bi,
            BJ=bj,
            num_warps=8,
        )
        return out

    configs = [(64, 16, 16), (32, 16, 16), (64, 8, 16), (64, 16, 8)]
    first = triton_payoff(bt=64, bi=16, bj=16)
    _sync()
    print(
        "triton max_abs_diff:",
        int((first - ref).abs().max().item()),
        "checksum:",
        int(first.sum(dtype=torch.int64).item()),
    )
    for bt, bi, bj in configs:
        times, out = _event_time(
            lambda bt=bt, bi=bi, bj=bj: triton_payoff(bt=bt, bi=bi, bj=bj),
            repeats=6,
            warmup=2,
        )
        _summarize(f"triton tiled payoff BT={bt} BI={bi} BJ={bj}", times)

    def total_triton() -> torch.Tensor:
        r = rank_7_cards_single_batch_triton(cards).view(runouts.shape[0], NUM_HANDS).contiguous()
        out = torch.empty((NUM_HANDS, NUM_HANDS), device=device, dtype=torch.int32)
        out.zero_()
        bt, bi, bj = 32, 16, 16
        grid = (
            triton.cdiv(NUM_HANDS, bi),
            triton.cdiv(NUM_HANDS, bj),
            triton.cdiv(runouts.shape[0], bt),
        )
        _payoff_tile_kernel[grid](
            r,
            out,
            runouts.shape[0],
            NUM_HANDS,
            BT=bt,
            BI=bi,
            BJ=bj,
            num_warps=8,
        )
        return out

    total_triton_times, total_triton_out = _event_time(total_triton, repeats=10, warmup=3)
    _summarize("total rank + triton tiled payoff", total_triton_times)
    print("total triton max_abs_diff:", int((total_triton_out - ref).abs().max().item()))

    def total_torch() -> torch.Tensor:
        r = rank_7_cards_single_batch_triton(cards).view(runouts.shape[0], NUM_HANDS)
        return (
            (r[:, :, None] > r[:, None, :]).sum(dim=0, dtype=torch.int32)
            - (r[:, :, None] < r[:, None, :]).sum(dim=0, dtype=torch.int32)
        )

    total_times, total = _event_time(total_torch, repeats=10, warmup=3)
    _summarize("total rank + torch payoff matrix", total_times)
    print("final checksum:", int(total.sum(dtype=torch.int64).item()))


if __name__ == "__main__":
    main()
