#!/usr/bin/env python3
"""Precompute a preflop all-in matchup payoff table.

The saved table has shape ``[1326, 1326]``. Row ``h`` is the hero hole-card
combo, column ``v`` is the opponent combo, and each entry is:

    E_board[sign(rank(h, board) - rank(v, board))]

over valid five-card boards that collide with neither hand. Ties contribute
zero. Incompatible hole-card pairs are stored as zero.

At CFR time, dequantize ``payoff_i16`` with ``payoff_i16.float() / 32768``.
Then a preflop all-in-call leaf can use this table as a dense matvec:

    numer = villain_beliefs @ payoff.T
    denom = calculate_unblocked_mass(villain_beliefs)
    ev = numer / denom.clamp_min(eps)

The exact CUDA path is intended for offline generation:

    uv run python scripts/precompute_preflop_allin_table.py --output /tmp/preflop.pt.zst

Use ``--sample-boards`` for a quick smoke test or approximate table.
"""

from __future__ import annotations

import argparse
import io
import math
import time
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path

import torch
import zstandard as zstd

from p2.env.card_utils import (
    NUM_HANDS,
    board_allowed_hands,
    combo_compatible_tensor,
)
from p2.env.rules import rank_hands as rank_hands_torch

TOTAL_PREFLOP_BOARDS = math.comb(52, 5)
EXACT_DENOMINATOR = math.comb(48, 5)


def _rank_hands(board: torch.Tensor, use_triton: bool) -> torch.Tensor:
    if use_triton:
        from p2.env.rules_triton import rank_hands_triton

        ranks, _ = rank_hands_triton(board)
    else:
        ranks, _ = rank_hands_torch(board)
    return ranks.to(torch.int32).contiguous()


def _iter_board_chunks(
    *,
    board_chunk: int,
    sample_boards: int | None,
    seed: int,
) -> Iterator[torch.Tensor]:
    if sample_boards is not None and sample_boards < TOTAL_PREFLOP_BOARDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for start in range(0, sample_boards, board_chunk):
            count = min(board_chunk, sample_boards - start)
            keys = torch.rand(count, 52, generator=generator)
            boards = keys.topk(5, dim=1).indices.sort(dim=1).values.to(torch.long)
            yield boards.contiguous()
        return

    buf: list[tuple[int, int, int, int, int]] = []
    for combo in combinations(range(52), 5):
        buf.append(combo)
        if len(buf) == board_chunk:
            yield torch.tensor(buf, dtype=torch.long)
            buf.clear()
    if buf:
        yield torch.tensor(buf, dtype=torch.long)


def precompute_payoff(
    *,
    device: torch.device,
    board_chunk: int,
    hero_block: int,
    sample_boards: int | None,
    seed: int,
    use_triton: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, object]]:
    if board_chunk <= 0:
        raise ValueError("--board-chunk must be positive")
    if hero_block <= 0:
        raise ValueError("--hero-block must be positive")

    is_exact = sample_boards is None or sample_boards >= TOTAL_PREFLOP_BOARDS
    num_boards = TOTAL_PREFLOP_BOARDS if is_exact else int(sample_boards)

    compatible = combo_compatible_tensor(device=device).contiguous()
    compatible_cpu = compatible.cpu()
    score_cpu = torch.zeros(NUM_HANDS, NUM_HANDS, dtype=torch.int32)
    counts_cpu = (
        None
        if is_exact
        else torch.zeros(NUM_HANDS, NUM_HANDS, dtype=torch.int32)
    )

    started = time.perf_counter()
    chunks = math.ceil(num_boards / board_chunk)
    done = 0
    for chunk_idx, boards_cpu in enumerate(
        _iter_board_chunks(
            board_chunk=board_chunk,
            sample_boards=sample_boards,
            seed=seed,
        ),
        start=1,
    ):
        done += boards_cpu.shape[0]
        board = boards_cpu.to(device=device, non_blocking=True)
        ranks = _rank_hands(board.int(), use_triton=use_triton)
        allowed = board_allowed_hands(board).to(device=device).contiguous()

        for hero_start in range(0, NUM_HANDS, hero_block):
            hero_end = min(hero_start + hero_block, NUM_HANDS)
            hero_ranks = ranks[:, hero_start:hero_end, None]
            villain_ranks = ranks[:, None, :]
            valid = (
                allowed[:, hero_start:hero_end, None]
                & allowed[:, None, :]
                & compatible[hero_start:hero_end][None, :, :]
            )

            wins = (hero_ranks > villain_ranks) & valid
            losses = (hero_ranks < villain_ranks) & valid
            block_score = wins.sum(dim=0, dtype=torch.int32) - losses.sum(
                dim=0, dtype=torch.int32
            )
            score_cpu[hero_start:hero_end] += block_score.cpu()

            if counts_cpu is not None:
                counts_cpu[hero_start:hero_end] += valid.sum(
                    dim=0, dtype=torch.int32
                ).cpu()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - started
        rate = done / max(elapsed, 1e-9)
        remaining = (num_boards - done) / max(rate, 1e-9)
        print(
            f"chunk {chunk_idx:>5}/{chunks:<5} boards={done:,}/{num_boards:,} "
            f"rate={rate:,.1f} boards/s eta={remaining / 60:.1f} min",
            flush=True,
        )

    score_cpu.masked_fill_(~compatible_cpu, 0)

    payoff = score_cpu.float()
    if counts_cpu is None:
        payoff[compatible_cpu] /= EXACT_DENOMINATOR
    else:
        payoff = torch.where(
            counts_cpu > 0,
            payoff / counts_cpu.clamp(min=1).float(),
            torch.zeros_like(payoff),
        )
        counts_cpu.masked_fill_(~compatible_cpu, 0)
    payoff.masked_fill_(~compatible_cpu, 0.0)

    # Remove one-ULP accumulation asymmetries in approximate mode and make the
    # invariant explicit for downstream tests.
    payoff = 0.5 * (payoff - payoff.T)
    payoff.masked_fill_(~compatible_cpu, 0.0)

    metadata: dict[str, object] = {
        "semantics": "row hero combo, column opponent combo; +1 win, 0 tie, -1 loss averaged over valid boards",
        "combo_order": "p2.env.card_utils.hand_combos_tensor",
        "num_hands": NUM_HANDS,
        "num_boards": int(num_boards),
        "exact": bool(is_exact),
        "exact_denominator": EXACT_DENOMINATOR,
        "sample_seed": int(seed),
        "board_chunk": int(board_chunk),
        "hero_block": int(hero_block),
        "device": str(device),
        "ranker": "triton" if use_triton else "torch",
    }
    return payoff.contiguous(), score_cpu, counts_cpu, metadata


def _quantize_payoff_i16(payoff: torch.Tensor) -> torch.Tensor:
    """Quantize expected payoff to int16 with dequant scale 1/32768."""
    return (
        (payoff * 32768.0)
        .round()
        .clamp(torch.iinfo(torch.int16).min, torch.iinfo(torch.int16).max)
        .to(torch.int16)
        .contiguous()
    )


def _zstd_torch_save(payload: dict[str, object], output: Path, level: int) -> int:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    compressed = zstd.ZstdCompressor(level=level).compress(buffer.getvalue())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(compressed)
    return len(compressed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the torch.save payload.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for computation.",
    )
    parser.add_argument(
        "--board-chunk",
        type=int,
        default=256,
        help="Number of boards ranked per chunk.",
    )
    parser.add_argument(
        "--hero-block",
        type=int,
        default=64,
        help="Number of hero combo rows accumulated per block.",
    )
    parser.add_argument(
        "--sample-boards",
        type=int,
        default=None,
        help="Use a deterministic subset of boards instead of the exact full table.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=10,
        help="zstd compression level for the saved payload.",
    )
    parser.add_argument(
        "--save-score",
        action="store_true",
        help="Also save the int32 wins-minus-losses table.",
    )
    parser.add_argument(
        "--save-counts",
        action="store_true",
        help="Also save sampled-board valid counts. Exact mode uses a scalar denominator.",
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Use the PyTorch rank_hands implementation even on CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    use_triton = device.type == "cuda" and not args.no_triton

    if use_triton:
        try:
            import triton  # noqa: F401
        except ImportError:
            print("Triton is unavailable; falling back to PyTorch rank_hands.")
            use_triton = False

    if device.type == "cpu" and args.sample_boards is None:
        print(
            "Warning: exact CPU generation is expected to be very slow. "
            "Use CUDA or pass --sample-boards for a smoke test.",
            flush=True,
        )

    payoff, score, counts, metadata = precompute_payoff(
        device=device,
        board_chunk=args.board_chunk,
        hero_block=args.hero_block,
        sample_boards=args.sample_boards,
        seed=args.seed,
        use_triton=use_triton,
    )

    payoff_i16 = _quantize_payoff_i16(payoff.to(torch.float32))
    payload: dict[str, object] = {
        "payoff_i16": payoff_i16,
        "metadata": metadata
        | {
            "stored_table": {
                "name": "payoff_i16",
                "dtype": "int16",
                "quantization": "round(clamp(payoff * 32768, -32768, 32767)).to(int16)",
                "dequantization": "payoff_i16.float() / 32768",
            },
            "compression": "zstd",
            "zstd_level": int(args.zstd_level),
        },
    }
    if args.save_score:
        payload["wins_minus_losses"] = score
    if args.save_counts and counts is not None:
        payload["counts"] = counts

    compressed_bytes = _zstd_torch_save(payload, args.output, args.zstd_level)
    raw_i16_bytes = payoff_i16.numel() * payoff_i16.element_size()
    print(f"wrote {args.output}")
    print(
        f"raw int16 table: {raw_i16_bytes:,} bytes "
        f"({raw_i16_bytes / 1024 / 1024:.2f} MiB)"
    )
    print(
        f"zstd file: {compressed_bytes:,} bytes "
        f"({compressed_bytes / 1024 / 1024:.2f} MiB)"
    )
    print(f"metadata: {payload['metadata']}")


if __name__ == "__main__":
    main()
