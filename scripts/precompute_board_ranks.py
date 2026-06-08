from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p2.env.card_utils import NUM_HANDS, canonical_full_boards_with_weights
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import (
    rank_hand_scores_triton,
    triton_is_available as rules_triton_available,
)

NUM_CANONICAL_FULL_BOARDS = 134_459


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _rank_scores(board: torch.Tensor, *, use_triton: bool) -> torch.Tensor:
    if board.device.type == "cuda" and use_triton:
        return rank_hand_scores_triton(board).contiguous()
    ranks, _ = rank_hands_torch(board.int())
    return ranks.to(torch.int32).contiguous()


def _dense_u16_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Convert comparable int32 scores to per-board dense rank ids.

    The all-in sampler only needs ``>``, ``==``, and tie counts. Dense ranks
    preserve ordering and equality while fitting in uint16. Blocked combos may
    receive arbitrary ranks; downstream code must still use board masks to
    exclude them, as the online scorer does today.
    """
    sorted_scores, sorted_idx = scores.sort(dim=1)
    group_start = torch.ones_like(sorted_scores, dtype=torch.bool)
    group_start[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    sorted_dense = torch.cumsum(group_start.to(torch.int32), dim=1)
    dense = torch.empty_like(sorted_dense)
    dense.scatter_(1, sorted_idx, sorted_dense)
    return dense.to(torch.uint16)


def _canonical_boards_cpu() -> tuple[torch.Tensor, torch.Tensor]:
    boards, weights = canonical_full_boards_with_weights()
    if boards.shape != (NUM_CANONICAL_FULL_BOARDS, 5):
        raise RuntimeError(
            f"expected {NUM_CANONICAL_FULL_BOARDS} canonical boards, "
            f"got {tuple(boards.shape)}"
        )
    return boards, weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute dense uint16 ranks for all 1326 hands on canonical full boards."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/allin_canonical_board_ranks_u16.bin"),
        help="Raw uint16 output file with shape [num_boards, 1326].",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="JSON metadata path. Defaults to OUTPUT.with_suffix(OUTPUT.suffix + '.json').",
    )
    parser.add_argument("--device", default="cuda", help="Device for chunk ranking.")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--max-boards",
        type=int,
        default=0,
        help="Only precompute this many canonical boards; for smoke tests.",
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Use the PyTorch ranker even when CUDA/Triton is available.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output: Path = args.output
    metadata_output = args.metadata_output
    if metadata_output is None:
        metadata_output = output.with_suffix(output.suffix + ".json")
    if not args.force and (output.exists() or metadata_output.exists()):
        raise FileExistsError(f"{output} or {metadata_output} already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    use_triton = (
        device.type == "cuda" and rules_triton_available() and not bool(args.no_triton)
    )
    if device.type == "cuda":
        torch.cuda.set_device(device.index or torch.cuda.current_device())

    boards, orbit_weights = _canonical_boards_cpu()
    num_boards = (
        NUM_CANONICAL_FULL_BOARDS
        if args.max_boards <= 0
        else min(args.max_boards, NUM_CANONICAL_FULL_BOARDS)
    )
    boards = boards[:num_boards]
    orbit_weights = orbit_weights[:num_boards]

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_metadata = metadata_output.with_suffix(metadata_output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()
    if tmp_metadata.exists():
        tmp_metadata.unlink()

    ranks_mm = np.memmap(
        tmp_output,
        mode="w+",
        dtype=np.uint16,
        shape=(num_boards, NUM_HANDS),
    )
    start_time = time.perf_counter()
    written = 0
    while written < num_boards:
        end = min(written + args.chunk_size, num_boards)
        chunk_boards = boards[written:end].to(device=device, non_blocking=True)
        scores = _rank_scores(chunk_boards, use_triton=use_triton)
        dense = _dense_u16_ranks(scores)
        ranks_mm[written:end] = dense.cpu().numpy()
        written = end
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time
        boards_per_s = written / max(elapsed, 1.0e-9)
        print(
            f"[{written:>8}/{num_boards}] "
            f"{boards_per_s:,.0f} boards/s elapsed={elapsed / 60.0:.2f}m",
            flush=True,
        )

    ranks_mm.flush()
    del ranks_mm
    metadata = {
        "format": "raw_memmap",
        "dtype": "uint16",
        "shape": [int(num_boards), int(NUM_HANDS)],
        "num_boards": int(num_boards),
        "num_hands": int(NUM_HANDS),
        "board_order": (
            "canonical_full_boards_with_weights(), lexicographic by representative"
        ),
        "canonical_board_count": int(NUM_CANONICAL_FULL_BOARDS),
        "raw_board_orbit_weight_sum": float(orbit_weights.sum().item()),
        "orbit_weights": (
            "not stored; regenerate via "
            "p2.env.card_utils.canonical_full_boards_with_weights()"
        ),
        "rank_encoding": "per-board dense rank ids; larger is stronger; ties equal",
        "blocked_combo_values": "undefined; consumers must mask board-blocked combos",
        "device": str(device),
        "use_triton": bool(use_triton),
        "chunk_size": int(args.chunk_size),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    tmp_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    tmp_output.replace(output)
    tmp_metadata.replace(metadata_output)
    print(f"wrote {output}", flush=True)
    print(f"wrote {metadata_output}", flush=True)


if __name__ == "__main__":
    main()
