"""Extract a small sample of spots from spots.pt for the WebGPU benchmark.

Picks spots at the start of each street (actions_this_round == 0, no folds,
no all-ins) so they can be reconstructed by the TS evaluator using only
publicCards + heroHand + heroPlayer + button. Writes a JSON file consumed by
src/benchSpots.ts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
STREET_BOARD_COUNT = {0: 0, 1: 3, 2: 4, 3: 5}


def extract(spots_path: Path, per_street: int, seed: int) -> list[dict]:
    data = torch.load(spots_path, weights_only=False)
    s = data["spots"]
    n = s["street"].shape[0]

    has_folded = s["has_folded"].any(dim=1)
    is_allin = s["is_allin"].any(dim=1)
    clean = (s["actions_this_round"] == 0) & ~has_folded & ~is_allin

    rng = torch.Generator().manual_seed(seed)
    out: list[dict] = []
    for street in (0, 1, 2, 3):
        mask = clean & (s["street"] == street)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        take = min(per_street, idx.numel())
        perm = torch.randperm(idx.numel(), generator=rng)[:take]
        picks = idx[perm].tolist()
        for i in picks:
            board_count = STREET_BOARD_COUNT[street]
            board = s["board_indices"][i, :board_count].tolist()
            hole = s["hole_indices"][i].tolist()  # [[a,b],[c,d]]
            hero_player = int(s["to_act"][i].item())
            out.append(
                {
                    "spot_index": int(i),
                    "street": int(street),
                    "street_name": STREET_NAMES[street],
                    "button": int(s["button"][i].item()),
                    "hero_player": hero_player,
                    "hero_hand": [int(c) for c in hole[hero_player]],
                    "public_cards": [int(c) for c in board],
                    "pot": int(s["pot"][i].item()),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spots",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "spots.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "bench_spots.json",
    )
    parser.add_argument("--per-street", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spots = extract(args.spots, args.per_street, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spots, indent=2) + "\n")
    print(f"wrote {len(spots)} spots to {args.out}")


if __name__ == "__main__":
    main()
