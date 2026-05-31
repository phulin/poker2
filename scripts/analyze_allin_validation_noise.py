from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from p2.allin.sampler import PreflopAllInEstimatorWorkspace, estimate_preflop_allin_values
from p2.allin.training_data import (
    FEATURE_KEYS,
    MANIFEST_NAME,
    TARGET_KEY,
    PregeneratedAllInDataset,
)
from p2.env.card_utils import NUM_HANDS


def _side_pot_counts(committed: torch.Tensor, folded_mask: torch.Tensor) -> torch.Tensor:
    levels = committed.sort(dim=1).values
    previous = torch.cat((torch.zeros_like(levels[:, :1]), levels[:, :-1]), dim=1)
    widths = (levels - previous).clamp_min(0.0)
    participants = committed[:, None, :] >= levels[:, :, None]
    eligible = participants & (~folded_mask[:, None, :])
    positive_layers = (widths > 0.0) & eligible.any(dim=2)
    return positive_layers.sum(dim=1).sub(1).clamp_min(0).long()


def _empty_bucket(size: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "rows": torch.zeros(size, dtype=torch.float64, device=device),
        "all_sse": torch.zeros(size, dtype=torch.float64, device=device),
        "all_sae": torch.zeros(size, dtype=torch.float64, device=device),
        "all_count": torch.zeros(size, dtype=torch.float64, device=device),
        "live_sse": torch.zeros(size, dtype=torch.float64, device=device),
        "live_sae": torch.zeros(size, dtype=torch.float64, device=device),
        "live_count": torch.zeros(size, dtype=torch.float64, device=device),
    }


def _accumulate_bucket(
    bucket: dict[str, torch.Tensor],
    ids: torch.Tensor,
    *,
    all_sse: torch.Tensor,
    all_sae: torch.Tensor,
    all_count: torch.Tensor,
    live_sse: torch.Tensor,
    live_sae: torch.Tensor,
    live_count: torch.Tensor,
) -> None:
    size = bucket["rows"].numel()
    ones = torch.ones_like(all_sse, dtype=torch.float64)
    bucket["rows"] += torch.bincount(ids, weights=ones, minlength=size).to(torch.float64)
    bucket["all_sse"] += torch.bincount(ids, weights=all_sse, minlength=size).to(
        torch.float64
    )
    bucket["all_sae"] += torch.bincount(ids, weights=all_sae, minlength=size).to(
        torch.float64
    )
    bucket["all_count"] += torch.bincount(ids, weights=all_count, minlength=size).to(
        torch.float64
    )
    bucket["live_sse"] += torch.bincount(ids, weights=live_sse, minlength=size).to(
        torch.float64
    )
    bucket["live_sae"] += torch.bincount(ids, weights=live_sae, minlength=size).to(
        torch.float64
    )
    bucket["live_count"] += torch.bincount(ids, weights=live_count, minlength=size).to(
        torch.float64
    )


def _bucket_entries(
    bucket: dict[str, torch.Tensor],
    *,
    labels: list[str],
) -> list[dict[str, float | int | str]]:
    rows = bucket["rows"].detach().cpu()
    all_sse = bucket["all_sse"].detach().cpu()
    all_sae = bucket["all_sae"].detach().cpu()
    all_count = bucket["all_count"].detach().cpu()
    live_sse = bucket["live_sse"].detach().cpu()
    live_sae = bucket["live_sae"].detach().cpu()
    live_count = bucket["live_count"].detach().cpu()
    entries: list[dict[str, float | int | str]] = []
    for idx, label in enumerate(labels):
        row_count = int(rows[idx].item())
        if row_count == 0:
            continue
        all_n = float(all_count[idx].item())
        live_n = float(live_count[idx].item())
        all_mse = float(all_sse[idx].item() / all_n) if all_n > 0 else 0.0
        live_mse = float(live_sse[idx].item() / live_n) if live_n > 0 else 0.0
        entries.append(
            {
                "bucket": label,
                "rows": row_count,
                "all_entries_mse": all_mse,
                "all_entries_noise_floor_mse": 0.5 * all_mse,
                "all_entries_mae": float(all_sae[idx].item() / all_n)
                if all_n > 0
                else 0.0,
                "live_entries_mse": live_mse,
                "live_entries_noise_floor_mse": 0.5 * live_mse,
                "live_entries_mae": float(live_sae[idx].item() / live_n)
                if live_n > 0
                else 0.0,
            }
        )
    return entries


def _batch_to_cpu_tensors(
    batch: Any,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "beliefs": batch.beliefs.detach().cpu(),
        "starting_stacks": batch.starting_stacks.detach().cpu(),
        "committed": batch.committed.detach().cpu(),
        "stacks_after": batch.stacks_after.detach().cpu(),
        "allin_mask": batch.allin_mask.detach().cpu(),
        "folded_mask": batch.folded_mask.detach().cpu(),
        "scale": batch.scale.detach().cpu(),
        TARGET_KEY: targets.detach().cpu(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate independent all-in validation targets for fixed features "
            "and report target-target noise by live player and side-pot buckets."
        )
    )
    parser.add_argument(
        "--validation-data",
        default="outputs/allin_validation_data_8192_s1048576_b131072_bc64",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/allin_validation_noise_s2097152",
    )
    parser.add_argument("--seed", type=int, default=2_097_152)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--intermediate-examples",
        type=int,
        default=0,
        help="Write manifest_<N>.json once this many examples have been saved.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Override source manifest sample_count; 0 keeps the source value.",
    )
    parser.add_argument(
        "--board-samples",
        type=int,
        default=0,
        help="Override source manifest board_samples; 0 keeps the source value.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} already exists; pass --force to replace")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    dataset = PregeneratedAllInDataset(args.validation_data)
    source_manifest = dataset.manifest
    cfg = source_manifest["config"]
    players = int(source_manifest["players"])
    examples = int(source_manifest["examples"])
    if args.max_examples > 0:
        examples = min(examples, int(args.max_examples))
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    workspace = PreflopAllInEstimatorWorkspace()
    tuple_samples = int(cfg.get("tuple_samples", 0))
    tuple_samples_arg = tuple_samples if tuple_samples > 0 else None
    sample_count = int(args.sample_count) if args.sample_count > 0 else int(cfg["sample_count"])
    board_samples = (
        int(args.board_samples) if args.board_samples > 0 else int(cfg["board_samples"])
    )
    if sample_count <= 0 or board_samples <= 0:
        raise ValueError("sample_count and board_samples must be positive")
    board_chunk = int(cfg["board_chunk"])
    hand_chunk = int(cfg["hand_chunk"])
    tuple_tries = int(cfg["tuple_tries"])
    generation_config = {
        **cfg,
        "sample_count": sample_count,
        "board_samples": board_samples,
    }

    def write_manifest(
        path: Path,
        *,
        written_examples: int,
        shards: list[dict[str, int | str]],
        elapsed: float,
    ) -> None:
        manifest = {
            "format": source_manifest["format"],
            "source_validation_data": str(Path(args.validation_data)),
            "source_manifest": source_manifest,
            "noise_report": "noise_report.json",
            "generation": {
                "seed": int(args.seed),
                "device": str(device),
                "elapsed_seconds": elapsed,
                "config": generation_config,
            },
            "examples": int(written_examples),
            "players": players,
            "hands": int(source_manifest["hands"]),
            "feature_keys": list(FEATURE_KEYS),
            "target_key": TARGET_KEY,
            "shards": shards,
        }
        path.write_text(json.dumps(manifest, indent=2) + "\n")

    live_bucket = _empty_bucket(players + 1, device=device)
    side_bucket = _empty_bucket(players, device=device)
    matrix_bucket = _empty_bucket((players + 1) * players, device=device)
    overall = _empty_bucket(1, device=device)

    output_shards: list[dict[str, int | str]] = []
    started = time.perf_counter()
    row_start = 0
    shard_idx = 0
    while row_start < examples:
        cur = min(int(args.batch_size), examples - row_start)
        batch, targets_a = dataset.get_batch(row_start, cur, device=device)
        targets_b, _ = estimate_preflop_allin_values(
            batch,
            sample_count=sample_count,
            board_samples=board_samples,
            tuple_samples=tuple_samples_arg,
            tuple_tries=tuple_tries,
            board_chunk=board_chunk,
            hand_chunk=hand_chunk,
            generator=generator,
            preflop_table_path=cfg["preflop_table_path"],
            use_exact_two_player=bool(cfg.get("use_exact_two_player", True)),
            compute_stats=False,
            workspace=workspace,
        )

        delta = targets_b - targets_a
        abs_delta = delta.abs()
        sq_delta = delta.square()
        live_mask = (~batch.folded_mask)[:, :, None].expand_as(delta)
        live_counts = (~batch.folded_mask).sum(dim=1).long()
        side_counts = _side_pot_counts(batch.committed, batch.folded_mask)
        matrix_ids = live_counts * players + side_counts

        all_sse = sq_delta.sum(dim=(1, 2)).to(torch.float64)
        all_sae = abs_delta.sum(dim=(1, 2)).to(torch.float64)
        all_count = torch.full(
            (cur,),
            float(players * NUM_HANDS),
            dtype=torch.float64,
            device=device,
        )
        live_sse = sq_delta.masked_fill(~live_mask, 0.0).sum(dim=(1, 2)).to(torch.float64)
        live_sae = abs_delta.masked_fill(~live_mask, 0.0).sum(dim=(1, 2)).to(
            torch.float64
        )
        live_count = (live_counts * NUM_HANDS).to(torch.float64)

        _accumulate_bucket(
            overall,
            torch.zeros(cur, dtype=torch.long, device=device),
            all_sse=all_sse,
            all_sae=all_sae,
            all_count=all_count,
            live_sse=live_sse,
            live_sae=live_sae,
            live_count=live_count,
        )
        _accumulate_bucket(
            live_bucket,
            live_counts,
            all_sse=all_sse,
            all_sae=all_sae,
            all_count=all_count,
            live_sse=live_sse,
            live_sae=live_sae,
            live_count=live_count,
        )
        _accumulate_bucket(
            side_bucket,
            side_counts,
            all_sse=all_sse,
            all_sae=all_sae,
            all_count=all_count,
            live_sse=live_sse,
            live_sae=live_sae,
            live_count=live_count,
        )
        _accumulate_bucket(
            matrix_bucket,
            matrix_ids,
            all_sse=all_sse,
            all_sae=all_sae,
            all_count=all_count,
            live_sse=live_sse,
            live_sae=live_sae,
            live_count=live_count,
        )

        shard_name = f"shard_{shard_idx:06d}.pt"
        torch.save(_batch_to_cpu_tensors(batch, targets_b), output_dir / shard_name)
        output_shards.append(
            {
                "file": shard_name,
                "start": row_start,
                "end": row_start + cur,
                "examples": cur,
            }
        )
        row_start += cur
        shard_idx += 1
        elapsed = time.perf_counter() - started
        if args.intermediate_examples > 0 and row_start == args.intermediate_examples:
            write_manifest(
                output_dir / f"manifest_{row_start}.json",
                written_examples=row_start,
                shards=list(output_shards),
                elapsed=elapsed,
            )
        print(
            f"rows={row_start}/{examples} elapsed={elapsed:.1f}s "
            f"rate={row_start / max(elapsed, 1.0e-9):.2f} rows/s",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    matrix_labels = [
        f"live{live}_sp{side}"
        for live in range(players + 1)
        for side in range(players)
    ]
    report = {
        "source_validation_data": str(Path(args.validation_data)),
        "output_dir": str(output_dir),
        "seed": int(args.seed),
        "device": str(device),
        "examples": int(examples),
        "players": players,
        "elapsed_seconds": elapsed,
        "sample_count": sample_count,
        "board_samples": board_samples,
        "tuple_samples": int((sample_count + board_samples - 1) // board_samples)
        if tuple_samples_arg is None
        else int(tuple_samples_arg),
        "tuple_tries": tuple_tries,
        "board_chunk": board_chunk,
        "hand_chunk": hand_chunk,
        "overall": _bucket_entries(overall, labels=["overall"])[0],
        "by_live_count": _bucket_entries(
            live_bucket,
            labels=[f"live{idx}" for idx in range(players + 1)],
        ),
        "by_side_pots": _bucket_entries(
            side_bucket,
            labels=[f"sp{idx}" for idx in range(players)],
        ),
        "by_live_count_and_side_pots": _bucket_entries(
            matrix_bucket,
            labels=matrix_labels,
        ),
    }
    (output_dir / "noise_report.json").write_text(json.dumps(report, indent=2) + "\n")

    write_manifest(
        output_dir / MANIFEST_NAME,
        written_examples=int(examples),
        shards=output_shards,
        elapsed=elapsed,
    )
    print(json.dumps(report["overall"], indent=2), flush=True)
    dataset.close()


if __name__ == "__main__":
    main()
