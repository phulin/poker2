#!/usr/bin/env python3
"""Create a globally row-shuffled copy of preflop street-closed states."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_INPUT = "outputs/preflop_street_closed_states/live_pair_all"
DEFAULT_OUTPUT = "outputs/preflop_street_closed_states/live_pair_all_shuffled_seed20260628"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _state_bucket(manifest: dict[str, Any]) -> dict[str, Any]:
    for item in manifest.get("buckets", []):
        if item.get("label") == "preflop_street_closed":
            return dict(item)
    buckets = list(manifest.get("buckets", []))
    if len(buckets) == 1:
        return dict(buckets[0])
    raise ValueError("manifest must contain a preflop_street_closed bucket")


def _allocate_like(
    example: dict[str, torch.Tensor],
    *,
    rows: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, tensor in example.items():
        out[name] = torch.empty(
            (rows, *tuple(tensor.shape[1:])),
            dtype=tensor.dtype,
            device="cpu",
        )
    return out


def _copy_rows(
    dst: dict[str, torch.Tensor],
    src: dict[str, torch.Tensor],
    start: int,
) -> None:
    end = start + int(next(iter(src.values())).shape[0])
    for name, tensor in src.items():
        dst[name][start:end].copy_(tensor)


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing source manifest: {manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace")
    output_bucket_dir = output_dir / "preflop_street_closed"
    output_bucket_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    bucket = _state_bucket(manifest)
    shards = list(bucket.get("shards", []))
    if not shards:
        raise ValueError("source bucket has no shards")
    total_rows = int(bucket.get("num_rows", manifest.get("num_rows", 0)))
    if total_rows <= 0:
        raise ValueError("source bucket has no rows")

    first = torch.load(
        input_dir / "preflop_street_closed" / str(shards[0]["path"]),
        map_location="cpu",
        weights_only=True,
    )
    states = _allocate_like(first["states"], rows=total_rows)
    metadata = _allocate_like(first["metadata_tensors"], rows=total_rows)

    offset = 0
    for index, shard in enumerate(shards):
        path = input_dir / "preflop_street_closed" / str(shard["path"])
        payload = first if index == 0 else torch.load(path, map_location="cpu", weights_only=True)
        rows = int(payload.get("num_rows", next(iter(payload["states"].values())).shape[0]))
        _copy_rows(states, payload["states"], offset)
        _copy_rows(metadata, payload["metadata_tensors"], offset)
        offset += rows
        print(f"loaded {index + 1:,}/{len(shards):,} shards rows={offset:,}", flush=True)
    if offset != total_rows:
        raise RuntimeError(f"manifest rows={total_rows} but loaded rows={offset}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    permutation = torch.randperm(total_rows, generator=generator)

    out_shards: list[dict[str, Any]] = []
    written = 0
    for index, source_shard in enumerate(shards):
        rows = int(source_shard["num_rows"])
        indices = permutation[written : written + rows]
        out_states = {name: tensor.index_select(0, indices) for name, tensor in states.items()}
        out_metadata = {
            name: tensor.index_select(0, indices) for name, tensor in metadata.items()
        }
        filename = f"states_{index:06d}.pt"
        torch.save(
            {
                "kind": "preflop_street_closed_states_shard",
                "num_rows": rows,
                "states": out_states,
                "metadata_tensors": out_metadata,
            },
            output_bucket_dir / filename,
        )
        out_shards.append({"num_rows": rows, "path": filename})
        written += rows
        print(f"wrote {index + 1:,}/{len(shards):,} shards rows={written:,}", flush=True)
    if written != total_rows:
        raise RuntimeError(f"wrote rows={written}, expected {total_rows}")

    out_manifest = copy.deepcopy(manifest)
    for item in out_manifest.get("buckets", []):
        if item.get("label") == bucket.get("label", "preflop_street_closed"):
            item["shards"] = out_shards
            item["num_rows"] = total_rows
            break
    out_manifest["num_rows"] = total_rows
    out_manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    out_manifest["complete"] = True
    out_manifest["shuffle"] = {
        "kind": "global_example_permutation",
        "seed": int(args.seed),
        "source_dataset": str(input_dir.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(out_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote shuffled dataset: {output_dir} rows={total_rows:,}", flush=True)


if __name__ == "__main__":
    main()
