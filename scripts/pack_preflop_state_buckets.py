#!/usr/bin/env python3
"""Pack pregenerated preflop public-state buckets into one shard per bucket."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _load_manifest(root: Path, *, allow_partial: bool) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() and allow_partial:
        manifest_path = root / "manifest.partial.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found under {root}; use --allow-partial for active runs"
        )
    return json.loads(manifest_path.read_text())


def _bucket_label(item: dict[str, Any]) -> str:
    label = item.get("label")
    if label is not None:
        return str(label)
    return f"frontier_{int(item['action_count'])}"


def _manifest_bucket_items(
    manifest: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    if manifest.get("buckets"):
        return "buckets", list(manifest["buckets"])
    if manifest.get("frontiers"):
        return "frontiers", list(manifest["frontiers"])
    raise ValueError("Manifest has no buckets/frontiers to pack")


def _pack_bucket(
    *,
    source_root: Path,
    output_root: Path,
    item: dict[str, Any],
    field_names: list[str],
    overwrite: bool,
) -> dict[str, Any]:
    label = _bucket_label(item)
    source_bucket_dir = source_root / label
    output_bucket_dir = output_root / label
    output_bucket_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_bucket_dir / "states.pt"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")

    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in field_names}
    rows = 0
    shards = list(item.get("shards", []))
    if not shards:
        raise ValueError(f"{label} has no shards")

    for shard_index, shard in enumerate(shards):
        shard_path = source_bucket_dir / str(shard.get("path", shard.get("file")))
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        states = payload["states"]
        if shard_index == 0:
            missing = [name for name in field_names if name not in states]
            if missing:
                raise KeyError(f"{label} missing state fields: {missing}")
        shard_rows = int(payload.get("num_rows", next(iter(states.values())).shape[0]))
        rows += shard_rows
        for name in field_names:
            chunks[name].append(states[name])

    packed_states = {
        name: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
        for name, parts in chunks.items()
    }
    torch.save(
        {
            "kind": "preflop_policy_public_states_packed_bucket",
            "source_dataset": str(source_root),
            "bucket": label,
            "num_rows": rows,
            "states": packed_states,
        },
        output_path,
    )

    packed_item = copy.deepcopy(item)
    packed_item["num_rows"] = rows
    packed_item["shards"] = [{"path": "states.pt", "num_rows": rows}]

    del chunks, packed_states
    gc.collect()
    return packed_item


def pack_dataset(args: argparse.Namespace) -> None:
    source_root = Path(args.state_dataset)
    output_root = Path(args.output_dir)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("output dir must be different from the source dataset")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(source_root, allow_partial=args.allow_partial)
    bucket_key, bucket_items = _manifest_bucket_items(manifest)
    selected = set(args.buckets) if args.buckets else None
    field_names = list(manifest.get("state_fields") or [])
    if not field_names:
        first_item = bucket_items[0]
        first_label = _bucket_label(first_item)
        first_shard = first_item["shards"][0]
        first_path = (
            source_root
            / first_label
            / str(first_shard.get("path", first_shard.get("file")))
        )
        first_payload = torch.load(first_path, map_location="cpu", weights_only=True)
        field_names = list(first_payload["states"].keys())

    packed_items: list[dict[str, Any]] = []
    for item in bucket_items:
        label = _bucket_label(item)
        if selected is not None and label not in selected:
            continue
        print(f"packing {label} from {len(item.get('shards', []))} shards", flush=True)
        packed_item = _pack_bucket(
            source_root=source_root,
            output_root=output_root,
            item=item,
            field_names=field_names,
            overwrite=args.overwrite,
        )
        packed_items.append(packed_item)
        print(
            f"packed {label}: rows={int(packed_item['num_rows']):,}",
            flush=True,
        )

    packed_manifest = copy.deepcopy(manifest)
    packed_manifest[bucket_key] = packed_items
    packed_manifest["kind"] = f"{manifest.get('kind', 'preflop_policy_states')}_packed"
    packed_manifest["source_dataset"] = str(source_root)
    packed_manifest["packed_at"] = datetime.now(timezone.utc).isoformat()
    packed_manifest["num_rows"] = sum(
        int(item.get("num_rows", 0)) for item in packed_items
    )
    packed_manifest["complete"] = True
    with (output_root / "manifest.json").open("w") as f:
        json.dump(packed_manifest, f, indent=2, sort_keys=True)

    print(
        f"done: rows={int(packed_manifest['num_rows']):,} output={output_root}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--buckets", nargs="*", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    pack_dataset(parse_args())


if __name__ == "__main__":
    main()
