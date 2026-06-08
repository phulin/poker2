from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from p2.allin.training_data import FEATURE_KEYS, MANIFEST_NAME, TARGET_KEY
from p2.env.card_utils import NUM_HANDS, PREFLOP_HANDS, collapse_1326_to_169


def _manifest_path(path: Path) -> Path:
    return path / MANIFEST_NAME if path.is_dir() else path


def _convert_shard(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    beliefs = tensors["beliefs"]
    targets = tensors[TARGET_KEY]
    if beliefs.shape[-1] != NUM_HANDS:
        raise ValueError(f"expected beliefs final axis {NUM_HANDS}, got {beliefs.shape}")
    if targets.shape[-1] != NUM_HANDS:
        raise ValueError(f"expected targets final axis {NUM_HANDS}, got {targets.shape}")

    converted = {key: tensors[key] for key in FEATURE_KEYS if key != "beliefs"}
    converted["beliefs"] = collapse_1326_to_169(beliefs, reduction="sum").contiguous()
    converted[TARGET_KEY] = collapse_1326_to_169(
        targets,
        reduction="mean",
    ).contiguous()
    return {key: converted[key] for key in (*FEATURE_KEYS, TARGET_KEY)}


def convert_allin_dataset_to_169(
    source: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    source_manifest_path = _manifest_path(Path(source))
    source_root = source_manifest_path.parent
    output = Path(output_dir)
    output_manifest_path = output / MANIFEST_NAME
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    manifest = json.loads(source_manifest_path.read_text())
    if manifest.get("format") != "p2.allin.training_data.v1":
        raise ValueError(f"unsupported all-in dataset format in {source_manifest_path}")
    if int(manifest["hands"]) != NUM_HANDS:
        raise ValueError(f"source dataset must have {NUM_HANDS} hands")

    start_time = time.perf_counter()
    shards: list[dict[str, Any]] = []
    for shard_idx, shard in enumerate(manifest["shards"]):
        shard_start = time.perf_counter()
        tensors = torch.load(source_root / shard["file"], map_location="cpu")
        converted = _convert_shard(tensors)
        torch.save(converted, output / shard["file"])
        shard_info = dict(shard)
        shards.append(shard_info)
        elapsed = time.perf_counter() - shard_start
        print(
            f"[{shard_idx + 1:>4}/{len(manifest['shards'])}] "
            f"wrote={shard['file']} examples={shard['examples']} "
            f"shard={elapsed:.2f}s",
            flush=True,
        )

    converted_manifest = dict(manifest)
    converted_manifest["hands"] = PREFLOP_HANDS
    converted_manifest["shards"] = shards
    converted_config = dict(converted_manifest.get("config", {}))
    converted_config["range_hand_dim"] = PREFLOP_HANDS
    converted_manifest["config"] = converted_config
    converted_manifest["conversion"] = {
        "kind": "allin_1326_to_169",
        "source_manifest": str(source_manifest_path),
        "belief_reduction": "sum",
        "target_reduction": "mean",
        "source_hands": NUM_HANDS,
        "target_hands": PREFLOP_HANDS,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    output_manifest_path.write_text(
        json.dumps(converted_manifest, indent=2, sort_keys=True) + "\n"
    )
    return converted_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pregenerated 1326-combo all-in dataset to native 169."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Source dataset directory or manifest.json path.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the converted 169-hand dataset.",
    )
    parser.add_argument("--force", action="store_true", help="Replace output_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = convert_allin_dataset_to_169(
        args.source,
        args.output_dir,
        force=args.force,
    )
    print(
        f"wrote {manifest['examples']} examples to {args.output_dir / MANIFEST_NAME}",
        flush=True,
    )


if __name__ == "__main__":
    main()
