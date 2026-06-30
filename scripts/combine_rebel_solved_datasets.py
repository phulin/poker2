#!/usr/bin/env python3
"""Combine compatible ReBeL solved datasets into one manifest/shard set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Literal


STREAMS = ("value", "policy")
LINK_MODES = ("symlink", "hardlink", "copy")
MODEL_DEFAULTS = {
    "belief_low_rank_board_conditioned": False,
    "belief_low_rank_dim": 0,
    "belief_second_moment": False,
    "belief_skip_matching_encoder": False,
    "board_conditioned_hand_embedding_dim": 0,
    "card_token_value_head_dim": 0,
    "context_range_stats": False,
    "cross_range_rank": 0,
    "postflop_multi_token_trunk": False,
    "value_hand_basis_rank": 0,
    "value_head_rank": 0,
    "value_per_hand_residual": False,
    "value_strength_bucket_count": 0,
    "value_strength_bucket_film": False,
    "value_strength_bucket_relative": False,
}


def _dataset_dir(path: Path) -> Path:
    return path.parent if path.name == "manifest.json" else path


def _load_manifest(path: Path) -> dict[str, Any]:
    dataset_dir = _dataset_dir(path)
    return json.loads((dataset_dir / "manifest.json").read_text())


def _count_dict_sum(dicts: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for count_dict in dicts:
        for key, value in count_dict.items():
            out[str(key)] = out.get(str(key), 0) + int(value)
    return {key: out[key] for key in sorted(out, key=lambda item: int(item))}


def _sum_count_section(manifests: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        stream: _count_dict_sum(
            [
                {
                    str(count_key): int(count_value)
                    for count_key, count_value in manifest.get(key, {})
                    .get(stream, {})
                    .items()
                }
                for manifest in manifests
            ]
        )
        for stream in (*STREAMS, "total")
    }


def _same_or_raise(manifests: list[dict[str, Any]], key: str) -> Any:
    first = manifests[0].get(key)
    for index, manifest in enumerate(manifests[1:], start=1):
        if manifest.get(key) != first:
            raise ValueError(f"manifest {index} has incompatible {key!r}")
    return first


def _normalized_model_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config = dict(manifest.get("model_config", {}))
    for key, value in MODEL_DEFAULTS.items():
        config.setdefault(key, value)
    return config


def _same_model_config_or_raise(manifests: list[dict[str, Any]]) -> None:
    first = _normalized_model_config(manifests[0])
    for index, manifest in enumerate(manifests[1:], start=1):
        if _normalized_model_config(manifest) != first:
            raise ValueError(f"manifest {index} has incompatible 'model_config'")


def _copy_link_or_symlink(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unsupported link mode {mode!r}")


def combine_datasets(
    inputs: list[Path],
    output: Path,
    *,
    link_mode: Literal["symlink", "hardlink", "copy"] = "symlink",
) -> dict[str, Any]:
    if len(inputs) < 2:
        raise ValueError("at least two input datasets are required")
    output = _dataset_dir(output)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists")
    output.mkdir(parents=True, exist_ok=True)

    input_dirs = [_dataset_dir(path) for path in inputs]
    manifests = [_load_manifest(path) for path in input_dirs]
    for key in (
        "format",
        "num_players",
        "hands",
        "num_actions",
        "context_length",
        "street_support",
        "included_streets",
        "storage_float_dtype",
        "model_family",
        "feature_encoder",
        "action_schedule",
        "root_source",
        "root_source_codes",
        "root_streets",
        "quality",
        "env_config",
        "search_config",
    ):
        _same_or_raise(manifests, key)
    _same_model_config_or_raise(manifests)

    combined = dict(manifests[0])
    combined["combined_from"] = [
        {
            "path": str(path),
            "value_examples": int(manifest.get("value_examples", 0)),
            "policy_examples": int(manifest.get("policy_examples", 0)),
            "generator": manifest.get("generator"),
        }
        for path, manifest in zip(input_dirs, manifests, strict=True)
    ]
    combined["value_examples"] = sum(
        int(manifest.get("value_examples", 0)) for manifest in manifests
    )
    combined["policy_examples"] = sum(
        int(manifest.get("policy_examples", 0)) for manifest in manifests
    )
    for key in (
        "street_counts",
        "node_depth_counts",
        "target_source_counts",
        "leaf_target_source_counts",
        "root_source_counts",
    ):
        combined[key] = _sum_count_section(manifests, key)

    combined_shards: dict[str, list[dict[str, Any]]] = {"value": [], "policy": []}
    for stream in STREAMS:
        stream_dir = output / stream
        stream_dir.mkdir(parents=True, exist_ok=True)
        cursor = 0
        shard_idx = 0
        for input_dir, manifest in zip(input_dirs, manifests, strict=True):
            for shard in manifest.get("shards", {}).get(stream, []):
                src = input_dir / shard["file"]
                if not src.exists():
                    raise FileNotFoundError(src)
                rows = int(shard["end"]) - int(shard["start"])
                rel_path = f"{stream}/shard_{shard_idx:06d}.pt"
                dst = output / rel_path
                _copy_link_or_symlink(src, dst, link_mode)
                combined_shards[stream].append(
                    {"file": rel_path, "start": cursor, "end": cursor + rows}
                )
                cursor += rows
                shard_idx += 1
        expected = int(combined[f"{stream}_examples"])
        if cursor != expected:
            raise ValueError(
                f"{stream} shard rows total {cursor} does not match manifest "
                f"examples {expected}"
            )
    combined["shards"] = combined_shards
    combined["generator"] = {
        "combined": True,
        "link_mode": link_mode,
        "source_count": len(input_dirs),
    }

    manifest_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--link-mode",
        choices=LINK_MODES,
        default="symlink",
        help="How to materialize source shards in the combined dataset.",
    )
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = combine_datasets(args.inputs, args.output, link_mode=args.link_mode)
    print(
        f"wrote {manifest['value_examples']} value and "
        f"{manifest['policy_examples']} policy examples to {args.output}"
    )


if __name__ == "__main__":
    main()
