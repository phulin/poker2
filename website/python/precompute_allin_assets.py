from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from p2.env.card_utils import NUM_HANDS, combo_suit_permutation_tensor
from p2.search.allin_payoff import (
    I16_SCALE,
    canonical_flop_data_cpu,
    compute_postflop_payoff_quantized,
    compute_postflop_payoff_quantized_triton_batched,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _write_u32(path: Path, values: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = values.detach().cpu().to(torch.int64).numpy().astype("<u4", copy=False)
    path.write_bytes(arr.tobytes(order="C"))


def _write_i16(path: Path, values: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = values.detach().cpu().to(torch.int16).numpy()
    if arr.dtype.byteorder not in ("<", "="):
        arr = arr.astype("<i2", copy=False)
    path.write_bytes(arr.tobytes(order="C"))


def _flop_table_name(index: int) -> str:
    return f"{index:04d}.i16"


def _compute_flop_batch(
    flops: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if device.type == "cuda":
        return compute_postflop_payoff_quantized_triton_batched(
            flops.to(device),
            dtype=torch.int16,
        )
    tables = [
        compute_postflop_payoff_quantized(flop.to(device), dtype=torch.int16).cpu()
        for flop in flops
    ]
    return torch.stack(tables, dim=0)


def precompute_flop_assets(
    out: Path,
    *,
    device: torch.device,
    batch_size: int,
    limit: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    canonical_flops, actual_to_canon, actual_perm = canonical_flop_data_cpu()
    canonical_total = int(canonical_flops.shape[0])
    if limit is not None:
        canonical_flops = canonical_flops[:limit]

    flop_dir = out / "flop"
    tables_dir = flop_dir / "tables"
    _write_u32(flop_dir / "actual_to_canon.u32", actual_to_canon)
    _write_u32(flop_dir / "actual_perm.u32", actual_perm)
    _write_u32(flop_dir / "combo_perms.u32", combo_suit_permutation_tensor())

    total = int(canonical_flops.shape[0])
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = canonical_flops[start:end].contiguous()
        pending: list[int] = []
        pending_flops: list[torch.Tensor] = []
        for local, flop in enumerate(batch):
            index = start + local
            path = tables_dir / _flop_table_name(index)
            if path.exists() and not overwrite:
                continue
            pending.append(index)
            pending_flops.append(flop)
        if not pending:
            continue

        computed = _compute_flop_batch(
            torch.stack(pending_flops, dim=0),
            device=device,
        ).cpu()
        for row, index in enumerate(pending):
            _write_i16(tables_dir / _flop_table_name(index), computed[row])
        print(f"wrote flop tables {pending[0]}..{pending[-1]} ({len(pending)})")

    return {
        "actualToCanonFile": "allin/flop/actual_to_canon.u32",
        "actualPermFile": "allin/flop/actual_perm.u32",
        "comboPermsFile": "allin/flop/combo_perms.u32",
        "tablePathTemplate": "allin/flop/tables/{id4}.i16",
        "dtype": "int16",
        "scale": I16_SCALE,
        "canonicalCount": canonical_total,
        "generatedCount": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute WebGPU all-in payoff table assets.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("website/public/models/rebel_latest/allin"),
        help="Output all-in asset directory, usually under an exported model directory.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for table generation; use cuda for practical full-flop generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of canonical flops to generate per batched CUDA call.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N canonical flop tables; intended for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = _device(args.device)
    if device.type != "cuda" and args.limit is None:
        raise RuntimeError(
            "full flop asset generation requires CUDA; pass --limit for a CPU smoke run"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    flop_manifest = precompute_flop_assets(
        args.out,
        device=device,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    manifest = {
        "enabled": True,
        "scale": I16_SCALE,
        "flop": {
            key: value
            for key, value in flop_manifest.items()
            if key != "canonicalCount"
            and key != "generatedCount"
        },
        "metadata": {
            "format": "p2.webgpu_allin_assets",
            "numHands": NUM_HANDS,
            "canonicalFlops": flop_manifest["canonicalCount"],
            "generatedFlops": flop_manifest["generatedCount"],
        },
    }
    manifest_path = args.out / "allin_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "allInManifest": str(manifest_path),
                "canonicalFlops": flop_manifest["canonicalCount"],
                "generatedFlops": flop_manifest["generatedCount"],
                "tableBytesEach": NUM_HANDS * NUM_HANDS * 2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
