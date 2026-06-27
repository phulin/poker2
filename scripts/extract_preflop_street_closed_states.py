#!/usr/bin/env python3
"""Extract preflop states one legal action away from closing the street."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

from p2.env.pbs_env import PBSEnv
from p2.stages.preflop_backward_induction import (
    STATE_FIELDS,
    _copy_public_states_to_env,
)


DEFAULT_STATE_DATASET = (
    "outputs/preflop_policy_states/"
    "eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find saved preflop public states where at least one legal action "
            "closes the preflop street, step each such legal closing action "
            "without force-HU mutation, and save the resulting street-closed "
            "states."
        )
    )
    parser.add_argument("--state-dataset", default=DEFAULT_STATE_DATASET)
    parser.add_argument(
        "--output-dir",
        default="outputs/preflop_street_closed_states/live_pair_5m",
    )
    parser.add_argument(
        "--buckets",
        nargs="*",
        default=None,
        help="Input bucket labels to scan. Defaults to all buckets/frontiers.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=5_000_000,
        help="Global output row cap. Use 0 or a negative value for no cap.",
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--shard-size", type=int, default=250_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_manifest(root: Path, *, allow_partial: bool) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists() and allow_partial:
        manifest_path = root / "manifest.partial.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest found under {root}; use --allow-partial for active runs"
        )
    return json.loads(manifest_path.read_text())


def _bucket_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("buckets", "frontiers"):
        for item in manifest.get(key, []):
            label = item.get("label")
            if label is None and "action_count" in item:
                label = f"frontier_{int(item['action_count'])}"
            if label is None:
                continue
            cloned = dict(item)
            cloned["label"] = str(label)
            items.append(cloned)
    return items


def _iter_shard_paths(
    *,
    root: Path,
    item: dict[str, Any],
) -> Iterator[Path]:
    label = str(item["label"])
    shards = list(item.get("shards", []))
    if not shards:
        candidate = root / label / "states.pt"
        if candidate.exists():
            yield candidate
        return
    for shard in shards:
        rel = shard.get("path", shard.get("file"))
        if rel is None:
            continue
        yield root / label / str(rel)


def _make_env_from_manifest(
    manifest: dict[str, Any],
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
) -> PBSEnv:
    env_cfg = dict(manifest.get("env_config", {}))
    model = dict(manifest.get("model", {}))
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    env = PBSEnv(
        num_envs=num_envs,
        num_players=int(model.get("num_players", env_cfg.get("num_players", 6))),
        mean_stack=int(env_cfg.get("stack", 10000)),
        sb=int(env_cfg.get("sb", 50)),
        bb=int(env_cfg.get("bb", 100)),
        default_bet_bins=list(env_cfg.get("bet_bins", [0.5, 0.75, 1.0, 1.5, 2.0])),
        device=device,
        rng=rng,
        float_dtype=torch.float32,
        stack_mode=str(env_cfg.get("stack_mode", "fixed")),
        min_stack_bb=int(env_cfg.get("min_stack_bb", 10)),
        mid_stack_bb=int(env_cfg.get("mid_stack_bb", 200)),
        max_stack_bb=int(env_cfg.get("max_stack_bb", 400)),
        high_stack_mass_ratio=float(env_cfg.get("high_stack_mass_ratio", 1.0 / 3.0)),
        force_heads_up_preflop_flop=False,
    )
    env.reset()
    return env


def _slice_states(
    states: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    return {name: tensor[start:end] for name, tensor in states.items()}


def _gather_env_states(env: PBSEnv, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        name: getattr(env, name).index_select(0, indices).detach().cpu()
        for name in STATE_FIELDS
    }


class _StreetClosedWriter:
    def __init__(self, output_dir: Path, *, shard_size: int) -> None:
        self.output_dir = output_dir
        self.bucket_dir = output_dir / "preflop_street_closed"
        self.shard_size = int(shard_size)
        self.pending_states: list[dict[str, torch.Tensor]] = []
        self.pending_aux: list[dict[str, torch.Tensor]] = []
        self.pending_rows = 0
        self.total_rows = 0
        self.shards: list[dict[str, Any]] = []
        self.shard_index = 0
        self.bucket_dir.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        states: dict[str, torch.Tensor],
        *,
        aux: dict[str, torch.Tensor],
    ) -> None:
        rows = int(next(iter(states.values())).shape[0])
        if rows <= 0:
            return
        self.pending_states.append(states)
        self.pending_aux.append(aux)
        self.pending_rows += rows
        self.total_rows += rows
        while self.pending_rows >= self.shard_size:
            self.flush(max_rows=self.shard_size)

    def _take_pending(
        self,
        max_rows: int | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
        take = self.pending_rows if max_rows is None else min(self.pending_rows, max_rows)
        if take <= 0:
            raise ValueError("cannot flush an empty shard")
        out_states = {name: [] for name in STATE_FIELDS}
        out_aux: dict[str, list[torch.Tensor]] = {
            "source_bucket": [],
            "source_row": [],
            "closing_action": [],
        }
        remaining_states: list[dict[str, torch.Tensor]] = []
        remaining_aux: list[dict[str, torch.Tensor]] = []
        remaining_rows = 0
        needed = take
        for states, aux in zip(self.pending_states, self.pending_aux):
            rows = int(next(iter(states.values())).shape[0])
            use = min(rows, needed)
            if use:
                for name in STATE_FIELDS:
                    out_states[name].append(states[name][:use])
                for name in out_aux:
                    out_aux[name].append(aux[name][:use])
                needed -= use
            if use < rows:
                tail_states = {name: tensor[use:] for name, tensor in states.items()}
                tail_aux = {name: tensor[use:] for name, tensor in aux.items()}
                remaining_states.append(tail_states)
                remaining_aux.append(tail_aux)
                remaining_rows += rows - use
            if needed <= 0:
                continue
        self.pending_states = remaining_states
        self.pending_aux = remaining_aux
        self.pending_rows = remaining_rows
        return (
            {name: torch.cat(parts, dim=0) for name, parts in out_states.items()},
            {name: torch.cat(parts, dim=0) for name, parts in out_aux.items()},
            take,
        )

    def flush(self, *, max_rows: int | None = None) -> None:
        if self.pending_rows <= 0:
            return
        states, aux, rows = self._take_pending(max_rows)
        filename = f"states_{self.shard_index:06d}.pt"
        path = self.bucket_dir / filename
        torch.save(
            {
                "kind": "preflop_street_closed_states_shard",
                "num_rows": rows,
                "states": states,
                "metadata_tensors": aux,
            },
            path,
        )
        self.shards.append({"path": filename, "num_rows": rows})
        self.shard_index += 1


def _write_manifest(
    *,
    output_dir: Path,
    input_manifest: dict[str, Any],
    source_dataset: Path,
    bucket_rows: dict[str, int],
    writer: _StreetClosedWriter,
    args: argparse.Namespace,
) -> None:
    manifest = {
        "kind": "preflop_street_closed_states",
        "format_version": 1,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_dataset.resolve()),
        "source_kind": input_manifest.get("kind"),
        "source_manifest": input_manifest,
        "cap": None if int(args.cap) <= 0 else int(args.cap),
        "num_rows": int(writer.total_rows),
        "state_fields": list(STATE_FIELDS),
        "metadata_fields": ["source_bucket", "source_row", "closing_action"],
        "closure": {
            "source_street": 0,
            "new_street": 1,
            "force_heads_up_preflop_flop": False,
            "criterion": "legal action with PBSEnv.step_bins(...).new_streets == 1",
        },
        "env_config": input_manifest.get("env_config", {}),
        "model": input_manifest.get("model", {}),
        "buckets": [
            {
                "label": "preflop_street_closed",
                "num_rows": int(writer.total_rows),
                "source_bucket_rows": bucket_rows,
                "shards": writer.shards,
            }
        ],
    }
    tmp_path = output_dir / "manifest.json.tmp"
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(output_dir / "manifest.json")


@torch.no_grad()
def run(args: argparse.Namespace) -> Path:
    source_root = Path(args.state_dataset)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists and is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    manifest = _load_manifest(source_root, allow_partial=bool(args.allow_partial))
    items = _bucket_items(manifest)
    wanted = set(args.buckets or [item["label"] for item in items])
    items = [item for item in items if item["label"] in wanted]
    if not items:
        raise ValueError(f"No matching buckets found for {sorted(wanted)}")

    env = _make_env_from_manifest(
        manifest,
        num_envs=int(args.batch_size),
        device=device,
        seed=int(args.seed),
    )
    writer = _StreetClosedWriter(output_dir, shard_size=int(args.shard_size))
    bucket_rows = {str(item["label"]): 0 for item in items}
    bucket_ids = {str(item["label"]): i for i, item in enumerate(items)}
    cap = int(args.cap)
    capped = cap > 0

    for item in items:
        label = str(item["label"])
        source_row_base = 0
        for shard_path in _iter_shard_paths(root=source_root, item=item):
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            states = payload["states"]
            rows = int(payload.get("num_rows", next(iter(states.values())).shape[0]))
            for start in range(0, rows, int(args.batch_size)):
                if capped and writer.total_rows >= cap:
                    break
                end = min(start + int(args.batch_size), rows)
                batch_states = _slice_states(states, start, end)
                count = _copy_public_states_to_env(env, batch_states)
                active = (~env.done[:count]) & (env.street[:count] == 0)
                if not bool(active.any().item()):
                    continue
                bin_amounts, legal = env.legal_bins_amounts_and_mask()
                for action in range(env.num_bet_bins):
                    if capped and writer.total_rows >= cap:
                        break
                    _copy_public_states_to_env(env, batch_states)
                    action_mask = active & legal[:count, action]
                    if not bool(action_mask.any().item()):
                        continue
                    bin_indices = torch.full(
                        (env.N,), -1, dtype=torch.long, device=device
                    )
                    bin_indices[:count] = torch.where(
                        action_mask,
                        torch.full((count,), action, dtype=torch.long, device=device),
                        torch.full((count,), -1, dtype=torch.long, device=device),
                    )
                    _rewards, new_streets, _dealt_cards = env.step_bins(
                        bin_indices,
                        bin_amounts=bin_amounts,
                        legal_masks=legal,
                    )
                    closed = (
                        action_mask
                        & (new_streets[:count] == 1)
                        & (env.street[:count] == 1)
                        & (~env.done[:count])
                    )
                    if not bool(closed.any().item()):
                        continue
                    indices = torch.where(closed)[0]
                    remaining = cap - writer.total_rows if capped else int(indices.numel())
                    if capped and int(indices.numel()) > remaining:
                        indices = indices[:remaining]
                    closed_states = _gather_env_states(env, indices)
                    source_rows = torch.arange(
                        source_row_base + start,
                        source_row_base + end,
                        dtype=torch.long,
                    ).index_select(0, indices.cpu())
                    aux = {
                        "source_bucket": torch.full(
                            (int(indices.numel()),),
                            bucket_ids[label],
                            dtype=torch.long,
                        ),
                        "source_row": source_rows,
                        "closing_action": torch.full(
                            (int(indices.numel()),), action, dtype=torch.long
                        ),
                    }
                    writer.append(closed_states, aux=aux)
                    bucket_rows[label] += int(indices.numel())
                if writer.total_rows and writer.total_rows % 100_000 == 0:
                    print(f"extracted {writer.total_rows:,} states", flush=True)
            source_row_base += rows
            if capped and writer.total_rows >= cap:
                break
        print(
            f"{label}: saved {bucket_rows[label]:,} street-closed states",
            flush=True,
        )
        if capped and writer.total_rows >= cap:
            break

    writer.flush()
    _write_manifest(
        output_dir=output_dir,
        input_manifest=manifest,
        source_dataset=source_root,
        bucket_rows=bucket_rows,
        writer=writer,
        args=args,
    )
    print(f"wrote {writer.total_rows:,} states to {output_dir}", flush=True)
    return output_dir


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
