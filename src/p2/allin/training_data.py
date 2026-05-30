from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from p2.allin.data import PreflopAllInBatch, make_random_preflop_allin_batch
from p2.allin.sampler import DEFAULT_PREFLOP_ALLIN_TABLE, estimate_preflop_allin_values
from p2.env.card_utils import NUM_HANDS


FEATURE_KEYS = (
    "beliefs",
    "starting_stacks",
    "committed",
    "stacks_after",
    "allin_mask",
    "folded_mask",
    "scale",
)
TARGET_KEY = "allin_values"
MANIFEST_NAME = "manifest.json"


@dataclass
class AllInDataGenConfig:
    players: int = 4
    sample_count: int = 50_000
    board_samples: int = 256
    tuple_samples: int = 0
    tuple_tries: int = 4
    board_chunk: int = 8
    hand_chunk: int = 128
    bb: int = 100
    min_stack_bb: int = 10
    mid_stack_bb: int = 200
    max_stack_bb: int = 400
    high_stack_mass_ratio: float = 1.0 / 3.0
    concentration: float = 1.0
    folded_commit_max_frac: float = 0.35
    preflop_table_path: str = str(DEFAULT_PREFLOP_ALLIN_TABLE)
    use_exact_two_player: bool = True


def batch_to_tensors(
    batch: PreflopAllInBatch,
    allin_values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "beliefs": batch.beliefs,
        "starting_stacks": batch.starting_stacks,
        "committed": batch.committed,
        "stacks_after": batch.stacks_after,
        "allin_mask": batch.allin_mask,
        "folded_mask": batch.folded_mask,
        "scale": batch.scale,
        TARGET_KEY: allin_values,
    }


def tensors_to_batch(
    tensors: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[PreflopAllInBatch, torch.Tensor]:
    batch = PreflopAllInBatch(
        beliefs=tensors["beliefs"].to(device, non_blocking=True),
        starting_stacks=tensors["starting_stacks"].to(device, non_blocking=True),
        committed=tensors["committed"].to(device, non_blocking=True),
        stacks_after=tensors["stacks_after"].to(device, non_blocking=True),
        allin_mask=tensors["allin_mask"].to(device, non_blocking=True),
        folded_mask=tensors["folded_mask"].to(device, non_blocking=True),
        scale=tensors["scale"].to(device, non_blocking=True),
    )
    return batch, tensors[TARGET_KEY].to(device, non_blocking=True)


def _slice_tensors(
    tensors: dict[str, torch.Tensor],
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in tensors.items()}


def _concat_tensor_chunks(
    chunks: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not chunks:
        raise ValueError("cannot concatenate an empty shard")
    return {
        key: torch.cat([chunk[key] for chunk in chunks], dim=0).contiguous()
        for key in (*FEATURE_KEYS, TARGET_KEY)
    }


def generate_allin_training_chunk(
    count: int,
    cfg: AllInDataGenConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None,
    compute_stats: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    batch = make_random_preflop_allin_batch(
        count,
        cfg.players,
        bb=cfg.bb,
        min_stack_bb=cfg.min_stack_bb,
        mid_stack_bb=cfg.mid_stack_bb,
        max_stack_bb=cfg.max_stack_bb,
        high_stack_mass_ratio=cfg.high_stack_mass_ratio,
        concentration=cfg.concentration,
        folded_commit_max_frac=cfg.folded_commit_max_frac,
        device=device,
        generator=generator,
    )
    targets, diag = estimate_preflop_allin_values(
        batch,
        sample_count=cfg.sample_count,
        board_samples=cfg.board_samples,
        tuple_samples=cfg.tuple_samples if cfg.tuple_samples > 0 else None,
        tuple_tries=cfg.tuple_tries,
        board_chunk=cfg.board_chunk,
        hand_chunk=cfg.hand_chunk,
        generator=generator,
        preflop_table_path=cfg.preflop_table_path,
        use_exact_two_player=cfg.use_exact_two_player,
        compute_stats=compute_stats,
    )
    return batch_to_tensors(batch, targets), diag


def save_allin_training_dataset(
    output_dir: str | Path,
    *,
    examples: int,
    shard_size: int,
    generation_batch_size: int,
    cfg: AllInDataGenConfig,
    device: torch.device,
    generator: torch.Generator | None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if examples <= 0:
        raise ValueError("examples must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists")

    shards: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    produced = 0
    shard_idx = 0
    while produced < examples:
        shard_examples = min(shard_size, examples - produced)
        chunks: list[dict[str, torch.Tensor]] = []
        diag: dict[str, float] = {}
        shard_start = time.perf_counter()
        remaining = shard_examples
        while remaining > 0:
            cur = min(generation_batch_size, remaining)
            chunk, diag = generate_allin_training_chunk(
                cur,
                cfg,
                device=device,
                generator=generator,
                compute_stats=(progress is not None),
            )
            chunks.append({key: value.cpu() for key, value in chunk.items()})
            remaining -= cur

        shard = _concat_tensor_chunks(chunks)
        filename = f"shard_{shard_idx:06d}.pt"
        torch.save(shard, output / filename)
        produced += shard_examples
        shard_info = {
            "file": filename,
            "examples": shard_examples,
            "start": produced - shard_examples,
            "end": produced,
        }
        shards.append(shard_info)
        shard_idx += 1

        if progress is not None:
            progress(
                {
                    "examples_done": produced,
                    "examples_total": examples,
                    "shard": shard_info,
                    "shard_seconds": time.perf_counter() - shard_start,
                    "total_seconds": time.perf_counter() - start_time,
                    "last_target_diag": diag,
                }
            )

    manifest = {
        "format": "p2.allin.training_data.v1",
        "examples": examples,
        "players": cfg.players,
        "hands": NUM_HANDS,
        "feature_keys": list(FEATURE_KEYS),
        "target_key": TARGET_KEY,
        "config": asdict(cfg),
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class PregeneratedAllInDataset:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / MANIFEST_NAME if self.path.is_dir() else self.path
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format") != "p2.allin.training_data.v1":
            raise ValueError(f"unsupported pregenerated data format in {manifest_path}")
        self.examples = int(self.manifest["examples"])
        self.players = int(self.manifest["players"])
        self.hands = int(self.manifest["hands"])
        if self.hands != NUM_HANDS:
            raise ValueError(f"expected {NUM_HANDS} hands, got {self.hands}")
        self.shards = list(self.manifest["shards"])
        self._loaded_index: int | None = None
        self._loaded_tensors: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.examples

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        if self._loaded_index != shard_idx:
            shard_path = self.root / self.shards[shard_idx]["file"]
            self._loaded_tensors = torch.load(shard_path, map_location="cpu")
            self._loaded_index = shard_idx
        assert self._loaded_tensors is not None
        return self._loaded_tensors

    def get_batch(
        self,
        start: int,
        count: int,
        *,
        device: torch.device,
    ) -> tuple[PreflopAllInBatch, torch.Tensor]:
        if start < 0 or count <= 0:
            raise ValueError("start must be nonnegative and count must be positive")
        end = start + count
        if end > self.examples:
            raise IndexError(
                f"pregenerated data exhausted: requested rows [{start}, {end}), "
                f"dataset has {self.examples}"
            )

        parts: list[dict[str, torch.Tensor]] = []
        for shard_idx, shard in enumerate(self.shards):
            shard_start = int(shard["start"])
            shard_end = int(shard["end"])
            if shard_end <= start:
                continue
            if shard_start >= end:
                break
            local_start = max(start, shard_start) - shard_start
            local_end = min(end, shard_end) - shard_start
            parts.append(_slice_tensors(self._load_shard(shard_idx), local_start, local_end))

        if not parts:
            raise IndexError(f"no pregenerated rows found for [{start}, {end})")
        tensors = parts[0] if len(parts) == 1 else _concat_tensor_chunks(parts)
        return tensors_to_batch(tensors, device=device)
