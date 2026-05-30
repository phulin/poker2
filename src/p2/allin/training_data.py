from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from p2.allin.data import PreflopAllInBatch, make_random_preflop_allin_batch
from p2.allin.sampler import DEFAULT_PREFLOP_ALLIN_TABLE, estimate_preflop_allin_values
from p2.env.card_utils import NUM_HANDS, combo_suit_permutation_inverse_tensor


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


def permute_allin_batch_by_suit(
    batch: PreflopAllInBatch,
    targets: torch.Tensor,
    *,
    suit_permutation_idxs: torch.Tensor,
) -> tuple[PreflopAllInBatch, torch.Tensor]:
    combo_permutations_inverse = combo_suit_permutation_inverse_tensor(
        device=suit_permutation_idxs.device
    )[suit_permutation_idxs]
    beliefs = torch.gather(
        batch.beliefs,
        2,
        combo_permutations_inverse[:, None, :].expand(
            -1,
            batch.beliefs.shape[1],
            -1,
        ),
    )
    permuted_targets = torch.gather(
        targets,
        2,
        combo_permutations_inverse[:, None, :].expand(
            -1,
            targets.shape[1],
            -1,
        ),
    )
    return (
        PreflopAllInBatch(
            beliefs=beliefs,
            starting_stacks=batch.starting_stacks,
            committed=batch.committed,
            stacks_after=batch.stacks_after,
            allin_mask=batch.allin_mask,
            folded_mask=batch.folded_mask,
            scale=batch.scale,
        ),
        permuted_targets,
    )


def permute_allin_batch_by_players(
    batch: PreflopAllInBatch,
    targets: torch.Tensor,
    *,
    player_permutations: torch.Tensor,
) -> tuple[PreflopAllInBatch, torch.Tensor]:
    if player_permutations.shape != batch.starting_stacks.shape:
        raise ValueError(
            f"player_permutations must have shape {tuple(batch.starting_stacks.shape)}"
        )

    player_permutations = player_permutations.to(
        device=batch.beliefs.device,
        dtype=torch.long,
        non_blocking=True,
    )
    belief_index = player_permutations[:, :, None].expand(
        -1,
        -1,
        batch.beliefs.shape[2],
    )
    target_index = player_permutations[:, :, None].expand(
        -1,
        -1,
        targets.shape[2],
    )
    return (
        PreflopAllInBatch(
            beliefs=torch.gather(batch.beliefs, 1, belief_index),
            starting_stacks=torch.gather(batch.starting_stacks, 1, player_permutations),
            committed=torch.gather(batch.committed, 1, player_permutations),
            stacks_after=torch.gather(batch.stacks_after, 1, player_permutations),
            allin_mask=torch.gather(batch.allin_mask, 1, player_permutations),
            folded_mask=torch.gather(batch.folded_mask, 1, player_permutations),
            scale=batch.scale,
        ),
        torch.gather(targets, 1, target_index),
    )


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


def _concat_batches(
    chunks: list[tuple[PreflopAllInBatch, torch.Tensor]],
) -> tuple[PreflopAllInBatch, torch.Tensor]:
    if not chunks:
        raise ValueError("cannot concatenate an empty batch")
    batches, targets = zip(*chunks, strict=True)
    return (
        PreflopAllInBatch(
            beliefs=torch.cat([batch.beliefs for batch in batches], dim=0).contiguous(),
            starting_stacks=torch.cat(
                [batch.starting_stacks for batch in batches], dim=0
            ).contiguous(),
            committed=torch.cat(
                [batch.committed for batch in batches], dim=0
            ).contiguous(),
            stacks_after=torch.cat(
                [batch.stacks_after for batch in batches], dim=0
            ).contiguous(),
            allin_mask=torch.cat(
                [batch.allin_mask for batch in batches], dim=0
            ).contiguous(),
            folded_mask=torch.cat(
                [batch.folded_mask for batch in batches], dim=0
            ).contiguous(),
            scale=torch.cat([batch.scale for batch in batches], dim=0).contiguous(),
        ),
        torch.cat(targets, dim=0).contiguous(),
    )


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
    def __init__(
        self,
        path: str | Path,
        *,
        pin_memory: bool = False,
        async_shard_prefetch: bool = False,
    ) -> None:
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
        self._pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="allin-shard-prefetch")
            if async_shard_prefetch
            else None
        )
        self._prefetched_index: int | None = None
        self._prefetched_future: Future[dict[str, torch.Tensor]] | None = None

    def __len__(self) -> int:
        return self.examples

    def close(self) -> None:
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_shard_from_disk(self, shard_idx: int) -> dict[str, torch.Tensor]:
        shard_path = self.root / self.shards[shard_idx]["file"]
        tensors = torch.load(shard_path, map_location="cpu")
        if self._pin_memory:
            tensors = {
                key: value.pin_memory() if isinstance(value, torch.Tensor) else value
                for key, value in tensors.items()
            }
        return tensors

    def _load_shard(self, shard_idx: int) -> dict[str, torch.Tensor]:
        if self._loaded_index != shard_idx:
            if (
                self._prefetched_index == shard_idx
                and self._prefetched_future is not None
            ):
                self._loaded_tensors = self._prefetched_future.result()
                self._prefetched_index = None
                self._prefetched_future = None
            else:
                self._loaded_tensors = self._load_shard_from_disk(shard_idx)
            self._loaded_index = shard_idx
        assert self._loaded_tensors is not None
        return self._loaded_tensors

    def _shard_index_for_row(self, row: int) -> int:
        if row < 0 or row >= self.examples:
            raise IndexError(f"row {row} outside pregenerated dataset")
        for shard_idx, shard in enumerate(self.shards):
            if int(shard["start"]) <= row < int(shard["end"]):
                return shard_idx
        raise IndexError(f"no shard contains pregenerated row {row}")

    def prefetch_shard_for_row(self, row: int) -> None:
        if self._prefetch_executor is None:
            return
        shard_idx = self._shard_index_for_row(row % self.examples)
        if self._loaded_index == shard_idx or self._prefetched_index == shard_idx:
            return
        if self._prefetched_future is not None and not self._prefetched_future.done():
            return
        self._prefetched_index = shard_idx
        self._prefetched_future = self._prefetch_executor.submit(
            self._load_shard_from_disk,
            shard_idx,
        )

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
            parts.append(
                _slice_tensors(self._load_shard(shard_idx), local_start, local_end)
            )

        if not parts:
            raise IndexError(f"no pregenerated rows found for [{start}, {end})")
        tensors = parts[0] if len(parts) == 1 else _concat_tensor_chunks(parts)
        return tensors_to_batch(tensors, device=device)

    def get_wrapped_batch(
        self,
        start: int,
        count: int,
        *,
        device: torch.device,
    ) -> tuple[PreflopAllInBatch, torch.Tensor]:
        if start < 0 or count <= 0:
            raise ValueError("start must be nonnegative and count must be positive")

        chunks: list[tuple[PreflopAllInBatch, torch.Tensor]] = []
        remaining = count
        row_start = start % self.examples
        while remaining > 0:
            cur = min(remaining, self.examples - row_start)
            chunks.append(self.get_batch(row_start, cur, device=device))
            remaining -= cur
            row_start = 0
        return chunks[0] if len(chunks) == 1 else _concat_batches(chunks)
