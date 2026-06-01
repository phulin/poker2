from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import torch

from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.rl.target_provenance import (
    TARGET_SOURCE_CFR_BACKUP,
    TARGET_SOURCE_CLOSING_NET,
    TARGET_SOURCE_EXACT_TERMINAL,
    TARGET_SOURCE_NAMES,
)


FORMAT_VERSION = "p2.rebel.solved_postflop.v1"
MANIFEST_NAME = "manifest.json"
StreamName = Literal["value", "policy"]
STREET_NAMES = {0: "preflop", 1: "flop", 2: "turn", 3: "river", 4: "terminal"}
SUPPORTED_STORAGE_FLOAT_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _storage_dtype_name(dtype: torch.dtype | str | None) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        name = dtype.removeprefix("torch.")
        if name not in SUPPORTED_STORAGE_FLOAT_DTYPES:
            raise ValueError(
                "storage_float_dtype must be one of "
                f"{sorted(SUPPORTED_STORAGE_FLOAT_DTYPES)}, got {dtype!r}"
            )
        return name
    for name, torch_dtype in SUPPORTED_STORAGE_FLOAT_DTYPES.items():
        if dtype == torch_dtype:
            return name
    raise ValueError(
        "storage_float_dtype must be one of "
        f"{sorted(SUPPORTED_STORAGE_FLOAT_DTYPES)}, got {dtype}"
    )


def _move_tensor(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    float_dtype: torch.dtype | None,
) -> torch.Tensor:
    if tensor.dtype.is_floating_point and float_dtype is not None:
        return tensor.to(device=device, dtype=float_dtype)
    return tensor.to(device=device)


def rebel_batch_to_tensors(
    batch: RebelBatch, *, storage_float_dtype: torch.dtype | str | None = None
) -> dict[str, torch.Tensor]:
    """Serialize a RebelBatch into plain tensors for shard storage."""

    dtype_name = _storage_dtype_name(storage_float_dtype)
    target_dtype = (
        SUPPORTED_STORAGE_FLOAT_DTYPES[dtype_name] if dtype_name is not None else None
    )

    def prepare(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().cpu()
        if target_dtype is not None and tensor.dtype.is_floating_point:
            tensor = tensor.to(target_dtype)
        return tensor

    tensors = {
        "features.context": prepare(batch.features.context),
        "features.street": prepare(batch.features.street),
        "features.to_act": prepare(batch.features.to_act),
        "features.board": prepare(batch.features.board),
        "features.beliefs": prepare(batch.features.beliefs),
        "legal_masks": prepare(batch.legal_masks),
    }
    if batch.value_targets is not None:
        tensors["value_targets"] = prepare(batch.value_targets)
    if batch.policy_targets is not None:
        tensors["policy_targets"] = prepare(batch.policy_targets)
    for key, value in batch.statistics.items():
        tensors[f"statistics.{key}"] = prepare(value)
    return tensors


def rebel_batch_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    device: torch.device | None = None,
    float_dtype: torch.dtype | None = torch.float32,
) -> RebelBatch:
    """Deserialize a tensor-only shard payload into a RebelBatch."""

    if device is None:
        device = torch.device("cpu")
    statistics_prefix = "statistics."
    statistics = {
        key[len(statistics_prefix) :]: _move_tensor(
            value, device=device, float_dtype=float_dtype
        )
        for key, value in tensors.items()
        if key.startswith(statistics_prefix)
    }
    return RebelBatch(
        features=MLPFeatures(
            context=_move_tensor(
                tensors["features.context"], device=device, float_dtype=float_dtype
            ),
            street=tensors["features.street"].to(device),
            to_act=tensors["features.to_act"].to(device),
            board=tensors["features.board"].to(device),
            beliefs=_move_tensor(
                tensors["features.beliefs"], device=device, float_dtype=float_dtype
            ),
        ),
        legal_masks=tensors["legal_masks"].to(device),
        value_targets=(
            _move_tensor(
                tensors["value_targets"], device=device, float_dtype=float_dtype
            )
            if "value_targets" in tensors
            else None
        ),
        policy_targets=(
            _move_tensor(
                tensors["policy_targets"], device=device, float_dtype=float_dtype
            )
            if "policy_targets" in tensors
            else None
        ),
        statistics=statistics,
    )


def _slice_tensors(
    tensors: Mapping[str, torch.Tensor], start: int, end: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:end] for key, value in tensors.items()}


def _index_tensors(
    tensors: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in tensors.items()}


def _concat_tensors(chunks: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not chunks:
        raise ValueError("Cannot concatenate an empty tensor chunk list")
    keys = set(chunks[0].keys())
    if any(set(chunk.keys()) != keys for chunk in chunks):
        raise ValueError("Shard tensor keys do not match")
    return {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in keys}


def _stream_target_key(stream: StreamName) -> str:
    return "value_targets" if stream == "value" else "policy_targets"


def _validate_stream_batch(batch: RebelBatch, stream: StreamName) -> None:
    if stream == "value":
        if batch.value_targets is None or batch.policy_targets is not None:
            raise ValueError("value stream shards must contain value targets only")
        return
    if batch.policy_targets is None or batch.value_targets is not None:
        raise ValueError("policy stream shards must contain policy targets only")


def _write_stream(
    root: Path,
    stream: StreamName,
    batches: Sequence[RebelBatch],
    *,
    storage_float_dtype: torch.dtype | str | None,
) -> tuple[list[dict[str, Any]], int]:
    stream_dir = root / stream
    stream_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    start = 0
    for shard_idx, batch in enumerate(batches):
        _validate_stream_batch(batch, stream)
        end = start + len(batch)
        rel_path = f"{stream}/shard_{shard_idx:06d}.pt"
        torch.save(
            rebel_batch_to_tensors(
                batch, storage_float_dtype=storage_float_dtype
            ),
            root / rel_path,
        )
        shards.append({"file": rel_path, "start": start, "end": end})
        start = end
    return shards, start


def _count_tensor_values(batches: Sequence[RebelBatch], key: str) -> dict[str, int]:
    counts: dict[int, int] = {}
    for batch in batches:
        if key == "street":
            values = batch.features.street
        else:
            values = batch.statistics.get(key)
            if values is None:
                continue
        unique, batch_counts = torch.unique(
            values.detach().cpu().reshape(-1), return_counts=True
        )
        for value, count in zip(unique.tolist(), batch_counts.tolist(), strict=True):
            counts[int(value)] = counts.get(int(value), 0) + int(count)
    return {str(key): counts[key] for key in sorted(counts)}


def _sum_stat_tensor(batches: Sequence[RebelBatch], key: str) -> int:
    total = 0
    for batch in batches:
        values = batch.statistics.get(key)
        if values is not None:
            total += int(values.detach().cpu().to(torch.long).sum().item())
    return total


def _leaf_target_source_counts(batches: Sequence[RebelBatch]) -> dict[str, int]:
    return {
        str(TARGET_SOURCE_CFR_BACKUP): _sum_stat_tensor(
            batches, f"leaf_target_source_{TARGET_SOURCE_CFR_BACKUP}_count"
        ),
        str(TARGET_SOURCE_EXACT_TERMINAL): _sum_stat_tensor(
            batches, f"leaf_target_source_{TARGET_SOURCE_EXACT_TERMINAL}_count"
        ),
        str(TARGET_SOURCE_CLOSING_NET): _sum_stat_tensor(
            batches, f"leaf_target_source_{TARGET_SOURCE_CLOSING_NET}_count"
        ),
    }


def _merge_count_dicts(*dicts: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in dicts:
        for key, value in counts.items():
            merged[key] = merged.get(key, 0) + int(value)
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def _street_names(streets: Sequence[int]) -> list[str]:
    return [STREET_NAMES.get(int(street), str(int(street))) for street in streets]


def write_rebel_solved_dataset(
    output_dir: str | Path,
    *,
    value_batches: Sequence[RebelBatch] = (),
    policy_batches: Sequence[RebelBatch] = (),
    metadata: Mapping[str, Any] | None = None,
    storage_float_dtype: torch.dtype | str | None = None,
) -> dict[str, Any]:
    """Write bounded solved ReBeL examples as tensor-only shards."""

    storage_dtype_name = _storage_dtype_name(storage_float_dtype) or "float32"
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} already exists")

    value_shards, value_examples = _write_stream(
        root,
        "value",
        value_batches,
        storage_float_dtype=storage_dtype_name,
    )
    policy_shards, policy_examples = _write_stream(
        root,
        "policy",
        policy_batches,
        storage_float_dtype=storage_dtype_name,
    )
    example_batch = next(iter(value_batches), None) or next(iter(policy_batches), None)
    if example_batch is None:
        raise ValueError("At least one value or policy batch is required")

    street_values: set[int] = set()
    for batch in (*value_batches, *policy_batches):
        street_values.update(int(x) for x in batch.features.street.unique().tolist())
    value_street_counts = _count_tensor_values(value_batches, "street")
    policy_street_counts = _count_tensor_values(policy_batches, "street")
    value_depth_counts = _count_tensor_values(value_batches, "node_depth")
    policy_depth_counts = _count_tensor_values(policy_batches, "node_depth")
    value_target_source_counts = _count_tensor_values(value_batches, "target_source")
    policy_target_source_counts = _count_tensor_values(policy_batches, "target_source")
    value_leaf_target_source_counts = _leaf_target_source_counts(value_batches)
    value_root_source_counts = _count_tensor_values(value_batches, "root_source")
    policy_root_source_counts = _count_tensor_values(policy_batches, "root_source")

    manifest: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "num_players": example_batch.features.num_players,
        "hands": NUM_HANDS,
        "num_actions": int(example_batch.legal_masks.shape[-1]),
        "context_length": int(example_batch.features.context.shape[-1]),
        "street_support": sorted(street_values),
        "included_streets": _street_names(sorted(street_values)),
        "street_counts": {
            "value": value_street_counts,
            "policy": policy_street_counts,
            "total": _merge_count_dicts(value_street_counts, policy_street_counts),
        },
        "node_depth_counts": {
            "value": value_depth_counts,
            "policy": policy_depth_counts,
            "total": _merge_count_dicts(value_depth_counts, policy_depth_counts),
        },
        "target_source_counts": {
            "value": value_target_source_counts,
            "policy": policy_target_source_counts,
            "total": _merge_count_dicts(
                value_target_source_counts, policy_target_source_counts
            ),
        },
        "target_source_names": {
            str(code): name for code, name in sorted(TARGET_SOURCE_NAMES.items())
        },
        "leaf_target_source_counts": {
            "value": value_leaf_target_source_counts,
            "policy": {},
            "total": value_leaf_target_source_counts,
        },
        "root_source_counts": {
            "value": value_root_source_counts,
            "policy": policy_root_source_counts,
            "total": _merge_count_dicts(
                value_root_source_counts, policy_root_source_counts
            ),
        },
        "storage_float_dtype": storage_dtype_name,
        "value_examples": int(value_examples),
        "policy_examples": int(policy_examples),
        "shards": {"value": value_shards, "policy": policy_shards},
    }
    if metadata is not None:
        manifest.update(dict(metadata))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class RebelSolvedDataset:
    """Reader for bounded postflop solved-example datasets."""

    def __init__(
        self,
        path: str | Path,
        *,
        num_players: int | None = None,
        num_actions: int | None = None,
        context_length: int | None = None,
        model_name: str | None = None,
        action_schedule: Mapping[str, Any] | None = None,
        street_support: Sequence[int] | None = None,
        pin_memory: bool = False,
        async_shard_prefetch: bool = False,
    ) -> None:
        self.path = Path(path)
        manifest_path = self.path / MANIFEST_NAME if self.path.is_dir() else self.path
        self.root = manifest_path.parent
        self.manifest = json.loads(manifest_path.read_text())
        self._validate_manifest(
            manifest_path,
            num_players=num_players,
            num_actions=num_actions,
            context_length=context_length,
            model_name=model_name,
            action_schedule=action_schedule,
            street_support=street_support,
        )
        self.examples = {
            "value": int(self.manifest.get("value_examples", 0)),
            "policy": int(self.manifest.get("policy_examples", 0)),
        }
        self.shards = {
            "value": list(self.manifest.get("shards", {}).get("value", [])),
            "policy": list(self.manifest.get("shards", {}).get("policy", [])),
        }
        self._loaded: dict[StreamName, tuple[int, dict[str, torch.Tensor]] | None] = {
            "value": None,
            "policy": None,
        }
        self._pin_memory = bool(pin_memory and torch.cuda.is_available())
        self.storage_float_dtype = str(
            self.manifest.get("storage_float_dtype", "float32")
        )
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="rebel-shard-prefetch")
            if async_shard_prefetch
            else None
        )
        self._prefetched: dict[
            StreamName, tuple[int, Future[dict[str, torch.Tensor]]] | None
        ] = {"value": None, "policy": None}

    def close(self) -> None:
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=True)
            self._prefetch_executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _validate_manifest(
        self,
        manifest_path: Path,
        *,
        num_players: int | None,
        num_actions: int | None,
        context_length: int | None,
        model_name: str | None,
        action_schedule: Mapping[str, Any] | None,
        street_support: Sequence[int] | None,
    ) -> None:
        if self.manifest.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported solved dataset format in {manifest_path}")
        if int(self.manifest.get("hands", -1)) != NUM_HANDS:
            raise ValueError(f"expected {NUM_HANDS} hands")
        storage_float_dtype = str(self.manifest.get("storage_float_dtype", "float32"))
        if storage_float_dtype not in SUPPORTED_STORAGE_FLOAT_DTYPES:
            raise ValueError(
                "manifest storage_float_dtype mismatch: "
                f"got {storage_float_dtype!r}"
            )
        checks = {
            "num_players": num_players,
            "num_actions": num_actions,
            "context_length": context_length,
        }
        for key, expected in checks.items():
            if expected is None:
                continue
            actual = int(self.manifest.get(key, -1))
            if actual != int(expected):
                raise ValueError(
                    f"manifest {key} mismatch: expected {expected}, got {actual}"
                )
        if model_name is not None:
            actual_model = self.manifest.get("model_family")
            if actual_model is None:
                model_config = self.manifest.get("model_config", {})
                if isinstance(model_config, Mapping):
                    actual_model = model_config.get("name")
            if actual_model != model_name:
                raise ValueError(
                    f"manifest model mismatch: expected {model_name}, got {actual_model}"
                )
        if action_schedule is not None:
            actual_schedule = self.manifest.get("action_schedule")
            if actual_schedule != dict(action_schedule):
                raise ValueError("manifest action_schedule mismatch")
        if street_support is not None:
            actual_streets = {int(x) for x in self.manifest.get("street_support", [])}
            expected_streets = {int(x) for x in street_support}
            if not actual_streets.issubset(expected_streets):
                raise ValueError(
                    "manifest street_support mismatch: "
                    f"expected subset of {sorted(expected_streets)}, "
                    f"got {sorted(actual_streets)}"
                )

    def __len__(self) -> int:
        return self.examples["value"] + self.examples["policy"]

    def stream_len(self, stream: StreamName) -> int:
        return self.examples[stream]

    def _load_shard_from_disk(
        self, stream: StreamName, shard_idx: int
    ) -> dict[str, torch.Tensor]:
        shard_path = self.root / self.shards[stream][shard_idx]["file"]
        tensors = torch.load(shard_path, map_location="cpu", weights_only=True)
        if self._pin_memory:
            tensors = {
                key: value.pin_memory() if isinstance(value, torch.Tensor) else value
                for key, value in tensors.items()
            }
        return tensors

    def _load_shard(self, stream: StreamName, shard_idx: int) -> dict[str, torch.Tensor]:
        loaded = self._loaded[stream]
        if loaded is None or loaded[0] != shard_idx:
            prefetched = self._prefetched[stream]
            if prefetched is not None and prefetched[0] == shard_idx:
                tensors = prefetched[1].result()
                self._prefetched[stream] = None
            else:
                tensors = self._load_shard_from_disk(stream, shard_idx)
            target_key = _stream_target_key(stream)
            if target_key not in tensors:
                raise ValueError(f"{stream} shard missing {target_key}")
            self._loaded[stream] = (shard_idx, tensors)
        loaded = self._loaded[stream]
        assert loaded is not None
        return loaded[1]

    def prefetch_shard_for_row(self, stream: StreamName, row: int) -> None:
        if self._prefetch_executor is None or self.examples[stream] == 0:
            return
        shard_idx = self._shard_index_for_row(stream, row % self.examples[stream])
        loaded = self._loaded[stream]
        prefetched = self._prefetched[stream]
        if loaded is not None and loaded[0] == shard_idx:
            return
        if prefetched is not None and prefetched[0] == shard_idx:
            return
        if prefetched is not None and not prefetched[1].done():
            return
        self._prefetched[stream] = (
            shard_idx,
            self._prefetch_executor.submit(
                self._load_shard_from_disk, stream, shard_idx
            ),
        )

    def _shard_index_for_row(self, stream: StreamName, row: int) -> int:
        if row < 0 or row >= self.examples[stream]:
            raise IndexError(f"{stream} row {row} outside solved dataset")
        for shard_idx, shard in enumerate(self.shards[stream]):
            if int(shard["start"]) <= row < int(shard["end"]):
                return shard_idx
        raise IndexError(f"no {stream} shard contains row {row}")

    def get_batch(
        self,
        stream: StreamName,
        start: int,
        count: int,
        *,
        device: torch.device | None = None,
        float_dtype: torch.dtype | None = torch.float32,
        wrap: bool = False,
    ) -> RebelBatch:
        if count <= 0 or start < 0:
            raise ValueError("start must be nonnegative and count must be positive")
        total = self.examples[stream]
        if total == 0:
            raise ValueError(f"{stream} stream is empty")
        if not wrap and start + count > total:
            raise IndexError(
                f"{stream} data exhausted: requested [{start}, {start + count}), "
                f"dataset has {total}"
            )

        chunks: list[dict[str, torch.Tensor]] = []
        remaining = count
        cursor = start
        while remaining > 0:
            row = cursor % total if wrap else cursor
            shard_idx = self._shard_index_for_row(stream, row)
            shard = self.shards[stream][shard_idx]
            shard_start = int(shard["start"])
            shard_end = int(shard["end"])
            local_start = row - shard_start
            take = min(remaining, shard_end - row)
            chunks.append(
                _slice_tensors(
                    self._load_shard(stream, shard_idx),
                    local_start,
                    local_start + take,
                )
            )
            remaining -= take
            cursor += take

        if wrap or cursor < total:
            self.prefetch_shard_for_row(stream, cursor % total)
        tensors = chunks[0] if len(chunks) == 1 else _concat_tensors(chunks)
        return rebel_batch_from_tensors(
            tensors, device=device, float_dtype=float_dtype
        )

    def sample_batch(
        self,
        stream: StreamName,
        count: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
        float_dtype: torch.dtype | None = torch.float32,
    ) -> RebelBatch:
        total = self.examples[stream]
        if count <= 0:
            raise ValueError("count must be positive")
        if total == 0:
            raise ValueError(f"{stream} stream is empty")
        rows = torch.randint(0, total, (count,), generator=generator)
        chunks = []
        for shard_idx, shard in enumerate(self.shards[stream]):
            shard_start = int(shard["start"])
            shard_end = int(shard["end"])
            mask = (rows >= shard_start) & (rows < shard_end)
            if not mask.any():
                continue
            local_rows = rows[mask] - shard_start
            chunks.append(_index_tensors(self._load_shard(stream, shard_idx), local_rows))
        tensors = chunks[0] if len(chunks) == 1 else _concat_tensors(chunks)
        return rebel_batch_from_tensors(
            tensors, device=device, float_dtype=float_dtype
        )
