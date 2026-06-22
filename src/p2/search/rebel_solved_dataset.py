from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import torch

from p2.env.card_utils import NUM_HANDS, PREFLOP_HANDS
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
SUPPORTED_HAND_DIMS = (NUM_HANDS, PREFLOP_HANDS)


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


def _infer_hand_dim_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    hand_dim: int | None = None,
) -> int:
    beliefs = tensors["features.beliefs"]
    if beliefs.dim() != 2:
        raise ValueError(f"features.beliefs must be 2-D, got {tuple(beliefs.shape)}")
    width = int(beliefs.shape[1])
    candidates: list[int] = []
    if hand_dim is not None:
        candidates.append(int(hand_dim))
    candidates.extend(SUPPORTED_HAND_DIMS)

    seen: set[int] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate not in SUPPORTED_HAND_DIMS or width % candidate != 0:
            continue
        value_targets = tensors.get("value_targets")
        if (
            isinstance(value_targets, torch.Tensor)
            and value_targets.dim() >= 3
            and int(value_targets.shape[-1]) != candidate
        ):
            continue
        policy_targets = tensors.get("policy_targets")
        if (
            isinstance(policy_targets, torch.Tensor)
            and policy_targets.dim() >= 3
            and int(policy_targets.shape[-2]) != candidate
        ):
            continue
        return candidate

    raise ValueError(
        "cannot infer solved dataset hand dimension from features.beliefs "
        f"width {width}; expected a multiple of one of {SUPPORTED_HAND_DIMS}"
    )


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
    hand_dim: int | None = None,
) -> RebelBatch:
    """Deserialize a tensor-only shard payload into a RebelBatch."""

    if device is None:
        device = torch.device("cpu")
    inferred_hand_dim = _infer_hand_dim_from_tensors(tensors, hand_dim=hand_dim)
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
            hand_dim=inferred_hand_dim,
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


def _add_count_dict(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[str(key)] = target.get(str(key), 0) + int(value)


def _street_names(streets: Sequence[int]) -> list[str]:
    return [STREET_NAMES.get(int(street), str(int(street))) for street in streets]


class RebelSolvedDatasetWriter:
    """Streaming writer for bounded solved ReBeL examples.

    Shards are written as batches arrive; only small manifest counters are kept
    in memory.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        storage_float_dtype: torch.dtype | str | None = None,
    ) -> None:
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME
        if self.manifest_path.exists():
            raise FileExistsError(f"{self.manifest_path} already exists")
        self.storage_dtype_name = _storage_dtype_name(storage_float_dtype) or "float32"
        self.shards: dict[StreamName, list[dict[str, Any]]] = {
            "value": [],
            "policy": [],
        }
        self.examples: dict[StreamName, int] = {"value": 0, "policy": 0}
        self.street_counts: dict[StreamName, dict[str, int]] = {
            "value": {},
            "policy": {},
        }
        self.depth_counts: dict[StreamName, dict[str, int]] = {
            "value": {},
            "policy": {},
        }
        self.target_source_counts: dict[StreamName, dict[str, int]] = {
            "value": {},
            "policy": {},
        }
        self.root_source_counts: dict[StreamName, dict[str, int]] = {
            "value": {},
            "policy": {},
        }
        self.leaf_target_source_counts: dict[str, int] = {}
        self.street_values: set[int] = set()
        self.example_batch: RebelBatch | None = None

    def append(self, stream: StreamName, batch: RebelBatch) -> None:
        _validate_stream_batch(batch, stream)
        if len(batch) == 0:
            return
        if self.example_batch is None:
            self.example_batch = batch

        stream_dir = self.root / stream
        stream_dir.mkdir(parents=True, exist_ok=True)
        shard_idx = len(self.shards[stream])
        start = self.examples[stream]
        end = start + len(batch)
        rel_path = f"{stream}/shard_{shard_idx:06d}.pt"
        torch.save(
            rebel_batch_to_tensors(
                batch, storage_float_dtype=self.storage_dtype_name
            ),
            self.root / rel_path,
        )
        self.shards[stream].append({"file": rel_path, "start": start, "end": end})
        self.examples[stream] = end

        self.street_values.update(int(x) for x in batch.features.street.unique().tolist())
        _add_count_dict(self.street_counts[stream], _count_tensor_values([batch], "street"))
        _add_count_dict(self.depth_counts[stream], _count_tensor_values([batch], "node_depth"))
        _add_count_dict(
            self.target_source_counts[stream],
            _count_tensor_values([batch], "target_source"),
        )
        _add_count_dict(
            self.root_source_counts[stream],
            _count_tensor_values([batch], "root_source"),
        )
        if stream == "value":
            _add_count_dict(
                self.leaf_target_source_counts,
                _leaf_target_source_counts([batch]),
            )

    def finalize(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        example_batch = self.example_batch
        if example_batch is None:
            raise ValueError("At least one value or policy batch is required")

        value_street_counts = self.street_counts["value"]
        policy_street_counts = self.street_counts["policy"]
        value_depth_counts = self.depth_counts["value"]
        policy_depth_counts = self.depth_counts["policy"]
        value_target_source_counts = self.target_source_counts["value"]
        policy_target_source_counts = self.target_source_counts["policy"]
        value_root_source_counts = self.root_source_counts["value"]
        policy_root_source_counts = self.root_source_counts["policy"]
        value_leaf_target_source_counts = dict(
            sorted(self.leaf_target_source_counts.items(), key=lambda item: int(item[0]))
        )

        manifest: dict[str, Any] = {
            "format": FORMAT_VERSION,
            "num_players": example_batch.features.num_players,
            "hands": example_batch.features.hand_dim,
            "num_actions": int(example_batch.legal_masks.shape[-1]),
            "context_length": int(example_batch.features.context.shape[-1]),
            "street_support": sorted(self.street_values),
            "included_streets": _street_names(sorted(self.street_values)),
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
            "storage_float_dtype": self.storage_dtype_name,
            "value_examples": int(self.examples["value"]),
            "policy_examples": int(self.examples["policy"]),
            "shards": self.shards,
        }
        if metadata is not None:
            manifest.update(dict(metadata))
            manifest["hands"] = example_batch.features.hand_dim
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def write_rebel_solved_dataset(
    output_dir: str | Path,
    *,
    value_batches: Sequence[RebelBatch] = (),
    policy_batches: Sequence[RebelBatch] = (),
    metadata: Mapping[str, Any] | None = None,
    storage_float_dtype: torch.dtype | str | None = None,
) -> dict[str, Any]:
    """Write bounded solved ReBeL examples as tensor-only shards."""

    writer = RebelSolvedDatasetWriter(
        output_dir, storage_float_dtype=storage_float_dtype
    )
    for batch in value_batches:
        writer.append("value", batch)
    for batch in policy_batches:
        writer.append("policy", batch)
    return writer.finalize(metadata)


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
        manifest_hands = int(self.manifest.get("hands", -1))
        if manifest_hands not in SUPPORTED_HAND_DIMS:
            raise ValueError(
                f"manifest hands must be one of {SUPPORTED_HAND_DIMS}, "
                f"got {manifest_hands}"
            )
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
            tensors,
            device=device,
            float_dtype=float_dtype,
            hand_dim=int(self.manifest.get("hands", NUM_HANDS)),
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
            tensors,
            device=device,
            float_dtype=float_dtype,
            hand_dim=int(self.manifest.get("hands", NUM_HANDS)),
        )
