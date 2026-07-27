"""Append-only per-game outcome records for real-hand evaluation matches.

Every single game played by :mod:`p2.eval.duplicate_match` is written as one
JSONL row so that rating models (TrueSkill/Elo/Bayesian) can be refit offline
without ever replaying poker. A sidecar ``<path>.manifest.json`` holds the run
configuration (agents, env, search fidelity, seed).

Performance note: rows are materialized from device tensors with a *single*
host transfer. All per-game columns (integer and floating point alike) are
concatenated into one ``float64`` matrix and moved to the CPU in one go;
float64 represents every integer field here exactly, and float32 rewards widen
losslessly. There is no per-game ``.item()`` anywhere in this module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


TERMINAL_FOLD = 0
TERMINAL_SHOWDOWN = 1
TERMINAL_TIMEOUT = 2

TERMINAL_NAMES = {
    TERMINAL_FOLD: "fold",
    TERMINAL_SHOWDOWN: "showdown",
    TERMINAL_TIMEOUT: "timeout",
}

STREET_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
    4: "showdown_street",
}

RECORD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GameRecord:
    """One played game, scored from agent A's perspective.

    ``reward_a_*`` are all the same signed quantity in three units:
    ``stacks`` is the env-native reward (chips / effective stack), ``chips`` is
    raw chips, and ``bb`` is big blinds.
    """

    eval_id: str
    schema_version: int
    game_index: int
    pair_id: int
    pair_half: int
    agent_a_id: str
    agent_b_id: str
    seat_a: int
    button: int
    reward_a_chips: float
    reward_a_bb: float
    reward_a_stacks: float
    terminal_type: str
    street: int
    street_name: str
    final_pot: int
    num_decisions: int
    board: list[int]
    hole_a: list[int]
    hole_b: list[int]
    hole_a_combo: int
    hole_b_combo: int
    winner: int
    effective_stack: int
    big_blind: int
    seed: int
    agent_a_fidelity: dict[str, Any] = field(default_factory=dict)
    agent_b_fidelity: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GameRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in row.items() if k in known})


# --------------------------------------------------------------------- batch

# Column layout of the single host transfer. Order matters only internally.
_INT_COLUMNS = (
    "pair_id",
    "pair_half",
    "seat_a",
    "button",
    "terminal_type",
    "street",
    "final_pot",
    "num_decisions",
    "winner",
    "effective_stack",
    "hole_a_combo",
    "hole_b_combo",
)
_FLOAT_COLUMNS = ("reward_a_stacks",)
_CARD_COLUMNS = (("board", 5), ("hole_a", 2), ("hole_b", 2))


@dataclass
class GameBatchTensors:
    """Per-game outcome columns, all on-device, all shaped ``[G]`` (or ``[G, k]``).

    Filled by the match player and converted to :class:`GameRecord` rows with a
    single device->host transfer via :meth:`to_records`.
    """

    pair_id: torch.Tensor
    pair_half: torch.Tensor
    seat_a: torch.Tensor
    button: torch.Tensor
    terminal_type: torch.Tensor
    street: torch.Tensor
    final_pot: torch.Tensor
    num_decisions: torch.Tensor
    winner: torch.Tensor
    effective_stack: torch.Tensor
    hole_a_combo: torch.Tensor
    hole_b_combo: torch.Tensor
    reward_a_stacks: torch.Tensor
    board: torch.Tensor  # [G, 5]
    hole_a: torch.Tensor  # [G, 2]
    hole_b: torch.Tensor  # [G, 2]

    def to_records(
        self,
        *,
        eval_id: str,
        agent_a_id: str,
        agent_b_id: str,
        big_blind: int,
        seed: int,
        agent_a_fidelity: dict[str, Any] | None = None,
        agent_b_fidelity: dict[str, Any] | None = None,
    ) -> list[GameRecord]:
        columns = [getattr(self, name).reshape(-1, 1) for name in _INT_COLUMNS]
        columns += [getattr(self, name).reshape(-1, 1) for name in _FLOAT_COLUMNS]
        columns += [
            getattr(self, name).reshape(len(self.pair_id), width)
            for name, width in _CARD_COLUMNS
        ]
        packed = torch.cat([c.to(torch.float64) for c in columns], dim=1)
        # THE single host transfer for the whole batch.
        host = packed.cpu().tolist()

        n_int = len(_INT_COLUMNS)
        n_float = len(_FLOAT_COLUMNS)
        fid_a = dict(agent_a_fidelity or {})
        fid_b = dict(agent_b_fidelity or {})

        records: list[GameRecord] = []
        for game_index, row in enumerate(host):
            ints = [int(v) for v in row[:n_int]]
            floats = row[n_int : n_int + n_float]
            cursor = n_int + n_float
            cards: dict[str, list[int]] = {}
            for name, width in _CARD_COLUMNS:
                cards[name] = [int(v) for v in row[cursor : cursor + width]]
                cursor += width
            values = dict(zip(_INT_COLUMNS, ints, strict=True))
            reward_stacks = float(floats[0])
            effective_stack = values["effective_stack"]
            reward_chips = reward_stacks * effective_stack
            street = values["street"]
            records.append(
                GameRecord(
                    eval_id=eval_id,
                    schema_version=RECORD_SCHEMA_VERSION,
                    game_index=game_index,
                    pair_id=values["pair_id"],
                    pair_half=values["pair_half"],
                    agent_a_id=agent_a_id,
                    agent_b_id=agent_b_id,
                    seat_a=values["seat_a"],
                    button=values["button"],
                    reward_a_chips=reward_chips,
                    reward_a_bb=reward_chips / float(big_blind),
                    reward_a_stacks=reward_stacks,
                    terminal_type=TERMINAL_NAMES[values["terminal_type"]],
                    street=street,
                    street_name=STREET_NAMES.get(street, f"street_{street}"),
                    final_pot=values["final_pot"],
                    num_decisions=values["num_decisions"],
                    board=cards["board"],
                    hole_a=cards["hole_a"],
                    hole_b=cards["hole_b"],
                    hole_a_combo=values["hole_a_combo"],
                    hole_b_combo=values["hole_b_combo"],
                    winner=values["winner"],
                    effective_stack=effective_stack,
                    big_blind=int(big_blind),
                    seed=int(seed),
                    agent_a_fidelity=fid_a,
                    agent_b_fidelity=fid_b,
                )
            )
        return records


# -------------------------------------------------------------------- writer


class RecordWriter:
    """Append-only JSONL writer with a sidecar run-config manifest."""

    def __init__(self, path: str | Path, manifest: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
        if manifest is not None:
            self.write_manifest(manifest)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        payload = dict(manifest)
        payload.setdefault("schema_version", RECORD_SCHEMA_VERSION)
        payload.setdefault("records_path", str(self.path))
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def append(self, records: Iterable[GameRecord]) -> int:
        count = 0
        with self.path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), sort_keys=True))
                handle.write("\n")
                count += 1
        return count


def load_records(path: str | Path) -> list[GameRecord]:
    """Read a JSONL record file back into :class:`GameRecord` objects."""
    rows: list[GameRecord] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(GameRecord.from_dict(json.loads(line)))
    return rows


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read the sidecar manifest for a JSONL record file."""
    p = Path(path)
    manifest_path = p.with_suffix(p.suffix + ".manifest.json")
    return json.loads(manifest_path.read_text())


def pair_differences(
    records: Sequence[GameRecord], unit: str = "bb"
) -> tuple[list[int], list[float]]:
    """Duplicate-scored per-pair results for agent A, from recorded games.

    For each pair, agent A's reward in half 0 plus its reward in half 1 is the
    difference between how agent A and agent B did on the *same* deal (in half 1
    agent A holds what agent B held, from the same position). We report half
    that sum so the number stays on a per-game scale.

    Returns:
        ``(pair_ids, differences)`` sorted by pair id, using only complete pairs.
    """
    attr = {
        "bb": "reward_a_bb",
        "chips": "reward_a_chips",
        "stacks": "reward_a_stacks",
    }[unit]

    halves: dict[int, dict[int, float]] = {}
    for record in records:
        halves.setdefault(record.pair_id, {})[record.pair_half] = float(
            getattr(record, attr)
        )

    pair_ids = sorted(pid for pid, h in halves.items() if {0, 1} <= set(h))
    diffs = [0.5 * (halves[pid][0] + halves[pid][1]) for pid in pair_ids]
    return pair_ids, diffs
