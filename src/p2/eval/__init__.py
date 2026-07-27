"""Real-hand, real-chip agent evaluation (duplicate matches + game records)."""

from p2.eval.agents import (
    AgentIdentity,
    CallAgent,
    FoldAgent,
    MatchAgent,
    RandomAgent,
    SearchAgent,
)
from p2.eval.duplicate_match import MatchResult, play_duplicate_match
from p2.eval.records import (
    GameBatchTensors,
    GameRecord,
    RecordWriter,
    load_manifest,
    load_records,
    pair_differences,
)

__all__ = [
    "AgentIdentity",
    "CallAgent",
    "FoldAgent",
    "GameBatchTensors",
    "GameRecord",
    "MatchAgent",
    "MatchResult",
    "RandomAgent",
    "RecordWriter",
    "SearchAgent",
    "load_manifest",
    "load_records",
    "pair_differences",
    "play_duplicate_match",
]
