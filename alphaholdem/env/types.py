from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Card is represented as integer 0..51, rank 0..12, suit 0..3


@dataclass
class Action:
    kind: str  # "fold", "check", "call", "bet", "raise", "allin"
    amount: int = 0  # chips committed in this action (for bet/raise/allin)


@dataclass
class PlayerState:
    stack: int
    stack_after_posting: int = 0
    committed: int = 0
    hole_cards: List[int] = field(default_factory=list)
    has_folded: bool = False
    is_allin: bool = False


@dataclass
class GameState:
    button: int  # 0 or 1
    street: str  # "preflop", "flop", "turn", "river", "showdown"
    deck: List[int]
    board: List[int] = field(default_factory=list)
    pot: int = 0
    to_act: int = 0
    small_blind: int = 50
    big_blind: int = 100
    min_raise: int = 0
    last_aggressive_amount: int = 0
    players: Tuple[PlayerState, PlayerState] = field(
        default_factory=lambda: (PlayerState(0), PlayerState(0))
    )
    terminal: bool = False
    winner: Optional[int] = None  # 0/1 or None for split
    # (street, actor, kind, amount, to_call_at_time, total_committed_at_time)
    action_history: List[Tuple[str, int, str, int, int, int]] = field(
        default_factory=list
    )
