from enum import Enum
from dataclasses import dataclass
import torch


class ScalarContext(Enum):
    ACTOR = 0
    POSITION = 1
    ACTIONS_ROUND = 2
    POT = 3
    MIN_RAISE = 4
    NUM_SCALAR_CONTEXT = 5


class PlayerContext(Enum):
    STACK = 0
    COMMITTED = 1
    SPR = 2
    NUM_PLAYER_CONTEXT = 3


def context_length(num_players: int) -> int:
    return (
        ScalarContext.NUM_SCALAR_CONTEXT.value
        + num_players * PlayerContext.NUM_PLAYER_CONTEXT.value
    )
