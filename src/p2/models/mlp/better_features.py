from enum import Enum


class ScalarContext(Enum):
    ACTOR = 0
    POSITION = 1
    STREET = 2
    ACTIONS_ROUND = 3
    POT = 4
    MIN_RAISE = 5
    NUM_SCALAR_CONTEXT = 6


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
