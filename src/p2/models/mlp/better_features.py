from enum import Enum


class ScalarContext(Enum):
    ACTOR = 0
    POSITION = 1
    ACTIONS_ROUND = 2
    POT = 3
    MIN_RAISE = 4
    LOG_STACK_DEPTH_BB = 5
    LOG_POT_BB = 6
    NUM_SCALAR_CONTEXT = 7


class ValueScalarContext(Enum):
    ACTOR = 0
    POSITION = 1
    CHANCE_PHASE = 2
    POT = 3
    MIN_RAISE = 4
    LOG_STACK_DEPTH_BB = 5
    LOG_POT_BB = 6
    NUM_SCALAR_CONTEXT = 7


class ChancePhase(Enum):
    POST_CHANCE = 0
    PRE_CHANCE = 1


class PlayerContext(Enum):
    STACK = 0
    COMMITTED = 1
    SPR = 2
    LOG_COMMITTED_BB = 3
    NUM_PLAYER_CONTEXT = 4


def context_length(num_players: int) -> int:
    return policy_context_length(num_players)


def policy_context_length(num_players: int) -> int:
    return (
        ScalarContext.NUM_SCALAR_CONTEXT.value
        + num_players * PlayerContext.NUM_PLAYER_CONTEXT.value
    )


def value_context_length(num_players: int) -> int:
    return (
        ValueScalarContext.NUM_SCALAR_CONTEXT.value
        + num_players * PlayerContext.NUM_PLAYER_CONTEXT.value
    )
