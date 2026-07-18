from enum import Enum


class ScalarContext(Enum):
    ACTIONS_ROUND = 0
    POT = 1
    MIN_RAISE = 2
    LOG_STACK_DEPTH_BB = 3
    LOG_POT_BB = 4
    MAX_COMMITTED = 5
    LAST_AGGRESSIVE_AMOUNT = 6
    NUM_LEGAL_ACTIONS = 7
    CAN_FOLD = 8
    CAN_RAISE = 9
    CAN_ALLIN = 10
    NUM_SCALAR_CONTEXT = 11


class ValueScalarContext(Enum):
    CHANCE_PHASE = 0
    POT = 1
    MIN_RAISE = 2
    LOG_STACK_DEPTH_BB = 3
    LOG_POT_BB = 4
    MAX_COMMITTED = 5
    LAST_AGGRESSIVE_AMOUNT = 6
    NUM_LEGAL_ACTIONS = 7
    CAN_FOLD = 8
    CAN_RAISE = 9
    CAN_ALLIN = 10
    NUM_SCALAR_CONTEXT = 11


class ChancePhase(Enum):
    POST_CHANCE = 0
    PRE_CHANCE = 1


class PlayerContext(Enum):
    STACK = 0
    COMMITTED = 1
    SPR = 2
    LOG_COMMITTED_BB = 3
    ALL_IN = 4
    IS_ACTOR = 5
    REL_POS_TO_BUTTON = 6
    TO_CALL_SCALE = 7
    TO_CALL_POT = 8
    STACK_AFTER_CALL_SCALE = 9
    NUM_PLAYER_CONTEXT = 10


class MultiwayScalarContext(Enum):
    ACTOR = 0
    POSITION = 1
    ACTIONS_ROUND = 2
    POT = 3
    MIN_RAISE = 4
    LOG_STACK_DEPTH_BB = 5
    LOG_POT_BB = 6
    MAX_COMMITTED = 7
    LAST_AGGRESSIVE_AMOUNT = 8
    UNOPENED_OR_CHECKED_TO_ACTOR = 9
    NUM_LEGAL_ACTIONS = 10
    CAN_FOLD = 11
    CAN_CALL = 12
    CAN_RAISE = 13
    CAN_ALLIN = 14
    NUM_SCALAR_CONTEXT = 15


class MultiwayValueScalarContext(Enum):
    ACTOR = 0
    POSITION = 1
    CHANCE_PHASE = 2
    POT = 3
    MIN_RAISE = 4
    LOG_STACK_DEPTH_BB = 5
    LOG_POT_BB = 6
    MAX_COMMITTED = 7
    LAST_AGGRESSIVE_AMOUNT = 8
    UNOPENED_OR_CHECKED_TO_ACTOR = 9
    NUM_LEGAL_ACTIONS = 10
    CAN_FOLD = 11
    CAN_CALL = 12
    CAN_RAISE = 13
    CAN_ALLIN = 14
    NUM_SCALAR_CONTEXT = 15


class MultiwayPlayerContext(Enum):
    STACK = 0
    COMMITTED = 1
    SPR = 2
    LOG_COMMITTED_BB = 3
    FOLDED = 4
    ALL_IN = 5
    ACTED_THIS_ROUND = 6
    IS_ACTOR = 7
    REL_POS_TO_ACTOR = 8
    REL_POS_TO_BUTTON = 9
    TO_CALL_SCALE = 10
    TO_CALL_POT = 11
    STACK_AFTER_CALL_SCALE = 12
    NUM_PLAYER_CONTEXT = 13


def context_length(num_players: int) -> int:
    return policy_context_length(num_players)


def policy_context_length(num_players: int) -> int:
    if num_players == 2:
        return (
            ScalarContext.NUM_SCALAR_CONTEXT.value
            + num_players * PlayerContext.NUM_PLAYER_CONTEXT.value
        )
    return (
        MultiwayScalarContext.NUM_SCALAR_CONTEXT.value
        + num_players * MultiwayPlayerContext.NUM_PLAYER_CONTEXT.value
    )


def value_context_length(num_players: int) -> int:
    if num_players == 2:
        return (
            ValueScalarContext.NUM_SCALAR_CONTEXT.value
            + num_players * PlayerContext.NUM_PLAYER_CONTEXT.value
        )
    return (
        MultiwayValueScalarContext.NUM_SCALAR_CONTEXT.value
        + num_players * MultiwayPlayerContext.NUM_PLAYER_CONTEXT.value
    )


def context_schemas(num_players: int, *, value: bool = False):
    if num_players == 2:
        return (ValueScalarContext if value else ScalarContext), PlayerContext
    return (
        MultiwayValueScalarContext if value else MultiwayScalarContext
    ), MultiwayPlayerContext
