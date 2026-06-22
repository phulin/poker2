from enum import Enum


class ScalarContext(Enum):
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


class ValueScalarContext(Enum):
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


class ChancePhase(Enum):
    POST_CHANCE = 0
    PRE_CHANCE = 1


class PlayerContext(Enum):
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
    return (
        ScalarContext.NUM_SCALAR_CONTEXT.value
        + num_players * PlayerContext.NUM_PLAYER_CONTEXT.value
    )


def value_context_length(num_players: int) -> int:
    return (
        ValueScalarContext.NUM_SCALAR_CONTEXT.value
        + num_players * PlayerContext.NUM_PLAYER_CONTEXT.value
    )
