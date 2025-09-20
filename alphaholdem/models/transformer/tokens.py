"""Token definitions for transformer-based poker models."""

from enum import Enum


CLS_INDEX = 0
HOLE0_INDEX = 2
HOLE1_INDEX = 3


class Special(Enum):
    """Special tokens used in the variable-length transformer sequence."""

    CLS = 0
    CONTEXT = 1
    STREET_PREFLOP = 2
    STREET_FLOP = 3
    STREET_TURN = 4
    STREET_RIVER = 5
    NUM_SPECIAL = 6


class Context(Enum):
    """Context token indices for the transformer model."""

    POT = 0
    STACK_P0 = 1
    STACK_P1 = 2
    EFFECTIVE_STACK_P0 = 3
    EFFECTIVE_STACK_P1 = 4
    SPR_P0 = 5
    SPR_P1 = 6
    COMMITTED_P0 = 7
    COMMITTED_P1 = 8
    POSITION = 9
    ACTIONS_ROUND = 10
    MIN_RAISE = 11
    BET_TO_CALL = 12
    NUM_CONTEXT = 13


class Cls(Enum):
    """CLS token indices for the transformer model."""

    SB = 0
    BB = 1
    HERO_POSITION = 2
    NUM_CLS = 3


def get_special_token_id_offset() -> int:
    """value_offset where special tokens start."""

    return 0


def get_card_token_id_offset() -> int:
    """value_offset where card tokens start."""

    return get_special_token_id_offset() + Special.NUM_SPECIAL.value


def get_action_token_id_offset() -> int:
    """value_offset where action tokens start."""

    return get_card_token_id_offset() + 52
