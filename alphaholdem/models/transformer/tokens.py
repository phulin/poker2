"""Token definitions for transformer-based poker models."""

from enum import Enum


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
    COMMITTED_P0 = 3
    COMMITTED_P1 = 4
    POSITION = 5
    STREET = 6
    ACTIONS_ROUND = 7
    MIN_RAISE = 8
    BET_TO_CALL = 9
    NUM_CONTEXT = 10
