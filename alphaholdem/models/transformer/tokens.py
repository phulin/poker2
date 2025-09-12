"""Token definitions for transformer-based poker models."""

from enum import Enum


class Special(Enum):
    """Special tokens for the transformer model."""

    CLS = 0
    SEP = 1
    MASK = 2
    PAD = 3
    NUM_SPECIAL = 4


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
