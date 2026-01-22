"""Token definitions for transformer-based poker models."""

from enum import Enum

# Beginning token sequence indices
CLS_INDEX = 0
GAME_INDEX = 1
HOLE0_INDEX = 2
HOLE1_INDEX = 3


class Special(Enum):
    """Special tokens used in the variable-length transformer sequence."""

    CLS = 0
    GAME = 1
    CONTEXT = 2
    STREET_PREFLOP = 3
    STREET_FLOP = 4
    STREET_TURN = 5
    STREET_RIVER = 6
    NUM_SPECIAL = 7


class Context(Enum):
    """Context token indices for the transformer model."""

    POSITION = 0
    ACTIONS_ROUND = 1
    POT = 2
    STACK_P0 = 3
    STACK_P1 = 4
    COMMITTED_P0 = 5
    COMMITTED_P1 = 6
    MIN_RAISE = 7
    BET_TO_CALL = 8
    NUM_RAW_CONTEXT = 9
    # Above are stored as ints in SED/replay buffer; embedded as scaled floats.
    # Below are only in embeddings.
    EFFECTIVE_STACK_P0 = 9
    EFFECTIVE_STACK_P1 = 10
    SPR_P0 = 11
    SPR_P1 = 12
    NUM_CONTEXT = 13


class Game(Enum):
    """CLS token indices for the transformer model."""

    SB = 0
    BB = 1
    HERO_POSITION = 2
    NUM_RAW_GAME = 3
    # below are only in embeddings.
    SCALED_BB = 3
    SCALED_SB = 4
    NUM_GAME = 5


def get_special_token_id_offset() -> int:
    """value_offset where special tokens start."""

    return 0


def get_card_token_id_offset() -> int:
    """value_offset where card tokens start."""

    return get_special_token_id_offset() + Special.NUM_SPECIAL.value


def get_action_token_id_offset() -> int:
    """value_offset where action tokens start."""

    return get_card_token_id_offset() + 52
