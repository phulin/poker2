from __future__ import annotations

from typing import Any, List
import torch

from ..core.interfaces import Encoder
from ..core.registry import register_card_encoder
from ..env import rules


def _cards_to_planes(cards: List[int]) -> torch.Tensor:
    planes = torch.zeros((4, 13), dtype=torch.float32)
    for c in cards:
        planes[rules.suit(c), rules.rank(c)] = 1.0
    return planes


@register_card_encoder("cards_planes_v1")
class CardsPlanesV1(Encoder):
    def encode_cards(self, game_state: Any, seat: int) -> Any:
        # Channels: hole, flop, turn, river, public, all
        hole = _cards_to_planes(game_state.players[seat].hole_cards)
        flop = _cards_to_planes(game_state.board[:3]) if len(game_state.board) >= 3 else torch.zeros((4, 13), dtype=torch.float32)
        turn = _cards_to_planes(game_state.board[3:4]) if len(game_state.board) >= 4 else torch.zeros((4, 13), dtype=torch.float32)
        river = _cards_to_planes(game_state.board[4:5]) if len(game_state.board) >= 5 else torch.zeros((4, 13), dtype=torch.float32)
        public = _cards_to_planes(game_state.board)
        all_cards = _cards_to_planes(game_state.players[seat].hole_cards + game_state.board)
        return torch.stack([hole, flop, turn, river, public, all_cards], dim=0)

    def encode_actions(self, game_state: Any, seat: int, num_bet_bins: int) -> Any:
        raise NotImplementedError("Use actions encoder class for actions tensor")
