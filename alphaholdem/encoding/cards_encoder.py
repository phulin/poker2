from __future__ import annotations

from typing import Any, List, Optional
import torch

from ..core.interfaces import Encoder
from ..core.registry import register_card_encoder
from ..env import rules
from ..core.structured_config import Config


def _cards_to_planes(
    cards: List[int], device: Optional[torch.device] = None
) -> torch.Tensor:
    planes = torch.zeros((4, 13), dtype=torch.float32, device=device)
    for c in cards:
        planes[rules.suit(c), rules.rank(c)] = 1.0
    return planes


@register_card_encoder("cards_planes_v1")
class CardsPlanesV1(Encoder):
    def __init__(self, config: Config | None = None):
        # Store config (not currently used but kept for consistency)
        self.cfg: Config | None = config

    def encode_cards(
        self, game_state: Any, seat: int, device: Optional[torch.device] = None
    ) -> Any:
        # Channels: hole, flop, turn, river, public, all
        hole = _cards_to_planes(game_state.players[seat].hole_cards, device=device)
        flop = (
            _cards_to_planes(game_state.board[:3], device=device)
            if len(game_state.board) >= 3
            else torch.zeros((4, 13), dtype=torch.float32, device=device)
        )
        turn = (
            _cards_to_planes(game_state.board[3:4], device=device)
            if len(game_state.board) >= 4
            else torch.zeros((4, 13), dtype=torch.float32, device=device)
        )
        river = (
            _cards_to_planes(game_state.board[4:5], device=device)
            if len(game_state.board) >= 5
            else torch.zeros((4, 13), dtype=torch.float32, device=device)
        )
        public = _cards_to_planes(game_state.board, device=device)
        all_cards = _cards_to_planes(
            game_state.players[seat].hole_cards + game_state.board,
            device=device,
        )
        return torch.stack([hole, flop, turn, river, public, all_cards], dim=0)

    def encode_actions(self, game_state: Any, seat: int, num_bet_bins: int) -> Any:
        raise NotImplementedError("Use actions encoder class for actions tensor")
