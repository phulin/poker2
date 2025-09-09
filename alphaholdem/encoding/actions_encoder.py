from __future__ import annotations

from typing import Any, List, Optional
import torch

from ..core.interfaces import Encoder
from ..core.registry import register_action_encoder
from ..env.types import GameState, Action
from ..encoding.action_mapping import _action_to_bin_idx
from ..core.structured_config import Config


@register_action_encoder("actions_hu_v1")
class ActionsHUEncoderV1(Encoder):
    def __init__(
        self, history_actions_per_round: int = 6, config: Config | None = None
    ):
        self.history_actions_per_round = history_actions_per_round
        # Store config for access to bet_bins
        self.cfg: Config | None = config

    def encode_cards(
        self, game_state: Any, seat: int, device: Optional[torch.device] = None
    ) -> Any:
        raise NotImplementedError("Use card encoder for cards tensor")

    def encode_actions(
        self,
        game_state: Any,
        seat: int,
        device: Optional[torch.device] = None,
    ) -> Any:
        # Derive num_bet_bins from config
        num_bet_bins = (
            len(self.cfg.env.bet_bins) + 3 if self.cfg else 8
        )  # Default to 8 if no config
        dtype = (
            torch.bfloat16
            if self.cfg
            and self.cfg.train.use_mixed_precision
            and device
            and device.type in ["cuda", "mps"]
            else torch.float32
        )

        rounds = ["preflop", "flop", "turn", "river"]
        channels: List[torch.Tensor] = []
        for _ in rounds:
            for _ in range(self.history_actions_per_round):
                channels.append(
                    torch.zeros((4, num_bet_bins), dtype=dtype, device=device)
                )

        # Populate historical planes per round: player-specific and sum
        if game_state.action_history:
            slots = self.history_actions_per_round
            for street_i, street in enumerate(rounds):
                events = [e for e in game_state.action_history if e[0] == street]
                events = events[-slots:]
                for idx, evt in enumerate(reversed(events)):
                    _, actor, kind, amount, _, _ = evt
                    bin_idx = _action_to_bin_idx(
                        Action(kind, amount), game_state, num_bet_bins
                    )
                    if bin_idx is None:
                        continue
                    ch = street_i * slots + (slots - 1 - idx)
                    mat = channels[ch]
                    mat[actor, bin_idx] = 1.0
                    mat[2, bin_idx] = 1.0  # sum plane
                    channels[ch] = mat

        # Fill current legal actions for the next action slot in current round
        legal = self._get_legal_mask(
            game_state, seat, num_bet_bins, dtype=dtype, device=device
        )
        round_idx = (
            max(0, rounds.index(game_state.street))
            if game_state.street in rounds
            else 0
        )
        ch_idx = round_idx * self.history_actions_per_round + (
            self.history_actions_per_round - 1
        )
        mat = channels[ch_idx]
        # Ensure legal mask lives on same device and dtype
        if legal.device != mat.device:
            legal = legal.to(mat.device)
        mat[3, :] = legal
        channels[ch_idx] = mat
        return torch.stack(channels, dim=0)

    def _get_legal_mask(
        self,
        game_state: GameState,
        seat: int,
        num_bet_bins: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate legal action mask for current state."""
        mask = torch.zeros(num_bet_bins, dtype=dtype, device=device)
        if hasattr(game_state, "env") and game_state.env is not None:
            # Prefer env-provided legal bins for consistency and speed
            legal_bins = game_state.env.legal_action_bins(num_bet_bins)
            if len(legal_bins) > 0:
                mask[legal_bins] = 1.0
            return mask
        # Fallback when no env is attached (e.g., unit tests)
        mask[1] = 1.0  # check/call
        mask[num_bet_bins - 1] = 1.0  # all-in
        return mask

    def _action_to_bin(
        self, action: Action, game_state: GameState, num_bet_bins: int
    ) -> int | None:
        """Map Action to discrete bin index using total_committed and config bet_bins."""
        if action.kind == "fold":
            return 0
        elif action.kind in ("check", "call"):
            return 1
        elif action.kind == "allin":
            return num_bet_bins - 1

        total_committed = (
            game_state.pot
            + game_state.players[0].committed
            + game_state.players[1].committed
        )
        if total_committed == 0:
            return 1

        if action.kind == "bet":
            fraction = action.amount / total_committed
        elif action.kind == "raise":
            me = game_state.to_act
            opp = 1 - me
            to_call = (
                game_state.players[opp].committed - game_state.players[me].committed
            )
            raise_part = max(0, action.amount - max(0, to_call))
            fraction = raise_part / total_committed if total_committed > 0 else 0.0
        else:
            return None

        bins = list(self.cfg.bet_bins)
        nearest = min(range(len(bins)), key=lambda i: abs(fraction - bins[i]))
        return 2 + nearest
