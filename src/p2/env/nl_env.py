from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from p2.env import rules
from p2.env.hunl_tensor_env import DEFAULT_BET_BINS
from p2.env.types import Action

STREETS = ("preflop", "flop", "turn", "river", "showdown")


@dataclass
class NLPlayerState:
    stack: int
    stack_after_posting: int = 0
    committed: int = 0
    hole_cards: list[int] = field(default_factory=list)
    has_folded: bool = False
    is_allin: bool = False
    acted_this_round: bool = False


@dataclass
class NLGameState:
    button: int
    street: str
    deck: list[int]
    board: list[int] = field(default_factory=list)
    pot: int = 0
    to_act: int = 0
    small_blind: int = 50
    big_blind: int = 100
    min_raise: int = 0
    last_aggressive_amount: int = 0
    players: tuple[NLPlayerState, ...] = field(default_factory=tuple)
    terminal: bool = False
    winner: int | None = None
    winners: tuple[int, ...] = field(default_factory=tuple)
    action_history: list[tuple[str, int, str, int, int, int]] = field(
        default_factory=list
    )


class NLEnv:
    """Readable multiway no-limit Hold 'Em reference environment.

    This class prioritizes explicit scalar semantics for tests and debugging.
    It can deal private cards and resolve showdowns, but it can also run with
    ``deal_hole_cards=False`` to mirror public-belief-state betting mechanics.
    """

    def __init__(
        self,
        starting_stack: int = 20000,
        num_players: int = 6,
        sb: int = 50,
        bb: int = 100,
        default_bet_bins: list[float] | None = None,
        seed: int | None = None,
        deal_hole_cards: bool = True,
    ) -> None:
        if num_players < 2:
            raise ValueError("num_players must be at least 2")
        if starting_stack < bb:
            raise ValueError("starting_stack must cover the big blind")
        self.starting_stack = int(starting_stack)
        self.num_players = int(num_players)
        self.sb = int(sb)
        self.bb = int(bb)
        self.default_bet_bins = default_bet_bins or DEFAULT_BET_BINS
        self.num_bet_bins = len(self.default_bet_bins) + 3
        self.deal_hole_cards = bool(deal_hole_cards)
        self.rng = random.Random(seed)
        self.state: NLGameState | None = None

    def reset(
        self,
        seed: int | None = None,
        force_button: int | None = None,
        force_deck: list[int] | None = None,
    ) -> NLGameState:
        if seed is not None:
            self.rng.seed(seed)

        deck = rules.new_shuffled_deck(self.rng)
        if force_deck is not None:
            forced = list(force_deck)
            remaining = [card for card in deck if card not in set(forced)]
            deck = forced + remaining

        button = self.rng.randrange(self.num_players) if force_button is None else force_button
        players = [NLPlayerState(stack=self.starting_stack) for _ in range(self.num_players)]

        if self.deal_hole_cards:
            for player in players:
                player.hole_cards = [deck.pop(0), deck.pop(0)]

        sb_player = self._small_blind_player(button)
        bb_player = self._big_blind_player(button)
        for player_idx, blind in ((sb_player, self.sb), (bb_player, self.bb)):
            posted = min(blind, players[player_idx].stack)
            players[player_idx].stack -= posted
            players[player_idx].stack_after_posting = players[player_idx].stack
            players[player_idx].committed += posted
            players[player_idx].is_allin = players[player_idx].stack == 0

        to_act = self._preflop_first_actor(button)
        state = NLGameState(
            button=button,
            street="preflop",
            deck=deck,
            board=[],
            pot=self.sb + self.bb,
            to_act=to_act,
            small_blind=self.sb,
            big_blind=self.bb,
            min_raise=self.bb,
            last_aggressive_amount=self.bb,
            players=tuple(players),
            terminal=False,
        )
        state.env = self
        self.state = state
        return state

    def legal_actions(self) -> list[Action]:
        amounts, mask = self.legal_bins_amounts_and_mask()
        actions: list[Action] = []
        for bin_idx, legal in enumerate(mask):
            if not legal:
                continue
            if bin_idx == 0:
                actions.append(Action("fold"))
            elif bin_idx == 1:
                to_call = self._to_call(self._require_state().to_act)
                actions.append(Action("call" if to_call > 0 else "check", min(to_call, self._actor().stack)))
            elif bin_idx == self.num_bet_bins - 1:
                actions.append(Action("allin", amounts[bin_idx]))
            else:
                to_call = self._to_call(self._require_state().to_act)
                actions.append(Action("raise" if to_call > 0 else "bet", amounts[bin_idx]))
        return actions

    def legal_action_bins(self, num_bet_bins: int | None = None) -> list[int]:
        if num_bet_bins is not None and num_bet_bins != self.num_bet_bins:
            raise ValueError("NLEnv uses the bet-bin count from default_bet_bins")
        _, mask = self.legal_bins_amounts_and_mask()
        return [idx for idx, legal in enumerate(mask) if legal]

    def legal_bins_amounts_and_mask(self) -> tuple[list[int], list[bool]]:
        state = self._require_state()
        amounts = [-1] * self.num_bet_bins
        mask = [False] * self.num_bet_bins
        if state.terminal:
            return amounts, mask

        actor = state.to_act
        player = state.players[actor]
        if player.has_folded or player.is_allin or player.stack <= 0:
            return amounts, mask

        to_call = self._to_call(actor)
        mask[0] = to_call > 0
        amounts[1] = min(to_call, player.stack)
        mask[1] = True

        can_raise = player.stack > to_call and self._players_who_can_respond(actor) > 0
        for offset, mult in enumerate(self.default_bet_bins):
            bin_idx = 2 + offset
            amount = to_call + int(state.pot * mult)
            amounts[bin_idx] = amount
            raise_increment = amount - to_call
            mask[bin_idx] = (
                can_raise
                and amount < player.stack
                and raise_increment >= state.min_raise
            )

        amounts[-1] = player.stack
        mask[-1] = player.stack > 0
        return amounts, mask

    def step_bins(self, bin_idx: int) -> tuple[NLGameState, list[float], bool, dict[str, Any]]:
        amounts, mask = self.legal_bins_amounts_and_mask()
        if bin_idx < 0:
            return self._require_state(), [0.0] * self.num_players, False, {}
        if bin_idx >= self.num_bet_bins or not mask[bin_idx]:
            raise ValueError(f"Illegal action bin {bin_idx}")
        if bin_idx == 0:
            action = Action("fold")
        elif bin_idx == 1:
            action = Action("call" if amounts[1] > 0 else "check", amounts[1])
        elif bin_idx == self.num_bet_bins - 1:
            action = Action("allin", amounts[-1])
        else:
            action = Action("raise" if self._to_call(self._require_state().to_act) > 0 else "bet", amounts[bin_idx])
        return self.step(action)

    def step(self, action: Action) -> tuple[NLGameState, list[float], bool, dict[str, Any]]:
        state = self._require_state()
        if state.terminal:
            return state, [0.0] * self.num_players, True, {}

        actor = state.to_act
        players = [self._copy_player(player) for player in state.players]
        player = players[actor]
        to_call = max(p.committed for p in players) - player.committed
        total_committed_before = state.pot
        action_amount = 0
        aggressive = False

        if action.kind == "fold":
            player.has_folded = True
        elif action.kind in ("check", "call"):
            action_amount = min(to_call, player.stack)
            self._put_chips(player, action_amount)
        elif action.kind in ("bet", "raise", "allin"):
            action_amount = min(max(action.amount, 0), player.stack)
            old_max = max(p.committed for p in players)
            self._put_chips(player, action_amount)
            raise_increment = player.committed - old_max
            aggressive = raise_increment > 0
            if aggressive:
                state.min_raise = max(state.min_raise, raise_increment)
                state.last_aggressive_amount = player.committed
        else:
            raise ValueError(f"Unknown action {action.kind}")

        players[actor] = player
        pot = state.pot + action_amount
        if aggressive:
            for p in players:
                p.acted_this_round = False
        player.acted_this_round = True

        history = list(state.action_history)
        history.append(
            (state.street, actor, action.kind, action_amount, to_call, total_committed_before)
        )

        terminal, winners = self._terminal_status(players, state.street)
        next_street = state.street
        board = list(state.board)
        deck = list(state.deck)
        to_act = actor
        min_raise = state.min_raise
        last_aggressive_amount = state.last_aggressive_amount

        if terminal:
            to_act = actor
        elif self._no_more_betting(players):
            board = self._deal_to_full_board(deck, board)
            next_street = "showdown"
            terminal = True
            winners = self._showdown_winners(players, board)
        elif self._round_closed(players):
            for p in players:
                p.committed = 0
                p.acted_this_round = False
            min_raise = self.bb
            last_aggressive_amount = 0
            if state.street == "river":
                next_street = "showdown"
                terminal = True
                winners = self._showdown_winners(players, board)
            else:
                next_street = STREETS[STREETS.index(state.street) + 1]
                board = self._deal_next_street(deck, board, next_street)
                to_act = self._first_postflop_actor(players, state.button)
        else:
            to_act = self._next_actor(players, actor)

        winner = winners[0] if len(winners) == 1 else None
        new_state = NLGameState(
            button=state.button,
            street=next_street,
            deck=deck,
            board=board,
            pot=pot,
            to_act=to_act,
            small_blind=self.sb,
            big_blind=self.bb,
            min_raise=min_raise,
            last_aggressive_amount=last_aggressive_amount,
            players=tuple(players),
            terminal=terminal,
            winner=winner,
            winners=tuple(winners),
            action_history=history,
        )
        new_state.env = self
        self.state = new_state
        rewards = self._rewards(new_state) if terminal else [0.0] * self.num_players
        return new_state, rewards, terminal, {}

    def _small_blind_player(self, button: int) -> int:
        return button if self.num_players == 2 else (button + 1) % self.num_players

    def _big_blind_player(self, button: int) -> int:
        return (button + 1) % self.num_players if self.num_players == 2 else (button + 2) % self.num_players

    def _preflop_first_actor(self, button: int) -> int:
        return self._small_blind_player(button) if self.num_players == 2 else (self._big_blind_player(button) + 1) % self.num_players

    def _first_postflop_actor(self, players: list[NLPlayerState], button: int) -> int:
        return self._next_actor(players, button)

    def _next_actor(self, players: list[NLPlayerState], actor: int) -> int:
        for offset in range(1, self.num_players + 1):
            idx = (actor + offset) % self.num_players
            if not players[idx].has_folded and not players[idx].is_allin and players[idx].stack > 0:
                return idx
        return actor

    def _round_closed(self, players: list[NLPlayerState]) -> bool:
        eligible = [p for p in players if not p.has_folded and not p.is_allin and p.stack > 0]
        if not eligible:
            return False
        max_committed = max(p.committed for p in players)
        return all(p.acted_this_round and p.committed == max_committed for p in eligible)

    def _no_more_betting(self, players: list[NLPlayerState]) -> bool:
        live = [p for p in players if not p.has_folded]
        eligible = [p for p in live if not p.is_allin and p.stack > 0]
        return len(live) > 1 and len(eligible) == 0

    def _terminal_status(
        self, players: list[NLPlayerState], street: str
    ) -> tuple[bool, tuple[int, ...]]:
        live = tuple(idx for idx, player in enumerate(players) if not player.has_folded)
        if len(live) <= 1:
            return True, live
        if street == "showdown":
            return True, live
        return False, ()

    def _showdown_winners(
        self, players: list[NLPlayerState], board: list[int]
    ) -> tuple[int, ...]:
        live = [idx for idx, player in enumerate(players) if not player.has_folded]
        if not self.deal_hole_cards or len(board) < 5:
            return tuple(live)

        best = live[0]
        winners = [best]
        for idx in live[1:]:
            cmp = rules.compare_7(players[idx].hole_cards + board, players[best].hole_cards + board)
            if cmp > 0:
                best = idx
                winners = [idx]
            elif cmp == 0:
                winners.append(idx)
        return tuple(winners)

    def _deal_next_street(
        self, deck: list[int], board: list[int], next_street: str
    ) -> list[int]:
        if next_street == "flop":
            board.extend([deck.pop(0), deck.pop(0), deck.pop(0)])
        elif next_street in {"turn", "river"}:
            board.append(deck.pop(0))
        return board

    def _deal_to_full_board(self, deck: list[int], board: list[int]) -> list[int]:
        while len(board) < 5:
            board.append(deck.pop(0))
        return board

    def _to_call(self, actor: int) -> int:
        state = self._require_state()
        return max(player.committed for player in state.players) - state.players[actor].committed

    def _players_who_can_respond(self, actor: int) -> int:
        state = self._require_state()
        return sum(
            not player.has_folded and not player.is_allin and player.stack > 0
            for idx, player in enumerate(state.players)
            if idx != actor
        )

    def _actor(self) -> NLPlayerState:
        state = self._require_state()
        return state.players[state.to_act]

    def _put_chips(self, player: NLPlayerState, amount: int) -> None:
        amount = min(max(amount, 0), player.stack)
        player.stack -= amount
        player.committed += amount
        player.is_allin = player.stack == 0

    def _rewards(self, state: NLGameState) -> list[float]:
        rewards = []
        split_count = max(len(state.winners), 1)
        for idx, player in enumerate(state.players):
            share = state.pot / split_count if idx in state.winners else 0.0
            rewards.append((player.stack + share - self.starting_stack) / self.starting_stack)
        return rewards

    def _copy_player(self, player: NLPlayerState) -> NLPlayerState:
        return NLPlayerState(
            stack=player.stack,
            stack_after_posting=player.stack_after_posting,
            committed=player.committed,
            hole_cards=list(player.hole_cards),
            has_folded=player.has_folded,
            is_allin=player.is_allin,
            acted_this_round=player.acted_this_round,
        )

    def _require_state(self) -> NLGameState:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        return self.state
