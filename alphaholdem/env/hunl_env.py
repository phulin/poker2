from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple
import random

from .types import Action, GameState, PlayerState
from . import rules

STREETS = ("preflop", "flop", "turn", "river", "showdown")


class HUNLEnv:
    def __init__(self, starting_stack: int = 20000, sb: int = 50, bb: int = 100, seed: int = 0):
        self.starting_stack = starting_stack
        self.sb = sb
        self.bb = bb
        self.rng = random.Random(seed)
        self.state: GameState | None = None

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        deck = rules.new_shuffled_deck(self.rng)
        p0 = PlayerState(stack=self.starting_stack)
        p1 = PlayerState(stack=self.starting_stack)
        # blinds
        button = self.rng.randrange(2)
        # deal hole cards
        p0.hole_cards = [deck.pop(), deck.pop()]
        p1.hole_cards = [deck.pop(), deck.pop()]
        # post blinds
        p_sb = 0 if button == 0 else 1
        p_bb = 1 - p_sb
        p_states = [p0, p1]
        p_states[p_sb].stack -= self.sb
        p_states[p_sb].committed += self.sb
        p_states[p_bb].stack -= self.bb
        p_states[p_bb].committed += self.bb
        pot = self.sb + self.bb
        to_act = p_sb  # SB acts first pre-flop in heads-up after posting
        min_raise = self.bb
        last_aggr = self.bb
        self.state = GameState(
            button=button,
            street="preflop",
            deck=deck,
            board=[],
            pot=pot,
            to_act=to_act,
            small_blind=self.sb,
            big_blind=self.bb,
            min_raise=min_raise,
            last_aggressive_amount=last_aggr,
            players=(p_states[0], p_states[1]),
            terminal=False,
        )
        # Attach env reference to state for legal action lookup
        self.state.env = self
        return self.state

    def legal_actions(self) -> List[Action]:
        s = self._require_state()
        if s.terminal:
            return []
        me = s.to_act
        opp = 1 - me
        me_p = s.players[me]
        opp_p = s.players[opp]
        to_call = opp_p.committed - me_p.committed
        actions: List[Action] = []
        if to_call <= 0:
            actions.append(Action("check"))
            # can bet if stacks allow
            if me_p.stack > 0 and opp_p.stack > 0:
                # minimum bet is big blind on preflop when unopened, otherwise min_raise
                min_bet = max(s.big_blind, s.min_raise) if s.street == "preflop" else max(1, s.min_raise)
                min_bet = min(min_bet, me_p.stack)
                if min_bet > 0:
                    actions.append(Action("bet", amount=min_bet))
                # all-in as an option
                if me_p.stack > 0:
                    actions.append(Action("allin", amount=me_p.stack))
        else:
            # can fold or call
            actions.append(Action("fold"))
            call_amt = min(to_call, me_p.stack)
            if call_amt > 0:
                actions.append(Action("call", amount=call_amt))
            # can raise if both players have chips
            if me_p.stack > call_amt and opp_p.stack > 0:
                min_raise = max(s.min_raise, s.last_aggressive_amount)
                raise_amt = call_amt + min_raise
                if raise_amt <= me_p.stack:
                    actions.append(Action("raise", amount=raise_amt))
                # all-in raise
                actions.append(Action("allin", amount=me_p.stack))
        return actions

    def step(self, action: Action) -> Tuple[GameState, int, bool, Dict[str, Any]]:
        s = self._require_state()
        if s.terminal:
            return s, 0, True, {}
        me = s.to_act
        opp = 1 - me
        me_p = replace(s.players[me])
        opp_p = replace(s.players[opp])
        board = list(s.board)
        deck = list(s.deck)
        pot = s.pot
        min_raise = s.min_raise
        last_aggr = s.last_aggressive_amount
        street = s.street

        to_call = opp_p.committed - me_p.committed

        if action.kind == "fold":
            me_p.has_folded = True
            pot += me_p.committed + opp_p.committed
            me_p.committed = 0
            opp_p.committed = 0
            winner = opp
            new_state = GameState(
                button=s.button,
                street=street,
                deck=deck,
                board=board,
                pot=pot,
                to_act=me,
                small_blind=s.small_blind,
                big_blind=s.big_blind,
                min_raise=min_raise,
                last_aggressive_amount=last_aggr,
                players=(me_p, opp_p) if me == 0 else (opp_p, me_p),
                terminal=True,
                winner=winner,
            )
            new_state.env = self
            self.state = new_state
            reward = self._terminal_rewards(new_state, me)
            return new_state, reward, True, {}

        if action.kind == "check":
            # move to next player / next street if round closed
            next_to_act = opp
            # if opponent also checked previously (to_call==0 and last action was check), we may need a round closure.
            # For simplicity, detect closure when both committed equal and no pending to_call after both act. We'll handle
            # closure at transition function after updating to_act.
        elif action.kind == "call":
            call_amt = min(to_call, me_p.stack)
            me_p.stack -= call_amt
            me_p.committed += call_amt
            pot += call_amt
            next_to_act = opp
        elif action.kind in ("bet", "raise", "allin"):
            # determine total to put in now (for unopened, treat as bet; else raise over to_call)
            target = action.amount
            put_in = target
            # commit call part first if needed
            if to_call > 0:
                call_amt = min(to_call, me_p.stack)
                me_p.stack -= call_amt
                me_p.committed += call_amt
                pot += call_amt
                put_in -= call_amt
            # now add raise/bet part
            put_in = max(0, min(put_in, me_p.stack))
            me_p.stack -= put_in
            me_p.committed += put_in
            pot += put_in
            # update min_raise and last aggressive amount
            last_aggr = max(last_aggr, put_in)
            min_raise = max(min_raise, last_aggr)
            if me_p.stack == 0:
                me_p.is_allin = True
            next_to_act = opp
        else:
            raise ValueError(f"Unknown action {action.kind}")

        # Update players back into tuple in seat order
        players = list(s.players)
        players[me] = me_p
        players[opp] = opp_p

        # Determine if betting round is closed
        round_closed = self._round_closed(players)

        # Advance street if closed
        if round_closed:
            # move committed to pot and reset committed
            pot += players[0].committed + players[1].committed
            players[0].committed = 0
            players[1].committed = 0
            # next street
            if street == "preflop":
                street = "flop"
                board.extend([deck.pop(), deck.pop(), deck.pop()])
                next_to_act = 1 - s.button  # postflop, button acts last
                min_raise = s.big_blind
                last_aggr = 0
            elif street == "flop":
                street = "turn"
                board.append(deck.pop())
                next_to_act = 1 - s.button
                min_raise = s.big_blind
                last_aggr = 0
            elif street == "turn":
                street = "river"
                board.append(deck.pop())
                next_to_act = 1 - s.button
                min_raise = s.big_blind
                last_aggr = 0
            elif street == "river":
                street = "showdown"
            else:
                pass

        # If showdown, evaluate or award all-in resolution if any player is all-in and remaining streets auto-dealt
        terminal = False
        winner = None
        if street == "showdown":
            # showdown: evaluate if both live; otherwise award to non-folded
            if not players[0].has_folded and not players[1].has_folded:
                # ensure board has 5
                while len(board) < 5:
                    board.append(deck.pop())
                cmp = rules.compare_7(players[0].hole_cards + board, players[1].hole_cards + board)
                if cmp > 0:
                    winner = 0
                elif cmp < 0:
                    winner = 1
                else:
                    winner = None  # split
            else:
                winner = 0 if players[1].has_folded else 1 if players[0].has_folded else None
            terminal = True

        new_state = GameState(
            button=s.button,
            street=street,
            deck=deck,
            board=board,
            pot=pot,
            to_act=next_to_act,
            small_blind=s.small_blind,
            big_blind=s.big_blind,
            min_raise=min_raise,
            last_aggressive_amount=last_aggr,
            players=(players[0], players[1]),
            terminal=terminal,
            winner=winner,
        )
        new_state.env = self
        self.state = new_state
        done = terminal
        reward = self._terminal_rewards(new_state, me) if done else 0
        return new_state, reward, done, {}

    def _round_closed(self, players: List[PlayerState]) -> bool:
        # Closed if both players have equal committed OR any player folded OR both all-in
        if players[0].has_folded or players[1].has_folded:
            return True
        if players[0].is_allin and players[1].is_allin:
            return True
        # Round is closed when both players have equal committed amounts
        # (including when both are 0 after checking)
        return players[0].committed == players[1].committed

    def _terminal_rewards(self, s: GameState, perspective: int) -> int:
        # Positive if perspective player wins chips
        # For simplicity, assign whole pot to winner or split evenly on tie
        if s.winner is None:
            return 0
        return s.pot if s.winner == perspective else -s.pot

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        return self.state
