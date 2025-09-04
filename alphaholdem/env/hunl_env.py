from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple
import random

from line_profiler import profile

from .types import Action, GameState, PlayerState
from ..core.config_loader import get_config
from . import rules

STREETS = ("preflop", "flop", "turn", "river", "showdown")


class HUNLEnv:
    def __init__(
        self, starting_stack: int = 20000, sb: int = 50, bb: int = 100, seed: int = None
    ):
        self.starting_stack = starting_stack
        self.sb = sb
        self.bb = bb
        self.rng = random.Random(seed)
        self.state: GameState | None = None
        self.actions_this_round = 0  # Count actions in current betting round

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
        p_states[p_sb].stack_after_posting = p_states[p_sb].stack
        p_states[p_sb].committed += self.sb
        p_states[p_bb].stack -= self.bb
        p_states[p_bb].stack_after_posting = p_states[p_bb].stack
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
        self.actions_this_round = 0  # Reset action counter
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

        # Calculate total committed chips (pot + both players' committed amounts)
        total_committed = s.pot + me_p.committed + opp_p.committed

        # Always can check if no bet to call
        if to_call <= 0:
            actions.append(Action("check"))

        # Always can fold if there's a bet to call
        if to_call > 0:
            actions.append(Action("fold"))
            call_amt = min(to_call, me_p.stack)
            if call_amt > 0:
                actions.append(Action("call", amount=call_amt))

        # Can bet/raise if both players have chips
        can_bet_raise = me_p.stack > 0 and opp_p.stack > 0
        if can_bet_raise and total_committed > 0:
            # Calculate bet/raise sizes as fractions of total committed
            if to_call <= 0:
                # Betting - no call amount needed
                bet_sizes = [
                    int(total_committed * 0.5),  # 1/2 total committed
                    int(total_committed * 0.75),  # 3/4 total committed
                    total_committed,  # total committed
                    int(total_committed * 1.5),  # 1.5x total committed
                    int(total_committed * 2.0),  # 2x total committed
                ]
                for bet_size in bet_sizes:
                    if 0 < bet_size <= me_p.stack:
                        actions.append(Action("bet", amount=bet_size))
            else:
                # Raising - need to include call amount
                call_amt = min(to_call, me_p.stack)
                raise_sizes = [
                    call_amt + int(total_committed * 0.5),  # call + 1/2 total committed
                    call_amt
                    + int(total_committed * 0.75),  # call + 3/4 total committed
                    call_amt + total_committed,  # call + total committed
                    call_amt
                    + int(total_committed * 1.5),  # call + 1.5x total committed
                    call_amt + int(total_committed * 2.0),  # call + 2x total committed
                ]
                for raise_size in raise_sizes:
                    if raise_size > call_amt and raise_size <= me_p.stack:
                        actions.append(Action("raise", amount=raise_size))

        # Always can all-in if player has chips
        if me_p.stack > 0:
            actions.append(Action("allin", amount=me_p.stack))

        return actions

    def legal_action_bins(self, num_bet_bins: int) -> List[int]:
        """Return legal discrete action bin indices without constructing concrete amounts.

        Bins: 0=fold, 1=check/call, 2..(nb-2)=bet/raise presets, (nb-1)=all-in
        """
        s = self._require_state()
        if s.terminal:
            return []
        me = s.to_act
        opp = 1 - me
        me_p = s.players[me]
        opp_p = s.players[opp]
        to_call = opp_p.committed - me_p.committed

        bins: List[int] = []

        # Fold only when facing a bet
        if to_call > 0:
            bins.append(0)

        # Check or call
        if to_call <= 0:
            bins.append(1)  # check
        else:
            call_amt = min(to_call, me_p.stack)
            if call_amt > 0:
                bins.append(1)  # call

        # Bet/Raise options are available if both players have chips
        can_bet_raise = me_p.stack > 0 and opp_p.stack > 0
        if can_bet_raise:
            cfg = get_config(None)
            bet_mults = list(cfg.bet_bins)
            total_committed = s.pot + me_p.committed + opp_p.committed
            candidate_bins: List[int] = []
            realized_actions: set[tuple] = set()
            for i, mult in enumerate(bet_mults):
                bin_idx = 2 + i
                if bin_idx >= num_bet_bins - 1:
                    break
                # Compute target amount for this bin
                base = int(total_committed * mult) if total_committed > 0 else 0
                if to_call <= 0:
                    amount = base
                    kind = "bet"
                    # Legal if amount is between 1 and stack-1 (stack goes to all-in bin)
                    if amount <= 0 or amount >= me_p.stack:
                        continue
                else:
                    amount = base + to_call
                    kind = "raise"
                    # Must have chips beyond call
                    if me_p.stack <= to_call:
                        continue
                    # Must exceed call and satisfy min-raise unless only all-in is possible
                    min_raise_inc = max(1, s.min_raise)
                    if amount <= to_call:
                        continue
                    raise_inc = amount - to_call
                    if raise_inc < min_raise_inc:
                        # allow only if this bin equals all-in, otherwise skip
                        if amount < me_p.stack:
                            continue
                # Do not exceed stack (those map to all-in bin)
                if amount >= me_p.stack:
                    continue
                # Deduplicate bins mapping to identical concrete action
                key = (kind, amount)
                if key in realized_actions:
                    continue
                realized_actions.add(key)
                candidate_bins.append(bin_idx)
            bins.extend(candidate_bins)

            # All-in if chips remain for both us and opponent
            bins.append(num_bet_bins - 1)

        return bins

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
        street_at_action = street  # capture street before any potential advancement

        to_call = opp_p.committed - me_p.committed

        # Capture action context BEFORE applying effects
        total_committed_before = pot + me_p.committed + opp_p.committed
        action_amount = 0

        if action.kind == "fold":
            me_p.has_folded = True
            pot += me_p.committed + opp_p.committed
            me_p.committed = 0
            opp_p.committed = 0
            winner = opp
            # append to history
            history = list(s.action_history)
            history.append(
                (
                    street_at_action,
                    me,
                    "fold",
                    action_amount,
                    to_call,
                    total_committed_before,
                )
            )
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
                action_history=history,
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
            action_amount = call_amt
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
            action_amount = target
            # update min_raise and last aggressive amount
            # last_aggr should track the total amount bet/raised, not just the raise increment
            total_bet = (
                me_p.committed
            )  # This is the total amount committed by the current player
            last_aggr = max(last_aggr, total_bet)
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

        # Increment action counter BEFORE checking if round is closed
        self.actions_this_round += 1

        # Determine if betting round is closed
        round_closed = self._round_closed(players)

        # Advance street if closed
        if round_closed:
            # Note: committed chips are already in the pot from the action step
            # We just need to reset committed amounts for the new street
            players[0].committed = 0
            players[1].committed = 0
            # next street
            if street == "preflop":
                street = "flop"
                board.extend([deck.pop(), deck.pop(), deck.pop()])
                next_to_act = 1 - s.button  # postflop, button acts last
                min_raise = s.big_blind
                last_aggr = 0
                self.actions_this_round = 0  # Reset for new street
            elif street == "flop":
                street = "turn"
                board.append(deck.pop())
                next_to_act = 1 - s.button
                min_raise = s.big_blind
                last_aggr = 0
                self.actions_this_round = 0  # Reset for new street
            elif street == "turn":
                street = "river"
                board.append(deck.pop())
                next_to_act = 1 - s.button
                min_raise = s.big_blind
                last_aggr = 0
                self.actions_this_round = 0  # Reset for new street
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
                cmp = rules.compare_7(
                    players[0].hole_cards + board, players[1].hole_cards + board
                )
                if cmp > 0:
                    winner = 0
                elif cmp < 0:
                    winner = 1
                else:
                    winner = None  # split
            else:
                winner = (
                    0 if players[1].has_folded else 1 if players[0].has_folded else None
                )
            terminal = True

        # Append action to history (after computing updates above)
        history = list(s.action_history)
        action_kind = action.kind
        history.append(
            (
                street_at_action,
                me,
                action_kind,
                int(action_amount),
                int(to_call),
                int(total_committed_before),
            )
        )

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
            action_history=history,
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
        # AND at least 2 actions have been taken (both players have acted)
        return (
            players[0].committed == players[1].committed
            and self.actions_this_round >= 2
        )

    def _terminal_rewards(self, s: GameState, perspective: int) -> int:
        # Positive if perspective player wins chips
        # For simplicity, assign whole pot to winner or split evenly on tie
        pot_share = (
            s.pot if s.winner == perspective else s.pot / 2 if s.winner is None else 0
        )
        raw_reward = s.players[perspective].stack + pot_share - self.starting_stack
        # Scale rewards by 100 big blinds to stabilize learning
        scale = float(self.bb) * 100.0
        return float(raw_reward) / scale

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("Environment not reset")
        return self.state
