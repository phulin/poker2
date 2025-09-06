from __future__ import annotations

from typing import Optional, Tuple

import torch

from . import rules


class HUNLTensorEnv:
    """Tensorized, batched HUNL environment (first pass).

    Maintains key game scalars/vectors as tensors of shape [N] or [N, 2]:
      - stacks, committed, has_folded, is_allin: [N, 2]
      - pot, to_act, street, actions_this_round, min_raise, last_aggr: [N]
      - board: [N, 5] (card ints, -1 for empty)
      - done mask: [N] bool

    Non-tensor state kept per-env where variable-length is needed:
      - deck: list of lists of ints (shuffled), one per env

    This initial version supports discrete bin stepping compatible with bet_bins.
    """

    # Type signatures for fields
    N: int
    starting_stack: int
    sb: int
    bb: int
    device: torch.device
    bet_bins: list[float]
    rng: torch.Generator
    # deck tensor and per-env draw position
    deck: torch.Tensor
    deck_pos: torch.Tensor
    button: torch.Tensor
    street: torch.Tensor
    to_act: torch.Tensor
    pot: torch.Tensor
    min_raise: torch.Tensor
    last_aggr: torch.Tensor
    actions_this_round: torch.Tensor
    stacks: torch.Tensor
    committed: torch.Tensor
    has_folded: torch.Tensor
    is_allin: torch.Tensor
    board_onehot: torch.Tensor
    done: torch.Tensor
    winner: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        starting_stack: int,
        sb: int,
        bb: int,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        bet_bins: Optional[list[float]] = None,
    ) -> None:
        assert num_envs > 0
        self.N = num_envs
        self.starting_stack = int(starting_stack)
        self.sb = int(sb)
        self.bb = int(bb)
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.bet_bins = bet_bins or [0.5, 0.75, 1.0, 1.5, 2.0]

        self.rng = torch.Generator(device="cpu")
        if seed is not None:
            self.rng.manual_seed(int(seed))

        # Per-env decks as tensor and draw positions
        self.deck = torch.empty(self.N, 52, dtype=torch.long)
        self.deck_pos = torch.zeros(self.N, dtype=torch.long)

        # Tensors (initialized in reset)
        self.button = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.street = torch.zeros(self.N, dtype=torch.long, device=self.device)  # 0..4
        self.to_act = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.pot = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.min_raise = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.last_aggr = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.actions_this_round = torch.zeros(
            self.N, dtype=torch.long, device=self.device
        )

        self.stacks = torch.zeros(self.N, 2, dtype=torch.long, device=self.device)
        self.committed = torch.zeros(self.N, 2, dtype=torch.long, device=self.device)
        self.has_folded = torch.zeros(self.N, 2, dtype=torch.bool, device=self.device)
        self.is_allin = torch.zeros(self.N, 2, dtype=torch.bool, device=self.device)

        # Board and hole cards stored as one-hot [4,13]
        # board_onehot: [N, 5, 4, 13], hole_onehot: [N, 2 players, 2 cards, 4, 13]
        self.board_onehot = torch.zeros(
            (self.N, 5, 4, 13), dtype=torch.float32, device=self.device
        )
        self.done = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self.winner = torch.full(
            (self.N,), -1, dtype=torch.long, device=self.device
        )  # -1 split
        # Hole cards one-hot
        self.hole_onehot = torch.zeros(
            (self.N, 2, 2, 4, 13), dtype=torch.float32, device=self.device
        )

    # --- Reset -----------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.manual_seed(int(seed))

        # New shuffled decks and deal
        # Vectorized reset for all environments

        # Shuffle decks for all envs and reset draw positions
        self.deck = torch.stack(
            [torch.randperm(52, generator=self.rng) for _ in range(self.N)], dim=0
        )
        self.deck_pos[:] = 0

        # Randomize button for all envs
        button = torch.randint(0, 2, (self.N,), generator=self.rng)
        p_sb = (button == 0).long() * 0 + (
            button == 1
        ).long() * 1  # 0 if button==0 else 1
        p_bb = 1 - p_sb

        # Stacks and committed
        stacks = torch.full((self.N, 2), self.starting_stack, dtype=torch.long)
        committed = torch.zeros((self.N, 2), dtype=torch.long)

        # Deal hole cards: for each env, assign 4 cards in deck order
        # [N, 4] indices into deck for each env
        card_indices = torch.arange(4).unsqueeze(0).expand(self.N, 4)
        cards = self.deck[torch.arange(self.N).unsqueeze(1), card_indices]  # [N, 4]
        self.deck_pos[:] = 4

        # Assign hole cards
        _c0_1 = cards[:, 0]
        _c0_2 = cards[:, 1]
        _c1_1 = cards[:, 2]
        _c1_2 = cards[:, 3]

        # Post blinds (vectorized)
        # For each env, subtract sb/bb from correct player
        # p_sb and p_bb are [N] with values 0 or 1
        stacks[torch.arange(self.N), p_sb] -= self.sb
        committed[torch.arange(self.N), p_sb] += self.sb
        stacks[torch.arange(self.N), p_bb] -= self.bb
        committed[torch.arange(self.N), p_bb] += self.bb

        # Write per-env scalars
        self.button[:] = button
        self.street[:] = 0  # preflop
        self.to_act[:] = p_sb
        self.pot[:] = self.sb + self.bb
        self.min_raise[:] = self.bb
        self.last_aggr[:] = self.bb
        self.actions_this_round[:] = 0
        self.stacks[:, 0] = stacks[:, 0]
        self.stacks[:, 1] = stacks[:, 1]
        self.committed[:, 0] = committed[:, 0]
        self.committed[:, 1] = committed[:, 1]
        self.has_folded[:, :] = False
        self.is_allin[:, :] = False
        self.board_onehot[:, :, :, :] = 0.0
        self.done[:] = False
        self.winner[:] = -1

        # Zero out hole_onehot
        self.hole_onehot[:, :, :, :, :] = 0.0

        # Set hole_onehot for all envs
        for idx, (c0_1, c0_2, c1_1, c1_2) in enumerate(
            zip(_c0_1.tolist(), _c0_2.tolist(), _c1_1.tolist(), _c1_2.tolist())
        ):
            self.hole_onehot[idx, 0, 0, rules.suit(c0_1), rules.rank(c0_1)] = 1.0
            self.hole_onehot[idx, 0, 1, rules.suit(c0_2), rules.rank(c0_2)] = 1.0
            self.hole_onehot[idx, 1, 0, rules.suit(c1_1), rules.rank(c1_1)] = 1.0
            self.hole_onehot[idx, 1, 1, rules.suit(c1_2), rules.rank(c1_2)] = 1.0
        # Note: Hole cards are implicit in deck/history; full tensorization of hole cards
        # can be added later (e.g., hole_cards tensor [N, 2, 2]).

    # --- Legality ---------------------------------------------------------------

    def legal_action_bins_mask(self, num_bet_bins: int) -> torch.Tensor:
        """Return [N, num_bet_bins] mask of legal bins.

        Follows the single-env improved logic approximately.
        """
        N = self.N
        mask = torch.zeros(N, num_bet_bins, dtype=torch.bool, device=self.device)

        me = self.to_act.clone()
        opp = 1 - me
        # Gather per-env views
        me_stack = self.stacks.gather(1, me.view(N, 1)).squeeze(1)
        opp_stack = self.stacks.gather(1, opp.view(N, 1)).squeeze(1)
        me_comm = self.committed.gather(1, me.view(N, 1)).squeeze(1)
        opp_comm = self.committed.gather(1, opp.view(N, 1)).squeeze(1)
        to_call = opp_comm - me_comm

        total_committed = self.pot + me_comm + opp_comm

        # Fold if facing bet
        mask[:, 0] = to_call > 0
        # Check/Call
        can_check = to_call <= 0
        can_call = (to_call > 0) & (me_stack > 0)
        mask[:, 1] = can_check | can_call

        # Bet/Raise bins 2..nb-2
        can_bet_raise = (me_stack > 0) & (opp_stack > 0)
        # Candidate base sizes per bin using total_committed * mult
        for i, mult in enumerate(self.bet_bins):
            b = 2 + i
            if b >= num_bet_bins - 1:
                break
            base = (total_committed.to(torch.float32) * float(mult)).to(torch.long)
            if_mask = can_bet_raise.clone()
            if_mask &= total_committed > 0
            # Bet case
            bet_ok = (to_call <= 0) & (base >= self.bb) & (base < me_stack)
            # Raise case: include call amount, must exceed to_call
            raise_amt = base + to_call
            raise_inc = base
            meets_min = raise_inc >= self.min_raise
            can_only_allin = me_stack <= (to_call + self.min_raise)
            raise_ok = (
                (to_call > 0)
                & (me_stack > to_call)
                & (raise_amt > to_call)
                & (meets_min | can_only_allin)
            )
            mask[:, b] = if_mask & (bet_ok | raise_ok)

        # All-in
        mask[:, num_bet_bins - 1] = me_stack > 0

        return mask

    # --- Step ------------------------------------------------------------------

    def step_bins(
        self, bin_indices: torch.Tensor, num_bet_bins: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step all envs using discrete bin indices tensor [N].

        Returns (rewards [N], dones [N], to_act [N], info dict)
        Rewards are scaled by 100bb consistent with HUNLEnv.
        """
        assert bin_indices.shape[0] == self.N
        N = self.N
        device = self.device

        me = self.to_act
        opp = 1 - me

        me_stack = self.stacks.gather(1, me.view(N, 1)).squeeze(1)
        opp_stack = self.stacks.gather(1, opp.view(N, 1)).squeeze(1)
        me_comm = self.committed.gather(1, me.view(N, 1)).squeeze(1)
        opp_comm = self.committed.gather(1, opp.view(N, 1)).squeeze(1)
        to_call = opp_comm - me_comm

        rewards = torch.zeros(N, dtype=torch.float32, device=device)

        # Group masks by action type
        is_fold = bin_indices == 0
        is_check_call = bin_indices == 1
        is_allin = bin_indices == (num_bet_bins - 1)
        is_bet_raise = ~(is_fold | is_check_call | is_allin)

        # Fold: immediate terminal, award pot to opp
        if is_fold.any():
            idx = is_fold.nonzero(as_tuple=False).squeeze(1)
            # Winner is opp
            self.winner[idx] = opp[idx]
            self.done[idx] = True
            # Reward equals current stack - starting_stack (committed lost to pot/opponent)
            scale = float(self.bb) * 100.0
            rewards[idx] = (
                self.stacks[idx, me[idx]].to(torch.float32) - float(self.starting_stack)
            ) / scale

        # Check/Call
        if is_check_call.any():
            idx = is_check_call.nonzero(as_tuple=False).squeeze(1)
            call_amt = torch.minimum(to_call[idx], me_stack[idx])
            self.stacks[idx, me[idx]] -= call_amt
            self.committed[idx, me[idx]] += call_amt
            self.pot[idx] += call_amt

        # All-in
        if is_allin.any():
            idx = is_allin.nonzero(as_tuple=False).squeeze(1)
            amt = me_stack[idx]
            # Pay call first if needed
            call_amt = torch.clamp(to_call[idx], min=0, max=amt)
            self.stacks[idx, me[idx]] -= call_amt
            self.committed[idx, me[idx]] += call_amt
            self.pot[idx] += call_amt
            amt = amt - call_amt
            # Then shove remaining
            self.stacks[idx, me[idx]] -= amt
            self.committed[idx, me[idx]] += amt
            self.pot[idx] += amt
            self.is_allin[idx, me[idx]] = True

        # Bet/Raise bins: approximate mapping using total_committed * mult
        if is_bet_raise.any():
            idx = is_bet_raise.nonzero(as_tuple=False).squeeze(1)
            bin_local = bin_indices[idx]
            mult_idx = (bin_local - 2).clamp(min=0, max=len(self.bet_bins) - 1)
            mult = torch.tensor(
                [self.bet_bins[j] for j in mult_idx.tolist()],
                dtype=torch.float32,
                device=device,
            )
            total_committed = (self.pot[idx] + me_comm[idx] + opp_comm[idx]).to(
                torch.float32
            )
            base = (total_committed * mult).to(torch.long)
            # Separate bet vs raise by to_call
            idx_bet = idx[to_call[idx] <= 0]
            if idx_bet.numel() > 0:
                amt = base[to_call[idx] <= 0]
                amt = amt.clamp(min=self.bb)
                amt = torch.minimum(amt, self.stacks[idx_bet, me[idx_bet]])
                self.stacks[idx_bet, me[idx_bet]] -= amt
                self.committed[idx_bet, me[idx_bet]] += amt
                self.pot[idx_bet] += amt
                self.last_aggr[idx_bet] = amt
                self.min_raise[idx_bet] = torch.maximum(self.min_raise[idx_bet], amt)
            idx_raise = idx[to_call[idx] > 0]
            if idx_raise.numel() > 0:
                base_r = base[to_call[idx] > 0]
                call_amt = torch.minimum(
                    to_call[idx_raise], self.stacks[idx_raise, me[idx_raise]]
                )
                # Pay call
                self.stacks[idx_raise, me[idx_raise]] -= call_amt
                self.committed[idx_raise, me[idx_raise]] += call_amt
                self.pot[idx_raise] += call_amt
                # Raise part
                avail = self.stacks[idx_raise, me[idx_raise]]
                min_req = self.min_raise[idx_raise]
                raise_part = torch.minimum(base_r, avail)
                need = (raise_part < min_req) & (avail > min_req)
                raise_part = torch.where(need, min_req, raise_part)
                self.stacks[idx_raise, me[idx_raise]] -= raise_part
                self.committed[idx_raise, me[idx_raise]] += raise_part
                self.pot[idx_raise] += raise_part
                self.last_aggr[idx_raise] = raise_part
                self.min_raise[idx_raise] = torch.maximum(
                    self.min_raise[idx_raise], raise_part
                )

        # Advance to next actor (simple alternation)
        self.to_act = 1 - self.to_act
        self.actions_this_round += (~self.done).to(torch.long)

        # Round closure: equal committed and both acted
        equal_committed = self.committed[:, 0] == self.committed[:, 1]
        round_closed = equal_committed & (self.actions_this_round >= 2)
        rc_idx = round_closed.nonzero(as_tuple=False).squeeze(1)
        if rc_idx.numel() > 0:
            # Reset committed
            self.pot[rc_idx] += 0  # already in pot
            self.committed[rc_idx, :] = 0
            self.actions_this_round[rc_idx] = 0
            # Advance street and deal
            s = self.street[rc_idx]
            # flop
            flop_mask = s == 0
            if flop_mask.any():
                ids = rc_idx[flop_mask]
                for i in ids.tolist():
                    c0 = int(self.deck[i, self.deck_pos[i]].item())
                    self.deck_pos[i] += 1
                    c1 = int(self.deck[i, self.deck_pos[i]].item())
                    self.deck_pos[i] += 1
                    c2 = int(self.deck[i, self.deck_pos[i]].item())
                    self.deck_pos[i] += 1
                    self.board_onehot[i, 0, rules.suit(c0), rules.rank(c0)] = 1.0
                    self.board_onehot[i, 1, rules.suit(c1), rules.rank(c1)] = 1.0
                    self.board_onehot[i, 2, rules.suit(c2), rules.rank(c2)] = 1.0
                self.street[ids] = 1
                self.to_act[ids] = 1 - self.button[ids]
            # turn
            turn_mask = s == 1
            if turn_mask.any():
                ids = rc_idx[turn_mask]
                for i in ids.tolist():
                    c = int(self.deck[i, self.deck_pos[i]].item())
                    self.deck_pos[i] += 1
                    self.board_onehot[i, 3, rules.suit(c), rules.rank(c)] = 1.0
                self.street[ids] = 2
                self.to_act[ids] = 1 - self.button[ids]
            # river
            river_mask = s == 2
            if river_mask.any():
                ids = rc_idx[river_mask]
                for i in ids.tolist():
                    c = int(self.deck[i, self.deck_pos[i]].item())
                    self.deck_pos[i] += 1
                    self.board_onehot[i, 4, rules.suit(c), rules.rank(c)] = 1.0
                self.street[ids] = 3
                self.to_act[ids] = 1 - self.button[ids]
            # showdown
            sd_mask = s == 3
            if sd_mask.any():
                ids = rc_idx[sd_mask]
                # Evaluate winners or award folds
                # Vectorized showdown resolution for all ids at once

                # Only consider ids where neither player has folded
                not_folded_mask = (~self.has_folded[ids, 0]) & (
                    ~self.has_folded[ids, 1]
                )
                ids_showdown = ids[not_folded_mask]
                ids_folded = ids[~not_folded_mask]

                # Build one-hot 7-card planes for each player: 2 hole + 5 board
                # Shape: [num_ids, 4, 13]
                num_sd = ids_showdown.shape[0]
                if num_sd > 0:
                    # Player 0
                    a_plane = (
                        self.hole_onehot[ids_showdown, 0, 0]
                        + self.hole_onehot[ids_showdown, 0, 1]
                        + self.board_onehot[ids_showdown].sum(dim=1)
                    )
                    # Player 1
                    b_plane = (
                        self.hole_onehot[ids_showdown, 1, 0]
                        + self.hole_onehot[ids_showdown, 1, 1]
                        + self.board_onehot[ids_showdown].sum(dim=1)
                    )
                    # Clamp planes to 0/1
                    a_plane = (a_plane > 0.5).to(torch.float32)
                    b_plane = (b_plane > 0.5).to(torch.float32)
                    # Compare hands in batch
                    cmp = rules.compare_7_batch_onehot(
                        a_plane, b_plane
                    )  # shape [num_sd]
                    # Set winners
                    self.winner[ids_showdown[cmp > 0]] = 0
                    self.winner[ids_showdown[cmp < 0]] = 1
                    self.winner[ids_showdown[cmp == 0]] = -1

                # Mark all as done
                self.done[ids] = True

                # Compute scaled reward for actor of this step
                m = me[ids].long()
                scale = float(self.bb) * 100.0
                # Pot shares
                winner_ids = self.winner[ids]
                pot = self.pot[ids].float()
                stacks = self.stacks[ids, m].float()
                # pot_share: full pot if winner == m, half if tie, else 0
                pot_share = torch.where(
                    winner_ids == m,
                    pot,
                    torch.where(winner_ids == -1, pot / 2.0, torch.zeros_like(pot)),
                )
                rewards[ids] = (stacks + pot_share - float(self.starting_stack)) / scale
        return rewards, self.done.clone(), self.to_act.clone(), {}

    def reset_done(self, seed: Optional[int] = None) -> None:
        """Reset only environments with done=True; keeps others unchanged."""
        if seed is not None:
            self.rng.manual_seed(int(seed))
        ids = torch.nonzero(self.done, as_tuple=False).squeeze(1).tolist()
        for i in ids:
            deck = rules.new_shuffled_deck()
            button = int(torch.randint(0, 2, (), generator=self.rng).item())
            p_sb = 0 if button == 0 else 1
            p_bb = 1 - p_sb
            stacks = [self.starting_stack, self.starting_stack]
            committed = [0, 0]
            # Deal
            _c0_1 = deck.pop()
            _c0_2 = deck.pop()
            _c1_1 = deck.pop()
            _c1_2 = deck.pop()
            # Blinds
            stacks[p_sb] -= self.sb
            committed[p_sb] += self.sb
            stacks[p_bb] -= self.bb
            committed[p_bb] += self.bb
            # Write
            self.button[i] = button
            self.street[i] = 0
            self.to_act[i] = p_sb
            self.pot[i] = self.sb + self.bb
            self.min_raise[i] = self.bb
            self.last_aggr[i] = self.bb
            self.actions_this_round[i] = 0
            self.stacks[i, 0] = stacks[0]
            self.stacks[i, 1] = stacks[1]
            self.committed[i, 0] = committed[0]
            self.committed[i, 1] = committed[1]
            self.has_folded[i, :] = False
            self.is_allin[i, :] = False
            self.board_onehot[i, :, :, :] = 0.0
            self.hole_onehot[i, :, :, :, :] = 0.0
            self.hole_onehot[i, 0, 0, rules.suit(_c0_1), rules.rank(_c0_1)] = 1.0
            self.hole_onehot[i, 0, 1, rules.suit(_c0_2), rules.rank(_c0_2)] = 1.0
            self.hole_onehot[i, 1, 0, rules.suit(_c1_1), rules.rank(_c1_1)] = 1.0
            self.hole_onehot[i, 1, 1, rules.suit(_c1_2), rules.rank(_c1_2)] = 1.0
            self.done[i] = False
            self.winner[i] = -1
            self._decks[i] = deck
