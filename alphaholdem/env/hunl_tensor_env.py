from __future__ import annotations

from typing import Optional, Tuple

try:
    from line_profiler import profile
except ImportError:  # pragma: no cover

    def profile(f):
        return f


import torch

from . import rules


class HUNLTensorEnv:
    """Tensorized, batched HUNL environment.

    Maintains key game scalars/vectors as tensors of shape [N] or [N, 2]:
      - stacks, committed, has_folded, is_allin: [N, 2]
      - pot, to_act, street, actions_this_round, min_raise, last_aggr: [N]
      - board_onehot: [N, 5, 4, 13], hole_onehot: [N, 2, 2, 4, 13]
      - done mask: [N] bool, winner: [N] in {0,1,-1}

    Decks are stored as a tensor [N, 52] with per-env draw positions [N].
    Discrete bin stepping is compatible with bet_bins.
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
    # action history
    # shape: [N, 4 streets, S rounds per street, 4 rows (p0,p1,sum,legal), num_bins]
    # rows: 0=p0 action one-hot, 1=p1 action one-hot, 2=sum, 3=legal mask
    action_history: torch.Tensor
    history_slots: int

    def __init__(
        self,
        num_envs: int,
        starting_stack: int,
        sb: int,
        bb: int,
        bet_bins: list[float],
        device: Optional[torch.device] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        assert num_envs > 0
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.N = num_envs
        self.arange_n = torch.arange(self.N, device=self.device)
        self.starting_stack = int(starting_stack)
        self.sb = int(sb)
        self.bb = int(bb)
        self.bet_bins = bet_bins
        # Cache bet bins as tensor for fast indexing
        self.bet_bins_t = torch.tensor(
            self.bet_bins, dtype=torch.float32, device=self.device
        )
        # Number of discrete bins: 0=fold, 1=check/call, 2..(B-2)=presets, (B-1)=all-in
        self.num_bet_bins = len(self.bet_bins) + 3

        # Use provided RNG or create a new one
        if rng is not None:
            self.rng = rng
        else:
            self.rng = torch.Generator(device=self.device)

        # Per-env decks as tensor and draw positions
        self.deck = torch.empty(self.N, 9, dtype=torch.long, device=self.device)
        self.deck_pos = torch.zeros(self.N, dtype=torch.long, device=self.device)

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
        self.acted_since_reset = torch.zeros(
            self.N, dtype=torch.bool, device=self.device
        )

        self.stacks = torch.zeros(self.N, 2, dtype=torch.long, device=self.device)
        self.committed = torch.zeros(self.N, 2, dtype=torch.long, device=self.device)
        self.has_folded = torch.zeros(self.N, 2, dtype=torch.bool, device=self.device)
        self.is_allin = torch.zeros(self.N, 2, dtype=torch.bool, device=self.device)

        # Board and hole cards stored as one-hot [4,13]
        # board_onehot: [N, 5, 4, 13], hole_onehot: [N, 2 players, 2 cards, 4, 13]
        self.board_onehot = torch.zeros(
            (self.N, 5, 4, 13), dtype=torch.long, device=self.device
        )

        # Chip tracking for delta calculations - single tensor for both players
        # chips_placed[env_idx, player] = total chips placed by that player in that environment
        self.chips_placed = torch.zeros(self.N, 2, dtype=torch.long, device=self.device)
        self.hole_onehot = torch.zeros(
            (self.N, 2, 2, 4, 13), dtype=torch.long, device=self.device
        )
        self.done = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self.winner = torch.full(
            (self.N,), -1, dtype=torch.long, device=self.device
        )  # -1 split
        # history config
        self.history_slots = 6
        self.action_history = torch.zeros(
            (self.N, 4, self.history_slots, 4, self.num_bet_bins),
            dtype=torch.bool,
            device=self.device,
        )

        # Cache one-hot card encodings for all 52 cards
        # Precompute full 4x13 one-hot matrices for all cards 0-51
        all_cards = torch.arange(52, device=self.device)
        suits = all_cards // 13  # [52] suit indices 0-3
        ranks = all_cards % 13  # [52] rank indices 0-12

        # Create full one-hot cache: [52, 4, 13]
        self.card_onehot_cache = torch.zeros(
            52, 4, 13, dtype=torch.long, device=self.device
        )
        self.card_onehot_cache[all_cards, suits, ranks] = 1

    # --- Reset -----------------------------------------------------------------

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:

        # Determine which environments to reset
        if mask is None:
            # Reset all environments
            ids = slice(None)
            num_reset = self.N
        else:
            # Reset only environments specified by mask
            ids = mask
            if ids.numel() == 0:
                return  # Nothing to reset
            num_reset = ids.numel()

        # Shuffle decks for specified environments
        random_vals = torch.rand(num_reset, 52, generator=self.rng, device=self.device)
        # Only need 9 cards.
        _, deck_cards = torch.topk(random_vals, 9, dim=1)
        self.deck[ids] = deck_cards

        # Deal hole cards: for each env, assign 4 cards in deck order
        # [num_reset, 4] indices into deck for each env
        cards = self.deck[ids, :4]
        self.deck_pos[ids] = 4

        # Randomize button for specified environments
        button = torch.randint(
            0, 2, (num_reset,), generator=self.rng, device=self.device
        )
        p_sb = button
        p_bb = 1 - button

        # Assign hole cards
        _c0_1 = cards[:, 0]
        _c0_2 = cards[:, 1]
        _c1_1 = cards[:, 2]
        _c1_2 = cards[:, 3]

        # Reset stacks and post blinds (vectorized)
        # For each env, subtract sb/bb from correct player
        # p_sb and p_bb are [num_reset] with values 0 or 1
        self.stacks[ids, p_sb] = self.starting_stack - self.sb
        self.committed[ids, p_sb] = self.sb
        self.stacks[ids, p_bb] = self.starting_stack - self.bb
        self.committed[ids, p_bb] = self.bb

        # Write per-env scalars
        self.button[ids] = button
        self.street[ids] = 0  # preflop
        self.to_act[ids] = p_sb
        self.pot[ids] = self.sb + self.bb
        self.min_raise[ids] = self.bb
        self.last_aggr[ids] = self.bb
        self.actions_this_round[ids] = 0
        self.acted_since_reset[ids] = False
        self.has_folded[ids, :] = False
        self.is_allin[ids, :] = False
        self.done[ids] = False
        self.winner[ids] = -1

        # Zero out board_onehot and hole_onehot
        self.board_onehot[ids, :, :, :] = 0.0
        self.hole_onehot[ids, :, :, :, :] = 0.0

        # Set hole_onehot for specified environments using cached one-hot matrices
        # Direct assignment from cache: [num_reset, 4, 13] -> [num_reset, 2, 2, 4, 13]
        self.hole_onehot[ids, 0, 0] = self.card_onehot_cache[_c0_1]
        self.hole_onehot[ids, 0, 1] = self.card_onehot_cache[_c0_2]
        self.hole_onehot[ids, 1, 0] = self.card_onehot_cache[_c1_1]
        self.hole_onehot[ids, 1, 1] = self.card_onehot_cache[_c1_2]

        # Reset history planes for specified environments
        self.action_history[ids].zero_()

        # Reset chip tracking for specified environments and account for posted blinds
        self.chips_placed[ids] = 0
        self.chips_placed[ids, p_sb] = self.sb
        self.chips_placed[ids, p_bb] = self.bb

    # --- Legality ---------------------------------------------------------------

    def _compute_bin_amounts_and_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (amounts, mask) for discrete bins, allowing non-integer bets, no deduplication.

        amounts: [N, B] with -1 for non-preset bins or illegal presets
        mask:    [N, B] bool mask of legal bins (fold/check-call/presets/all-in)
        """
        N = self.N
        device = self.device
        B = self.num_bet_bins
        mask = torch.zeros(N, B, dtype=torch.bool, device=device)

        me = self.to_act
        opp = 1 - me
        me_stack = self.stacks[self.arange_n, me]
        opp_stack = self.stacks[self.arange_n, opp]
        me_committed = self.committed[self.arange_n, me]
        opp_committed = self.committed[self.arange_n, opp]
        to_call = opp_committed - me_committed

        total_committed = self.pot + me_committed + opp_committed

        # Fold if facing bet
        mask[:, 0] = to_call > 0

        # Check/Call
        can_check = to_call <= 0
        can_call = (to_call > 0) & (me_stack > 0)
        mask[:, 1] = can_check | can_call

        # Pre-compute candidate concrete amounts for preset bins 2..B-2
        can_bet_raise = (me_stack > 0) & (opp_stack > 0)
        amounts = torch.full((N, B), -1, dtype=torch.long, device=device)

        # Vectorized version using self.bet_bins_t
        # bet_bins_t: [B-3] tensor of multipliers (float)
        num_presets = self.bet_bins_t.shape[0]
        # [N, B-3] for all preset bins
        base = (total_committed.unsqueeze(1) * self.bet_bins_t.unsqueeze(0)).to(
            torch.long
        )  # [N, B-3]
        amt = torch.full(
            (N, num_presets), -1, dtype=torch.long, device=device
        )  # [N, B-3]

        # Bet case
        bet_case = (to_call <= 0) & can_bet_raise & (total_committed > 0)  # [N]
        bet_case = bet_case.unsqueeze(1).expand(-1, num_presets)  # [N, B-3]
        bet_amt = base.clamp(min=self.bb)  # [N, B-3]
        # Cap bet to opponent's effective stack to avoid unmatched excess in HU
        bet_amt = torch.minimum(bet_amt, opp_stack.unsqueeze(1))
        bet_amt = torch.minimum(
            bet_amt, (me_stack - 1).unsqueeze(1)
        )  # strictly below all-in
        bet_legal = bet_case & (bet_amt > 0)
        amt = torch.where(bet_legal, bet_amt, amt)

        # Raise case
        raise_case = (
            (to_call > 0) & can_bet_raise & (me_stack > to_call) & (total_committed > 0)
        )  # [N]
        raise_case = raise_case.unsqueeze(1).expand(-1, num_presets)  # [N, B-3]
        raise_amt = base + to_call.unsqueeze(1)  # [N, B-3]
        raise_inc = base  # [N, B-3]
        meets_min = raise_inc >= self.min_raise.unsqueeze(1)  # [N, B-3]
        can_only_allin = me_stack.unsqueeze(1) <= (
            to_call.unsqueeze(1) + self.min_raise.unsqueeze(1)
        )  # [N, B-3]
        # Legal if exceeds call and either meets min-raise or only all-in remains
        raise_legal = (
            raise_case
            & (raise_amt > to_call.unsqueeze(1))
            & (meets_min | can_only_allin)
        )
        # Do not exceed stack (those map to all-in bin)
        # Cap raise to both our stack and opponent's stack (HU, no side-pot)
        raise_amt = torch.minimum(raise_amt, (me_stack - 1).unsqueeze(1))
        raise_amt = torch.minimum(raise_amt, opp_stack.unsqueeze(1))
        raise_legal = raise_legal & (raise_amt > to_call.unsqueeze(1))
        amt = torch.where(raise_legal, raise_amt, amt)

        # Write to amounts and mask for bins 2..B-2
        amounts[:, 2 : 2 + num_presets] = amt
        mask[:, 2 : 2 + num_presets] = amt > 0

        # All-in always available if we have chips
        mask[:, B - 1] = me_stack > 0

        # If any player is all-in in an env, restrict to check/call only
        any_allin = self.is_allin[:, 0] | self.is_allin[:, 1]
        if any_allin.any():
            rows = any_allin
            mask[rows, :] = False
            mask[rows, 1] = True

        return amounts, mask

    def legal_action_bins_mask(self) -> torch.Tensor:
        """Return [N, B] mask of legal bins with deduplication."""
        _, mask = self._compute_bin_amounts_and_mask()
        return mask

    def bin_amounts(self) -> torch.Tensor:
        """Return [N, B] concrete amounts for bins; -1 where not applicable."""
        amounts, _ = self._compute_bin_amounts_and_mask()
        return amounts

    # --- Helper APIs -----------------------------------------------------------

    def active_indices(self) -> torch.Tensor:
        """Return [M] tensor of indices for environments that are not done."""
        return torch.where(~self.done)[0]

    def states_summary(self) -> dict:
        """Lightweight snapshot for debugging/logging (avoids large tensors)."""
        return {
            "street": self.street,
            "to_act": self.to_act,
            "pot": self.pot,
            "min_raise": self.min_raise,
            "done": self.done,
        }

    def get_action_history(self) -> torch.Tensor:
        """Return action history planes tensor if allocated, else None."""
        return self.action_history

    def bet(self, env_indices: torch.Tensor, chips: torch.Tensor) -> None:
        """
        Place a bet for the current player in specified environments.

        Args:
            env_indices: Environment indices to update [M]
            chips: Amount of chips to bet [M]
        """

        player = self.to_act[env_indices]

        # Update stacks: subtract chips from player's stack
        self.stacks[env_indices, player] -= chips

        # Update committed: add chips to player's committed amount
        self.committed[env_indices, player] += chips

        # Update pot: add chips to pot
        self.pot[env_indices] += chips

        # Update chips_placed: track total chips placed by this player
        self.chips_placed[env_indices, player] += chips

    # --- Step ------------------------------------------------------------------

    @profile
    def step_bins(
        self, bin_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all envs using discrete bet bin indices tensor [N]. -1 means no action.

        Returns (rewards [N], dones [N], to_act [N], chips_placed [N])
        Rewards are scaled by 100bb consistent with HUNLEnv.
        """
        assert bin_indices.shape[0] == self.N
        N = self.N
        device = self.device

        me = self.to_act
        opp = 1 - me

        me_stack = self.stacks.gather(1, me.view(N, 1)).squeeze(1)
        me_comm = self.committed.gather(1, me.view(N, 1)).squeeze(1)
        opp_comm = self.committed.gather(1, opp.view(N, 1)).squeeze(1)
        to_call = opp_comm - me_comm

        committed_before = me_comm.clone()

        rewards = torch.zeros(N, dtype=torch.float32, device=device)

        # Group masks by action type
        is_fold = bin_indices == 0
        is_check_call = bin_indices == 1
        is_allin = bin_indices == (self.num_bet_bins - 1)
        is_bet_raise = ~(is_fold | is_check_call | is_allin)

        # -1 bet bin index means no action
        acting_mask = bin_indices >= 0
        acting = torch.where(acting_mask)[0]

        # We've now acted since reset
        self.acted_since_reset[acting] = True

        # Record current action into history (actor rows and sum row)
        round_idx = self.street.clamp(min=0, max=3)[acting]
        slot_idx = torch.clamp(self.actions_this_round, max=self.history_slots - 1)[
            acting
        ]
        # clear actor rows at this slot
        self.action_history[acting, round_idx, slot_idx, 0, :] = 0
        self.action_history[acting, round_idx, slot_idx, 1, :] = 0
        # set actor-specific one-hot
        self.action_history[
            acting, round_idx, slot_idx, me[acting], bin_indices[acting]
        ] = 1
        # sum row
        self.action_history[acting, round_idx, slot_idx, 2, bin_indices[acting]] = 1

        # Fold: immediate terminal, award pot to opp
        idx = torch.where(is_fold)[0]
        # Winner is opp
        self.winner[idx] = opp[idx]
        self.done[idx] = True
        # Reward equals current stack - starting_stack (committed lost to pot/opponent)
        scale = float(self.bb) * 100.0
        rewards[idx] = (
            self.stacks[idx, me[idx]].to(torch.float32) - float(self.starting_stack)
        ) / scale

        # Check/Call
        idx = torch.where(is_check_call)[0]
        call_amt = torch.minimum(to_call[idx], me_stack[idx])
        self.bet(idx, call_amt)

        # All-in
        idx = torch.where(is_allin)[0]
        amt = me_stack[idx]
        # Pay call first if needed
        call_amt = torch.clamp(to_call[idx], min=0)
        call_amt = torch.minimum(call_amt, amt)
        if call_amt.any():
            self.bet(idx, call_amt)
        # Then shove remaining
        remaining_amt = amt - call_amt
        if remaining_amt.any():
            self.bet(idx, remaining_amt)
        self.is_allin[idx, me[idx]] = True

        # Bet/Raise bins: approximate mapping using total_committed * mult
        if is_bet_raise.any():
            idx = torch.where(is_bet_raise)[0]
            bin_local = bin_indices[idx]
            mult_idx = (bin_local - 2).clamp(min=0, max=len(self.bet_bins) - 1)
            mult = self.bet_bins_t[mult_idx]
            total_committed = self.pot[idx] + me_comm[idx] + opp_comm[idx]
            base = (total_committed * mult).to(torch.long)
            # Separate bet vs raise by to_call
            idx_bet = idx[to_call[idx] <= 0]
            if idx_bet.numel() > 0:
                amt = base[to_call[idx] <= 0]
                amt = amt.clamp(min=self.bb)
                # Cap to opponent's effective stack to avoid over-betting beyond coverage
                amt = torch.minimum(amt, self.stacks[idx_bet, 1 - me[idx_bet]])
                # Also cannot exceed our own stack (non-allin path)
                amt = torch.minimum(amt, self.stacks[idx_bet, me[idx_bet]])
                self.bet(idx_bet, amt)
                self.last_aggr[idx_bet] = amt
                self.min_raise[idx_bet] = torch.maximum(self.min_raise[idx_bet], amt)
            idx_raise = idx[to_call[idx] > 0]
            if idx_raise.numel() > 0:
                base_r = base[to_call[idx] > 0]
                call_amt = torch.minimum(
                    to_call[idx_raise], self.stacks[idx_raise, me[idx_raise]]
                )
                # Pay call
                self.bet(idx_raise, call_amt)
                # Raise part
                avail = self.stacks[idx_raise, me[idx_raise]]
                # Opponent coverage after call
                opp_avail = self.stacks[idx_raise, 1 - me[idx_raise]]
                min_req = self.min_raise[idx_raise]
                raise_part = torch.minimum(base_r, avail)
                # Cap to opponent coverage (HU, no side-pot)
                raise_part = torch.minimum(raise_part, opp_avail)
                need = (raise_part < min_req) & (avail > min_req)
                raise_part = torch.where(need, min_req, raise_part)
                self.bet(idx_raise, raise_part)
                self.last_aggr[idx_raise] = raise_part
                self.min_raise[idx_raise] = torch.maximum(
                    self.min_raise[idx_raise], raise_part
                )

        # Advance to next actor (simple alternation)
        self.to_act[acting] = 1 - self.to_act[acting]
        self.actions_this_round[acting] += ~self.done[acting]
        self.acted_since_reset[acting] = True

        # Write legal mask for next decision into history legal row
        next_street = self.street[acting].clamp(min=0, max=3)
        next_slot = torch.clamp(
            self.actions_this_round[acting], max=self.history_slots - 1
        )
        legal_mask = self.legal_action_bins_mask()
        self.action_history[acting, next_street, next_slot, 3, :] = legal_mask[acting]

        # Chips placed by the actor in this step (after updates)
        placed_chips = (
            self.committed.gather(1, me.view(N, 1)).squeeze(1) - committed_before
        )

        # Round closure: equal committed and both acted
        equal_committed = self.committed[:, 0] == self.committed[:, 1]
        round_closed = equal_committed & (self.actions_this_round >= 2)
        rc_idx = torch.where(round_closed)[0]
        if rc_idx.numel() > 0:
            # Reset committed
            self.pot[rc_idx] += 0  # already in pot
            self.committed[rc_idx, :] = 0
            self.actions_this_round[rc_idx] = 0
            # Advance street and deal
            s = self.street[rc_idx]

            # flop (vectorized dealing of 3 cards)
            flop_mask = s == 0
            ids = rc_idx[flop_mask]
            pos = self.deck_pos[ids]
            self.deck_pos[ids] = pos + 3
            self.board_onehot[ids, 0] = self.card_onehot_cache[self.deck[ids, pos]]
            self.board_onehot[ids, 1] = self.card_onehot_cache[self.deck[ids, pos + 1]]
            self.board_onehot[ids, 2] = self.card_onehot_cache[self.deck[ids, pos + 2]]

            self.street[ids] = 1
            self.to_act[ids] = 1 - self.button[ids]
            self.min_raise[ids] = self.bb
            self.last_aggr[ids] = 0

            # turn
            turn_mask = s == 1
            ids = rc_idx[turn_mask]
            pos = self.deck_pos[ids]
            c = self.deck[ids, pos]
            self.deck_pos[ids] = pos + 1
            # Use cached one-hot matrix for turn card
            self.board_onehot[ids, 3] = self.card_onehot_cache[c]
            self.street[ids] = 2
            self.to_act[ids] = 1 - self.button[ids]
            self.min_raise[ids] = self.bb
            self.last_aggr[ids] = 0

            # river
            river_mask = s == 2
            ids = rc_idx[river_mask]
            pos = self.deck_pos[ids]
            c = self.deck[ids, pos]
            self.deck_pos[ids] = pos + 1
            # Use cached one-hot matrix for river card
            self.board_onehot[ids, 4] = self.card_onehot_cache[c]
            self.street[ids] = 3
            self.to_act[ids] = 1 - self.button[ids]
            self.min_raise[ids] = self.bb
            self.last_aggr[ids] = 0

            # showdown
            sd_mask = s == 3
            ids = rc_idx[sd_mask]
            # Evaluate winners or award folds
            # Vectorized showdown resolution for all ids at once

            # Only consider ids where neither player has folded
            not_folded_mask = (~self.has_folded[ids, 0]) & (~self.has_folded[ids, 1])
            ids_showdown = ids[not_folded_mask]

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
                a_plane = a_plane.clamp(0, 1)
                b_plane = b_plane.clamp(0, 1)
                # Compare hands in batch
                cmp = rules.compare_7_batch(a_plane, b_plane)  # shape [num_sd]
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

        return rewards, self.done, self.to_act, placed_chips

    def reset_done(self) -> None:
        """Reset only environments with done=True; keeps others unchanged."""
        ids = torch.where(self.done)[0]
        if ids.numel() == 0:
            return  # Nothing to reset
        self.reset(mask=ids)
