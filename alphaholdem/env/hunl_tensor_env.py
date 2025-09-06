from __future__ import annotations

from typing import Optional, Tuple
import random

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
    # action history (allocated lazily when num_bet_bins known)
    # shape: [N, 4 rounds, S slots, 4 rows (p0,p1,sum,legal), num_bins]
    # rows: 0=p0 action one-hot, 1=p1 action one-hot, 2=sum, 3=legal mask
    action_history: object
    history_slots: int

    def __init__(
        self,
        num_envs: int,
        starting_stack: int,
        sb: int,
        bb: int,
        bet_bins: list[float],
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
        shuffle_mode: str = "torch",
    ) -> None:
        assert num_envs > 0
        self.N = num_envs
        self.starting_stack = int(starting_stack)
        self.sb = int(sb)
        self.bb = int(bb)
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.bet_bins = bet_bins
        self.shuffle_mode = shuffle_mode
        # Number of discrete bins: 0=fold, 1=check/call, 2..(B-2)=presets, (B-1)=all-in
        self.num_bet_bins = len(self.bet_bins) + 3

        self.rng = torch.Generator(device="cpu")
        if seed is not None:
            self.rng.manual_seed(int(seed))

        # Per-env decks as tensor and draw positions
        self.deck = torch.empty(self.N, 52, dtype=torch.long, device=self.device)
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
        # history config
        self.history_slots = 6
        self.action_history = None

    # --- Reset -----------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng.manual_seed(int(seed))

        # New shuffled decks and deal
        # Vectorized reset for all environments

        # Shuffle decks for all envs and reset draw positions
        if self.shuffle_mode == "python":
            # Use Python RNG to match HUNLEnv shuffles for parity tests
            base_seed = int(seed) if seed is not None else random.randrange(1 << 30)
            decks = []
            for i in range(self.N):
                rng = random.Random(base_seed + i * 9973)
                d = rules.new_shuffled_deck(rng)
                decks.append(d)
            self.deck = torch.tensor(decks, dtype=torch.long, device=self.device)
        else:
            self.deck = torch.stack(
                [torch.randperm(52, generator=self.rng) for _ in range(self.N)], dim=0
            ).to(self.device)
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
        card_indices = (
            torch.arange(4, device=self.device).unsqueeze(0).expand(self.N, 4)
        )
        cards = self.deck[
            torch.arange(self.N, device=self.device).unsqueeze(1), card_indices
        ]  # [N, 4]
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
        # Vectorized set of hole one-hots
        s0_1, r0_1 = _c0_1 // 13, _c0_1 % 13
        s0_2, r0_2 = _c0_2 // 13, _c0_2 % 13
        s1_1, r1_1 = _c1_1 // 13, _c1_1 % 13
        s1_2, r1_2 = _c1_2 // 13, _c1_2 % 13
        n_idx = torch.arange(self.N, device=self.device)
        self.hole_onehot[n_idx, 0, 0, s0_1, r0_1] = 1.0
        self.hole_onehot[n_idx, 0, 1, s0_2, r0_2] = 1.0
        self.hole_onehot[n_idx, 1, 0, s1_1, r1_1] = 1.0
        self.hole_onehot[n_idx, 1, 1, s1_2, r1_2] = 1.0
        # Reset history planes if already allocated
        if (
            isinstance(self.action_history, torch.Tensor)
            and self.action_history.numel() > 0
        ):
            self.action_history.zero_()
        # Note: Hole cards are implicit in deck/history; full tensorization of hole cards
        # can be added later (e.g., hole_cards tensor [N, 2, 2]).

    # --- Legality ---------------------------------------------------------------

    def _compute_bin_amounts_and_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (amounts, mask) for discrete bins with per-env deduplication.

        amounts: [N, B] with -1 for non-preset bins or illegal presets
        mask:    [N, B] bool mask of legal bins (fold/check-call/presets/all-in)
        """
        N = self.N
        device = self.device
        B = self.num_bet_bins
        mask = torch.zeros(N, B, dtype=torch.bool, device=device)

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

        # Pre-compute candidate concrete amounts for preset bins 2..B-2
        can_bet_raise = (me_stack > 0) & (opp_stack > 0)
        amounts = torch.full((N, B), -1, dtype=torch.long, device=device)
        for i, mult in enumerate(self.bet_bins):
            b = 2 + i
            if b >= B - 1:
                break
            base = (total_committed.to(torch.float32) * float(mult)).to(torch.long)
            # Start with illegal by default
            amt = torch.full((N,), -1, dtype=torch.long, device=device)
            # Bet case
            bet_case = (to_call <= 0) & can_bet_raise & (total_committed > 0)
            bet_amt = base.clamp(min=self.bb)
            # Cap bet to opponent's effective stack to avoid unmatched excess in HU
            bet_amt = torch.minimum(bet_amt, opp_stack)
            bet_amt = torch.minimum(bet_amt, me_stack - 1)  # strictly below all-in
            bet_legal = bet_case & (bet_amt > 0)
            amt = torch.where(bet_legal, bet_amt, amt)
            # Raise case
            raise_case = (
                (to_call > 0)
                & can_bet_raise
                & (me_stack > to_call)
                & (total_committed > 0)
            )
            raise_amt = base + to_call
            raise_inc = base
            meets_min = raise_inc >= self.min_raise
            can_only_allin = me_stack <= (to_call + self.min_raise)
            # Legal if exceeds call and either meets min-raise or only all-in remains
            raise_legal = (
                raise_case & (raise_amt > to_call) & (meets_min | can_only_allin)
            )
            # Do not exceed stack (those map to all-in bin)
            # Cap raise to both our stack and opponent's stack (HU, no side-pot)
            raise_amt = torch.minimum(raise_amt, me_stack - 1)
            raise_amt = torch.minimum(raise_amt, opp_stack)
            raise_legal = raise_legal & (raise_amt > to_call)
            amt = torch.where(raise_legal, raise_amt, amt)
            amounts[:, b] = amt

        # Deduplicate by concrete amount per environment, keep first occurrence
        seen_amounts = torch.full((N, B), -1, dtype=torch.long, device=device)
        seen_ptr = torch.zeros(N, dtype=torch.long, device=device)
        for b in range(2, max(2, B - 1)):
            cand = amounts[:, b]
            is_candidate = cand >= 0
            # cand equals any previously seen amount?
            dup = (cand.view(N, 1) == seen_amounts).any(dim=1)
            is_new = is_candidate & (~dup)
            mask[:, b] = is_new
            if is_new.any():
                idx = torch.nonzero(is_new, as_tuple=False).squeeze(1)
                seen_amounts[idx, seen_ptr[idx]] = cand[idx]
                seen_ptr[idx] = seen_ptr[idx] + 1

        # All-in always available if we have chips
        mask[:, B - 1] = me_stack > 0

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
        return torch.nonzero(~self.done, as_tuple=False).squeeze(1)

    def states_summary(self) -> dict:
        """Lightweight snapshot for debugging/logging (avoids large tensors)."""
        return {
            "street": self.street.clone(),
            "to_act": self.to_act.clone(),
            "pot": self.pot.clone(),
            "min_raise": self.min_raise.clone(),
            "done": self.done.clone(),
        }

    def get_action_history(self) -> torch.Tensor | None:
        """Return action history planes tensor if allocated, else None."""
        return self.action_history

    # --- Step ------------------------------------------------------------------

    def step_bins(
        self, bin_indices: torch.Tensor
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

        # Lazily allocate action history tensor when num_bet_bins is known
        if (
            (not isinstance(self.action_history, torch.Tensor))
            or self.action_history.numel() == 0
            or self.action_history.size(-1) != self.num_bet_bins
        ):
            self.action_history = torch.zeros(
                (self.N, 4, self.history_slots, 4, self.num_bet_bins),
                dtype=torch.float32,
                device=self.device,
            )

        # Group masks by action type
        is_fold = bin_indices == 0
        is_check_call = bin_indices == 1
        is_allin = bin_indices == (self.num_bet_bins - 1)
        is_bet_raise = ~(is_fold | is_check_call | is_allin)

        # Record current action into history (actor rows and sum row)
        n_idx = torch.arange(N, device=self.device)
        round_idx = self.street.clamp(min=0, max=3)
        slot_idx = torch.clamp(self.actions_this_round, max=self.history_slots - 1)
        # clear actor rows at this slot
        self.action_history[n_idx, round_idx, slot_idx, 0, :] = 0.0
        self.action_history[n_idx, round_idx, slot_idx, 1, :] = 0.0
        # set actor-specific one-hot
        self.action_history[n_idx, round_idx, slot_idx, me, bin_indices] = 1.0
        # sum row
        self.action_history[n_idx, round_idx, slot_idx, 2, bin_indices] = 1.0

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
            # Avoid mixed scalar/tensor clamp; clamp to non-negative, then cap by amt
            call_amt = torch.clamp(to_call[idx], min=0)
            call_amt = torch.minimum(call_amt, amt)
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
                # Cap to opponent's effective stack to avoid over-betting beyond coverage
                amt = torch.minimum(amt, self.stacks[idx_bet, 1 - me[idx_bet]])
                # Also cannot exceed our own stack (non-allin path)
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
                # Opponent coverage after call
                opp_avail = self.stacks[idx_raise, 1 - me[idx_raise]]
                min_req = self.min_raise[idx_raise]
                raise_part = torch.minimum(base_r, avail)
                # Cap to opponent coverage (HU, no side-pot)
                raise_part = torch.minimum(raise_part, opp_avail)
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

        # Write legal mask for next decision into history legal row
        next_round = self.street.clamp(min=0, max=3)
        next_slot = torch.clamp(self.actions_this_round, max=self.history_slots - 1)
        legal_mask = self.legal_action_bins_mask().to(torch.float32)
        self.action_history[n_idx, next_round, next_slot, 3, :] = legal_mask

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
            # flop (vectorized dealing of 3 cards)
            flop_mask = s == 0
            if flop_mask.any():
                ids = rc_idx[flop_mask]
                pos = self.deck_pos[ids]
                a = self.deck[ids, pos]
                b = self.deck[ids, pos + 1]
                c = self.deck[ids, pos + 2]
                self.deck_pos[ids] = pos + 3
                a_s, a_r = a // 13, a % 13
                b_s, b_r = b // 13, b % 13
                c_s, c_r = c // 13, c % 13
                self.board_onehot[ids, 0, a_s, a_r] = 1.0
                self.board_onehot[ids, 1, b_s, b_r] = 1.0
                self.board_onehot[ids, 2, c_s, c_r] = 1.0
                self.street[ids] = 1
                self.to_act[ids] = 1 - self.button[ids]
                self.min_raise[ids] = self.bb
                self.last_aggr[ids] = 0
            # turn
            turn_mask = s == 1
            if turn_mask.any():
                ids = rc_idx[turn_mask]
                pos = self.deck_pos[ids]
                c = self.deck[ids, pos]
                self.deck_pos[ids] = pos + 1
                c_s, c_r = c // 13, c % 13
                self.board_onehot[ids, 3, c_s, c_r] = 1.0
                self.street[ids] = 2
                self.to_act[ids] = 1 - self.button[ids]
                self.min_raise[ids] = self.bb
                self.last_aggr[ids] = 0
            # river
            river_mask = s == 2
            if river_mask.any():
                ids = rc_idx[river_mask]
                pos = self.deck_pos[ids]
                c = self.deck[ids, pos]
                self.deck_pos[ids] = pos + 1
                c_s, c_r = c // 13, c % 13
                self.board_onehot[ids, 4, c_s, c_r] = 1.0
                self.street[ids] = 3
                self.to_act[ids] = 1 - self.button[ids]
                self.min_raise[ids] = self.bb
                self.last_aggr[ids] = 0
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
        # Auto-runout: if any env has a player all-in and not done, deal to river and resolve
        auto_mask = (self.is_allin[:, 0] | self.is_allin[:, 1]) & (~self.done)
        auto_ids = torch.nonzero(auto_mask, as_tuple=False).squeeze(1)
        if auto_ids.numel() > 0:
            for i in auto_ids.tolist():
                # Deal remaining community cards to reach 5
                current_cards = int(
                    self.board_onehot[i].sum().item()
                )  # equals num cards
                while current_cards < 5:
                    c = int(self.deck[i, self.deck_pos[i]].item())
                    self.deck_pos[i] += 1
                    # place into first empty slot
                    for k in range(5):
                        if self.board_onehot[i, k].sum().item() == 0.0:
                            self.board_onehot[i, k, rules.suit(c), rules.rank(c)] = 1.0
                            break
                    current_cards += 1
                # Set river street and resolve showdown if not folded
                self.street[i] = 3
                if not self.has_folded[i, 0] and not self.has_folded[i, 1]:
                    a_plane = (
                        self.hole_onehot[i, 0, 0]
                        + self.hole_onehot[i, 0, 1]
                        + self.board_onehot[i].sum(dim=0)
                    )
                    b_plane = (
                        self.hole_onehot[i, 1, 0]
                        + self.hole_onehot[i, 1, 1]
                        + self.board_onehot[i].sum(dim=0)
                    )
                    a_plane = (a_plane > 0.5).to(torch.float32).unsqueeze(0)
                    b_plane = (b_plane > 0.5).to(torch.float32).unsqueeze(0)
                    cmp = rules.compare_7_batch_onehot(a_plane, b_plane)[0].item()
                    if cmp > 0:
                        self.winner[i] = 0
                    elif cmp < 0:
                        self.winner[i] = 1
                    else:
                        self.winner[i] = -1
                # Mark done and assign reward for current actor perspective
                self.done[i] = True
                m = int(me[i].item())
                scale = float(self.bb) * 100.0
                pot_share = 0.0
                if self.winner[i] == m:
                    pot_share = float(self.pot[i].item())
                elif self.winner[i] == -1:
                    pot_share = float(self.pot[i].item()) / 2.0
                rewards[i] = (
                    float(self.stacks[i, m].item())
                    + pot_share
                    - float(self.starting_stack)
                ) / scale

        return rewards, self.done.clone(), self.to_act.clone(), {}

    def reset_done(self, seed: Optional[int] = None) -> None:
        """Reset only environments with done=True; keeps others unchanged."""
        if seed is not None:
            self.rng.manual_seed(int(seed))
        ids = torch.nonzero(self.done, as_tuple=False).squeeze(1).tolist()
        for i in ids:
            # Shuffle deck tensor and reset draw pos
            if self.shuffle_mode == "python":
                base_seed = int(seed) if seed is not None else random.randrange(1 << 30)
                rng = random.Random(base_seed + i * 9973)
                self.deck[i] = torch.tensor(
                    rules.new_shuffled_deck(rng), dtype=torch.long, device=self.device
                )
            else:
                self.deck[i] = torch.randperm(52, generator=self.rng).to(self.device)
            self.deck_pos[i] = 0
            button = int(torch.randint(0, 2, (), generator=self.rng).item())
            p_sb = 0 if button == 0 else 1
            p_bb = 1 - p_sb
            stacks = [self.starting_stack, self.starting_stack]
            committed = [0, 0]
            # Deal
            _c0_1 = int(self.deck[i, self.deck_pos[i]].item())
            self.deck_pos[i] += 1
            _c0_2 = int(self.deck[i, self.deck_pos[i]].item())
            self.deck_pos[i] += 1
            _c1_1 = int(self.deck[i, self.deck_pos[i]].item())
            self.deck_pos[i] += 1
            _c1_2 = int(self.deck[i, self.deck_pos[i]].item())
            self.deck_pos[i] += 1
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
            # deck handled via tensorized storage
