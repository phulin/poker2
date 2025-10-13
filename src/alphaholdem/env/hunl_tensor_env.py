from __future__ import annotations

from typing import Optional, Tuple

import torch

from alphaholdem.env import rules
from alphaholdem.utils.profiling import profile

DEBUG_STEP_TABLE_ENVS = 3
DEFAULT_BET_BINS = [0.5, 0.75, 1.0, 1.5, 2.0]


class HUNLTensorEnv:
    """Tensorized, batched HUNL environment.

    Maintains key game scalars/vectors as tensors of shape [N] or [N, 2]:
      - stacks, committed, has_folded, is_allin: [N, 2]
      - pot, to_act, street, actions_this_round, min_raise: [N]
      - board_onehot: [N, 5, 4, 13], hole_onehot: [N, 2 players, 2 cards, 4, 13]
      - done mask: [N] bool, winner: [N] in {0,1,2}

    Decks are stored as a tensor [N, 52] with per-env draw positions [N].
    Discrete bin stepping is compatible with bet_bins.
    """

    # Type signatures for fields
    N: int
    starting_stack: int
    sb: int
    bb: int
    device: torch.device
    rng: torch.Generator
    # deck tensor and per-env draw position
    deck: torch.Tensor
    deck_pos: torch.Tensor
    button: torch.Tensor
    street: torch.Tensor
    to_act: torch.Tensor
    pot: torch.Tensor
    min_raise: torch.Tensor
    actions_this_round: torch.Tensor
    stacks: torch.Tensor
    committed: torch.Tensor
    has_folded: torch.Tensor
    is_allin: torch.Tensor
    board_onehot: torch.Tensor
    hole_onehot: torch.Tensor
    board_indices: torch.Tensor
    hole_indices: torch.Tensor
    done: torch.Tensor
    winner: torch.Tensor

    def __init__(
        self,
        num_envs: int,
        starting_stack: int,
        sb: int,
        bb: int,
        default_bet_bins: Optional[list[float]] = None,
        device: Optional[torch.device] = None,
        rng: Optional[torch.Generator] = None,
        float_dtype: torch.dtype = torch.float32,
        debug_step_table: bool = False,
        flop_showdown: bool = False,
    ) -> None:
        assert num_envs > 0
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.float_dtype = float_dtype
        self.N = num_envs
        self.arange_n = torch.arange(self.N, device=self.device)
        self.starting_stack = int(starting_stack)
        self.sb = int(sb)
        self.bb = int(bb)
        self.default_bet_bins = default_bet_bins or DEFAULT_BET_BINS
        self.num_bet_bins = len(self.default_bet_bins) + 3
        self.scale = float(self.bb) * 100.0
        self.debug_step_table = debug_step_table
        self.flop_showdown = flop_showdown

        # Use provided RNG or create a new one
        if rng is not None:
            self.rng = rng
        else:
            self.rng = torch.Generator(device=self.device)

        # Per-env decks as tensor and draw positions
        self.deck = torch.zeros(self.N, 9, dtype=torch.long, device=self.device)
        self.deck_pos = torch.zeros(self.N, dtype=torch.long, device=self.device)

        # Tensors (initialized in reset)
        self.button = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.street = torch.zeros(self.N, dtype=torch.long, device=self.device)  # 0..4
        self.to_act = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.pot = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.min_raise = torch.zeros(self.N, dtype=torch.long, device=self.device)
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

        # Board and hole cards stored as one-hot [4,13] and as indices [0-51]
        # board_onehot: [N, 5, 4, 13], hole_onehot: [N, 2 players, 2 cards, 4, 13]
        # board_indices: [N, 5], hole_indices: [N, 2 players, 2 cards]
        self.board_onehot = torch.zeros(
            (self.N, 5, 4, 13), dtype=torch.bool, device=self.device
        )
        self.hole_onehot = torch.zeros(
            (self.N, 2, 2, 4, 13), dtype=torch.bool, device=self.device
        )
        self.board_indices = torch.full(
            (self.N, 5), -1, dtype=torch.long, device=self.device
        )  # -1 means no card
        self.hole_indices = torch.full(
            (self.N, 2, 2), -1, dtype=torch.long, device=self.device
        )  # -1 means no card

        # Chip tracking for delta calculations - single tensor for both players
        # chips_placed[env_idx, player] = total chips placed by that player in that environment
        self.chips_placed = torch.zeros(self.N, 2, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        self.winner = torch.full(
            (self.N,), -1, dtype=torch.long, device=self.device
        )  # -1 split

        # Cache one-hot card encodings for all 52 cards
        # Precompute full 4x13 one-hot matrices for all cards 0-51
        all_cards = torch.arange(52, device=self.device)
        suits = all_cards // 13  # [52] suit indices 0-3
        ranks = all_cards % 13  # [52] rank indices 0-12

        # Create full one-hot cache: [52, 4, 13]
        self.card_onehot_cache = torch.zeros(
            52, 4, 13, dtype=torch.bool, device=self.device
        )
        self.card_onehot_cache[all_cards, suits, ranks] = True

    # --- Reset -----------------------------------------------------------------

    def reset(
        self,
        env_indices: Optional[torch.Tensor] = None,
        force_button: Optional[torch.Tensor] = None,
        force_deck: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset the environment.

        Args:
            env_indices: Optional[torch.Tensor] - Environment indices to reset.
            force_button: Optional[torch.Tensor] - Force button for each environment.
            force_deck: Optional[torch.Tensor] - Force deck for each environment.
                Shape: [num_reset, 1-9]. If shorter than 9, the rest of the deck is shuffled.
        """
        if self.debug_step_table:
            print("=" * 49 + " RESET " + "=" * 49)
        # Determine which environments to reset
        if env_indices is None:
            # Reset all environments
            ids = torch.arange(self.N, device=self.device)
            num_reset = self.N
        else:
            # Reset only environments specified by mask
            ids = env_indices
            if ids.numel() == 0:
                return  # Nothing to reset
            num_reset = ids.numel()

        # Force the forced cards to the front of the deck.
        decks = torch.arange(52, device=self.device).repeat(num_reset, 1)
        forced = 0 if force_deck is None else force_deck.shape[1]

        if force_button is not None:
            button = force_button.clone()
        else:
            button = torch.randint(
                0, 2, (num_reset,), generator=self.rng, device=self.device
            )
        p_sb = button
        p_bb = 1 - button

        if force_deck is not None:
            # guarantees swaps only go forward in deck. sorted has sorted[i] >= i.
            force_deck_sorted = force_deck.sort(dim=1)[0]
            for i in range(forced):
                decks[ids, i] = decks[ids, force_deck_sorted[:, i]]
                decks[ids, force_deck_sorted[:, i]] = decks[ids, i]
            # now unsort the forced portion.
            decks[ids, :forced] = force_deck

        # Pick enough of the remaining cards to get to 9 cards.
        assert forced <= 9
        left_to_shuffle = 9 - forced
        decks_left = decks[:, forced:]
        if left_to_shuffle > 0:
            random_vals = torch.rand(
                num_reset, 52 - forced, generator=self.rng, device=self.device
            )
            _, deck_cards = torch.topk(random_vals, left_to_shuffle, dim=1)
            decks_left[:, :left_to_shuffle] = decks_left.gather(1, deck_cards)

        self.deck[ids] = decks[:, :9]

        # Deal hole cards: for each env, assign 4 cards in deck order
        # [num_reset, 4] indices into deck for each env
        cards = self.deck[ids, :4]
        self.deck_pos[ids] = 4

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
        self.actions_this_round[ids] = 0
        self.acted_since_reset[ids] = False
        self.has_folded[ids, :] = False
        self.is_allin[ids, :] = False
        self.done[ids] = False
        self.winner[ids] = -1

        # Zero out board_onehot, hole_onehot, and card indices
        self.board_onehot[ids, :, :, :] = False
        self.hole_onehot[ids, :, :, :, :] = False
        self.board_indices[ids, :] = -1
        self.hole_indices[ids, :, :] = -1

        # Set hole_onehot for specified environments using cached one-hot matrices
        # Direct assignment from cache: [num_reset, 4, 13] -> [num_reset, 2, 2, 4, 13]
        self.hole_onehot[ids, 0, 0] = self.card_onehot_cache[_c0_1]
        self.hole_onehot[ids, 0, 1] = self.card_onehot_cache[_c0_2]
        self.hole_onehot[ids, 1, 0] = self.card_onehot_cache[_c1_1]
        self.hole_onehot[ids, 1, 1] = self.card_onehot_cache[_c1_2]

        # Set hole card indices
        self.hole_indices[ids, 0, 0] = _c0_1
        self.hole_indices[ids, 0, 1] = _c0_2
        self.hole_indices[ids, 1, 0] = _c1_1
        self.hole_indices[ids, 1, 1] = _c1_2

        # Reset chip tracking for specified environments and account for posted blinds
        self.chips_placed[ids, p_sb] = self.sb
        self.chips_placed[ids, p_bb] = self.bb

    # --- Legality ---------------------------------------------------------------

    def legal_bins_amounts_and_mask(
        self, bet_bins: Optional[list[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (amounts, mask) for discrete bins, allowing integer bets, no deduplication.
        Amounts is the concrete additional amount the player would commit to the pot.

        Note: Bet bin deduplication is intentionally not implemented for performance reasons.
        While multiple preset bins may map to the same concrete amount (especially near
        all-in situations), deduplicating them would require expensive tensor operations
        and has minimal benefit for training stability. The policy can learn to handle
        multiple bins mapping to the same action effectively.

        amounts: [N, B] with -1 for non-preset bins or illegal presets
        mask:    [N, B] bool mask of legal bins (fold/check-call/presets/all-in)
        """

        if bet_bins is None:
            bet_bins = self.default_bet_bins

        N = self.N
        device = self.device
        B = len(bet_bins) + 3
        amounts = torch.full((N, B), -1, dtype=torch.long, device=device)
        mask = torch.zeros(N, B, dtype=torch.bool, device=device)

        me = self.to_act
        opp = 1 - me
        me_stack = self.stacks[self.arange_n, me]
        opp_stack = self.stacks[self.arange_n, opp]
        me_committed = self.committed[self.arange_n, me]
        opp_committed = self.committed[self.arange_n, opp]
        to_call = opp_committed - me_committed

        # Fold if facing bet
        mask[:, 0] = to_call > 0

        # Check/Call: you can always check or call, even if it might put you all in.
        mask[:, 1] = 1

        # Pre-compute candidate concrete amounts for preset bins 2..B-2
        bet_bins_t = torch.tensor(
            [0] * 2 + bet_bins + [0], dtype=torch.float32, device=self.device
        )
        can_bet_raise = (me_stack > 0) & (opp_stack > 0)
        additional_amounts = (
            bet_bins_t[2:-1].view(1, B - 3) * self.pot.view(N, 1)
        ).long()
        bet_raise_amounts = to_call.view(N, 1) + additional_amounts  # [N, B-3]

        # Either in raise or bet case, additional amount above call must be at least min_raise.
        bet_raise_legal = (
            can_bet_raise.view(N, 1)
            & (bet_raise_amounts <= me_stack.view(N, 1))
            & (additional_amounts >= self.min_raise.view(N, 1))
        )

        # Write to amounts and mask for bins 2..B-2
        amounts[:, 2:-1] = bet_raise_amounts
        mask[:, 2:-1] = bet_raise_legal

        # All-in always available if we have chips
        amounts[:, B - 1] = me_stack
        mask[:, B - 1] = me_stack > 0

        me_allin = torch.where(self.is_allin[self.arange_n, me])[0]
        opp_allin = torch.where(self.is_allin[self.arange_n, opp])[0]
        mask[opp_allin, 0:2] = True  # only fold/call are legal when opp is all-in
        mask[opp_allin, 2:] = False
        # if both allin, this will override the opp_allin path.
        mask[me_allin, :] = False
        mask[me_allin, 1] = True  # only call is legal when we are all-in

        return amounts, mask

    def legal_bins_mask(self, bet_bins: Optional[list[float]] = None) -> torch.Tensor:
        """Return [N, B] mask of legal bins with deduplication."""
        _, mask = self.legal_bins_amounts_and_mask(bet_bins)
        return mask

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

    def legal_mask_bins_for(
        self, indices: torch.Tensor, bet_bins: Optional[list[float]] = None
    ) -> torch.Tensor:
        """Return legal mask for a subset of environments: shape [len(indices), B]."""
        if indices.numel() == 0:
            return torch.zeros(
                0, len(self.default_bet_bins) + 3, dtype=torch.bool, device=self.device
            )
        full_mask = self.legal_bins_mask(bet_bins)
        return full_mask[indices]

    # --- Sanity helpers -------------------------------------------------------
    def sanity_check(
        self,
        indices: Optional[torch.Tensor] = None,
        label: Optional[str] = None,
    ) -> None:
        """Validate deck and deck_pos bounds for all or a subset of rows.

        Args:
            indices: Optional subset of env rows to check. If None, checks all.
            label: Optional context label for assertion messages.
        """
        if indices is None:
            deck = self.deck
            deck_pos = self.deck_pos
        else:
            if indices.numel() == 0:
                return
            deck = self.deck[indices]
            deck_pos = self.deck_pos[indices]

        ctx = f" ({label})" if label else ""
        assert (deck >= 0).all() and (deck < 52).all(), (
            f"Sanity check failed{ctx}: deck out of range: "
            f"min={int(deck.min().item())}, max={int(deck.max().item())}"
        )
        # deck_pos can be 0..9 inclusive (up to 9 staged cards)
        assert (deck_pos >= 0).all() and (deck_pos <= 9).all(), (
            f"Sanity check failed{ctx}: deck_pos out of range: "
            f"min={int(deck_pos.min().item())}, max={int(deck_pos.max().item())}"
        )

    def copy_state_from(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        copy_deck: bool = True,
    ) -> None:
        """Vectorized copy of state rows from src_env[src_indices] to self[dst_indices]."""
        if src_indices.numel() == 0:
            return
        assert src_indices.shape[0] == dst_indices.shape[0]
        # Disallow overlap when copying within the same env to avoid aliasing
        if src_env is self:
            smin = int(src_indices.min().item())
            smax = int(src_indices.max().item())
            dmin = int(dst_indices.min().item())
            dmax = int(dst_indices.max().item())
            # Ranges must not overlap
            if not (smax < dmin or dmax < smin):
                raise AssertionError(
                    "copy_state_from requires non-overlapping index ranges when copying within the same env"
                )

        # Scalars / vectors
        self.button[dst_indices] = src_env.button[src_indices]
        self.street[dst_indices] = src_env.street[src_indices]
        self.to_act[dst_indices] = src_env.to_act[src_indices]
        self.pot[dst_indices] = src_env.pot[src_indices]
        self.min_raise[dst_indices] = src_env.min_raise[src_indices]
        self.actions_this_round[dst_indices] = src_env.actions_this_round[src_indices]
        self.acted_since_reset[dst_indices] = src_env.acted_since_reset[src_indices]

        # 2-player tensors
        self.stacks[dst_indices] = src_env.stacks[src_indices]
        self.committed[dst_indices] = src_env.committed[src_indices]
        self.has_folded[dst_indices] = src_env.has_folded[src_indices]
        self.is_allin[dst_indices] = src_env.is_allin[src_indices]
        self.chips_placed[dst_indices] = src_env.chips_placed[src_indices]

        # Card state
        self.board_onehot[dst_indices] = src_env.board_onehot[src_indices]
        self.hole_onehot[dst_indices] = src_env.hole_onehot[src_indices]
        self.board_indices[dst_indices] = src_env.board_indices[src_indices]
        self.hole_indices[dst_indices] = src_env.hole_indices[src_indices]

        # Done/winner
        self.done[dst_indices] = src_env.done[src_indices]
        self.winner[dst_indices] = src_env.winner[src_indices]

        if copy_deck:
            # Sanity check via helper on source rows
            src_env.sanity_check(indices=src_indices, label="copy_state_from src")
            self.deck[dst_indices] = src_env.deck[src_indices]
            self.deck_pos[dst_indices] = src_env.deck_pos[src_indices]
            # Optional: verify destination rows after copy
            self.sanity_check(indices=dst_indices, label="copy_state_from dst")

    def clone_states(
        self, dst_children: torch.Tensor, src_parents: torch.Tensor
    ) -> None:
        """Clone rows within the same env from parents to children (vectorized)."""
        if dst_children.numel() == 0:
            return
        assert dst_children.shape[0] == src_parents.shape[0]
        # Disallow overlap in ranges to avoid aliasing issues
        smin = int(src_parents.min().item())
        smax = int(src_parents.max().item())
        dmin = int(dst_children.min().item())
        dmax = int(dst_children.max().item())
        assert (
            smax < dmin or dmax < smin
        ), "clone_states requires non-overlapping index ranges between src_parents and dst_children"
        # Snapshot source slices to avoid overlap aliasing during assignment
        self.copy_state_from(self, src_parents, dst_children, copy_deck=True)

    def get_action_history(self) -> torch.Tensor:
        """Return action history planes tensor if allocated, else None."""
        # TODO: Make this work again.
        return torch.zeros(
            self.N,
            4,
            6,
            4,
            len(self.default_bet_bins) + 3,
            dtype=torch.bool,
            device=self.device,
        )

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

    def finish_and_assign_rewards(
        self, env_indices: torch.Tensor, winners: torch.Tensor
    ) -> None:
        """Assign winners to specified environments. Rewards are from p0's perspective."""
        self.winner[env_indices] = winners
        self.done[env_indices] = True

        pot = self.pot[env_indices].to(self.float_dtype)
        my_stack = self.stacks[env_indices, 0].to(self.float_dtype)
        # pot_share: full pot if winner == m, half if tie, else 0
        pot_share = torch.where(
            winners == 0,
            pot,
            torch.where(winners == 2, pot / 2.0, 0),
        )

        return (my_stack + pot_share - float(self.starting_stack)) / self.scale

    # --- Step ------------------------------------------------------------------

    def step_bins(
        self,
        bin_indices: torch.Tensor,
        bin_amounts: Optional[torch.Tensor] = None,
        legal_masks: Optional[torch.Tensor] = None,
        bet_bins: Optional[list[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if bet_bins is None:
            bet_bins = self.default_bet_bins
        if bin_amounts is None or legal_masks is None:
            bin_amounts, legal_masks = self.legal_bins_amounts_and_mask(bet_bins)

        num_bins = len(bet_bins) + 3
        all_in_index = num_bins - 1

        action_indices = torch.full_like(bin_indices, -1)
        action_indices[bin_indices == 0] = 0  # fold
        action_indices[bin_indices == 1] = 1  # check/call

        bet_mask = (bin_indices >= 2) & (bin_indices < all_in_index)
        bet_indices = torch.where(bet_mask)[0]
        bet_amounts = torch.zeros_like(bin_indices)
        if bet_indices.numel() > 0:
            bet_amounts[bet_indices] = bin_amounts[
                bet_indices, bin_indices[bet_indices]
            ]
            action_indices[bet_indices] = 2

        action_indices[bin_indices == all_in_index] = 3  # all-in

        invalid_mask = (bin_indices >= 0) & (action_indices == -1)
        if invalid_mask.any():
            raise ValueError(
                f"Received unsupported bet bin indices: {bin_indices[invalid_mask]}"
            )

        return self.step(action_indices, bet_amounts)

    def step(
        self, action_indices: torch.Tensor, bet_amounts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step all envs using discrete bet bin indices tensor [N]. -1 means no action.
        bin_indices: [N] bet bin indices
        bin_amounts: [N, B] concrete amounts for bins; -1 where not applicable
        legal_masks: [N, B] bool mask of legal bins (fold/check-call/presets/all-in)

        Returns:
          - rewards [N]
          - dones [N]
          - to_act [N]
          - new_streets [N]: -1 if street did not advance, otherwise 1=flop, 2=turn, 3=river
          - dealt_cards [N, 3]: indices of newly dealt cards this step; -1 where not applicable
        Rewards are from p0's perspective, scaled by 100bb.
        """
        assert action_indices.shape[0] == self.N
        N = self.N
        device = self.device

        actor_idx = self.to_act
        other_idx = 1 - self.to_act
        actor_stack = self.stacks.gather(1, actor_idx.view(N, 1)).squeeze(1)
        actor_committed = self.committed.gather(1, actor_idx.view(N, 1)).squeeze(1)
        other_committed = self.committed.gather(1, other_idx.view(N, 1)).squeeze(1)
        to_call = other_committed - actor_committed

        rewards = torch.zeros(N, dtype=self.float_dtype, device=device)
        # Track street advancement and cards dealt this step
        new_streets = torch.full((N,), -1, dtype=torch.long, device=device)
        dealt_cards = torch.full((N, 3), -1, dtype=torch.long, device=device)

        # Group masks by action type
        is_fold = action_indices == 0
        is_check_call = action_indices == 1
        is_bet_raise = action_indices == 2
        is_allin = action_indices == 3

        # -1 bet bin index means no action
        acting_mask = action_indices >= 0
        acting = torch.where(acting_mask)[0]
        if self.done[acting].any():
            print(
                f"Warning: trying to act while done in {self.done[acting].sum()} envs."
            )

        if (to_call[acting] < 0).any():
            print(f"Warning: to_call < 0 in {(to_call[acting] < 0).sum()} envs.")

        # We've now acted since reset
        self.acted_since_reset[acting] = True

        if self.debug_step_table:
            starting_street = self.street[:DEBUG_STEP_TABLE_ENVS].clone()
            starting_to_act = self.to_act[:DEBUG_STEP_TABLE_ENVS].clone()

        # Handle different actions.
        # Fold: immediate terminal, award pot to opp
        action_idx = torch.where(is_fold)[0]
        # Winner is opp
        rewards[action_idx] = self.finish_and_assign_rewards(
            action_idx, other_idx[action_idx]
        )

        # Check/Call
        action_idx = torch.where(is_check_call)[0]
        call_amount = torch.minimum(to_call[action_idx], actor_stack[action_idx])
        self.bet(action_idx, call_amount)

        # All-in
        action_idx = torch.where(is_allin)[0]
        all_in_amount = actor_stack[action_idx]
        self.bet(action_idx, all_in_amount)
        self.is_allin[action_idx, actor_idx[action_idx]] = True

        # Bet/Raise bins: approximate mapping using pot * mult (pot includes committed)
        # Semantics: e.g. 0.5x pot means raise ABOVE the current call amount by 0.5x pot
        # So if you bet $10, opponent raises to $20 total, half pot means you put $10 to
        # call the $20 and then half of the starting pot ($30 total => $15 raise).
        action_idx = torch.where(is_bet_raise)[0]
        bet_raise_amount = bet_amounts[action_idx]
        self.bet(action_idx, bet_raise_amount)
        self.min_raise[action_idx] = torch.maximum(
            self.min_raise[action_idx], bet_raise_amount - to_call[action_idx]
        )

        if self.debug_step_table:
            after_betting_committed = self.committed[:DEBUG_STEP_TABLE_ENVS].clone()

        # Advance to next actor (simple alternation)
        self.to_act[acting] = 1 - self.to_act[acting]
        self.actions_this_round[acting] += 1
        self.acted_since_reset[acting] = True

        # Round closure: equal committed (or 1 player all-in) and both acted
        equal_committed = self.committed[:, 0] == self.committed[:, 1]
        all_in_committed = (
            (self.is_allin[:, 0] & self.is_allin[:, 1])
            | (self.is_allin[:, 0] & (self.committed[:, 0] <= self.committed[:, 1]))
            | (self.is_allin[:, 1] & (self.committed[:, 1] <= self.committed[:, 0]))
        )
        # NB: Don't need to handle folds since then we're totally done with the hand
        round_closed = (
            acting_mask  # must have acted
            & ~self.done  # this will be newly true if we've folded.
            & (equal_committed | all_in_committed)
            & (self.actions_this_round >= 2)
        )
        round_closed_idx = torch.where(round_closed)[0]
        if round_closed_idx.numel() > 0:
            stack_difference = (
                self.stacks[round_closed_idx, 0] != self.stacks[round_closed_idx, 1]
            )
            if stack_difference.any():
                print(f"Warning: stacks not equal in {stack_difference.numel()} envs.")

            # Reset committed
            self.committed[round_closed_idx, :] = 0
            self.actions_this_round[round_closed_idx] = 0
            self.to_act[round_closed_idx] = 1 - self.button[round_closed_idx]
            self.min_raise[round_closed_idx] = (
                self.bb
            )  # Reset min_raise to big blind for new street
            # Advance street and deal
            s = self.street[round_closed_idx]

            # showdown after river (or right after flop in flop showdown mode)
            sd_mask = s == (0 if self.flop_showdown else 3)
            showdown_ids = round_closed_idx[sd_mask]
            # Evaluate winners or award folds
            # Vectorized showdown resolution for all showdown_ids at once

            # Build one-hot 7-card planes for each player: 2 hole + 5 board
            # Shape: [num_sd, 4, 13]
            N_sd = showdown_ids.numel()
            if N_sd > 0:
                ab_plane = self.hole_onehot[showdown_ids].any(
                    dim=2
                ) | self.board_onehot[showdown_ids].any(dim=1).unsqueeze(1)
                cmp = rules.compare_7_single_batch(ab_plane)  # shape [num_sd]
                # Set winners
                self.winner[showdown_ids[cmp > 0]] = 0
                self.winner[showdown_ids[cmp < 0]] = 1
                self.winner[showdown_ids[cmp == 0]] = 2

                rewards[showdown_ids] = self.finish_and_assign_rewards(
                    showdown_ids, self.winner[showdown_ids]
                )

            # flop (vectorized dealing of 3 cards)
            flop_mask = s == 0
            flop_ids = round_closed_idx[flop_mask]
            pos = self.deck_pos[flop_ids]
            # (debug assertions removed)
            self.deck_pos[flop_ids] = pos + 3
            self.board_onehot[flop_ids, 0] = self.card_onehot_cache[
                self.deck[flop_ids, pos]
            ]
            self.board_onehot[flop_ids, 1] = self.card_onehot_cache[
                self.deck[flop_ids, pos + 1]
            ]
            self.board_onehot[flop_ids, 2] = self.card_onehot_cache[
                self.deck[flop_ids, pos + 2]
            ]
            # Set flop card indices
            self.board_indices[flop_ids, 0] = self.deck[flop_ids, pos]
            self.board_indices[flop_ids, 1] = self.deck[flop_ids, pos + 1]
            self.board_indices[flop_ids, 2] = self.deck[flop_ids, pos + 2]
            # Record returned advancement/card indices
            if flop_ids.numel() > 0:
                new_streets[flop_ids] = 1
                dealt_cards[flop_ids, 0] = self.board_indices[flop_ids, 0]
                dealt_cards[flop_ids, 1] = self.board_indices[flop_ids, 1]
                dealt_cards[flop_ids, 2] = self.board_indices[flop_ids, 2]

            # turn
            turn_mask = s == 1
            turn_ids = round_closed_idx[turn_mask]
            pos = self.deck_pos[turn_ids]
            # (debug assertions removed)
            c = self.deck[turn_ids, pos]
            self.deck_pos[turn_ids] = pos + 1
            self.board_onehot[turn_ids, 3] = self.card_onehot_cache[c]
            # Set turn card index
            self.board_indices[turn_ids, 3] = c
            # Record returned advancement/card indices
            if turn_ids.numel() > 0:
                new_streets[turn_ids] = 2
                dealt_cards[turn_ids, 0] = c

            # river
            river_mask = s == 2
            river_ids = round_closed_idx[river_mask]
            pos = self.deck_pos[river_ids]
            # (debug assertions removed)
            c = self.deck[river_ids, pos]
            self.deck_pos[river_ids] = pos + 1
            # Use cached one-hot matrix for river card
            self.board_onehot[river_ids, 4] = self.card_onehot_cache[c]
            # Set river card index
            self.board_indices[river_ids, 4] = c
            # Record returned advancement/card indices
            if river_ids.numel() > 0:
                new_streets[river_ids] = 3
                dealt_cards[river_ids, 0] = c

            # Advance street
            self.street[round_closed_idx] += 1

        if self.debug_step_table:
            self._print_debug_table(
                action_indices[:3],
                rewards[:3],
                acting[:3],
                starting_street[:3],
                starting_to_act[:3],
                after_betting_committed[:3],
            )

        return rewards, self.done, self.to_act, new_streets, dealt_cards

    def _print_debug_table(
        self,
        bin_indices: torch.Tensor,
        rewards: torch.Tensor,
        active_indices: torch.Tensor,
        starting_street: torch.Tensor,
        starting_to_act: torch.Tensor,
        after_betting_committed: torch.Tensor,
    ) -> None:
        """Print a condensed debug row for all envs: [street] [actor] [action] [reward] [done] [pot] [committed] [stacks]."""
        # Convert tensors to CPU for printing
        bin_indices_cpu = bin_indices.cpu()
        rewards_cpu = rewards.cpu()
        dones_cpu = self.done.cpu()
        to_act_cpu = starting_to_act.cpu()
        street_cpu = starting_street.cpu()
        pot_cpu = self.pot.cpu()
        committed_cpu = after_betting_committed.cpu().clamp(0, 999)
        stacks_cpu = self.stacks.cpu()

        # Street codes
        # 0=preflop(p),1=flop(f),2=turn(t),3=river(r),4=showdown(s)
        street_codes = "pftrs"

        # Build compact token per env with fixed-width columns
        tokens = []
        for i in range(bin_indices_cpu.shape[0]):
            s = street_codes[int(street_cpu[i].item())]
            actor = str(int(to_act_cpu[i].item()))

            # Action
            bi = int(bin_indices_cpu[i].item())
            if bi < 0:
                act = "-"
            elif bi == 0:
                act = "f"
            elif bi == 1:
                act = "c"
            elif bi == 3:
                act = "ALL"
            else:
                # TODO: FIX
                mult_idx = bi - 2
                if 0 <= mult_idx < len(self.default_bet_bins):
                    mult = self.default_bet_bins[mult_idx]
                    # Format like 0.5, 1.0, 1.5 etc
                    act = f"{mult:.2f}".rstrip("0").rstrip(".")
                else:
                    act = f"bin{bi}"

            # Format values with proper padding
            r = f"{float(rewards_cpu[i].item()):6.3f}"
            done = "✓" if bool(dones_cpu[i].item()) else "✗"
            pot = f"{int(pot_cpu[i].item()):4d}"
            comm0 = f"{int(committed_cpu[i, 0].item()):3d}"
            comm1 = f"{int(committed_cpu[i, 1].item()):3d}"
            stack0 = f"{int(stacks_cpu[i, 0].item()):4d}"
            stack1 = f"{int(stacks_cpu[i, 1].item()):4d}"

            # Format with fixed widths: street(1), actor(1), action(4), reward(6), done(1), pot(4), committed(8), stacks(12)
            if s == " ":
                token = " " * 44
            elif (
                self.street[i].item() == 4 and abs(rewards_cpu[i].item()) < 0.0001
            ):  # showdown - show hands instead of normal row
                hands_str = self._format_hands_for_debug(i)
                token = f"s {hands_str}"
            else:
                token = f"{s:1} {actor:1} {act:<4} {r:6} {done:1} {pot:4} ({comm0},{comm1}) ({stack0},{stack1})"
            tokens.append(token)

        # Print header showing env indices once per row for readability
        print(
            " | ".join(tokens)
            + " | "
            + " ".join(str(i) for i in active_indices.tolist())
        )

    def _format_hands_for_debug(self, env_idx: int) -> str:
        """Format the two hands for debug output at showdown."""

        # Convert hole cards to readable format
        def card_to_str(card_onehot):
            # card_onehot is [4, 13] - find the suit and rank
            suit_idx = card_onehot.sum(dim=1).argmax().item()
            rank_idx = card_onehot.sum(dim=0).argmax().item()

            suits = ["♠", "♥", "♦", "♣"]
            ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

            return f"{ranks[rank_idx]}{suits[suit_idx]}"

        # Get hole cards for both players
        hole_cards = self.hole_onehot[env_idx]  # [2, 2, 4, 13]

        # Player 0 hand
        p0_card1 = card_to_str(hole_cards[0, 0])
        p0_card2 = card_to_str(hole_cards[0, 1])

        # Player 1 hand
        p1_card1 = card_to_str(hole_cards[1, 0])
        p1_card2 = card_to_str(hole_cards[1, 1])

        # Get board cards
        board_cards = []
        board_onehot = self.board_onehot[env_idx]  # [5, 4, 13]
        for i in range(5):
            if board_onehot[i].sum() > 0:  # Card is present
                board_cards.append(card_to_str(board_onehot[i]))

        board_str = " ".join(board_cards) if board_cards else "No board"

        return (
            f"     {p0_card1} {p0_card2} = {board_str} = {p1_card1} {p1_card2}       "
        )

    def get_hole_card_indices(self, env_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get hole card indices for both players in a specific environment.

        Args:
            env_idx: Environment index

        Returns:
            Tuple of (player_0_cards, player_1_cards) where each is [2] tensor of card indices
        """
        p0_cards = self.hole_indices[env_idx, 0]  # [2]
        p1_cards = self.hole_indices[env_idx, 1]  # [2]
        return p0_cards, p1_cards

    def get_board_card_indices(self, env_idx: int) -> torch.Tensor:
        """Get board card indices for a specific environment.

        Args:
            env_idx: Environment index

        Returns:
            Tensor of board card indices [5], with -1 for empty slots
        """
        return self.board_indices[env_idx]  # [5]

    def get_visible_card_indices(self, env_idx: int, player: int) -> torch.Tensor:
        """Get all visible card indices for a specific player in a specific environment.

        Args:
            env_idx: Environment index
            player: Player index (0 or 1)

        Returns:
            Tensor of visible card indices, including hole cards and board cards
        """
        hole_cards = self.hole_indices[env_idx, player]  # [2]
        board_cards = self.board_indices[env_idx]  # [5]

        # Filter out -1 values (empty slots)
        hole_visible = hole_cards[hole_cards >= 0]
        board_visible = board_cards[board_cards >= 0]

        return torch.cat([hole_visible, board_visible])

    def get_all_card_indices(self, env_idx: int) -> torch.Tensor:
        """Get all card indices (hole + board) for a specific environment.

        Args:
            env_idx: Environment index

        Returns:
            Tensor of all card indices, including hole cards and board cards
        """
        hole_cards = self.hole_indices[env_idx].flatten()  # [4]
        board_cards = self.board_indices[env_idx]  # [5]

        # Filter out -1 values (empty slots)
        hole_visible = hole_cards[hole_cards >= 0]
        board_visible = board_cards[board_cards >= 0]

        return torch.cat([hole_visible, board_visible])

    def reset_done(self) -> None:
        """Reset only environments with done=True; keeps others unchanged."""
        ids = torch.where(self.done)[0]
        if ids.numel() == 0:
            return  # Nothing to reset
        self.reset(env_indices=ids)

    # --- Utility: slicing ------------------------------------------------------
    def __getitem__(self, indices: torch.Tensor | slice) -> "HUNLTensorEnv":
        """Return a new env containing copies of the selected rows.

        This creates a fresh HUNLTensorEnv of size len(indices) and copies state
        from the current env into it. The copy is deep for all state tensors.
        """
        if isinstance(indices, slice):
            indices = torch.arange(self.N, device=self.device)[indices]
        k = indices.numel()
        # Sanity: validate source rows before copy
        # Remove heavy debug asserts now that bug is fixed
        dst = HUNLTensorEnv(
            num_envs=k,
            starting_stack=self.starting_stack,
            sb=self.sb,
            bb=self.bb,
            default_bet_bins=self.default_bet_bins,
            device=self.device,
            rng=self.rng,
            float_dtype=self.float_dtype,
            debug_step_table=self.debug_step_table,
            flop_showdown=self.flop_showdown,
        )
        dst.copy_state_from(
            self, indices, torch.arange(k, device=self.device), copy_deck=True
        )
        return dst
