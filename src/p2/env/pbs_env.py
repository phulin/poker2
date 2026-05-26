from __future__ import annotations

import torch

from p2.env.hunl_tensor_env import DEFAULT_BET_BINS


class PBSEnv:
    """Vectorized public-belief-state no-limit Hold 'Em environment.

    The environment tracks multi-player public betting state and public board
    cards only. It intentionally does not deal private cards to seats.
    """

    N: int
    num_players: int
    mean_stack: int
    sb: int
    bb: int
    device: torch.device
    rng: torch.Generator

    def __init__(
        self,
        num_envs: int,
        num_players: int = 6,
        mean_stack: int | None = None,
        starting_stack: int | None = None,
        sb: int = 50,
        bb: int = 100,
        default_bet_bins: list[float] | None = None,
        device: torch.device | str | None = None,
        rng: torch.Generator | None = None,
        float_dtype: torch.dtype = torch.float32,
        init_state: bool = True,
    ) -> None:
        if num_envs < 0:
            raise ValueError("num_envs must be non-negative")
        if num_players < 2:
            raise ValueError("num_players must be at least 2")
        if mean_stack is None:
            if starting_stack is None:
                raise ValueError("mean_stack is required when starting_stack is not set")
            mean_stack = starting_stack
        elif starting_stack is not None and starting_stack != mean_stack:
            raise ValueError("mean_stack and starting_stack must match when both set")
        if mean_stack < bb:
            raise ValueError("mean_stack must cover the big blind")

        self.N = int(num_envs)
        self.num_players = int(num_players)
        self.mean_stack = int(mean_stack)
        self.sb = int(sb)
        self.bb = int(bb)
        self.default_bet_bins = default_bet_bins or DEFAULT_BET_BINS
        self.num_bet_bins = len(self.default_bet_bins) + 3
        self.device = torch.device(device) if device is not None else torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.float_dtype = float_dtype
        self.rng = rng if rng is not None else torch.Generator(device=self.device)

        self.arange_n = torch.arange(self.N, device=self.device)
        self.players = torch.arange(self.num_players, device=self.device)
        self.deck = torch.empty(self.N, 5, dtype=torch.long, device=self.device)
        self.deck_pos = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.button = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.street = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.to_act = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.last_to_act = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.pot = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.min_raise = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.last_aggressive_amount = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.actions_this_round = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.actions_last_round = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.stacks = torch.empty(self.N, self.num_players, dtype=torch.long, device=self.device)
        self.starting_stacks = torch.empty_like(self.stacks)
        self.scale = torch.empty(self.N, dtype=self.float_dtype, device=self.device)
        self.committed = torch.empty_like(self.stacks)
        self.chips_placed = torch.empty_like(self.stacks)
        self.has_folded = torch.empty(self.N, self.num_players, dtype=torch.bool, device=self.device)
        self.is_allin = torch.empty_like(self.has_folded)
        self.acted_this_round = torch.empty_like(self.has_folded)
        self.done = torch.empty(self.N, dtype=torch.bool, device=self.device)
        self.winner = torch.empty(self.N, dtype=torch.long, device=self.device)
        self.winners = torch.empty_like(self.has_folded)
        self.board_indices = torch.empty(self.N, 5, dtype=torch.long, device=self.device)
        self.last_board_indices = torch.empty_like(self.board_indices)
        self.board_onehot = torch.empty(self.N, 5, 4, 13, dtype=torch.bool, device=self.device)

        cards = torch.arange(52, device=self.device)
        self.card_onehot_cache = torch.zeros(52, 4, 13, dtype=torch.bool, device=self.device)
        self.card_onehot_cache[cards, cards // 13, cards % 13] = True

        if init_state:
            self.deck.zero_()
            self.deck_pos.zero_()
            self.button.zero_()
            self.street.zero_()
            self.to_act.zero_()
            self.last_to_act.zero_()
            self.pot.zero_()
            self.min_raise.zero_()
            self.last_aggressive_amount.zero_()
            self.actions_this_round.zero_()
            self.actions_last_round.zero_()
            self.stacks.zero_()
            self.starting_stacks.fill_(self.mean_stack)
            self.scale.fill_(float(self.mean_stack))
            self.committed.zero_()
            self.chips_placed.zero_()
            self.has_folded.zero_()
            self.is_allin.zero_()
            self.acted_this_round.zero_()
            self.done.zero_()
            self.winner.fill_(-1)
            self.winners.zero_()
            self.board_indices.fill_(-1)
            self.last_board_indices.fill_(-1)
            self.board_onehot.zero_()

    @classmethod
    def from_proto(
        cls, proto: PBSEnv, num_envs: int | None = None, init_state: bool = True
    ) -> PBSEnv:
        return PBSEnv(
            num_envs=proto.N if num_envs is None else num_envs,
            num_players=proto.num_players,
            mean_stack=proto.mean_stack,
            sb=proto.sb,
            bb=proto.bb,
            default_bet_bins=proto.default_bet_bins,
            device=proto.device,
            rng=proto.rng,
            float_dtype=proto.float_dtype,
            init_state=init_state,
        )

    def reset(
        self,
        env_indices: torch.Tensor | None = None,
        force_button: torch.Tensor | None = None,
        force_deck: torch.Tensor | None = None,
    ) -> None:
        ids = self.arange_n if env_indices is None else env_indices
        if ids.numel() == 0:
            return
        num_reset = ids.numel()

        if force_deck is None:
            scores = torch.rand(num_reset, 52, generator=self.rng, device=self.device)
            self.deck[ids] = torch.topk(scores, 5, dim=1).indices
        else:
            if force_deck.shape != (num_reset, 5):
                raise ValueError("force_deck must have shape [num_reset, 5]")
            self.deck[ids] = force_deck

        button = force_button if force_button is not None else torch.randint(
            0, self.num_players, (num_reset,), generator=self.rng, device=self.device
        )
        sb_player = self._small_blind_player(button)
        bb_player = self._big_blind_player(button)
        rows = torch.arange(num_reset, device=self.device)

        self.button[ids] = button
        self.street[ids] = 0
        self.to_act[ids] = self._preflop_first_actor(button)
        self.last_to_act[ids] = self.to_act[ids]
        self.pot[ids] = self.sb + self.bb
        self.min_raise[ids] = self.bb
        self.last_aggressive_amount[ids] = self.bb
        self.actions_this_round[ids] = 0
        self.actions_last_round[ids] = 0
        self.deck_pos[ids] = 0

        self.starting_stacks[ids] = self.mean_stack
        self.scale[ids] = float(self.mean_stack)
        self.stacks[ids] = self.mean_stack
        self.committed[ids] = 0
        self.chips_placed[ids] = 0
        self.stacks[ids[rows], sb_player] -= self.sb
        self.stacks[ids[rows], bb_player] -= self.bb
        self.committed[ids[rows], sb_player] = self.sb
        self.committed[ids[rows], bb_player] = self.bb
        self.chips_placed[ids[rows], sb_player] = self.sb
        self.chips_placed[ids[rows], bb_player] = self.bb
        self.has_folded[ids] = False
        self.is_allin[ids] = self.stacks[ids] == 0
        self.acted_this_round[ids] = False
        self.done[ids] = False
        self.winner[ids] = -1
        self.winners[ids] = False
        self.board_indices[ids] = -1
        self.last_board_indices[ids] = -1
        self.board_onehot[ids] = False

    def legal_bins_amounts_and_mask(
        self, bet_bins: list[float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if bet_bins is None:
            bet_bins = self.default_bet_bins
        B = len(bet_bins) + 3
        amounts = torch.full((self.N, B), -1, dtype=torch.long, device=self.device)
        mask = torch.zeros(self.N, B, dtype=torch.bool, device=self.device)

        actor = self.to_act
        actor_stack = self.stacks.gather(1, actor[:, None]).squeeze(1)
        actor_committed = self.committed.gather(1, actor[:, None]).squeeze(1)
        max_committed = self.committed.amax(dim=1)
        to_call = max_committed - actor_committed
        can_act = (
            ~self.done
            & ~self.has_folded.gather(1, actor[:, None]).squeeze(1)
            & ~self.is_allin.gather(1, actor[:, None]).squeeze(1)
            & (actor_stack > 0)
        )

        mask[:, 0] = can_act & (to_call > 0)
        amounts[:, 1] = torch.minimum(to_call, actor_stack)
        mask[:, 1] = can_act

        bet_bins_t = torch.tensor(bet_bins, dtype=torch.float32, device=self.device)
        additional = (self.pot[:, None].to(torch.float32) * bet_bins_t[None, :]).long()
        bet_amounts = to_call[:, None] + additional
        amounts[:, 2:-1] = bet_amounts

        live_can_respond = (
            ~self.has_folded
            & ~self.is_allin
            & (self.stacks > 0)
            & (self.players[None, :] != actor[:, None])
        ).any(dim=1)
        raise_increment = bet_amounts - to_call[:, None]
        preset_legal = (
            can_act[:, None]
            & live_can_respond[:, None]
            & (actor_stack[:, None] > to_call[:, None])
            & (bet_amounts < actor_stack[:, None])
            & (raise_increment >= self.min_raise[:, None])
        )
        mask[:, 2:-1] = preset_legal
        amounts[:, -1] = actor_stack
        mask[:, -1] = can_act & (actor_stack > 0)
        return amounts, mask

    def legal_bins_mask(self, bet_bins: list[float] | None = None) -> torch.Tensor:
        _, mask = self.legal_bins_amounts_and_mask(bet_bins)
        return mask

    def legal_mask_bins_for(
        self, indices: torch.Tensor, bet_bins: list[float] | None = None
    ) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.zeros(0, self.num_bet_bins, dtype=torch.bool, device=self.device)
        return self.legal_bins_mask(bet_bins)[indices]

    def active_indices(self) -> torch.Tensor:
        return torch.where(~self.done)[0]

    def states_summary(self) -> dict[str, torch.Tensor]:
        return {
            "street": self.street,
            "to_act": self.to_act,
            "pot": self.pot,
            "min_raise": self.min_raise,
            "done": self.done,
        }

    def step_bins(
        self,
        bin_indices: torch.Tensor,
        bin_amounts: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None,
        bet_bins: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bet_bins is None:
            bet_bins = self.default_bet_bins
        if bin_amounts is None:
            bin_amounts, legal_masks = self.legal_bins_amounts_and_mask(bet_bins)
        B = len(bet_bins) + 3
        all_in_idx = B - 1
        clamped_bins = bin_indices.clamp(0, all_in_idx)
        selected_amounts = bin_amounts.gather(1, clamped_bins[:, None]).squeeze(1)
        action_indices = torch.full_like(bin_indices, -1)
        action_indices = torch.where(bin_indices == 0, torch.zeros_like(action_indices), action_indices)
        action_indices = torch.where(bin_indices == 1, torch.ones_like(action_indices), action_indices)
        action_indices = torch.where(
            (bin_indices >= 2) & (bin_indices < all_in_idx),
            torch.full_like(action_indices, 2),
            action_indices,
        )
        action_indices = torch.where(
            bin_indices == all_in_idx,
            torch.full_like(action_indices, 3),
            action_indices,
        )
        return self.step(action_indices, selected_amounts)

    def step(
        self, action_indices: torch.Tensor, bet_amounts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action_indices.shape[0] != self.N:
            raise ValueError("action_indices must have shape [N]")
        actor = self.to_act
        rows = self.arange_n
        actor_stack = self.stacks.gather(1, actor[:, None]).squeeze(1)
        actor_committed = self.committed.gather(1, actor[:, None]).squeeze(1)
        max_committed_before = self.committed.amax(dim=1)
        to_call = max_committed_before - actor_committed
        acting = (action_indices >= 0) & ~self.done
        is_fold = acting & (action_indices == 0)
        is_call = acting & (action_indices == 1)
        is_bet_raise = acting & (action_indices == 2)
        is_allin = acting & (action_indices == 3)

        call_amount = torch.minimum(to_call, actor_stack)
        put_in = torch.zeros(self.N, dtype=torch.long, device=self.device)
        put_in = torch.where(is_call, call_amount, put_in)
        put_in = torch.where(is_bet_raise, bet_amounts.clamp_min(0).minimum(actor_stack), put_in)
        put_in = torch.where(is_allin, actor_stack, put_in)

        self.last_to_act[acting] = self.to_act[acting]
        self.stacks[rows, actor] -= put_in
        self.committed[rows, actor] += put_in
        self.chips_placed[rows, actor] += put_in
        self.pot += put_in
        self.has_folded[rows, actor] |= is_fold

        actor_stack_after = self.stacks.gather(1, actor[:, None]).squeeze(1)
        now_allin = acting & (put_in > 0) & (actor_stack_after == 0)
        self.is_allin[rows, actor] |= now_allin

        new_actor_committed = self.committed.gather(1, actor[:, None]).squeeze(1)
        raise_increment = new_actor_committed - max_committed_before
        aggressive = acting & (raise_increment > 0) & (is_bet_raise | is_allin)
        self.min_raise = torch.where(
            aggressive, torch.maximum(self.min_raise, raise_increment), self.min_raise
        )
        self.last_aggressive_amount = torch.where(
            aggressive, new_actor_committed, self.last_aggressive_amount
        )

        acted_after = self.acted_this_round.clone()
        acted_after[rows, actor] |= acting
        reset_acted = torch.zeros_like(acted_after)
        reset_acted[rows, actor] = acting
        self.acted_this_round = torch.where(aggressive[:, None], reset_acted, acted_after)
        self.actions_this_round += acting.to(torch.long)
        self.last_board_indices[acting] = self.board_indices[acting]

        live = ~self.has_folded
        live_count = live.sum(dim=1)
        terminal_fold = acting & (live_count <= 1)
        single_winner = live.to(torch.long).argmax(dim=1)
        self.done |= terminal_fold
        self.winner = torch.where(terminal_fold, single_winner, self.winner)
        self.winners |= terminal_fold[:, None] & live

        eligible = live & ~self.is_allin & (self.stacks > 0)
        eligible_count = eligible.sum(dim=1)
        max_committed = self.committed.amax(dim=1)
        matched = self.committed == max_committed[:, None]
        all_ready = ((~eligible) | (self.acted_this_round & matched)).all(dim=1)
        no_more_betting = acting & ~self.done & (live_count > 1) & (eligible_count == 0)
        round_closed = (
            acting
            & ~self.done
            & (eligible_count > 0)
            & all_ready
        )

        rewards = torch.zeros(self.N, self.num_players, dtype=self.float_dtype, device=self.device)
        new_streets = torch.full((self.N,), -1, dtype=torch.long, device=self.device)
        dealt_cards = torch.full((self.N, 5), -1, dtype=torch.long, device=self.device)
        rewards = self._assign_fold_rewards(rewards, terminal_fold)

        rewards = self._finish_no_more_betting(
            rewards, no_more_betting, new_streets, dealt_cards
        )
        rewards = self._advance_closed_rounds(
            rewards, round_closed, new_streets, dealt_cards
        )

        can_act = ~self.has_folded & ~self.is_allin & (self.stacks > 0)
        self.to_act = torch.where(
            acting & ~self.done,
            self._next_player_after(actor, can_act),
            self.to_act,
        )
        self.to_act = torch.where(
            round_closed & ~self.done,
            self._next_player_after(self.button, can_act),
            self.to_act,
        )
        return rewards, new_streets, dealt_cards

    def finish_and_assign_rewards(
        self, env_indices: torch.Tensor, winners: torch.Tensor
    ) -> torch.Tensor:
        winners_mask = torch.zeros(
            env_indices.numel(), self.num_players, dtype=torch.bool, device=self.device
        )
        winners_mask[torch.arange(env_indices.numel(), device=self.device), winners] = True
        self.done[env_indices] = True
        self.winner[env_indices] = winners
        self.winners[env_indices] = winners_mask
        rewards = torch.zeros(self.N, self.num_players, dtype=self.float_dtype, device=self.device)
        return self._reward_rows(env_indices, winners_mask, rewards)[env_indices]

    def reset_done(self) -> None:
        self.reset(torch.where(self.done)[0])

    def copy_state_from(
        self,
        src_env: PBSEnv,
        src_select: torch.Tensor | slice,
        dest_select: torch.Tensor | slice,
        copy_deck: bool = True,
    ) -> None:
        fields = (
            "button",
            "street",
            "to_act",
            "last_to_act",
            "pot",
            "min_raise",
            "last_aggressive_amount",
            "actions_this_round",
            "actions_last_round",
            "stacks",
            "starting_stacks",
            "scale",
            "committed",
            "chips_placed",
            "has_folded",
            "is_allin",
            "acted_this_round",
            "done",
            "winner",
            "winners",
            "board_indices",
            "last_board_indices",
            "board_onehot",
            "deck_pos",
        )
        for field in fields:
            getattr(self, field)[dest_select] = getattr(src_env, field)[src_select]
        if copy_deck:
            self.deck[dest_select] = src_env.deck[src_select]

    def repeat_interleave(self, repeats: torch.Tensor, output_size: int) -> PBSEnv:
        indices = self.arange_n.repeat_interleave(repeats.to(self.device), output_size=output_size)
        return self.gather_rows(indices)

    def gather_rows(self, indices: torch.Tensor) -> PBSEnv:
        dst = PBSEnv.from_proto(self, num_envs=indices.numel())
        dst.copy_state_from(self, indices, torch.arange(indices.numel(), device=self.device))
        return dst

    def __getitem__(self, indices: torch.Tensor | slice) -> PBSEnv:
        if isinstance(indices, slice):
            indices = self.arange_n[indices]
        return self.gather_rows(indices)

    def _small_blind_player(self, button: torch.Tensor) -> torch.Tensor:
        if self.num_players == 2:
            return button
        return (button + 1) % self.num_players

    def _big_blind_player(self, button: torch.Tensor) -> torch.Tensor:
        offset = 1 if self.num_players == 2 else 2
        return (button + offset) % self.num_players

    def _preflop_first_actor(self, button: torch.Tensor) -> torch.Tensor:
        if self.num_players == 2:
            return self._small_blind_player(button)
        return (self._big_blind_player(button) + 1) % self.num_players

    def _next_player_after(self, start: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(1, self.num_players + 1, device=self.device)
        candidates = (start[:, None] + offsets[None, :]) % self.num_players
        candidate_ok = eligible.gather(1, candidates)
        first = candidate_ok.to(torch.long).argmax(dim=1)
        has_any = candidate_ok.any(dim=1)
        next_player = candidates.gather(1, first[:, None]).squeeze(1)
        return torch.where(has_any, next_player, start)

    def _assign_fold_rewards(
        self, rewards: torch.Tensor, terminal_fold: torch.Tensor
    ) -> torch.Tensor:
        ids = torch.where(terminal_fold)[0]
        winners = self.winners[ids]
        return self._reward_rows(ids, winners, rewards)

    def _reward_rows(
        self, ids: torch.Tensor, winners: torch.Tensor, rewards: torch.Tensor
    ) -> torch.Tensor:
        winner_count = winners.sum(dim=1).clamp_min(1).to(self.float_dtype)
        pot_share = self.pot[ids, None].to(self.float_dtype) / winner_count[:, None]
        rewards[ids] = (
            self.stacks[ids].to(self.float_dtype)
            + torch.where(winners, pot_share, torch.zeros_like(pot_share))
            - self.starting_stacks[ids].to(self.float_dtype)
        ) / self.scale[ids, None]
        return rewards

    def _finish_no_more_betting(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        new_streets: torch.Tensor,
        dealt_cards: torch.Tensor,
    ) -> torch.Tensor:
        ids = torch.where(mask)[0]
        self.board_indices[ids] = self.deck[ids]
        self.board_onehot[ids] = self.card_onehot_cache[self.deck[ids]]
        self.deck_pos[ids] = 5
        self.street[ids] = 4
        self.done[ids] = True
        live = ~self.has_folded[ids]
        self.winners[ids] = live
        sole = live.sum(dim=1) == 1
        self.winner[ids] = torch.where(sole, live.to(torch.long).argmax(dim=1), -1)
        new_streets[ids] = 4
        dealt_cards[ids] = self.deck[ids]
        return self._reward_rows(ids, live, rewards)

    def _advance_closed_rounds(
        self,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        new_streets: torch.Tensor,
        dealt_cards: torch.Tensor,
    ) -> torch.Tensor:
        ids = torch.where(mask)[0]
        old_street = self.street[ids]
        self.committed[ids] = 0
        self.actions_last_round[ids] = self.actions_this_round[ids]
        self.actions_this_round[ids] = 0
        self.acted_this_round[ids] = False
        self.min_raise[ids] = self.bb
        self.last_aggressive_amount[ids] = 0

        river_done = old_street == 3
        river_ids = ids[river_done]
        live = ~self.has_folded[river_ids]
        self.done[river_ids] = True
        self.street[river_ids] = 4
        self.winners[river_ids] = live
        sole = live.sum(dim=1) == 1
        self.winner[river_ids] = torch.where(sole, live.to(torch.long).argmax(dim=1), -1)
        new_streets[river_ids] = 4
        rewards = self._reward_rows(river_ids, live, rewards)

        flop_ids = ids[old_street == 0]
        self._deal_flop(flop_ids, new_streets, dealt_cards)
        turn_ids = ids[old_street == 1]
        self._deal_one(turn_ids, 3, 2, new_streets, dealt_cards)
        river_card_ids = ids[old_street == 2]
        self._deal_one(river_card_ids, 4, 3, new_streets, dealt_cards)
        return rewards

    def _deal_flop(
        self, ids: torch.Tensor, new_streets: torch.Tensor, dealt_cards: torch.Tensor
    ) -> None:
        pos = self.deck_pos[ids]
        cards = self.deck[ids[:, None], pos[:, None] + torch.arange(3, device=self.device)]
        self.board_indices[ids, :3] = cards
        self.board_onehot[ids, :3] = self.card_onehot_cache[cards]
        self.deck_pos[ids] = pos + 3
        self.street[ids] = 1
        new_streets[ids] = 1
        dealt_cards[ids, :3] = cards

    def _deal_one(
        self,
        ids: torch.Tensor,
        board_slot: int,
        street_value: int,
        new_streets: torch.Tensor,
        dealt_cards: torch.Tensor,
    ) -> None:
        pos = self.deck_pos[ids]
        card = self.deck[ids, pos]
        self.board_indices[ids, board_slot] = card
        self.board_onehot[ids, board_slot] = self.card_onehot_cache[card]
        self.deck_pos[ids] = pos + 1
        self.street[ids] = street_value
        new_streets[ids] = street_value
        dealt_cards[ids, 0] = card
