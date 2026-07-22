"""Triton kernels for fused sparse subgame construction."""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def triton_is_available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _legal_counts_kernel(
        to_act,
        pot,
        min_raise,
        actions_this_round,
        stacks,
        committed,
        is_allin,
        done,
        allin_leaf,
        bet_bins,
        legal_mask,
        child_counts,
        parent_start,
        parent_count,
        stop_new_street: tl.constexpr,
        num_actions: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        local_mask = offs < parent_count
        row = parent_start + offs

        actor = tl.load(to_act + row, mask=local_mask, other=0)
        opp = 1 - actor
        me_stack = tl.load(stacks + row * 2 + actor, mask=local_mask, other=0)
        opp_stack = tl.load(stacks + row * 2 + opp, mask=local_mask, other=0)
        me_committed = tl.load(committed + row * 2 + actor, mask=local_mask, other=0)
        opp_committed = tl.load(committed + row * 2 + opp, mask=local_mask, other=0)
        to_call = opp_committed - me_committed
        pot_v = tl.load(pot + row, mask=local_mask, other=0)
        min_raise_v = tl.load(min_raise + row, mask=local_mask, other=0)
        me_allin = tl.load(is_allin + row * 2 + actor, mask=local_mask, other=0)
        opp_allin = tl.load(is_allin + row * 2 + opp, mask=local_mask, other=0)
        row_done = tl.load(done + row, mask=local_mask, other=1)
        row_allin_leaf = tl.load(allin_leaf + row, mask=local_mask, other=1)
        actions = tl.load(actions_this_round + row, mask=local_mask, other=0)
        active = local_mask & (~row_done) & (~row_allin_leaf)
        if stop_new_street:
            active = active & (actions != 0)

        count = tl.zeros((block,), dtype=tl.int64)
        for action in tl.static_range(0, num_actions):
            legal = tl.full((block,), False, dtype=tl.int1)
            if action == 0:
                legal = to_call > 0
            elif action == 1:
                legal = tl.full((block,), True, dtype=tl.int1)
            elif action == num_actions - 1:
                legal = me_stack > 0
            else:
                additional = (tl.load(bet_bins + action) * pot_v.to(tl.float32)).to(
                    tl.int64
                )
                amount = to_call + additional
                legal = (
                    (me_stack > 0)
                    & (opp_stack > 0)
                    & (amount <= me_stack)
                    & (additional >= min_raise_v)
                )

            legal = tl.where(
                opp_allin,
                (action == 0) | (action == 1),
                legal,
            )
            legal = tl.where(me_allin, action == 1, legal)
            legal = legal & active
            tl.store(
                legal_mask + row * num_actions + action,
                legal,
                mask=local_mask,
            )
            count += legal.to(tl.int64)

        tl.store(child_counts + offs, count, mask=local_mask)

    @triton.jit
    def _write_children_same_street_kernel(
        deck,
        deck_pos,
        button,
        street,
        to_act,
        last_to_act,
        pot,
        min_raise,
        last_aggressive_amount,
        actions_this_round,
        actions_last_round,
        acted_since_reset,
        stacks,
        committed,
        has_folded,
        is_allin,
        starting_stacks,
        scale,
        board_indices,
        last_board_indices,
        hole_indices,
        chips_placed,
        done,
        winner,
        legal_mask,
        child_offsets,
        parent_index,
        action_from_parent,
        rewards,
        allin_leaf,
        bet_bins,
        parent_start,
        parent_count,
        dst_start,
        bb,
        mean_stack,
        allin_abstraction: tl.constexpr,
        flop_showdown: tl.constexpr,
        num_actions: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        local_mask = offs < parent_count
        src = parent_start + offs
        child_base = dst_start + tl.load(child_offsets + offs, mask=local_mask, other=0)

        src_actor = tl.load(to_act + src, mask=local_mask, other=0)
        src_opp = 1 - src_actor
        src_button = tl.load(button + src, mask=local_mask, other=0)
        src_street = tl.load(street + src, mask=local_mask, other=0)
        src_pot = tl.load(pot + src, mask=local_mask, other=0)
        src_min_raise = tl.load(min_raise + src, mask=local_mask, other=0)
        src_last_aggressive = tl.load(
            last_aggressive_amount + src, mask=local_mask, other=0
        )
        src_actions = tl.load(actions_this_round + src, mask=local_mask, other=0)
        src_actions_last = tl.load(actions_last_round + src, mask=local_mask, other=0)
        src_deck_pos = tl.load(deck_pos + src, mask=local_mask, other=0)
        src_winner = tl.load(winner + src, mask=local_mask, other=-1)
        src_done = tl.load(done + src, mask=local_mask, other=0)
        src_scale = tl.load(scale + src, mask=local_mask, other=1.0)
        actor_stack = tl.load(stacks + src * 2 + src_actor, mask=local_mask, other=0)
        actor_committed = tl.load(
            committed + src * 2 + src_actor, mask=local_mask, other=0
        )
        other_committed = tl.load(
            committed + src * 2 + src_opp, mask=local_mask, other=0
        )
        to_call = other_committed - actor_committed

        rank = tl.zeros((block,), dtype=tl.int64)
        for action in tl.static_range(0, num_actions):
            legal = tl.load(
                legal_mask + src * num_actions + action,
                mask=local_mask,
                other=0,
            )
            dst = child_base + rank
            write = local_mask & legal

            new_pot = src_pot
            new_min_raise = src_min_raise
            new_last_aggressive = src_last_aggressive
            new_actions = src_actions + 1
            new_actions_last = src_actions_last
            new_to_act = src_opp
            new_last_to_act = src_actor
            new_done = src_done
            new_winner = src_winner
            reward = tl.zeros((block,), dtype=tl.float32)

            stack0 = tl.load(stacks + src * 2 + 0, mask=local_mask, other=0)
            stack1 = tl.load(stacks + src * 2 + 1, mask=local_mask, other=0)
            committed0 = tl.load(committed + src * 2 + 0, mask=local_mask, other=0)
            committed1 = tl.load(committed + src * 2 + 1, mask=local_mask, other=0)
            chips0 = tl.load(chips_placed + src * 2 + 0, mask=local_mask, other=0)
            chips1 = tl.load(chips_placed + src * 2 + 1, mask=local_mask, other=0)
            allin0 = tl.load(is_allin + src * 2 + 0, mask=local_mask, other=0)
            allin1 = tl.load(is_allin + src * 2 + 1, mask=local_mask, other=0)

            chips = tl.zeros((block,), dtype=tl.int64)
            if action == 0:
                new_done = tl.full((block,), True, dtype=tl.int1)
                new_winner = src_opp
                pot_share = tl.where(src_opp == 0, src_pot, 0).to(tl.float32)
                p0_start = tl.load(
                    starting_stacks + src * 2 + 0, mask=local_mask, other=mean_stack
                ).to(tl.float32)
                reward = (stack0.to(tl.float32) + pot_share - p0_start) / src_scale
            elif action == 1:
                chips = tl.minimum(to_call, actor_stack)
            elif action == num_actions - 1:
                chips = actor_stack
                allin0 = tl.where(src_actor == 0, True, allin0)
                allin1 = tl.where(src_actor == 1, True, allin1)
            else:
                additional = (tl.load(bet_bins + action) * src_pot.to(tl.float32)).to(
                    tl.int64
                )
                chips = to_call + additional
                new_min_raise = tl.maximum(src_min_raise, chips - to_call)

            stack0 = tl.where(src_actor == 0, stack0 - chips, stack0)
            stack1 = tl.where(src_actor == 1, stack1 - chips, stack1)
            committed0 = tl.where(src_actor == 0, committed0 + chips, committed0)
            committed1 = tl.where(src_actor == 1, committed1 + chips, committed1)
            chips0 = tl.where(src_actor == 0, chips0 + chips, chips0)
            chips1 = tl.where(src_actor == 1, chips1 + chips, chips1)
            actor_stack_after = tl.where(src_actor == 0, stack0, stack1)
            implicit_allin = (chips > 0) & (actor_stack_after == 0)
            allin0 = tl.where(implicit_allin & (src_actor == 0), True, allin0)
            allin1 = tl.where(implicit_allin & (src_actor == 1), True, allin1)
            new_pot = new_pot + chips
            aggressive = ((action > 1) & (action < num_actions - 1)) | (
                (action == num_actions - 1) & (chips > to_call)
            )
            actor_committed_after = tl.where(src_actor == 0, committed0, committed1)
            new_last_aggressive = tl.where(
                aggressive, actor_committed_after, new_last_aggressive
            )

            equal_committed = committed0 == committed1
            allin_committed = (
                (allin0 & allin1)
                | (allin0 & (committed0 <= committed1))
                | (allin1 & (committed1 <= committed0))
            )
            round_closed = (
                (~new_done) & (equal_committed | allin_committed) & (new_actions >= 2)
            )
            if flop_showdown:
                showdown = round_closed & (src_street == 0)
            else:
                showdown = round_closed & (src_street == 3)
            committed0 = tl.where(round_closed, 0, committed0)
            committed1 = tl.where(round_closed, 0, committed1)
            new_actions_last = tl.where(round_closed, new_actions, new_actions_last)
            new_actions = tl.where(round_closed, 0, new_actions)
            new_to_act = tl.where(round_closed, 1 - src_button, new_to_act)
            new_min_raise = tl.where(round_closed, bb, new_min_raise)
            new_last_aggressive = tl.where(round_closed, 0, new_last_aggressive)
            new_street = tl.where(round_closed, src_street + 1, src_street)
            new_done = tl.where(showdown, True, new_done)

            tl.store(button + dst, src_button, mask=write)
            tl.store(street + dst, new_street, mask=write)
            tl.store(to_act + dst, new_to_act, mask=write)
            tl.store(last_to_act + dst, new_last_to_act, mask=write)
            tl.store(pot + dst, new_pot, mask=write)
            tl.store(min_raise + dst, new_min_raise, mask=write)
            tl.store(
                last_aggressive_amount + dst, new_last_aggressive, mask=write
            )
            tl.store(actions_this_round + dst, new_actions, mask=write)
            tl.store(actions_last_round + dst, new_actions_last, mask=write)
            tl.store(deck_pos + dst, src_deck_pos, mask=write)
            tl.store(winner + dst, new_winner, mask=write)
            tl.store(acted_since_reset + dst, True, mask=write)
            tl.store(done + dst, new_done, mask=write)
            tl.store(scale + dst, src_scale, mask=write)

            tl.store(stacks + dst * 2 + 0, stack0, mask=write)
            tl.store(stacks + dst * 2 + 1, stack1, mask=write)
            tl.store(committed + dst * 2 + 0, committed0, mask=write)
            tl.store(committed + dst * 2 + 1, committed1, mask=write)
            tl.store(chips_placed + dst * 2 + 0, chips0, mask=write)
            tl.store(chips_placed + dst * 2 + 1, chips1, mask=write)
            tl.store(
                starting_stacks + dst * 2 + 0,
                tl.load(starting_stacks + src * 2 + 0, mask=local_mask, other=0),
                mask=write,
            )
            tl.store(
                starting_stacks + dst * 2 + 1,
                tl.load(starting_stacks + src * 2 + 1, mask=local_mask, other=0),
                mask=write,
            )
            tl.store(
                has_folded + dst * 2 + 0,
                tl.load(has_folded + src * 2 + 0, mask=local_mask, other=0),
                mask=write,
            )
            tl.store(
                has_folded + dst * 2 + 1,
                tl.load(has_folded + src * 2 + 1, mask=local_mask, other=0),
                mask=write,
            )
            tl.store(is_allin + dst * 2 + 0, allin0, mask=write)
            tl.store(is_allin + dst * 2 + 1, allin1, mask=write)

            tl.store(parent_index + dst, src, mask=write)
            tl.store(action_from_parent + dst, action, mask=write)
            tl.store(rewards + dst, reward, mask=write)

            parent_to_call = to_call
            allin_call_leaf = (
                allin_abstraction
                & (action == 1)
                & tl.load(is_allin + src * 2 + src_opp, mask=local_mask, other=0)
                & (parent_to_call > 0)
                & (src_street > 0)
                & (src_street < 3)
            )
            tl.store(allin_leaf + dst, allin_call_leaf, mask=write)

            rank += legal.to(tl.int64)

    @triton.jit
    def _write_children_same_street_flat_kernel(
        button,
        street,
        to_act,
        last_to_act,
        pot,
        min_raise,
        last_aggressive_amount,
        actions_this_round,
        actions_last_round,
        acted_since_reset,
        stacks,
        committed,
        is_allin,
        has_folded,
        starting_stacks,
        scale,
        chips_placed,
        done,
        winner,
        legal_mask,
        child_offsets,
        parent_index,
        action_from_parent,
        rewards,
        allin_leaf,
        bet_bins,
        parent_start,
        parent_count,
        dst_start,
        bb,
        mean_stack,
        allin_abstraction: tl.constexpr,
        flop_showdown: tl.constexpr,
        num_actions: tl.constexpr,
        block: tl.constexpr,
    ):
        slot = tl.program_id(0) * block + tl.arange(0, block)
        total_slots = parent_count * num_actions
        local_mask = slot < total_slots
        parent_local = slot // num_actions
        action = slot - parent_local * num_actions
        src = parent_start + parent_local

        legal = tl.load(
            legal_mask + src * num_actions + action,
            mask=local_mask,
            other=0,
        )
        write = local_mask & legal

        rank = tl.zeros((block,), dtype=tl.int64)
        for prev_action in tl.static_range(0, num_actions):
            prev_legal = tl.load(
                legal_mask + src * num_actions + prev_action,
                mask=local_mask & (action > prev_action),
                other=0,
            )
            rank += prev_legal.to(tl.int64)

        child_base = dst_start + tl.load(
            child_offsets + parent_local, mask=local_mask, other=0
        )
        dst = child_base + rank

        src_actor = tl.load(to_act + src, mask=local_mask, other=0)
        src_opp = 1 - src_actor
        src_button = tl.load(button + src, mask=local_mask, other=0)
        src_street = tl.load(street + src, mask=local_mask, other=0)
        src_pot = tl.load(pot + src, mask=local_mask, other=0)
        src_min_raise = tl.load(min_raise + src, mask=local_mask, other=0)
        src_last_aggressive = tl.load(
            last_aggressive_amount + src, mask=local_mask, other=0
        )
        src_actions = tl.load(actions_this_round + src, mask=local_mask, other=0)
        src_actions_last = tl.load(actions_last_round + src, mask=local_mask, other=0)
        src_winner = tl.load(winner + src, mask=local_mask, other=-1)
        src_done = tl.load(done + src, mask=local_mask, other=0)
        src_scale = tl.load(scale + src, mask=local_mask, other=1.0)

        stack0 = tl.load(stacks + src * 2 + 0, mask=local_mask, other=0)
        stack1 = tl.load(stacks + src * 2 + 1, mask=local_mask, other=0)
        committed0 = tl.load(committed + src * 2 + 0, mask=local_mask, other=0)
        committed1 = tl.load(committed + src * 2 + 1, mask=local_mask, other=0)
        chips0 = tl.load(chips_placed + src * 2 + 0, mask=local_mask, other=0)
        chips1 = tl.load(chips_placed + src * 2 + 1, mask=local_mask, other=0)
        allin0 = tl.load(is_allin + src * 2 + 0, mask=local_mask, other=0)
        allin1 = tl.load(is_allin + src * 2 + 1, mask=local_mask, other=0)
        start0 = tl.load(starting_stacks + src * 2 + 0, mask=local_mask, other=mean_stack)
        start1 = tl.load(starting_stacks + src * 2 + 1, mask=local_mask, other=mean_stack)

        actor_stack = tl.where(src_actor == 0, stack0, stack1)
        actor_committed = tl.where(src_actor == 0, committed0, committed1)
        other_committed = tl.where(src_actor == 0, committed1, committed0)
        to_call = other_committed - actor_committed

        is_fold = action == 0
        is_call = action == 1
        is_allin_action = action == (num_actions - 1)
        is_bet_action = ~(is_fold | is_call | is_allin_action)

        additional = (
            tl.load(bet_bins + action, mask=local_mask & is_bet_action, other=0.0)
            * src_pot.to(tl.float32)
        ).to(tl.int64)
        chips = tl.zeros((block,), dtype=tl.int64)
        chips = tl.where(is_call, tl.minimum(to_call, actor_stack), chips)
        chips = tl.where(is_allin_action, actor_stack, chips)
        chips = tl.where(is_bet_action, to_call + additional, chips)

        new_pot = src_pot + chips
        new_min_raise = tl.where(
            is_bet_action, tl.maximum(src_min_raise, chips - to_call), src_min_raise
        )
        aggressive = is_bet_action | (is_allin_action & (chips > to_call))
        new_actions = src_actions + 1
        new_actions_last = src_actions_last
        new_to_act = src_opp
        new_last_to_act = src_actor
        new_done = src_done | is_fold
        new_winner = tl.where(is_fold, src_opp, src_winner)
        pot_share = tl.where(src_opp == 0, src_pot, 0).to(tl.float32)
        fold_reward = (stack0.to(tl.float32) + pot_share - start0.to(tl.float32)) / src_scale
        reward = tl.where(is_fold, fold_reward, 0.0)

        stack0 = tl.where(src_actor == 0, stack0 - chips, stack0)
        stack1 = tl.where(src_actor == 1, stack1 - chips, stack1)
        committed0 = tl.where(src_actor == 0, committed0 + chips, committed0)
        committed1 = tl.where(src_actor == 1, committed1 + chips, committed1)
        actor_committed_after = tl.where(src_actor == 0, committed0, committed1)
        new_last_aggressive = tl.where(
            aggressive, actor_committed_after, src_last_aggressive
        )
        chips0 = tl.where(src_actor == 0, chips0 + chips, chips0)
        chips1 = tl.where(src_actor == 1, chips1 + chips, chips1)
        actor_stack_after = tl.where(src_actor == 0, stack0, stack1)
        implicit_allin = (chips > 0) & (actor_stack_after == 0)
        allin0 = tl.where(implicit_allin & (src_actor == 0), True, allin0)
        allin1 = tl.where(implicit_allin & (src_actor == 1), True, allin1)

        equal_committed = committed0 == committed1
        allin_committed = (
            (allin0 & allin1)
            | (allin0 & (committed0 <= committed1))
            | (allin1 & (committed1 <= committed0))
        )
        round_closed = (
            (~new_done) & (equal_committed | allin_committed) & (new_actions >= 2)
        )
        if flop_showdown:
            showdown = round_closed & (src_street == 0)
        else:
            showdown = round_closed & (src_street == 3)
        committed0 = tl.where(round_closed, 0, committed0)
        committed1 = tl.where(round_closed, 0, committed1)
        new_actions_last = tl.where(round_closed, new_actions, new_actions_last)
        new_actions = tl.where(round_closed, 0, new_actions)
        new_to_act = tl.where(round_closed, 1 - src_button, new_to_act)
        new_min_raise = tl.where(round_closed, bb, new_min_raise)
        new_last_aggressive = tl.where(round_closed, 0, new_last_aggressive)
        new_street = tl.where(round_closed, src_street + 1, src_street)
        new_done = tl.where(showdown, True, new_done)

        tl.store(button + dst, src_button, mask=write)
        tl.store(street + dst, new_street, mask=write)
        tl.store(to_act + dst, new_to_act, mask=write)
        tl.store(last_to_act + dst, new_last_to_act, mask=write)
        tl.store(pot + dst, new_pot, mask=write)
        tl.store(min_raise + dst, new_min_raise, mask=write)
        tl.store(last_aggressive_amount + dst, new_last_aggressive, mask=write)
        tl.store(actions_this_round + dst, new_actions, mask=write)
        tl.store(actions_last_round + dst, new_actions_last, mask=write)
        tl.store(winner + dst, new_winner, mask=write)
        tl.store(acted_since_reset + dst, True, mask=write)
        tl.store(done + dst, new_done, mask=write)
        tl.store(scale + dst, src_scale, mask=write)

        tl.store(stacks + dst * 2 + 0, stack0, mask=write)
        tl.store(stacks + dst * 2 + 1, stack1, mask=write)
        tl.store(committed + dst * 2 + 0, committed0, mask=write)
        tl.store(committed + dst * 2 + 1, committed1, mask=write)
        tl.store(chips_placed + dst * 2 + 0, chips0, mask=write)
        tl.store(chips_placed + dst * 2 + 1, chips1, mask=write)
        tl.store(starting_stacks + dst * 2 + 0, start0, mask=write)
        tl.store(starting_stacks + dst * 2 + 1, start1, mask=write)
        tl.store(is_allin + dst * 2 + 0, allin0, mask=write)
        tl.store(is_allin + dst * 2 + 1, allin1, mask=write)
        # Inherit folded state from the parent (same-street children are never
        # newly folded: a fold is terminal via done/winner, not has_folded).
        # Must be written -- the subgame arena is persistent and
        # _postprocess_model_leaf_values reads has_folded on every leaf.
        tl.store(
            has_folded + dst * 2 + 0,
            tl.load(has_folded + src * 2 + 0, mask=local_mask, other=0),
            mask=write,
        )
        tl.store(
            has_folded + dst * 2 + 1,
            tl.load(has_folded + src * 2 + 1, mask=local_mask, other=0),
            mask=write,
        )

        tl.store(parent_index + dst, src, mask=write)
        tl.store(action_from_parent + dst, action, mask=write)
        tl.store(rewards + dst, reward, mask=write)

        allin_call_leaf = (
            allin_abstraction
            & is_call
            & tl.load(is_allin + src * 2 + src_opp, mask=local_mask, other=0)
            & (to_call > 0)
            & (src_street > 0)
            & (src_street < 3)
        )
        tl.store(allin_leaf + dst, allin_call_leaf, mask=write)

    @triton.jit
    def _copy_child_cards_kernel(
        deck,
        deck_pos,
        board_indices,
        last_board_indices,
        hole_indices,
        parent_index,
        child_start,
        child_count,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < child_count
        dst = child_start + offs
        src = tl.load(parent_index + dst, mask=mask, other=0)

        for col in tl.static_range(0, 9):
            tl.store(
                deck + dst * 9 + col,
                tl.load(deck + src * 9 + col, mask=mask, other=0),
                mask=mask,
            )
        tl.store(deck_pos + dst, tl.load(deck_pos + src, mask=mask, other=0), mask=mask)
        for col in tl.static_range(0, 5):
            board_card = tl.load(board_indices + src * 5 + col, mask=mask, other=-1)
            tl.store(board_indices + dst * 5 + col, board_card, mask=mask)
            tl.store(last_board_indices + dst * 5 + col, board_card, mask=mask)
        for col in tl.static_range(0, 4):
            tl.store(
                hole_indices + dst * 4 + col,
                tl.load(hole_indices + src * 4 + col, mask=mask, other=-1),
                mask=mask,
            )

    @triton.jit
    def _finalize_tree_masks_kernel(
        actions_this_round,
        done,
        to_act,
        parent_index,
        allin_leaf,
        legal_mask,
        valid_mask,
        root_mask,
        new_street_mask,
        leaf_mask,
        child_mask,
        child_counts,
        prev_actor,
        total_nodes,
        root_nodes,
        top_nodes,
        num_actions: tl.constexpr,
        block: tl.constexpr,
    ):
        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < total_nodes
        is_root = offs < root_nodes
        actions = tl.load(actions_this_round + offs, mask=mask, other=1)
        row_done = tl.load(done + offs, mask=mask, other=1)
        row_allin_leaf = tl.load(allin_leaf + offs, mask=mask, other=0)
        new_street = (actions == 0) & (~is_root) & (~row_allin_leaf)
        leaf = row_done | new_street | (offs >= top_nodes) | row_allin_leaf

        parent = tl.load(parent_index + offs, mask=mask & (~is_root), other=0)
        prev = tl.load(to_act + parent, mask=mask & (~is_root), other=-1)
        prev = tl.where(is_root, -1, prev)

        count = tl.zeros((block,), dtype=tl.int64)
        for action in tl.static_range(0, num_actions):
            legal = tl.load(
                legal_mask + offs * num_actions + action,
                mask=mask,
                other=0,
            )
            child = legal & (~leaf)
            tl.store(child_mask + offs * num_actions + action, child, mask=mask)
            count += child.to(tl.int64)

        tl.store(valid_mask + offs, True, mask=mask)
        tl.store(root_mask + offs, is_root, mask=mask)
        tl.store(new_street_mask + offs, new_street, mask=mask)
        tl.store(leaf_mask + offs, leaf, mask=mask)
        tl.store(child_counts + offs, count, mask=mask)
        tl.store(prev_actor + offs, prev, mask=mask)

    @triton.jit
    def _root_allowed_from_board_indices_kernel(
        board_indices,
        combo_card_a,
        combo_card_b,
        out_allowed,
        out_allowed_prob,
        n_roots,
        num_hands: tl.constexpr,
        block_h: tl.constexpr,
    ):
        root = tl.program_id(0)
        h = tl.arange(0, block_h)
        hand_mask = h < num_hands
        card_a = tl.load(combo_card_a + h, mask=hand_mask, other=-2)
        card_b = tl.load(combo_card_b + h, mask=hand_mask, other=-3)
        allowed = hand_mask

        for col in tl.static_range(0, 5):
            board_card = tl.load(board_indices + root * 5 + col)
            blocks_hand = (board_card >= 0) & (
                (card_a == board_card) | (card_b == board_card)
            )
            allowed = allowed & (~blocks_hand)

        denom = tl.sum(allowed.to(tl.float32), axis=0)
        prob = allowed.to(tl.float32) / tl.maximum(denom, 1.0)
        tl.store(out_allowed + root * num_hands + h, allowed, mask=hand_mask)
        tl.store(out_allowed_prob + root * num_hands + h, prob, mask=hand_mask)

    @triton.jit
    def _init_policy_tensors_kernel(
        uniform_policy,
        cumulative_regrets,
        parent_index,
        child_counts,
        total_nodes,
        root_nodes,
        num_hands: tl.constexpr,
        block: tl.constexpr,
    ):
        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < (total_nodes * num_hands)
        node = offs // num_hands
        is_root = node < root_nodes
        parent = tl.load(parent_index + node, mask=mask & (~is_root), other=0)
        denom = tl.load(child_counts + parent, mask=mask & (~is_root), other=1)
        uniform = tl.where(is_root, 0.0, 1.0 / denom.to(tl.float32))
        tl.store(uniform_policy + offs, uniform, mask=mask)
        tl.store(cumulative_regrets + offs, 0.0, mask=mask)

    @triton.jit
    def _init_belief_value_tensors_kernel(
        beliefs,
        self_reach,
        latest_values,
        action_from_parent,
        done,
        rewards,
        total_nodes,
        root_nodes,
        num_hands: tl.constexpr,
        block: tl.constexpr,
    ):
        planes = 2
        stride_node = planes * num_hands
        offs = tl.program_id(0) * block + tl.arange(0, block)
        mask = offs < (total_nodes * stride_node)
        node = offs // stride_node
        rem = offs - node * stride_node
        player = rem // num_hands

        is_root = node < root_nodes
        tl.store(beliefs + offs, 0.0, mask=mask)
        tl.store(self_reach + offs, tl.where(is_root, 1.0, 0.0), mask=mask)

        action = tl.load(action_from_parent + node, mask=mask, other=-1)
        row_done = tl.load(done + node, mask=mask, other=0)
        folded = (action == 0) & row_done
        reward = tl.load(rewards + node, mask=mask & folded, other=0.0)
        value = tl.where(player == 0, reward, -reward)
        tl.store(latest_values + offs, tl.where(folded, value, 0.0), mask=mask)


def legal_counts_triton_(
    env,
    legal_mask: torch.Tensor,
    child_counts: torch.Tensor,
    allin_leaf: torch.Tensor,
    bet_bins: torch.Tensor,
    *,
    parent_start: int,
    parent_count: int,
    stop_new_street: bool,
    num_actions: int,
    block: int = 128,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if parent_count == 0:
        return
    grid = (triton.cdiv(parent_count, block),)
    _legal_counts_kernel[grid](
        env.to_act,
        env.pot,
        env.min_raise,
        env.actions_this_round,
        env.stacks,
        env.committed,
        env.is_allin,
        env.done,
        allin_leaf,
        bet_bins,
        legal_mask,
        child_counts,
        parent_start,
        parent_count,
        stop_new_street,
        num_actions,
        block,
        num_warps=4,
    )


def write_children_same_street_triton_legacy_(
    env,
    legal_mask: torch.Tensor,
    child_offsets: torch.Tensor,
    parent_index: torch.Tensor,
    action_from_parent: torch.Tensor,
    rewards: torch.Tensor,
    allin_leaf: torch.Tensor,
    bet_bins: torch.Tensor,
    *,
    parent_start: int,
    parent_count: int,
    dst_start: int,
    allin_abstraction: bool,
    block: int = 32,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if parent_count == 0:
        return
    grid = (triton.cdiv(parent_count, block),)
    _write_children_same_street_kernel[grid](
        env.deck,
        env.deck_pos,
        env.button,
        env.street,
        env.to_act,
        env.last_to_act,
        env.pot,
        env.min_raise,
        env.last_aggressive_amount,
        env.actions_this_round,
        env.actions_last_round,
        env.acted_since_reset,
        env.stacks,
        env.committed,
        env.has_folded,
        env.is_allin,
        env.starting_stacks,
        env.scale,
        env.board_indices,
        env.last_board_indices,
        env.hole_indices,
        env.chips_placed,
        env.done,
        env.winner,
        legal_mask,
        child_offsets,
        parent_index,
        action_from_parent,
        rewards,
        allin_leaf,
        bet_bins,
        parent_start,
        parent_count,
        dst_start,
        env.bb,
        env.mean_stack,
        allin_abstraction,
        env.flop_showdown,
        legal_mask.shape[1],
        block,
        num_warps=4,
    )


def write_children_same_street_triton_optimized_(
    env,
    legal_mask: torch.Tensor,
    child_offsets: torch.Tensor,
    parent_index: torch.Tensor,
    action_from_parent: torch.Tensor,
    rewards: torch.Tensor,
    allin_leaf: torch.Tensor,
    bet_bins: torch.Tensor,
    *,
    parent_start: int,
    parent_count: int,
    dst_start: int,
    allin_abstraction: bool,
    block: int = 128,
) -> None:
    """Write same-street children with one Triton lane per parent-action slot.

    Card state is copied by ``copy_child_cards_triton_`` immediately after this
    writer, so this kernel avoids writing ``deck_pos``, ``deck``,
    ``board_indices``, ``last_board_indices``, and ``hole_indices``.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if parent_count == 0:
        return
    num_actions = legal_mask.shape[1]
    grid = (triton.cdiv(parent_count * num_actions, block),)
    _write_children_same_street_flat_kernel[grid](
        env.button,
        env.street,
        env.to_act,
        env.last_to_act,
        env.pot,
        env.min_raise,
        env.last_aggressive_amount,
        env.actions_this_round,
        env.actions_last_round,
        env.acted_since_reset,
        env.stacks,
        env.committed,
        env.is_allin,
        env.has_folded,
        env.starting_stacks,
        env.scale,
        env.chips_placed,
        env.done,
        env.winner,
        legal_mask,
        child_offsets,
        parent_index,
        action_from_parent,
        rewards,
        allin_leaf,
        bet_bins,
        parent_start,
        parent_count,
        dst_start,
        env.bb,
        env.mean_stack,
        allin_abstraction,
        env.flop_showdown,
        num_actions,
        block,
        num_warps=4,
    )


def write_children_same_street_triton_(
    env,
    legal_mask: torch.Tensor,
    child_offsets: torch.Tensor,
    parent_index: torch.Tensor,
    action_from_parent: torch.Tensor,
    rewards: torch.Tensor,
    allin_leaf: torch.Tensor,
    bet_bins: torch.Tensor,
    *,
    parent_start: int,
    parent_count: int,
    dst_start: int,
    allin_abstraction: bool,
    block: int = 128,
) -> None:
    if os.environ.get("P2_WRITE_CHILDREN_KERNEL", "").lower() == "legacy":
        write_children_same_street_triton_legacy_(
            env,
            legal_mask,
            child_offsets,
            parent_index,
            action_from_parent,
            rewards,
            allin_leaf,
            bet_bins,
            parent_start=parent_start,
            parent_count=parent_count,
            dst_start=dst_start,
            allin_abstraction=allin_abstraction,
            block=32,
        )
        return
    write_children_same_street_triton_optimized_(
        env,
        legal_mask,
        child_offsets,
        parent_index,
        action_from_parent,
        rewards,
        allin_leaf,
        bet_bins,
        parent_start=parent_start,
        parent_count=parent_count,
        dst_start=dst_start,
        allin_abstraction=allin_abstraction,
        block=block,
    )


def copy_child_cards_triton_(
    env,
    parent_index: torch.Tensor,
    *,
    child_start: int,
    child_count: int,
    block: int = 64,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if child_count == 0:
        return
    grid = (triton.cdiv(child_count, block),)
    _copy_child_cards_kernel[grid](
        env.deck,
        env.deck_pos,
        env.board_indices,
        env.last_board_indices,
        env.hole_indices,
        parent_index,
        child_start,
        child_count,
        BLOCK=block,
        num_warps=4,
    )


def finalize_tree_masks_triton_(
    env,
    legal_mask: torch.Tensor,
    parent_index: torch.Tensor,
    allin_leaf: torch.Tensor,
    valid_mask: torch.Tensor,
    root_mask: torch.Tensor,
    new_street_mask: torch.Tensor,
    leaf_mask: torch.Tensor,
    child_mask: torch.Tensor,
    child_counts: torch.Tensor,
    prev_actor: torch.Tensor,
    *,
    total_nodes: int,
    root_nodes: int,
    top_nodes: int,
    num_actions: int,
    block: int = 128,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if total_nodes == 0:
        return
    grid = (triton.cdiv(total_nodes, block),)
    _finalize_tree_masks_kernel[grid](
        env.actions_this_round,
        env.done,
        env.to_act,
        parent_index,
        allin_leaf,
        legal_mask,
        valid_mask,
        root_mask,
        new_street_mask,
        leaf_mask,
        child_mask,
        child_counts,
        prev_actor,
        total_nodes,
        root_nodes,
        top_nodes,
        num_actions,
        block,
        num_warps=4,
    )


def root_allowed_from_board_indices_triton_(
    board_indices: torch.Tensor,
    combo_card_a: torch.Tensor,
    combo_card_b: torch.Tensor,
    out_allowed: torch.Tensor,
    out_allowed_prob: torch.Tensor,
    *,
    n_roots: int,
    num_hands: int,
    block_h: int = 2048,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if n_roots == 0:
        return
    grid = (n_roots,)
    _root_allowed_from_board_indices_kernel[grid](
        board_indices,
        combo_card_a,
        combo_card_b,
        out_allowed,
        out_allowed_prob,
        n_roots,
        num_hands,
        block_h,
        num_warps=8,
    )


def init_policy_tensors_triton_(
    uniform_policy: torch.Tensor,
    cumulative_regrets: torch.Tensor,
    parent_index: torch.Tensor,
    child_counts: torch.Tensor,
    *,
    total_nodes: int,
    root_nodes: int,
    num_hands: int,
    block: int = 256,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if total_nodes == 0:
        return
    grid = (triton.cdiv(total_nodes * num_hands, block),)
    _init_policy_tensors_kernel[grid](
        uniform_policy,
        cumulative_regrets,
        parent_index,
        child_counts,
        total_nodes,
        root_nodes,
        num_hands,
        block,
        num_warps=4,
    )


def init_belief_value_tensors_triton_(
    beliefs: torch.Tensor,
    self_reach: torch.Tensor,
    latest_values: torch.Tensor,
    action_from_parent: torch.Tensor,
    done: torch.Tensor,
    rewards: torch.Tensor,
    *,
    total_nodes: int,
    root_nodes: int,
    num_hands: int,
    block: int = 256,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if total_nodes == 0:
        return
    grid = (triton.cdiv(total_nodes * 2 * num_hands, block),)
    _init_belief_value_tensors_kernel[grid](
        beliefs,
        self_reach,
        latest_values,
        action_from_parent,
        done,
        rewards,
        total_nodes,
        root_nodes,
        num_hands,
        block,
        num_warps=4,
    )
