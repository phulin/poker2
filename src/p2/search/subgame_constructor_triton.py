"""Triton kernels for fused sparse subgame construction."""

from __future__ import annotations

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
        parent_start: tl.constexpr,
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
        parent_start: tl.constexpr,
        parent_count,
        dst_start: tl.constexpr,
        bb: tl.constexpr,
        mean_stack: tl.constexpr,
        has_preflop_table: tl.constexpr,
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
            new_pot = new_pot + chips

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
            new_street = tl.where(round_closed, src_street + 1, src_street)
            new_done = tl.where(showdown, True, new_done)

            tl.store(button + dst, src_button, mask=write)
            tl.store(street + dst, new_street, mask=write)
            tl.store(to_act + dst, new_to_act, mask=write)
            tl.store(last_to_act + dst, new_last_to_act, mask=write)
            tl.store(pot + dst, new_pot, mask=write)
            tl.store(min_raise + dst, new_min_raise, mask=write)
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
                & (src_street < 3)
            )
            if not has_preflop_table:
                allin_call_leaf = allin_call_leaf & (src_street > 0)
            tl.store(allin_leaf + dst, allin_call_leaf, mask=write)

            rank += legal.to(tl.int64)

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
    has_preflop_table: bool,
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
        has_preflop_table,
        allin_abstraction,
        env.flop_showdown,
        legal_mask.shape[1],
        block,
        num_warps=4,
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
