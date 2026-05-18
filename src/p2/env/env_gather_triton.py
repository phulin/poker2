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
    def _gather_env_scalars_kernel(
        idx,
        src_button,
        src_street,
        src_to_act,
        src_last_to_act,
        src_pot,
        src_min_raise,
        src_actions_this_round,
        src_actions_last_round,
        src_deck_pos,
        src_winner,
        src_acted_since_reset,
        src_done,
        src_scale,
        dst_button,
        dst_street,
        dst_to_act,
        dst_last_to_act,
        dst_pot,
        dst_min_raise,
        dst_actions_this_round,
        dst_actions_last_round,
        dst_deck_pos,
        dst_winner,
        dst_acted_since_reset,
        dst_done,
        dst_scale,
        N,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        src = tl.load(idx + offs, mask=mask, other=0)

        tl.store(dst_button + offs, tl.load(src_button + src, mask=mask), mask=mask)
        tl.store(dst_street + offs, tl.load(src_street + src, mask=mask), mask=mask)
        tl.store(dst_to_act + offs, tl.load(src_to_act + src, mask=mask), mask=mask)
        tl.store(
            dst_last_to_act + offs, tl.load(src_last_to_act + src, mask=mask), mask=mask
        )
        tl.store(dst_pot + offs, tl.load(src_pot + src, mask=mask), mask=mask)
        tl.store(
            dst_min_raise + offs, tl.load(src_min_raise + src, mask=mask), mask=mask
        )
        tl.store(
            dst_actions_this_round + offs,
            tl.load(src_actions_this_round + src, mask=mask),
            mask=mask,
        )
        tl.store(
            dst_actions_last_round + offs,
            tl.load(src_actions_last_round + src, mask=mask),
            mask=mask,
        )
        tl.store(
            dst_deck_pos + offs, tl.load(src_deck_pos + src, mask=mask), mask=mask
        )
        tl.store(dst_winner + offs, tl.load(src_winner + src, mask=mask), mask=mask)
        tl.store(
            dst_acted_since_reset + offs,
            tl.load(src_acted_since_reset + src, mask=mask),
            mask=mask,
        )
        tl.store(dst_done + offs, tl.load(src_done + src, mask=mask), mask=mask)
        tl.store(dst_scale + offs, tl.load(src_scale + src, mask=mask), mask=mask)

    @triton.jit
    def _gather_env_pair_kernel(
        idx,
        src_stacks,
        src_committed,
        src_chips_placed,
        src_starting_stacks,
        src_has_folded,
        src_is_allin,
        dst_stacks,
        dst_committed,
        dst_chips_placed,
        dst_starting_stacks,
        dst_has_folded,
        dst_is_allin,
        N,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < N
        src = tl.load(idx + rows, mask=mask, other=0)
        for col in tl.static_range(0, 2):
            dst_off = rows * 2 + col
            src_off = src * 2 + col
            tl.store(dst_stacks + dst_off, tl.load(src_stacks + src_off, mask=mask), mask=mask)
            tl.store(
                dst_committed + dst_off,
                tl.load(src_committed + src_off, mask=mask),
                mask=mask,
            )
            tl.store(
                dst_chips_placed + dst_off,
                tl.load(src_chips_placed + src_off, mask=mask),
                mask=mask,
            )
            tl.store(
                dst_starting_stacks + dst_off,
                tl.load(src_starting_stacks + src_off, mask=mask),
                mask=mask,
            )
            tl.store(
                dst_has_folded + dst_off,
                tl.load(src_has_folded + src_off, mask=mask),
                mask=mask,
            )
            tl.store(
                dst_is_allin + dst_off,
                tl.load(src_is_allin + src_off, mask=mask),
                mask=mask,
            )

    @triton.jit
    def _gather_env_small_cards_kernel(
        idx,
        src_deck,
        src_board_indices,
        src_last_board_indices,
        src_hole_indices,
        dst_deck,
        dst_board_indices,
        dst_last_board_indices,
        dst_hole_indices,
        N,
        BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < N
        src = tl.load(idx + rows, mask=mask, other=0)

        for col in tl.static_range(0, 9):
            tl.store(
                dst_deck + rows * 9 + col,
                tl.load(src_deck + src * 9 + col, mask=mask),
                mask=mask,
            )
        for col in tl.static_range(0, 5):
            tl.store(
                dst_board_indices + rows * 5 + col,
                tl.load(src_board_indices + src * 5 + col, mask=mask),
                mask=mask,
            )
            tl.store(
                dst_last_board_indices + rows * 5 + col,
                tl.load(src_last_board_indices + src * 5 + col, mask=mask),
                mask=mask,
            )
        for col in tl.static_range(0, 4):
            tl.store(
                dst_hole_indices + rows * 4 + col,
                tl.load(src_hole_indices + src * 4 + col, mask=mask),
                mask=mask,
            )

    @triton.jit
    def _gather_env_bool_matrix_kernel(
        idx,
        src_matrix,
        dst_matrix,
        N,
        WIDTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = row < N
        col_mask = col < WIDTH
        src = tl.load(idx + row, mask=row_mask, other=0)
        vals = tl.load(
            src_matrix + src[:, None] * WIDTH + col[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0,
        )
        tl.store(
            dst_matrix + row[:, None] * WIDTH + col[None, :],
            vals,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def gather_env_rows_triton(src_env, dst_env, indices: torch.Tensor) -> None:
    """Gather HUNLTensorEnv rows with a few Triton kernels.

    ``indices`` maps each destination row to a source row. The destination env
    must already be allocated with ``num_envs == len(indices)``.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if indices.device.type != "cuda":
        raise ValueError("gather_env_rows_triton requires CUDA indices.")
    if indices.numel() == 0:
        return
    idx = indices.contiguous()
    n = idx.numel()
    block = 256
    grid = (triton.cdiv(n, block),)

    _gather_env_scalars_kernel[grid](
        idx,
        src_env.button,
        src_env.street,
        src_env.to_act,
        src_env.last_to_act,
        src_env.pot,
        src_env.min_raise,
        src_env.actions_this_round,
        src_env.actions_last_round,
        src_env.deck_pos,
        src_env.winner,
        src_env.acted_since_reset,
        src_env.done,
        src_env.scale,
        dst_env.button,
        dst_env.street,
        dst_env.to_act,
        dst_env.last_to_act,
        dst_env.pot,
        dst_env.min_raise,
        dst_env.actions_this_round,
        dst_env.actions_last_round,
        dst_env.deck_pos,
        dst_env.winner,
        dst_env.acted_since_reset,
        dst_env.done,
        dst_env.scale,
        N=n,
        BLOCK=block,
    )
    _gather_env_pair_kernel[grid](
        idx,
        src_env.stacks,
        src_env.committed,
        src_env.chips_placed,
        src_env.starting_stacks,
        src_env.has_folded,
        src_env.is_allin,
        dst_env.stacks,
        dst_env.committed,
        dst_env.chips_placed,
        dst_env.starting_stacks,
        dst_env.has_folded,
        dst_env.is_allin,
        N=n,
        BLOCK=block,
    )
    _gather_env_small_cards_kernel[grid](
        idx,
        src_env.deck,
        src_env.board_indices,
        src_env.last_board_indices,
        src_env.hole_indices,
        dst_env.deck,
        dst_env.board_indices,
        dst_env.last_board_indices,
        dst_env.hole_indices,
        N=n,
        BLOCK=block,
    )

    mat_grid_board = (triton.cdiv(n, 16), triton.cdiv(260, 64))
    _gather_env_bool_matrix_kernel[mat_grid_board](
        idx,
        src_env.board_onehot,
        dst_env.board_onehot,
        N=n,
        WIDTH=260,
        BLOCK_M=16,
        BLOCK_N=64,
    )
    mat_grid_hole = (triton.cdiv(n, 16), triton.cdiv(208, 64))
    _gather_env_bool_matrix_kernel[mat_grid_hole](
        idx,
        src_env.hole_onehot,
        dst_env.hole_onehot,
        N=n,
        WIDTH=208,
        BLOCK_M=16,
        BLOCK_N=64,
    )
