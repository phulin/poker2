from __future__ import annotations

import copy

import torch

from p2.env.pbs_env import PBSEnv

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def triton_is_available() -> bool:
    return triton is not None


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


TRITON_PBS_ENV_ROW_FIELDS = (
    "deck",
    "deck_pos",
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
)


class TritonPBSEnvBuffer:
    """Size-doubling scratch buffer for reusable TritonPBSEnv row materialization."""

    def __init__(self, proto: PBSEnv, initial_capacity: int = 0) -> None:
        self.proto = proto
        self.capacity = 0
        self.env: TritonPBSEnv | None = None
        self._long_capacity = 0
        self._long_workspace: torch.Tensor | None = None
        if initial_capacity > 0:
            self.ensure_capacity(initial_capacity)

    def ensure_capacity(self, capacity: int) -> None:
        if self.env is not None and self.capacity >= capacity:
            return
        new_capacity = max(1, int(capacity), self.capacity * 2)
        self.capacity = new_capacity
        self.env = TritonPBSEnv.from_proto(
            self.proto, num_envs=new_capacity, init_state=False
        )

    def view(self, size: int) -> TritonPBSEnv:
        self.ensure_capacity(size)
        assert self.env is not None
        return self.env.capacity_view(size)

    def long_workspace(self, size: int) -> torch.Tensor:
        if self._long_workspace is None or self._long_capacity < size:
            new_capacity = max(1, int(size), self._long_capacity * 2)
            self._long_capacity = new_capacity
            self._long_workspace = torch.empty(
                new_capacity, dtype=torch.long, device=self.proto.device
            )
        return self._long_workspace[:size]


class TritonPBSEnv(PBSEnv):
    """CUDA/Triton variant of ``PBSEnv`` for the per-action main loop.

    Kernels cover legal mask/amount generation, bin-to-action mapping, the full
    step transition, reset state writes, and row copy/gather materialization.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._default_bet_bins_tensor = torch.tensor(
            self.default_bet_bins, dtype=torch.float32, device=self.device
        )

    def reset(
        self,
        env_indices: torch.Tensor | None = None,
        force_button: torch.Tensor | None = None,
        force_deck: torch.Tensor | None = None,
        force_starting_stacks: torch.Tensor | None = None,
    ) -> None:
        self._validate_triton_ready()
        ids = self.arange_n if env_indices is None else env_indices.to(self.device)
        if ids.numel() == 0:
            return
        num_reset = ids.numel()

        if force_deck is None:
            scores = torch.rand(num_reset, 52, generator=self.rng, device=self.device)
            decks = torch.topk(scores, 5, dim=1).indices.contiguous()
        else:
            decks = force_deck.to(self.device).contiguous()
            if decks.shape != (num_reset, 5):
                raise ValueError("force_deck must have shape [num_reset, 5]")

        if force_button is None:
            button = torch.randint(
                0,
                self.num_players,
                (num_reset,),
                generator=self.rng,
                device=self.device,
            )
        else:
            button = force_button.to(self.device).contiguous()

        if force_starting_stacks is None:
            starting_stacks = self._sample_starting_stacks(num_reset).contiguous()
        else:
            starting_stacks = force_starting_stacks.to(self.device).contiguous()
            if starting_stacks.shape != (num_reset, self.num_players):
                raise ValueError(
                    "force_starting_stacks must have shape [num_reset, num_players]"
                )

        _pbs_reset_kernel[(num_reset,)](
            ids.contiguous(),
            decks,
            button,
            starting_stacks,
            self.deck,
            self.deck_pos,
            self.button,
            self.street,
            self.to_act,
            self.last_to_act,
            self.pot,
            self.min_raise,
            self.last_aggressive_amount,
            self.actions_this_round,
            self.actions_last_round,
            self.stacks,
            self.starting_stacks,
            self.scale,
            self.committed,
            self.chips_placed,
            self.has_folded,
            self.is_allin,
            self.acted_this_round,
            self.done,
            self.winner,
            self.winners,
            self.board_indices,
            self.last_board_indices,
            self.board_onehot,
            self.sb,
            self.bb,
            self.num_players,
            BLOCK_P=_next_power_of_2(self.num_players),
            BLOCK_BOARD=_next_power_of_2(5 * 4 * 13),
            num_warps=4,
        )

    @classmethod
    def from_proto(
        cls, proto: PBSEnv, num_envs: int | None = None, init_state: bool = True
    ) -> TritonPBSEnv:
        return TritonPBSEnv(
            num_envs=proto.N if num_envs is None else num_envs,
            num_players=proto.num_players,
            mean_stack=proto.mean_stack,
            sb=proto.sb,
            bb=proto.bb,
            default_bet_bins=proto.default_bet_bins,
            device=proto.device,
            rng=proto.rng,
            float_dtype=proto.float_dtype,
            stack_mode=proto.stack_mode,
            min_stack_bb=proto.min_stack_bb,
            mid_stack_bb=proto.mid_stack_bb,
            max_stack_bb=proto.max_stack_bb,
            high_stack_mass_ratio=proto.high_stack_mass_ratio,
            force_heads_up_preflop_flop=proto.force_heads_up_preflop_flop,
            init_state=init_state,
        )

    def legal_bins_amounts_and_mask(
        self, bet_bins: list[float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_triton_ready()
        bet_bins_tensor = self._bet_bins_tensor(bet_bins)
        B = bet_bins_tensor.numel() + 3
        amounts = torch.empty((self.N, B), dtype=torch.long, device=self.device)
        mask = torch.empty((self.N, B), dtype=torch.bool, device=self.device)
        _pbs_legal_kernel[(self.N,)](
            bet_bins_tensor,
            self.to_act,
            self.stacks,
            self.committed,
            self.has_folded,
            self.is_allin,
            self.done,
            self.pot,
            self.min_raise,
            amounts,
            mask,
            self.num_players,
            B,
            BLOCK_P=_next_power_of_2(self.num_players),
            BLOCK_B=_next_power_of_2(B),
            num_warps=1,
        )
        return amounts, mask

    def step_bins(
        self,
        bin_indices: torch.Tensor,
        bin_amounts: torch.Tensor | None = None,
        legal_masks: torch.Tensor | None = None,
        bet_bins: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_triton_ready()
        if bin_amounts is None:
            bin_amounts, legal_masks = self.legal_bins_amounts_and_mask(bet_bins)
        B = bin_amounts.shape[1]
        action_indices = torch.empty_like(bin_indices)
        selected_amounts = torch.empty_like(bin_indices)
        _pbs_map_bins_kernel[(self.N,)](
            bin_indices.contiguous(),
            bin_amounts,
            action_indices,
            selected_amounts,
            B,
            num_warps=1,
        )
        return self.step(action_indices, selected_amounts)

    def step(
        self, action_indices: torch.Tensor, bet_amounts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_triton_ready()
        if action_indices.shape[0] != self.N:
            raise ValueError("action_indices must have shape [N]")
        if action_indices.device.type != "cuda" or bet_amounts.device.type != "cuda":
            raise ValueError("TritonPBSEnv actions must be CUDA tensors.")

        rewards = torch.empty(
            self.N, self.num_players, dtype=self.float_dtype, device=self.device
        )
        new_streets = torch.empty(self.N, dtype=torch.long, device=self.device)
        dealt_cards = torch.empty(self.N, 5, dtype=torch.long, device=self.device)
        _pbs_step_kernel[(self.N,)](
            action_indices.contiguous(),
            bet_amounts.contiguous(),
            self.deck,
            self.deck_pos,
            self.button,
            self.street,
            self.to_act,
            self.last_to_act,
            self.pot,
            self.min_raise,
            self.last_aggressive_amount,
            self.actions_this_round,
            self.actions_last_round,
            self.stacks,
            self.starting_stacks,
            self.scale,
            self.committed,
            self.chips_placed,
            self.has_folded,
            self.is_allin,
            self.acted_this_round,
            self.done,
            self.winner,
            self.winners,
            self.board_indices,
            self.last_board_indices,
            self.board_onehot,
            rewards,
            new_streets,
            dealt_cards,
            self.bb,
            self.num_players,
            self.force_heads_up_preflop_flop,
            BLOCK_P=_next_power_of_2(self.num_players),
            num_warps=1,
        )
        return rewards, new_streets, dealt_cards

    def copy_state_from(
        self,
        src_env: PBSEnv,
        src_select: torch.Tensor | slice,
        dest_select: torch.Tensor | slice,
        copy_deck: bool = True,
    ) -> None:
        self._validate_triton_ready()
        src_ids = self._select_to_tensor(src_select, src_env.N)
        dst_ids = self._select_to_tensor(dest_select, self.N)
        if src_ids.numel() == 0:
            return
        if src_ids.numel() != dst_ids.numel():
            raise ValueError("source and destination selections must have same length")
        if src_env.num_players != self.num_players:
            raise ValueError("source and destination must have same num_players")
        _pbs_copy_rows_kernel[(src_ids.numel(),)](
            src_ids.contiguous(),
            dst_ids.contiguous(),
            src_env.deck,
            src_env.deck_pos,
            src_env.button,
            src_env.street,
            src_env.to_act,
            src_env.last_to_act,
            src_env.pot,
            src_env.min_raise,
            src_env.last_aggressive_amount,
            src_env.actions_this_round,
            src_env.actions_last_round,
            src_env.stacks,
            src_env.starting_stacks,
            src_env.scale,
            src_env.committed,
            src_env.chips_placed,
            src_env.has_folded,
            src_env.is_allin,
            src_env.acted_this_round,
            src_env.done,
            src_env.winner,
            src_env.winners,
            src_env.board_indices,
            src_env.last_board_indices,
            src_env.board_onehot,
            self.deck,
            self.deck_pos,
            self.button,
            self.street,
            self.to_act,
            self.last_to_act,
            self.pot,
            self.min_raise,
            self.last_aggressive_amount,
            self.actions_this_round,
            self.actions_last_round,
            self.stacks,
            self.starting_stacks,
            self.scale,
            self.committed,
            self.chips_placed,
            self.has_folded,
            self.is_allin,
            self.acted_this_round,
            self.done,
            self.winner,
            self.winners,
            self.board_indices,
            self.last_board_indices,
            self.board_onehot,
            self.num_players,
            copy_deck,
            0,
            False,
            BLOCK_P=_next_power_of_2(self.num_players),
            BLOCK_BOARD=_next_power_of_2(5 * 4 * 13),
            num_warps=4,
        )

    def capacity_view(self, size: int) -> TritonPBSEnv:
        if size < 0 or size > self.N:
            raise ValueError(f"size must be in [0, {self.N}], got {size}")
        env = copy.copy(self)
        env.N = int(size)
        env.arange_n = self.arange_n[:size]
        for field in TRITON_PBS_ENV_ROW_FIELDS:
            setattr(env, field, getattr(self, field)[:size])
        return env

    def gather_rows_into(
        self,
        indices: torch.Tensor,
        dst_env: TritonPBSEnv,
        dst_start: int = 0,
        copy_deck: bool = True,
    ) -> None:
        indices = indices.to(self.device).contiguous()
        count = indices.numel()
        if count == 0:
            return
        if dst_start < 0 or dst_start + count > dst_env.N:
            raise ValueError("destination range is out of bounds")
        if dst_env.num_players != self.num_players:
            raise ValueError("source and destination must have same num_players")
        _pbs_copy_rows_kernel[(count,)](
            indices,
            dst_env.arange_n,
            self.deck,
            self.deck_pos,
            self.button,
            self.street,
            self.to_act,
            self.last_to_act,
            self.pot,
            self.min_raise,
            self.last_aggressive_amount,
            self.actions_this_round,
            self.actions_last_round,
            self.stacks,
            self.starting_stacks,
            self.scale,
            self.committed,
            self.chips_placed,
            self.has_folded,
            self.is_allin,
            self.acted_this_round,
            self.done,
            self.winner,
            self.winners,
            self.board_indices,
            self.last_board_indices,
            self.board_onehot,
            dst_env.deck,
            dst_env.deck_pos,
            dst_env.button,
            dst_env.street,
            dst_env.to_act,
            dst_env.last_to_act,
            dst_env.pot,
            dst_env.min_raise,
            dst_env.last_aggressive_amount,
            dst_env.actions_this_round,
            dst_env.actions_last_round,
            dst_env.stacks,
            dst_env.starting_stacks,
            dst_env.scale,
            dst_env.committed,
            dst_env.chips_placed,
            dst_env.has_folded,
            dst_env.is_allin,
            dst_env.acted_this_round,
            dst_env.done,
            dst_env.winner,
            dst_env.winners,
            dst_env.board_indices,
            dst_env.last_board_indices,
            dst_env.board_onehot,
            self.num_players,
            copy_deck,
            dst_start,
            True,
            BLOCK_P=_next_power_of_2(self.num_players),
            BLOCK_BOARD=_next_power_of_2(5 * 4 * 13),
            num_warps=4,
        )

    def repeat_interleave_into(
        self,
        repeats: torch.Tensor,
        dst_env: TritonPBSEnv,
        output_size: int,
        dst_start: int = 0,
        max_repeat: int | None = None,
        offsets_out: torch.Tensor | None = None,
    ) -> None:
        repeats = repeats.to(self.device).contiguous()
        if repeats.shape[0] != self.N:
            raise ValueError("repeats must have shape [N]")
        if output_size == 0:
            return
        if dst_start < 0 or dst_start + output_size > dst_env.N:
            raise ValueError("destination range is out of bounds")
        if offsets_out is None:
            offsets = torch.cumsum(repeats, dim=0) - repeats
        else:
            if offsets_out.shape[0] < self.N:
                raise ValueError("offsets_out must have at least N elements")
            offsets = offsets_out[: self.N]
            if offsets.device != repeats.device:
                raise ValueError("offsets_out must be on the env device")
            torch.cumsum(repeats, dim=0, out=offsets)
            offsets.sub_(repeats)
        if max_repeat is None:
            max_repeat = max(int(repeats.max().item()), 1)
        block_r = _next_power_of_2(max(max_repeat, 1))
        _pbs_repeat_rows_block_kernel[(self.N,)](
            repeats,
            offsets,
            self.deck,
            self.deck_pos,
            self.button,
            self.street,
            self.to_act,
            self.last_to_act,
            self.pot,
            self.min_raise,
            self.last_aggressive_amount,
            self.actions_this_round,
            self.actions_last_round,
            self.stacks,
            self.starting_stacks,
            self.scale,
            self.committed,
            self.chips_placed,
            self.has_folded,
            self.is_allin,
            self.acted_this_round,
            self.done,
            self.winner,
            self.winners,
            self.board_indices,
            self.last_board_indices,
            self.board_onehot,
            dst_env.deck,
            dst_env.deck_pos,
            dst_env.button,
            dst_env.street,
            dst_env.to_act,
            dst_env.last_to_act,
            dst_env.pot,
            dst_env.min_raise,
            dst_env.last_aggressive_amount,
            dst_env.actions_this_round,
            dst_env.actions_last_round,
            dst_env.stacks,
            dst_env.starting_stacks,
            dst_env.scale,
            dst_env.committed,
            dst_env.chips_placed,
            dst_env.has_folded,
            dst_env.is_allin,
            dst_env.acted_this_round,
            dst_env.done,
            dst_env.winner,
            dst_env.winners,
            dst_env.board_indices,
            dst_env.last_board_indices,
            dst_env.board_onehot,
            self.num_players,
            dst_start,
            BLOCK_R=block_r,
            BLOCK_P=_next_power_of_2(self.num_players),
            BLOCK_BOARD=_next_power_of_2(5 * 4 * 13),
            num_warps=4,
        )

    def repeat_interleave(self, repeats: torch.Tensor, output_size: int) -> PBSEnv:
        dst = type(self).from_proto(self, num_envs=output_size, init_state=False)
        if output_size > 0:
            self.repeat_interleave_into(repeats, dst, output_size)
        return dst

    def gather_rows(self, indices: torch.Tensor) -> PBSEnv:
        indices = indices.to(self.device).contiguous()
        dst = type(self).from_proto(self, num_envs=indices.numel(), init_state=False)
        self.gather_rows_into(indices, dst)
        return dst

    def _validate_triton_ready(self) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        if self.device.type != "cuda":
            raise ValueError("TritonPBSEnv requires CUDA tensors.")

    def _bet_bins_tensor(self, bet_bins: list[float] | None) -> torch.Tensor:
        if bet_bins is None or bet_bins == self.default_bet_bins:
            return self._default_bet_bins_tensor
        return torch.tensor(bet_bins, dtype=torch.float32, device=self.device)

    def _select_to_tensor(
        self, select: torch.Tensor | slice, upper: int
    ) -> torch.Tensor:
        if isinstance(select, slice):
            return torch.arange(upper, device=self.device)[select]
        return select.to(self.device)


if triton is not None:

    @triton.jit
    def _pbs_reset_kernel(
        ids_ptr,
        decks_ptr,
        button_in_ptr,
        starting_stacks_in_ptr,
        deck_ptr,
        deck_pos_ptr,
        button_ptr,
        street_ptr,
        to_act_ptr,
        last_to_act_ptr,
        pot_ptr,
        min_raise_ptr,
        last_aggressive_amount_ptr,
        actions_this_round_ptr,
        actions_last_round_ptr,
        stacks_ptr,
        starting_stacks_ptr,
        scale_ptr,
        committed_ptr,
        chips_placed_ptr,
        has_folded_ptr,
        is_allin_ptr,
        acted_ptr,
        done_ptr,
        winner_ptr,
        winners_ptr,
        board_indices_ptr,
        last_board_indices_ptr,
        board_onehot_ptr,
        sb: tl.constexpr,
        bb: tl.constexpr,
        P: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_BOARD: tl.constexpr,
    ):
        reset_row = tl.program_id(0)
        env = tl.load(ids_ptr + reset_row)
        button = tl.load(button_in_ptr + reset_row)
        sb_player = tl.where(P == 2, button, (button + 1) % P)
        bb_offset = tl.where(P == 2, 1, 2)
        bb_player = (button + bb_offset) % P
        first_actor = tl.where(P == 2, sb_player, (bb_player + 1) % P)

        deck_offsets = tl.arange(0, 8)
        deck_mask = deck_offsets < 5
        cards = tl.load(
            decks_ptr + reset_row * 5 + deck_offsets, mask=deck_mask, other=0
        )
        tl.store(deck_ptr + env * 5 + deck_offsets, cards, mask=deck_mask)
        tl.store(deck_pos_ptr + env, 0)
        tl.store(button_ptr + env, button)
        tl.store(street_ptr + env, 0)
        tl.store(to_act_ptr + env, first_actor)
        tl.store(last_to_act_ptr + env, first_actor)
        tl.store(min_raise_ptr + env, bb)
        tl.store(last_aggressive_amount_ptr + env, bb)
        tl.store(actions_this_round_ptr + env, 0)
        tl.store(actions_last_round_ptr + env, 0)
        tl.store(done_ptr + env, False)
        tl.store(winner_ptr + env, -1)

        p_offsets = tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P
        row_p = env * P
        starting_stack = tl.load(
            starting_stacks_in_ptr + reset_row * P + p_offsets,
            mask=p_mask,
            other=0,
        )
        blind = tl.where(
            p_offsets == sb_player, sb, tl.where(p_offsets == bb_player, bb, 0)
        )
        posted = tl.minimum(starting_stack, blind)
        stack = starting_stack - posted
        tl.store(pot_ptr + env, tl.sum(tl.where(p_mask, posted, 0), axis=0))
        min_start = tl.min(tl.where(p_mask, starting_stack, 2147483647), axis=0)
        tl.store(scale_ptr + env, min_start.to(tl.float32))
        tl.store(stacks_ptr + row_p + p_offsets, stack, mask=p_mask)
        tl.store(starting_stacks_ptr + row_p + p_offsets, starting_stack, mask=p_mask)
        tl.store(committed_ptr + row_p + p_offsets, posted, mask=p_mask)
        tl.store(chips_placed_ptr + row_p + p_offsets, posted, mask=p_mask)
        tl.store(has_folded_ptr + row_p + p_offsets, False, mask=p_mask)
        tl.store(is_allin_ptr + row_p + p_offsets, stack == 0, mask=p_mask)
        tl.store(acted_ptr + row_p + p_offsets, False, mask=p_mask)
        tl.store(winners_ptr + row_p + p_offsets, False, mask=p_mask)

        board_small = tl.arange(0, 8)
        board_small_mask = board_small < 5
        tl.store(board_indices_ptr + env * 5 + board_small, -1, mask=board_small_mask)
        tl.store(
            last_board_indices_ptr + env * 5 + board_small, -1, mask=board_small_mask
        )
        board_offsets = tl.arange(0, BLOCK_BOARD)
        board_mask = board_offsets < 260
        tl.store(board_onehot_ptr + env * 260 + board_offsets, False, mask=board_mask)

    @triton.jit
    def _pbs_copy_rows_kernel(
        src_ids_ptr,
        dst_ids_ptr,
        src_deck_ptr,
        src_deck_pos_ptr,
        src_button_ptr,
        src_street_ptr,
        src_to_act_ptr,
        src_last_to_act_ptr,
        src_pot_ptr,
        src_min_raise_ptr,
        src_last_aggressive_amount_ptr,
        src_actions_this_round_ptr,
        src_actions_last_round_ptr,
        src_stacks_ptr,
        src_starting_stacks_ptr,
        src_scale_ptr,
        src_committed_ptr,
        src_chips_placed_ptr,
        src_has_folded_ptr,
        src_is_allin_ptr,
        src_acted_ptr,
        src_done_ptr,
        src_winner_ptr,
        src_winners_ptr,
        src_board_indices_ptr,
        src_last_board_indices_ptr,
        src_board_onehot_ptr,
        dst_deck_ptr,
        dst_deck_pos_ptr,
        dst_button_ptr,
        dst_street_ptr,
        dst_to_act_ptr,
        dst_last_to_act_ptr,
        dst_pot_ptr,
        dst_min_raise_ptr,
        dst_last_aggressive_amount_ptr,
        dst_actions_this_round_ptr,
        dst_actions_last_round_ptr,
        dst_stacks_ptr,
        dst_starting_stacks_ptr,
        dst_scale_ptr,
        dst_committed_ptr,
        dst_chips_placed_ptr,
        dst_has_folded_ptr,
        dst_is_allin_ptr,
        dst_acted_ptr,
        dst_done_ptr,
        dst_winner_ptr,
        dst_winners_ptr,
        dst_board_indices_ptr,
        dst_last_board_indices_ptr,
        dst_board_onehot_ptr,
        P: tl.constexpr,
        COPY_DECK: tl.constexpr,
        DST_START: tl.constexpr,
        CONTIGUOUS_DST: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_BOARD: tl.constexpr,
    ):
        row = tl.program_id(0)
        src = tl.load(src_ids_ptr + row)
        if CONTIGUOUS_DST:
            dst = DST_START + row
        else:
            dst = tl.load(dst_ids_ptr + row)
        if COPY_DECK:
            deck_offsets = tl.arange(0, 8)
            deck_mask = deck_offsets < 5
            deck = tl.load(
                src_deck_ptr + src * 5 + deck_offsets, mask=deck_mask, other=0
            )
            tl.store(dst_deck_ptr + dst * 5 + deck_offsets, deck, mask=deck_mask)
        tl.store(dst_deck_pos_ptr + dst, tl.load(src_deck_pos_ptr + src))
        tl.store(dst_button_ptr + dst, tl.load(src_button_ptr + src))
        tl.store(dst_street_ptr + dst, tl.load(src_street_ptr + src))
        tl.store(dst_to_act_ptr + dst, tl.load(src_to_act_ptr + src))
        tl.store(dst_last_to_act_ptr + dst, tl.load(src_last_to_act_ptr + src))
        tl.store(dst_pot_ptr + dst, tl.load(src_pot_ptr + src))
        tl.store(dst_min_raise_ptr + dst, tl.load(src_min_raise_ptr + src))
        tl.store(
            dst_last_aggressive_amount_ptr + dst,
            tl.load(src_last_aggressive_amount_ptr + src),
        )
        tl.store(
            dst_actions_this_round_ptr + dst,
            tl.load(src_actions_this_round_ptr + src),
        )
        tl.store(
            dst_actions_last_round_ptr + dst,
            tl.load(src_actions_last_round_ptr + src),
        )
        tl.store(dst_scale_ptr + dst, tl.load(src_scale_ptr + src))
        tl.store(dst_done_ptr + dst, tl.load(src_done_ptr + src))
        tl.store(dst_winner_ptr + dst, tl.load(src_winner_ptr + src))

        p_offsets = tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P
        src_p = src * P
        dst_p = dst * P
        tl.store(
            dst_stacks_ptr + dst_p + p_offsets,
            tl.load(src_stacks_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )
        tl.store(
            dst_starting_stacks_ptr + dst_p + p_offsets,
            tl.load(src_starting_stacks_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )
        tl.store(
            dst_committed_ptr + dst_p + p_offsets,
            tl.load(src_committed_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )
        tl.store(
            dst_chips_placed_ptr + dst_p + p_offsets,
            tl.load(src_chips_placed_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )
        tl.store(
            dst_has_folded_ptr + dst_p + p_offsets,
            tl.load(src_has_folded_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )
        tl.store(
            dst_is_allin_ptr + dst_p + p_offsets,
            tl.load(src_is_allin_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )
        tl.store(
            dst_acted_ptr + dst_p + p_offsets,
            tl.load(src_acted_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )
        tl.store(
            dst_winners_ptr + dst_p + p_offsets,
            tl.load(src_winners_ptr + src_p + p_offsets, mask=p_mask, other=0),
            mask=p_mask,
        )

        board_small = tl.arange(0, 8)
        board_small_mask = board_small < 5
        tl.store(
            dst_board_indices_ptr + dst * 5 + board_small,
            tl.load(
                src_board_indices_ptr + src * 5 + board_small,
                mask=board_small_mask,
                other=-1,
            ),
            mask=board_small_mask,
        )
        tl.store(
            dst_last_board_indices_ptr + dst * 5 + board_small,
            tl.load(
                src_last_board_indices_ptr + src * 5 + board_small,
                mask=board_small_mask,
                other=-1,
            ),
            mask=board_small_mask,
        )
        board_offsets = tl.arange(0, BLOCK_BOARD)
        board_mask = board_offsets < 260
        tl.store(
            dst_board_onehot_ptr + dst * 260 + board_offsets,
            tl.load(
                src_board_onehot_ptr + src * 260 + board_offsets,
                mask=board_mask,
                other=0,
            ),
            mask=board_mask,
        )

    @triton.jit
    def _pbs_repeat_indices_kernel(
        repeats_ptr,
        offsets_ptr,
        indices_ptr,
        BLOCK_R: tl.constexpr,
    ):
        row = tl.program_id(0)
        repeats = tl.load(repeats_ptr + row)
        start = tl.load(offsets_ptr + row)
        offsets = tl.arange(0, BLOCK_R)
        mask = offsets < repeats
        tl.store(indices_ptr + start + offsets, row, mask=mask)

    @triton.jit
    def _pbs_repeat_rows_block_kernel(
        repeats_ptr,
        offsets_ptr,
        src_deck_ptr,
        src_deck_pos_ptr,
        src_button_ptr,
        src_street_ptr,
        src_to_act_ptr,
        src_last_to_act_ptr,
        src_pot_ptr,
        src_min_raise_ptr,
        src_last_aggressive_amount_ptr,
        src_actions_this_round_ptr,
        src_actions_last_round_ptr,
        src_stacks_ptr,
        src_starting_stacks_ptr,
        src_scale_ptr,
        src_committed_ptr,
        src_chips_placed_ptr,
        src_has_folded_ptr,
        src_is_allin_ptr,
        src_acted_ptr,
        src_done_ptr,
        src_winner_ptr,
        src_winners_ptr,
        src_board_indices_ptr,
        src_last_board_indices_ptr,
        src_board_onehot_ptr,
        dst_deck_ptr,
        dst_deck_pos_ptr,
        dst_button_ptr,
        dst_street_ptr,
        dst_to_act_ptr,
        dst_last_to_act_ptr,
        dst_pot_ptr,
        dst_min_raise_ptr,
        dst_last_aggressive_amount_ptr,
        dst_actions_this_round_ptr,
        dst_actions_last_round_ptr,
        dst_stacks_ptr,
        dst_starting_stacks_ptr,
        dst_scale_ptr,
        dst_committed_ptr,
        dst_chips_placed_ptr,
        dst_has_folded_ptr,
        dst_is_allin_ptr,
        dst_acted_ptr,
        dst_done_ptr,
        dst_winner_ptr,
        dst_winners_ptr,
        dst_board_indices_ptr,
        dst_last_board_indices_ptr,
        dst_board_onehot_ptr,
        P: tl.constexpr,
        DST_START: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_BOARD: tl.constexpr,
    ):
        src = tl.program_id(0)
        repeats = tl.load(repeats_ptr + src)
        r_offsets = tl.arange(0, BLOCK_R)
        r_mask = r_offsets < repeats
        dsts = DST_START + tl.load(offsets_ptr + src) + r_offsets

        tl.store(dst_deck_pos_ptr + dsts, tl.load(src_deck_pos_ptr + src), mask=r_mask)
        tl.store(dst_button_ptr + dsts, tl.load(src_button_ptr + src), mask=r_mask)
        tl.store(dst_street_ptr + dsts, tl.load(src_street_ptr + src), mask=r_mask)
        tl.store(dst_to_act_ptr + dsts, tl.load(src_to_act_ptr + src), mask=r_mask)
        tl.store(
            dst_last_to_act_ptr + dsts, tl.load(src_last_to_act_ptr + src), mask=r_mask
        )
        tl.store(dst_pot_ptr + dsts, tl.load(src_pot_ptr + src), mask=r_mask)
        tl.store(
            dst_min_raise_ptr + dsts, tl.load(src_min_raise_ptr + src), mask=r_mask
        )
        tl.store(
            dst_last_aggressive_amount_ptr + dsts,
            tl.load(src_last_aggressive_amount_ptr + src),
            mask=r_mask,
        )
        tl.store(
            dst_actions_this_round_ptr + dsts,
            tl.load(src_actions_this_round_ptr + src),
            mask=r_mask,
        )
        tl.store(
            dst_actions_last_round_ptr + dsts,
            tl.load(src_actions_last_round_ptr + src),
            mask=r_mask,
        )
        tl.store(dst_scale_ptr + dsts, tl.load(src_scale_ptr + src), mask=r_mask)
        tl.store(dst_done_ptr + dsts, tl.load(src_done_ptr + src), mask=r_mask)
        tl.store(dst_winner_ptr + dsts, tl.load(src_winner_ptr + src), mask=r_mask)

        deck_offsets = tl.arange(0, 8)
        deck_mask = deck_offsets < 5
        deck = tl.load(src_deck_ptr + src * 5 + deck_offsets, mask=deck_mask, other=0)
        tl.store(
            dst_deck_ptr + dsts[:, None] * 5 + deck_offsets[None, :],
            deck[None, :],
            mask=r_mask[:, None] & deck_mask[None, :],
        )

        p_offsets = tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P
        src_p = src * P
        dst_p = dsts[:, None] * P + p_offsets[None, :]
        tl.store(
            dst_stacks_ptr + dst_p,
            tl.load(src_stacks_ptr + src_p + p_offsets, mask=p_mask, other=0)[None, :],
            mask=r_mask[:, None] & p_mask[None, :],
        )
        tl.store(
            dst_starting_stacks_ptr + dst_p,
            tl.load(src_starting_stacks_ptr + src_p + p_offsets, mask=p_mask, other=0)[
                None, :
            ],
            mask=r_mask[:, None] & p_mask[None, :],
        )
        tl.store(
            dst_committed_ptr + dst_p,
            tl.load(src_committed_ptr + src_p + p_offsets, mask=p_mask, other=0)[
                None, :
            ],
            mask=r_mask[:, None] & p_mask[None, :],
        )
        tl.store(
            dst_chips_placed_ptr + dst_p,
            tl.load(src_chips_placed_ptr + src_p + p_offsets, mask=p_mask, other=0)[
                None, :
            ],
            mask=r_mask[:, None] & p_mask[None, :],
        )
        tl.store(
            dst_has_folded_ptr + dst_p,
            tl.load(src_has_folded_ptr + src_p + p_offsets, mask=p_mask, other=0)[
                None, :
            ],
            mask=r_mask[:, None] & p_mask[None, :],
        )
        tl.store(
            dst_is_allin_ptr + dst_p,
            tl.load(src_is_allin_ptr + src_p + p_offsets, mask=p_mask, other=0)[
                None, :
            ],
            mask=r_mask[:, None] & p_mask[None, :],
        )
        tl.store(
            dst_acted_ptr + dst_p,
            tl.load(src_acted_ptr + src_p + p_offsets, mask=p_mask, other=0)[None, :],
            mask=r_mask[:, None] & p_mask[None, :],
        )
        tl.store(
            dst_winners_ptr + dst_p,
            tl.load(src_winners_ptr + src_p + p_offsets, mask=p_mask, other=0)[None, :],
            mask=r_mask[:, None] & p_mask[None, :],
        )

        board_small = tl.arange(0, 8)
        board_small_mask = board_small < 5
        board = tl.load(
            src_board_indices_ptr + src * 5 + board_small,
            mask=board_small_mask,
            other=-1,
        )
        last_board = tl.load(
            src_last_board_indices_ptr + src * 5 + board_small,
            mask=board_small_mask,
            other=-1,
        )
        tl.store(
            dst_board_indices_ptr + dsts[:, None] * 5 + board_small[None, :],
            board[None, :],
            mask=r_mask[:, None] & board_small_mask[None, :],
        )
        tl.store(
            dst_last_board_indices_ptr + dsts[:, None] * 5 + board_small[None, :],
            last_board[None, :],
            mask=r_mask[:, None] & board_small_mask[None, :],
        )
        board_offsets = tl.arange(0, BLOCK_BOARD)
        board_mask = board_offsets < 260
        board_onehot = tl.load(
            src_board_onehot_ptr + src * 260 + board_offsets,
            mask=board_mask,
            other=0,
        )
        tl.store(
            dst_board_onehot_ptr + dsts[:, None] * 260 + board_offsets[None, :],
            board_onehot[None, :],
            mask=r_mask[:, None] & board_mask[None, :],
        )

    @triton.jit
    def _pbs_legal_kernel(
        bet_bins_ptr,
        to_act_ptr,
        stacks_ptr,
        committed_ptr,
        has_folded_ptr,
        is_allin_ptr,
        done_ptr,
        pot_ptr,
        min_raise_ptr,
        amounts_ptr,
        mask_ptr,
        P: tl.constexpr,
        B: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        row = tl.program_id(0)
        actor = tl.load(to_act_ptr + row)
        row_p = row * P
        p_offsets = tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P

        committed = tl.load(committed_ptr + row_p + p_offsets, mask=p_mask, other=0)
        stacks = tl.load(stacks_ptr + row_p + p_offsets, mask=p_mask, other=0)
        folded = tl.load(has_folded_ptr + row_p + p_offsets, mask=p_mask, other=1)
        allin = tl.load(is_allin_ptr + row_p + p_offsets, mask=p_mask, other=1)
        max_committed = tl.max(committed, axis=0)
        actor_stack = tl.load(stacks_ptr + row_p + actor)
        actor_committed = tl.load(committed_ptr + row_p + actor)
        actor_folded = tl.load(has_folded_ptr + row_p + actor)
        actor_allin = tl.load(is_allin_ptr + row_p + actor)
        done = tl.load(done_ptr + row)
        to_call = max_committed - actor_committed
        can_act = (~done) & (~actor_folded) & (~actor_allin) & (actor_stack > 0)

        b_offsets = tl.arange(0, BLOCK_B)
        b_mask = b_offsets < B
        tl.store(amounts_ptr + row * B + b_offsets, -1, mask=b_mask)
        tl.store(mask_ptr + row * B + b_offsets, False, mask=b_mask)

        tl.store(mask_ptr + row * B, can_act & (to_call > 0))
        tl.store(amounts_ptr + row * B + 1, tl.minimum(to_call, actor_stack))
        tl.store(mask_ptr + row * B + 1, can_act)

        responder = p_mask & (p_offsets != actor) & (~folded) & (~allin) & (stacks > 0)
        live_can_respond = tl.sum(tl.where(responder, 1, 0), axis=0) > 0
        pot = tl.load(pot_ptr + row).to(tl.float32)
        min_raise = tl.load(min_raise_ptr + row)
        preset_offsets = tl.arange(0, BLOCK_B)
        preset_mask = preset_offsets < (B - 3)
        mult = tl.load(bet_bins_ptr + preset_offsets, mask=preset_mask, other=0.0)
        additional = (pot * mult).to(tl.int64)
        bet_amount = to_call + additional
        raise_increment = bet_amount - to_call
        preset_legal = (
            preset_mask
            & can_act
            & live_can_respond
            & (actor_stack > to_call)
            & (bet_amount < actor_stack)
            & (raise_increment >= min_raise)
        )
        tl.store(
            amounts_ptr + row * B + 2 + preset_offsets, bet_amount, mask=preset_mask
        )
        tl.store(
            mask_ptr + row * B + 2 + preset_offsets, preset_legal, mask=preset_mask
        )
        tl.store(amounts_ptr + row * B + B - 1, actor_stack)
        tl.store(mask_ptr + row * B + B - 1, can_act & (actor_stack > 0))

    @triton.jit
    def _pbs_map_bins_kernel(
        bin_ptr,
        amounts_ptr,
        action_ptr,
        selected_amount_ptr,
        B: tl.constexpr,
    ):
        row = tl.program_id(0)
        bin_idx = tl.load(bin_ptr + row)
        valid = bin_idx >= 0
        clamped = tl.minimum(tl.maximum(bin_idx, 0), B - 1)
        selected = tl.load(amounts_ptr + row * B + clamped, mask=valid, other=0)
        action = tl.full((), -1, tl.int64)
        action = tl.where(bin_idx == 0, 0, action)
        action = tl.where(bin_idx == 1, 1, action)
        action = tl.where((bin_idx >= 2) & (bin_idx < B - 1), 2, action)
        action = tl.where(bin_idx == B - 1, 3, action)
        tl.store(action_ptr + row, action)
        tl.store(selected_amount_ptr + row, tl.where(valid, selected, 0))

    @triton.jit
    def _pbs_store_reward_row(
        row,
        row_p,
        p_offsets,
        p_mask,
        winners,
        stacks,
        starting_stacks,
        pot,
        scale,
        rewards_ptr,
    ):
        contributions = starting_stacks - stacks
        levels = contributions
        less = (
            (contributions[None, :] < levels[:, None])
            & p_mask[None, :]
            & p_mask[:, None]
        )
        previous = tl.max(tl.where(less, contributions[None, :], 0), axis=1)
        duplicate_count = tl.maximum(
            tl.sum(
                tl.where(
                    (contributions[None, :] == levels[:, None])
                    & p_mask[None, :]
                    & p_mask[:, None],
                    1,
                    0,
                ),
                axis=1,
            ),
            1,
        )
        widths = tl.maximum(levels - previous, 0).to(tl.float32) / duplicate_count.to(
            tl.float32
        )
        participants = (
            (contributions[None, :] >= levels[:, None])
            & p_mask[None, :]
            & p_mask[:, None]
        )
        participant_count = tl.sum(tl.where(participants, 1, 0), axis=1)
        layer_amount = widths * participant_count.to(tl.float32)
        eligible_winners = participants & winners[None, :]
        winner_count = tl.maximum(tl.sum(tl.where(eligible_winners, 1, 0), axis=1), 1)
        shares = (
            layer_amount[:, None]
            * eligible_winners.to(tl.float32)
            / winner_count[:, None].to(tl.float32)
        )
        payouts = tl.sum(shares, axis=0)
        denom = tl.maximum(starting_stacks.to(tl.float32), 1.0)
        reward = (
            stacks.to(tl.float32) + payouts - starting_stacks.to(tl.float32)
        ) / denom
        tl.store(rewards_ptr + row_p + p_offsets, reward, mask=p_mask)

    @triton.jit
    def _pbs_next_after(
        row_p, start, stacks, folded, allin, P: tl.constexpr, BLOCK_P: tl.constexpr
    ):
        d_offsets = tl.arange(0, BLOCK_P)
        d_mask = d_offsets < P
        distances = d_offsets + 1
        candidates = (start + distances) % P
        can_act = tl.load(stacks + row_p + candidates, mask=d_mask, other=0) > 0
        can_act = (
            can_act
            & (~tl.load(folded + row_p + candidates, mask=d_mask, other=1))
            & (~tl.load(allin + row_p + candidates, mask=d_mask, other=1))
        )
        best_distance = tl.min(tl.where(can_act & d_mask, distances, P + 1), axis=0)
        return tl.where(best_distance <= P, (start + best_distance) % P, start)

    @triton.jit
    def _pbs_write_card(
        row,
        slot,
        card,
        board_indices_ptr,
        board_onehot_ptr,
    ):
        tl.store(board_indices_ptr + row * 5 + slot, card)
        suit = card // 13
        rank = card - suit * 13
        tl.store(board_onehot_ptr + ((row * 5 + slot) * 4 + suit) * 13 + rank, True)

    @triton.jit
    def _pbs_step_kernel(
        action_ptr,
        bet_amount_ptr,
        deck_ptr,
        deck_pos_ptr,
        button_ptr,
        street_ptr,
        to_act_ptr,
        last_to_act_ptr,
        pot_ptr,
        min_raise_ptr,
        last_aggressive_amount_ptr,
        actions_this_round_ptr,
        actions_last_round_ptr,
        stacks_ptr,
        starting_stacks_ptr,
        scale_ptr,
        committed_ptr,
        chips_placed_ptr,
        has_folded_ptr,
        is_allin_ptr,
        acted_ptr,
        done_ptr,
        winner_ptr,
        winners_ptr,
        board_indices_ptr,
        last_board_indices_ptr,
        board_onehot_ptr,
        rewards_ptr,
        new_streets_ptr,
        dealt_cards_ptr,
        bb: tl.constexpr,
        P: tl.constexpr,
        force_heads_up_preflop_flop: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_p = row * P
        p_offsets = tl.arange(0, BLOCK_P)
        p_mask = p_offsets < P

        tl.store(new_streets_ptr + row, -1)
        board_offsets = tl.arange(0, 8)
        board_mask = board_offsets < 5
        tl.store(dealt_cards_ptr + row * 5 + board_offsets, -1, mask=board_mask)
        tl.store(rewards_ptr + row_p + p_offsets, 0.0, mask=p_mask)

        action = tl.load(action_ptr + row)
        done_before = tl.load(done_ptr + row)
        acting = (action >= 0) & (~done_before)
        actor = tl.load(to_act_ptr + row)
        actor_offset = row_p + actor

        last_board = tl.load(
            board_indices_ptr + row * 5 + board_offsets, mask=board_mask, other=-1
        )
        tl.store(
            last_board_indices_ptr + row * 5 + board_offsets,
            last_board,
            mask=acting & board_mask,
        )

        committed = tl.load(committed_ptr + row_p + p_offsets, mask=p_mask, other=0)
        chips_placed = tl.load(
            chips_placed_ptr + row_p + p_offsets, mask=p_mask, other=0
        )
        stacks = tl.load(stacks_ptr + row_p + p_offsets, mask=p_mask, other=0)
        folded = tl.load(has_folded_ptr + row_p + p_offsets, mask=p_mask, other=1)
        allin = tl.load(is_allin_ptr + row_p + p_offsets, mask=p_mask, other=1)
        acted = tl.load(acted_ptr + row_p + p_offsets, mask=p_mask, other=0)
        starting_stacks = tl.load(
            starting_stacks_ptr + row_p + p_offsets, mask=p_mask, other=0
        )
        max_committed_before = tl.max(committed, axis=0)
        actor_stack = tl.load(stacks_ptr + actor_offset)
        actor_committed = tl.load(committed_ptr + actor_offset)
        to_call = max_committed_before - actor_committed
        call_amount = tl.minimum(to_call, actor_stack)
        bet_amount = tl.minimum(
            tl.maximum(tl.load(bet_amount_ptr + row), 0), actor_stack
        )

        put_in = tl.full((), 0, tl.int64)
        put_in = tl.where(action == 1, call_amount, put_in)
        put_in = tl.where(action == 2, bet_amount, put_in)
        put_in = tl.where(action == 3, actor_stack, put_in)
        put_in = tl.where(acting, put_in, 0)

        actor_lane = p_offsets == actor
        new_actor_stack = actor_stack - put_in
        new_actor_committed = actor_committed + put_in
        new_actor_chips_placed = tl.load(chips_placed_ptr + actor_offset) + put_in
        stacks = tl.where(actor_lane & p_mask, new_actor_stack, stacks)
        committed = tl.where(actor_lane & p_mask, new_actor_committed, committed)
        chips_placed = tl.where(
            actor_lane & p_mask, new_actor_chips_placed, chips_placed
        )
        folded = tl.where(actor_lane & p_mask, folded | (action == 0), folded)
        allin = tl.where(
            actor_lane & p_mask,
            allin | ((put_in > 0) & (new_actor_stack == 0)),
            allin,
        )

        tl.store(last_to_act_ptr + row, actor, mask=acting)
        tl.store(stacks_ptr + actor_offset, new_actor_stack, mask=acting)
        tl.store(committed_ptr + actor_offset, new_actor_committed, mask=acting)
        tl.store(
            chips_placed_ptr + actor_offset,
            new_actor_chips_placed,
            mask=acting,
        )
        tl.store(pot_ptr + row, tl.load(pot_ptr + row) + put_in, mask=acting)
        tl.store(
            has_folded_ptr + row_p + p_offsets,
            folded,
            mask=acting & actor_lane & p_mask,
        )
        tl.store(
            is_allin_ptr + row_p + p_offsets, allin, mask=acting & actor_lane & p_mask
        )

        raise_increment = new_actor_committed - max_committed_before
        aggressive = acting & (raise_increment > 0) & ((action == 2) | (action == 3))
        old_min_raise = tl.load(min_raise_ptr + row)
        tl.store(
            min_raise_ptr + row,
            tl.where(
                aggressive, tl.maximum(old_min_raise, raise_increment), old_min_raise
            ),
            mask=acting,
        )
        old_last_aggressive = tl.load(last_aggressive_amount_ptr + row)
        tl.store(
            last_aggressive_amount_ptr + row,
            tl.where(aggressive, new_actor_committed, old_last_aggressive),
            mask=acting,
        )
        acted = tl.where(
            aggressive, actor_lane & p_mask, tl.where(actor_lane & p_mask, True, acted)
        )
        tl.store(acted_ptr + row_p + p_offsets, acted, mask=acting & p_mask)
        tl.store(
            actions_this_round_ptr + row,
            tl.load(actions_this_round_ptr + row) + 1,
            mask=acting,
        )

        live = p_mask & (~folded)
        live_count = tl.sum(tl.where(live, 1, 0), axis=0)
        terminal_fold = acting & (live_count <= 1)
        first_live = tl.min(tl.where(live, p_offsets, P), axis=0)
        tl.store(done_ptr + row, True, mask=terminal_fold)
        tl.store(winner_ptr + row, first_live, mask=terminal_fold)
        tl.store(winners_ptr + row_p + p_offsets, live, mask=terminal_fold & p_mask)

        done_after_fold = done_before | terminal_fold
        eligible = live & (~allin) & (stacks > 0)
        eligible_count = tl.sum(tl.where(eligible, 1, 0), axis=0)
        max_committed = tl.max(committed, axis=0)
        matched = committed == max_committed
        ready = (~eligible) | (acted & matched)
        all_ready = tl.min(tl.where(p_mask, ready.to(tl.int32), 1), axis=0) == 1
        no_more_betting = (
            acting & (~done_after_fold) & (live_count > 1) & (eligible_count == 0)
        )
        round_closed = acting & (~done_after_fold) & (eligible_count > 0) & all_ready

        pot = tl.load(pot_ptr + row)
        scale = tl.load(scale_ptr + row)
        tl.store(rewards_ptr + row_p + p_offsets, 0.0, mask=p_mask)
        if terminal_fold | no_more_betting:
            _pbs_store_reward_row(
                row,
                row_p,
                p_offsets,
                p_mask,
                live,
                stacks,
                starting_stacks,
                pot,
                scale,
                rewards_ptr,
            )

        old_street = tl.load(street_ptr + row)
        deck_pos = tl.load(deck_pos_ptr + row)
        if force_heads_up_preflop_flop:
            live_allin_count = tl.sum(tl.where(live & allin, 1, 0), axis=0)
            allin_call_leaf = (eligible_count == 1) & (live_allin_count > 0)
            force_hu = (
                round_closed & (old_street == 0) & (live_count > 2) & (~allin_call_leaf)
            )
            other_live = live & (~actor_lane)
            other_score = chips_placed * (P + 1) + (P - p_offsets)
            other_score = tl.where(other_live, other_score, -1)
            keep_other_score = tl.max(other_score, axis=0)
            keep_other = other_live & (other_score == keep_other_score)
            force_fold = force_hu & live & (~actor_lane) & (~keep_other)
            folded = folded | force_fold
            allin = allin & (~force_fold)
            tl.store(has_folded_ptr + row_p + p_offsets, folded, mask=force_hu & p_mask)
            tl.store(is_allin_ptr + row_p + p_offsets, allin, mask=force_hu & p_mask)

        if no_more_betting:
            c0 = tl.load(deck_ptr + row * 5)
            c1 = tl.load(deck_ptr + row * 5 + 1)
            c2 = tl.load(deck_ptr + row * 5 + 2)
            c3 = tl.load(deck_ptr + row * 5 + 3)
            c4 = tl.load(deck_ptr + row * 5 + 4)
            _pbs_write_card(row, 0, c0, board_indices_ptr, board_onehot_ptr)
            _pbs_write_card(row, 1, c1, board_indices_ptr, board_onehot_ptr)
            _pbs_write_card(row, 2, c2, board_indices_ptr, board_onehot_ptr)
            _pbs_write_card(row, 3, c3, board_indices_ptr, board_onehot_ptr)
            _pbs_write_card(row, 4, c4, board_indices_ptr, board_onehot_ptr)
            tl.store(deck_pos_ptr + row, 5)
            tl.store(street_ptr + row, 4)
            tl.store(done_ptr + row, True)
            tl.store(winners_ptr + row_p + p_offsets, live, mask=p_mask)
            tl.store(winner_ptr + row, tl.where(live_count == 1, first_live, -1))
            tl.store(new_streets_ptr + row, 4)
            tl.store(dealt_cards_ptr + row * 5, c0)
            tl.store(dealt_cards_ptr + row * 5 + 1, c1)
            tl.store(dealt_cards_ptr + row * 5 + 2, c2)
            tl.store(dealt_cards_ptr + row * 5 + 3, c3)
            tl.store(dealt_cards_ptr + row * 5 + 4, c4)

        if round_closed:
            tl.store(committed_ptr + row_p + p_offsets, 0, mask=p_mask)
            tl.store(
                actions_last_round_ptr + row, tl.load(actions_this_round_ptr + row)
            )
            tl.store(actions_this_round_ptr + row, 0)
            tl.store(acted_ptr + row_p + p_offsets, False, mask=p_mask)
            tl.store(min_raise_ptr + row, bb)
            tl.store(last_aggressive_amount_ptr + row, 0)
            if old_street == 3:
                tl.store(done_ptr + row, True)
                tl.store(street_ptr + row, 4)
                tl.store(winners_ptr + row_p + p_offsets, live, mask=p_mask)
                tl.store(winner_ptr + row, tl.where(live_count == 1, first_live, -1))
                tl.store(new_streets_ptr + row, 4)
                _pbs_store_reward_row(
                    row,
                    row_p,
                    p_offsets,
                    p_mask,
                    live,
                    stacks,
                    starting_stacks,
                    pot,
                    scale,
                    rewards_ptr,
                )
            elif old_street == 0:
                c0 = tl.load(deck_ptr + row * 5 + deck_pos)
                c1 = tl.load(deck_ptr + row * 5 + deck_pos + 1)
                c2 = tl.load(deck_ptr + row * 5 + deck_pos + 2)
                _pbs_write_card(row, 0, c0, board_indices_ptr, board_onehot_ptr)
                _pbs_write_card(row, 1, c1, board_indices_ptr, board_onehot_ptr)
                _pbs_write_card(row, 2, c2, board_indices_ptr, board_onehot_ptr)
                tl.store(deck_pos_ptr + row, deck_pos + 3)
                tl.store(street_ptr + row, 1)
                tl.store(new_streets_ptr + row, 1)
                tl.store(dealt_cards_ptr + row * 5, c0)
                tl.store(dealt_cards_ptr + row * 5 + 1, c1)
                tl.store(dealt_cards_ptr + row * 5 + 2, c2)
            elif old_street == 1:
                c = tl.load(deck_ptr + row * 5 + deck_pos)
                _pbs_write_card(row, 3, c, board_indices_ptr, board_onehot_ptr)
                tl.store(deck_pos_ptr + row, deck_pos + 1)
                tl.store(street_ptr + row, 2)
                tl.store(new_streets_ptr + row, 2)
                tl.store(dealt_cards_ptr + row * 5, c)
            elif old_street == 2:
                c = tl.load(deck_ptr + row * 5 + deck_pos)
                _pbs_write_card(row, 4, c, board_indices_ptr, board_onehot_ptr)
                tl.store(deck_pos_ptr + row, deck_pos + 1)
                tl.store(street_ptr + row, 3)
                tl.store(new_streets_ptr + row, 3)
                tl.store(dealt_cards_ptr + row * 5, c)

        next_after_actor = _pbs_next_after(
            row_p, actor, stacks_ptr, has_folded_ptr, is_allin_ptr, P, BLOCK_P
        )
        button = tl.load(button_ptr + row)
        next_after_button = _pbs_next_after(
            row_p, button, stacks_ptr, has_folded_ptr, is_allin_ptr, P, BLOCK_P
        )
        final_done = tl.load(done_ptr + row)
        tl.store(to_act_ptr + row, next_after_actor, mask=acting & (~final_done))
        tl.store(to_act_ptr + row, next_after_button, mask=round_closed & (~final_done))
