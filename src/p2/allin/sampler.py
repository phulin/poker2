from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from p2.allin.data import PreflopAllInBatch
from p2.allin.kernels import (
    AllinCdfWorkspace,
    allin_cdf_tuple_score_from_roots_accumulate_cuda_,
    allin_cdf_tuple_score_from_roots_masks_accumulate_cuda_,
    make_allin_cdf_workspace,
    triton_available,
)
from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import (
    rank_hand_scores_triton,
    triton_is_available as rules_triton_available,
)
from p2.search.allin_payoff import (
    I16_SCALE,
    allin_values_from_payoff_reference,
    load_preflop_payoff_i16,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREFLOP_ALLIN_TABLE = _REPO_ROOT / "outputs" / "preflop_allin_table.pt.zst"
NUM_FULL_BOARDS = 2_598_960


def _combo_masks(device: torch.device) -> torch.Tensor:
    combos = hand_combos_tensor(device=device)
    return ((1 << combos[:, 0]) | (1 << combos[:, 1])).to(torch.int64)


def _board_allowed_from_combo_masks(
    board: torch.Tensor,
    combo_masks: torch.Tensor,
) -> torch.Tensor:
    valid = board >= 0
    card_bits = torch.bitwise_left_shift(
        torch.ones_like(board, dtype=torch.int64),
        board.to(torch.int64).clamp_min(0),
    )
    board_masks = torch.where(valid, card_bits, torch.zeros_like(card_bits)).sum(dim=-1)
    return (combo_masks[None, :] & board_masks[:, None]) == 0


def _board_masks(board: torch.Tensor) -> torch.Tensor:
    valid = board >= 0
    card_bits = torch.bitwise_left_shift(
        torch.ones_like(board, dtype=torch.int64),
        board.to(torch.int64).clamp_min(0),
    )
    return torch.where(valid, card_bits, torch.zeros_like(card_bits)).sum(dim=-1)


def _sample_full_boards(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    scores = torch.rand(count, 52, device=device, generator=generator)
    return torch.topk(scores, 5, dim=1).indices


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    if device.type == "cuda" and device.index is None:
        return device.type, torch.cuda.current_device()
    return device.type, device.index


@lru_cache(maxsize=4)
def _full_boards_cached(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    cards = torch.arange(52, dtype=torch.long)
    boards = torch.combinations(cards, r=5).contiguous()
    return boards.to(device=device, non_blocking=True)


def _full_boards(device: torch.device) -> torch.Tensor:
    boards = _full_boards_cached(*_device_cache_key(device))
    if boards.shape != (NUM_FULL_BOARDS, 5):
        raise RuntimeError(
            f"expected {NUM_FULL_BOARDS} full boards, got {boards.shape}"
        )
    return boards


def _rank_hands(board: torch.Tensor) -> torch.Tensor:
    if board.device.type == "cuda" and rules_triton_available():
        return rank_hand_scores_triton(board).contiguous()
    ranks, _ = rank_hands_torch(board.int())
    return ranks.to(torch.int32).contiguous()


def _side_pot_layers(
    batch: PreflopAllInBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    contrib = batch.committed
    levels = contrib.sort(dim=1).values
    previous = torch.cat((torch.zeros_like(levels[:, :1]), levels[:, :-1]), dim=1)
    widths = (levels - previous).clamp_min(0.0)
    participants = contrib[:, None, :] >= levels[:, :, None]
    layer_amount = widths * participants.to(contrib.dtype).sum(dim=2)
    eligible = participants & (~batch.folded_mask[:, None, :])
    return layer_amount, eligible


def _slice_batch(batch: PreflopAllInBatch, rows: torch.Tensor) -> PreflopAllInBatch:
    return PreflopAllInBatch(
        beliefs=batch.beliefs.index_select(0, rows),
        starting_stacks=batch.starting_stacks.index_select(0, rows),
        committed=batch.committed.index_select(0, rows),
        stacks_after=batch.stacks_after.index_select(0, rows),
        allin_mask=batch.allin_mask.index_select(0, rows),
        folded_mask=batch.folded_mask.index_select(0, rows),
        scale=batch.scale.index_select(0, rows),
    )


@torch.no_grad()
def _estimate_two_player_preflop_exact(
    batch: PreflopAllInBatch,
    *,
    preflop_table_path: str | Path,
) -> torch.Tensor:
    """Exact preflop all-in values for rows with exactly two live players."""
    device = batch.beliefs.device
    B, P, H = batch.beliefs.shape
    if H != NUM_HANDS:
        raise ValueError(f"expected {NUM_HANDS} hands, got {H}")
    live_mask = ~batch.folded_mask
    live_counts = live_mask.sum(dim=1)
    if not torch.equal(live_counts, torch.full_like(live_counts, 2)):
        raise ValueError(
            "exact preflop all-in table requires exactly two live players per row"
        )

    live_coords = torch.nonzero(live_mask, as_tuple=False)
    live_indices = live_coords[:, 1].reshape(B, 2)
    pair_beliefs = batch.beliefs.gather(
        1,
        live_indices[:, :, None].expand(-1, -1, H),
    )
    table = load_preflop_payoff_i16(preflop_table_path, device)
    payoff_ev = allin_values_from_payoff_reference(
        table,
        pair_beliefs,
        scale=I16_SCALE,
    ).to(torch.float32)
    showdown_share = (0.5 * (payoff_ev + 1.0)).clamp(0.0, 1.0)

    layer_amount, eligible = _side_pot_layers(batch)
    pair_eligible = eligible.gather(
        2,
        live_indices[:, None, :].expand(-1, P, -1),
    )
    layer_live_count = pair_eligible.sum(dim=2)
    pair_payout = torch.zeros(B, 2, H, dtype=torch.float32, device=device)
    for slot in range(2):
        hero_eligible = pair_eligible[:, :, slot]
        uncontested = (
            layer_amount
            * hero_eligible.to(torch.float32)
            * (layer_live_count == 1).to(torch.float32)
        ).sum(dim=1)
        contested = (
            layer_amount
            * hero_eligible.to(torch.float32)
            * (layer_live_count == 2).to(torch.float32)
        ).sum(dim=1)
        pair_payout[:, slot] = (
            uncontested[:, None] + contested[:, None] * showdown_share[:, slot]
        )

    folded_value = (batch.stacks_after - batch.starting_stacks) / batch.scale[
        :, None
    ].clamp_min(1.0)
    values = folded_value[:, :, None].expand(-1, -1, H).clone()
    pair_starting = batch.starting_stacks.gather(1, live_indices)
    pair_after = batch.stacks_after.gather(1, live_indices)
    pair_values = (
        pair_after[:, :, None].to(torch.float32)
        + pair_payout
        - pair_starting[:, :, None].to(torch.float32)
    ) / batch.scale[:, None, None].to(torch.float32).clamp_min(1.0)
    values.scatter_(1, live_indices[:, :, None].expand(-1, -1, H), pair_values)
    return values.to(batch.beliefs.dtype)


def _resolve_sample_split(
    sample_count: int | None,
    board_samples: int | None,
    tuple_samples: int | None,
    *,
    exhaustive_boards: bool = False,
) -> tuple[int, int]:
    if sample_count is not None and sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if board_samples is not None and board_samples <= 0:
        raise ValueError("board_samples must be positive")
    if tuple_samples is not None and tuple_samples <= 0:
        raise ValueError("tuple_samples must be positive")
    if exhaustive_boards:
        if board_samples is not None and board_samples != NUM_FULL_BOARDS:
            raise ValueError(
                f"exhaustive board mode requires board_samples={NUM_FULL_BOARDS}"
            )
        if tuple_samples is None:
            raise ValueError(
                "tuple_samples is required in exhaustive board mode; it is the "
                "number of opponent tuple samples per full board"
            )
        return NUM_FULL_BOARDS, tuple_samples
    if sample_count is None:
        if board_samples is None or tuple_samples is None:
            raise ValueError(
                "board_samples and tuple_samples are required without sample_count"
            )
        return board_samples, tuple_samples
    if board_samples is None and tuple_samples is None:
        board_samples = min(256, sample_count)
        tuple_samples = (sample_count + board_samples - 1) // board_samples
    elif board_samples is None:
        assert tuple_samples is not None
        board_samples = (sample_count + tuple_samples - 1) // tuple_samples
    elif tuple_samples is None:
        tuple_samples = (sample_count + board_samples - 1) // board_samples
    return board_samples, tuple_samples


@dataclass
class PreflopAllInEstimatorWorkspace:
    """Reusable buffers for online preflop all-in target estimation."""

    cdf_workspace: AllinCdfWorkspace | None = None
    payout_sum: torch.Tensor | None = None
    denom_sum: torch.Tensor | None = None
    max_roots: int = 0
    players: int = 0
    hands: int = 0
    board_chunk: int = 0
    tuple_samples: int = 0
    tuple_tries: int = 0
    device: torch.device | None = None

    def ensure(
        self,
        *,
        roots: int,
        players: int,
        hands: int,
        board_chunk: int,
        tuple_samples: int,
        tuple_tries: int,
        board_samples: int,
        device: torch.device,
    ) -> tuple[AllinCdfWorkspace, torch.Tensor, torch.Tensor]:
        max_roots = max(int(roots), self.max_roots)
        full_chunk = min(int(board_chunk), int(board_samples))
        needs_new = (
            self.cdf_workspace is None
            or self.payout_sum is None
            or self.denom_sum is None
            or self.max_roots < roots
            or self.players != players
            or self.hands != hands
            or self.board_chunk != full_chunk
            or self.tuple_samples != tuple_samples
            or self.tuple_tries != tuple_tries
            or self.device != device
        )
        if needs_new:
            self.cdf_workspace = make_allin_cdf_workspace(
                max_rows=max_roots * full_chunk,
                players=players,
                sample_count=tuple_samples,
                tuple_tries=tuple_tries,
                device=device,
                hands=hands,
            )
            self.payout_sum = torch.empty(
                max_roots,
                players,
                hands,
                dtype=torch.float32,
                device=device,
            )
            self.denom_sum = torch.empty_like(self.payout_sum)
            self.max_roots = max_roots
            self.players = players
            self.hands = hands
            self.board_chunk = full_chunk
            self.tuple_samples = tuple_samples
            self.tuple_tries = tuple_tries
            self.device = device
        assert self.cdf_workspace is not None
        assert self.payout_sum is not None
        assert self.denom_sum is not None
        payout = self.payout_sum[:roots]
        denom = self.denom_sum[:roots]
        payout.zero_()
        denom.zero_()
        return self.cdf_workspace, payout, denom


@torch.no_grad()
def _estimate_preflop_allin_values_triton(
    batch: PreflopAllInBatch,
    *,
    board_samples: int,
    tuple_samples: int,
    tuple_tries: int,
    board_chunk: int,
    generator: torch.Generator | None,
    compute_stats: bool = True,
    workspace: PreflopAllInEstimatorWorkspace | None = None,
    use_board_masks: bool = True,
    skip_folded_heroes: bool = True,
    exhaustive_boards: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    start = time.perf_counter()
    device = batch.beliefs.device
    beliefs = batch.beliefs.to(torch.float32)
    B, P, H = beliefs.shape
    if H != NUM_HANDS:
        raise ValueError(f"expected {NUM_HANDS} hands, got {H}")

    combo_masks = _combo_masks(device)
    live_mask = ~batch.folded_mask
    layer_amount, eligible = _side_pot_layers(batch)
    # Convert per-root side tensors once; chunks just index_select into int8/f32.
    live_mask_i8 = live_mask.to(torch.int8)
    eligible_i8 = eligible.to(torch.int8)
    layer_amount_f32 = layer_amount.to(torch.float32)
    kernel_launch_seconds = 0.0

    if workspace is None:
        workspace = PreflopAllInEstimatorWorkspace()
    cdf_workspace, payout_sum, denom_sum = workspace.ensure(
        roots=B,
        players=P,
        hands=H,
        board_chunk=board_chunk,
        tuple_samples=tuple_samples,
        tuple_tries=tuple_tries,
        board_samples=board_samples,
        device=device,
    )
    full_chunk = min(board_chunk, board_samples)
    full_root_ids = torch.arange(B, device=device).repeat_interleave(full_chunk)
    full_live_rep = live_mask_i8.index_select(0, full_root_ids)
    full_layer_amount_rep = layer_amount_f32.index_select(0, full_root_ids)
    full_eligible_rep = eligible_i8.index_select(0, full_root_ids)

    done_boards = 0
    while done_boards < board_samples:
        cur_boards = min(board_chunk, board_samples - done_boards)
        row_count = B * cur_boards
        if cur_boards == full_chunk:
            root_ids = full_root_ids
            live_rep = full_live_rep
            layer_amount_rep = full_layer_amount_rep
            eligible_rep = full_eligible_rep
        else:
            root_ids = torch.arange(B, device=device).repeat_interleave(cur_boards)
            live_rep = live_mask_i8.index_select(0, root_ids)
            layer_amount_rep = layer_amount_f32.index_select(0, root_ids)
            eligible_rep = eligible_i8.index_select(0, root_ids)
        if exhaustive_boards:
            board_chunk_rows = _full_boards(device)[
                done_boards : done_boards + cur_boards
            ]
            boards = board_chunk_rows.repeat(B, 1)
            ranks = _rank_hands(board_chunk_rows).repeat(B, 1)
        else:
            boards = _sample_full_boards(row_count, device=device, generator=generator)
            ranks = _rank_hands(boards)
        launch_start = time.perf_counter()
        if use_board_masks:
            board_masks = _board_masks(boards).contiguous()
            allin_cdf_tuple_score_from_roots_masks_accumulate_cuda_(
                cdf_workspace,
                beliefs=beliefs,
                board_masks=board_masks,
                boards_per_root=cur_boards,
                hand_masks=combo_masks,
                hand_ranks=ranks,
                live_mask=live_rep,
                layer_amount=layer_amount_rep,
                eligible=eligible_rep,
                payout_sum=payout_sum,
                denom_sum=denom_sum,
                seed=(done_boards + 1) * 104_729,
                skip_folded_heroes=skip_folded_heroes,
            )
        else:
            allowed = _board_allowed_from_combo_masks(boards, combo_masks).contiguous()
            allin_cdf_tuple_score_from_roots_accumulate_cuda_(
                cdf_workspace,
                beliefs=beliefs,
                board_allowed=allowed,
                boards_per_root=cur_boards,
                hand_masks=combo_masks,
                hand_ranks=ranks,
                live_mask=live_rep,
                layer_amount=layer_amount_rep,
                eligible=eligible_rep,
                payout_sum=payout_sum,
                denom_sum=denom_sum,
                seed=(done_boards + 1) * 104_729,
            )
        kernel_launch_seconds += time.perf_counter() - launch_start
        done_boards += cur_boards

    expected_payout = torch.where(
        denom_sum > 0.0,
        payout_sum / denom_sum.clamp_min(1.0e-30),
        torch.zeros_like(payout_sum),
    )
    values = (
        batch.stacks_after[:, :, None].to(torch.float32)
        + expected_payout
        - batch.starting_stacks[:, :, None].to(torch.float32)
    ) / batch.scale[:, None, None].to(torch.float32).clamp_min(1.0)
    folded_value = (batch.stacks_after - batch.starting_stacks) / batch.scale[
        :, None
    ].clamp_min(1.0)
    values = torch.where(
        batch.folded_mask[:, :, None], folded_value[:, :, None], values
    )
    if not compute_stats:
        return values, {}
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total_samples = B * board_samples * tuple_samples
    diagnostics = {
        "target_seconds": elapsed,
        "target_kernel_launch_seconds": kernel_launch_seconds,
        "target_boards_per_second": float(B * board_samples / max(elapsed, 1.0e-9)),
        "target_samples_per_second": float(total_samples / max(elapsed, 1.0e-9)),
        "target_zero_denom_frac": float((denom_sum == 0).float().mean().item()),
        "target_value_mean": float(values.mean().item()),
        "target_value_std": float(values.std().item()),
        "target_board_samples": float(board_samples),
        "target_tuple_samples": float(tuple_samples),
        "target_exhaustive_boards": float(exhaustive_boards),
    }
    return values, diagnostics


@torch.no_grad()
def estimate_preflop_allin_values(
    batch: PreflopAllInBatch,
    *,
    sample_count: int | None = 50_000,
    board_samples: int | None = None,
    tuple_samples: int | None = None,
    tuple_tries: int = 4,
    board_chunk: int = 8,
    hand_chunk: int = 128,
    generator: torch.Generator | None = None,
    preflop_table_path: str | Path = DEFAULT_PREFLOP_ALLIN_TABLE,
    use_exact_two_player: bool = True,
    compute_stats: bool = True,
    workspace: PreflopAllInEstimatorWorkspace | None = None,
    use_board_masks: bool = True,
    skip_folded_heroes: bool = True,
    exhaustive_boards: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Estimate preflop all-in chip-normalized values by sampling full boards.

    This is a preflop extension of the fixed-river tuple-reject by-hand sampler:
    each sampled complete board becomes a river-board row, opponent tuples are
    sampled with rejection to avoid private-card collisions, and every hero hand
    is scored against those tuples. Side-pot payouts are accumulated per layer.

    When ``compute_stats`` is false the returned diagnostics dict is empty and the
    sampler skips the host-side reductions (``.item()`` D2H copies) and the final
    ``torch.cuda.synchronize`` used to populate it. This lets the caller's CPU run
    ahead and overlap the next step's work; pass ``True`` only on steps that
    actually log the diagnostics.

    When ``exhaustive_boards`` is true, ``tuple_samples`` is interpreted as a
    fixed number of opponent tuple samples per one of the 2,598,960 possible
    five-card boards. Random board sampling and ``sample_count`` are bypassed.
    """
    board_samples, tuple_samples = _resolve_sample_split(
        sample_count,
        board_samples,
        tuple_samples,
        exhaustive_boards=exhaustive_boards,
    )
    if tuple_tries <= 0:
        raise ValueError("tuple_tries must be positive")
    if board_chunk <= 0 or hand_chunk <= 0:
        raise ValueError("board_chunk and hand_chunk must be positive")

    start = time.perf_counter()
    device = batch.beliefs.device
    live_counts = (~batch.folded_mask).sum(dim=1)
    exact_mask = (
        live_counts == 2
        if use_exact_two_player
        else torch.zeros_like(live_counts, dtype=torch.bool)
    )
    if bool(exact_mask.any()):
        exact_rows = torch.nonzero(exact_mask, as_tuple=False).flatten()
        mc_rows = torch.nonzero(~exact_mask, as_tuple=False).flatten()
        values = torch.empty_like(batch.beliefs)
        exact_values = _estimate_two_player_preflop_exact(
            _slice_batch(batch, exact_rows),
            preflop_table_path=preflop_table_path,
        )
        values.index_copy_(0, exact_rows, exact_values)
        mc_diag: dict[str, float] = {}
        if mc_rows.numel() > 0:
            mc_values, mc_diag = estimate_preflop_allin_values(
                _slice_batch(batch, mc_rows),
                sample_count=None,
                board_samples=board_samples,
                tuple_samples=tuple_samples,
                tuple_tries=tuple_tries,
                board_chunk=board_chunk,
                hand_chunk=hand_chunk,
                generator=generator,
                preflop_table_path=preflop_table_path,
                use_exact_two_player=False,
                compute_stats=compute_stats,
                workspace=workspace,
                use_board_masks=use_board_masks,
                skip_folded_heroes=skip_folded_heroes,
                exhaustive_boards=exhaustive_boards,
            )
            values.index_copy_(0, mc_rows, mc_values)
        if not compute_stats:
            return values, {}
        elapsed = time.perf_counter() - start
        diagnostics = {
            "target_seconds": elapsed,
            "target_boards_per_second": mc_diag.get("target_boards_per_second", 0.0),
            "target_samples_per_second": mc_diag.get("target_samples_per_second", 0.0),
            "target_zero_denom_frac": mc_diag.get("target_zero_denom_frac", 0.0),
            "target_value_mean": float(values.mean().item()),
            "target_value_std": float(values.std().item()),
            "target_board_samples": float(board_samples),
            "target_tuple_samples": float(tuple_samples),
            "target_exhaustive_boards": float(exhaustive_boards),
            "target_exact_two_player_rows": float(exact_rows.numel()),
            "target_mc_rows": float(mc_rows.numel()),
        }
        if "target_kernel_launch_seconds" in mc_diag:
            diagnostics["target_kernel_launch_seconds"] = mc_diag[
                "target_kernel_launch_seconds"
            ]
        return values, diagnostics

    if device.type == "cuda" and triton_available():
        return _estimate_preflop_allin_values_triton(
            batch,
            board_samples=board_samples,
            tuple_samples=tuple_samples,
            tuple_tries=tuple_tries,
            board_chunk=board_chunk,
            generator=generator,
            compute_stats=compute_stats,
            workspace=workspace,
            use_board_masks=use_board_masks,
            skip_folded_heroes=skip_folded_heroes,
            exhaustive_boards=exhaustive_boards,
        )

    beliefs = batch.beliefs.to(torch.float32)
    B, P, H = beliefs.shape
    if H != NUM_HANDS:
        raise ValueError(f"expected {NUM_HANDS} hands, got {H}")

    combo_masks = _combo_masks(device)
    live_mask = ~batch.folded_mask
    layer_amount, eligible = _side_pot_layers(batch)
    payout_sum = torch.zeros(B, P, H, dtype=torch.float32, device=device)
    denom_sum = torch.zeros_like(payout_sum)
    full_chunk = min(board_chunk, board_samples)
    full_root_ids = torch.arange(B, device=device).repeat_interleave(full_chunk)
    full_live_rep = live_mask.index_select(0, full_root_ids)
    full_layer_amount_rep = layer_amount.index_select(0, full_root_ids)
    full_eligible_rep = eligible.index_select(0, full_root_ids)

    done_boards = 0
    while done_boards < board_samples:
        cur_boards = min(board_chunk, board_samples - done_boards)
        row_count = B * cur_boards
        if cur_boards == full_chunk:
            root_ids = full_root_ids
            live_rep = full_live_rep
            layer_amount_rep = full_layer_amount_rep
            eligible_rep = full_eligible_rep
        else:
            root_ids = torch.arange(B, device=device).repeat_interleave(cur_boards)
            live_rep = live_mask.index_select(0, root_ids)
            layer_amount_rep = layer_amount.index_select(0, root_ids)
            eligible_rep = eligible.index_select(0, root_ids)
        if exhaustive_boards:
            board_chunk_rows = _full_boards(device)[
                done_boards : done_boards + cur_boards
            ]
            boards = board_chunk_rows.repeat(B, 1)
            ranks = _rank_hands(board_chunk_rows).repeat(B, 1)
        else:
            boards = _sample_full_boards(row_count, device=device, generator=generator)
            ranks = _rank_hands(boards)
        allowed = _board_allowed_from_combo_masks(boards, combo_masks)
        board_beliefs = beliefs.index_select(0, root_ids)
        board_beliefs.masked_fill_(~allowed[:, None, :], 0.0)
        board_beliefs = board_beliefs / board_beliefs.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-30)
        flat_beliefs = board_beliefs.reshape(row_count * P, H)

        processed_samples = 0
        sample_chunk = min(64, tuple_samples)
        while processed_samples < tuple_samples:
            cur_samples = min(sample_chunk, tuple_samples - processed_samples)
            candidates = torch.multinomial(
                flat_beliefs,
                cur_samples * tuple_tries,
                replacement=True,
                generator=generator,
            ).reshape(row_count, P, tuple_tries, cur_samples)
            candidate_masks = combo_masks.index_select(
                0, candidates.reshape(-1)
            ).reshape(
                row_count,
                P,
                tuple_tries,
                cur_samples,
            )
            candidate_ranks = ranks.gather(
                1,
                candidates.reshape(row_count, P * tuple_tries * cur_samples),
            ).reshape(row_count, P, tuple_tries, cur_samples)

            for hero in range(P):
                hero_live = live_rep[:, hero]
                opp_live = live_rep.clone()
                opp_live[:, hero] = False
                alive = torch.zeros(
                    row_count, cur_samples, dtype=torch.bool, device=device
                )
                selected = torch.zeros(
                    row_count, P, cur_samples, dtype=torch.long, device=device
                )
                selected_ranks = torch.full(
                    (row_count, P, cur_samples),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
                for attempt in range(tuple_tries):
                    hands = candidates[:, :, attempt, :]
                    masks = candidate_masks[:, :, attempt, :]
                    valid = hero_live[:, None].expand(-1, cur_samples).clone()
                    for left in range(P):
                        for right in range(left + 1, P):
                            both = opp_live[:, left] & opp_live[:, right]
                            valid &= (~both[:, None]) | (
                                (masks[:, left] & masks[:, right]) == 0
                            )
                    take = (~alive) & valid
                    selected = torch.where(take[:, None, :], hands, selected)
                    selected_ranks = torch.where(
                        take[:, None, :],
                        candidate_ranks[:, :, attempt, :],
                        selected_ranks,
                    )
                    alive |= take

                selected_masks = combo_masks.index_select(
                    0, selected.reshape(-1)
                ).reshape(
                    row_count,
                    P,
                    cur_samples,
                )
                used_mask = torch.zeros(
                    row_count, cur_samples, dtype=torch.int64, device=device
                )
                for player in range(P):
                    used_mask |= torch.where(
                        opp_live[:, player, None],
                        selected_masks[:, player],
                        torch.zeros_like(used_mask),
                    )
                player_ids = torch.arange(P, device=device)
                opp_eligible = eligible_rep & (player_ids[None, None, :] != hero)
                layer_best_opp = torch.where(
                    opp_eligible[:, :, :, None],
                    selected_ranks[:, None, :, :],
                    torch.full((), -1, dtype=torch.int32, device=device),
                ).amax(dim=2)
                hero_layer_weight = (
                    eligible_rep[:, :, hero].to(torch.float32) * layer_amount_rep
                )

                for hand_start in range(0, H, hand_chunk):
                    hand_end = min(hand_start + hand_chunk, H)
                    hero_masks = combo_masks[hand_start:hand_end]
                    hero_ranks = ranks[:, hand_start:hand_end]
                    compatible = (
                        alive[:, None, :]
                        & allowed[:, hand_start:hand_end, None]
                        & ((used_mask[:, None, :] & hero_masks[None, :, None]) == 0)
                    )
                    payout = torch.zeros(
                        row_count,
                        hand_end - hand_start,
                        cur_samples,
                        dtype=torch.float32,
                        device=device,
                    )
                    layer_tie_count = 1.0 + (
                        opp_eligible[:, :, None, :, None]
                        & (
                            selected_ranks[:, None, None, :, :]
                            == hero_ranks[:, None, :, None, None]
                        )
                    ).to(torch.float32).sum(dim=3)
                    wins = hero_ranks[:, None, :, None] > layer_best_opp[:, :, None, :]
                    ties = hero_ranks[:, None, :, None] == layer_best_opp[:, :, None, :]
                    share = torch.where(
                        wins,
                        torch.ones_like(layer_tie_count),
                        torch.where(
                            ties,
                            1.0 / layer_tie_count,
                            torch.zeros_like(layer_tie_count),
                        ),
                    )
                    payout = (hero_layer_weight[:, :, None, None] * share).sum(dim=1)

                    comp_f = compatible.to(torch.float32)
                    payout_part = (comp_f * payout).sum(dim=2)
                    denom_part = comp_f.sum(dim=2)
                    # root_ids == arange(B).repeat_interleave(cur_boards), so a
                    # segmented sum replaces the per-root atomic scatter-add.
                    payout_sum[:, hero, hand_start:hand_end] += payout_part.reshape(
                        B, cur_boards, -1
                    ).sum(dim=1)
                    denom_sum[:, hero, hand_start:hand_end] += denom_part.reshape(
                        B, cur_boards, -1
                    ).sum(dim=1)

            processed_samples += cur_samples

        done_boards += cur_boards

    expected_payout = torch.where(
        denom_sum > 0.0,
        payout_sum / denom_sum.clamp_min(1.0e-30),
        torch.zeros_like(payout_sum),
    )
    values = (
        batch.stacks_after[:, :, None].to(torch.float32)
        + expected_payout
        - batch.starting_stacks[:, :, None].to(torch.float32)
    ) / batch.scale[:, None, None].to(torch.float32).clamp_min(1.0)
    folded_value = (batch.stacks_after - batch.starting_stacks) / batch.scale[
        :, None
    ].clamp_min(1.0)
    values = torch.where(
        batch.folded_mask[:, :, None], folded_value[:, :, None], values
    )
    if not compute_stats:
        return values, {}
    elapsed = time.perf_counter() - start
    diagnostics = {
        "target_seconds": elapsed,
        "target_boards_per_second": float(B * board_samples / max(elapsed, 1.0e-9)),
        "target_samples_per_second": float(
            B * board_samples * tuple_samples / max(elapsed, 1.0e-9)
        ),
        "target_zero_denom_frac": float((denom_sum == 0).float().mean().item()),
        "target_value_mean": float(values.mean().item()),
        "target_value_std": float(values.std().item()),
        "target_board_samples": float(board_samples),
        "target_tuple_samples": float(tuple_samples),
        "target_exhaustive_boards": float(exhaustive_boards),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return values, diagnostics
