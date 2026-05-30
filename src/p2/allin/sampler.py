from __future__ import annotations

import time
from pathlib import Path

import torch

from p2.allin.data import PreflopAllInBatch
from p2.allin.kernels import (
    allin_alias_tuple_score_cuda_into,
    make_allin_alias_workspace,
    triton_available,
)
from p2.env.card_utils import NUM_HANDS, board_allowed_hands, hand_combos_tensor
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import (
    rank_7_cards_single_batch_triton,
    triton_is_available as rules_triton_available,
)
from p2.search.allin_payoff import (
    I16_SCALE,
    allin_values_from_payoff_reference,
    load_preflop_payoff_i16,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PREFLOP_ALLIN_TABLE = _REPO_ROOT / "outputs" / "preflop_allin_table.pt.zst"


def _combo_masks(device: torch.device) -> torch.Tensor:
    combos = hand_combos_tensor(device=device)
    return ((1 << combos[:, 0]) | (1 << combos[:, 1])).to(torch.int64)


def _sample_full_boards(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    scores = torch.rand(count, 52, device=device, generator=generator)
    return torch.topk(scores, 5, dim=1).indices.sort(dim=1).values


def _rank_hands(board: torch.Tensor) -> torch.Tensor:
    if board.device.type == "cuda" and rules_triton_available():
        combos = hand_combos_tensor(device=board.device)
        rows = board.shape[0]
        holes = combos[None, :, :].expand(rows, NUM_HANDS, 2)
        boards = board[:, None, :].expand(rows, NUM_HANDS, 5)
        cards = torch.cat((holes, boards), dim=-1).reshape(-1, 7)
        return rank_7_cards_single_batch_triton(cards).view(rows, NUM_HANDS).contiguous()
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
    live_counts = batch.allin_mask.sum(dim=1)
    if not torch.equal(live_counts, torch.full_like(live_counts, 2)):
        raise ValueError("exact preflop all-in table requires exactly two live players per row")

    live_coords = torch.nonzero(batch.allin_mask, as_tuple=False)
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

    folded_value = (
        batch.stacks_after - batch.starting_stacks
    ) / batch.scale[:, None].clamp_min(1.0)
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
) -> tuple[int, int]:
    if sample_count is not None and sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if board_samples is not None and board_samples <= 0:
        raise ValueError("board_samples must be positive")
    if tuple_samples is not None and tuple_samples <= 0:
        raise ValueError("tuple_samples must be positive")
    if sample_count is None:
        if board_samples is None or tuple_samples is None:
            raise ValueError("board_samples and tuple_samples are required without sample_count")
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
    payout_sum = torch.zeros(B, P, H, dtype=torch.float32, device=device)
    denom_sum = torch.zeros_like(payout_sum)
    kernel_launch_seconds = 0.0

    workspace = make_allin_alias_workspace(
        max_rows=B * min(board_chunk, board_samples),
        players=P,
        sample_count=tuple_samples,
        tuple_tries=tuple_tries,
        device=device,
        hands=H,
    )

    done_boards = 0
    while done_boards < board_samples:
        cur_boards = min(board_chunk, board_samples - done_boards)
        row_count = B * cur_boards
        root_ids = torch.arange(B, device=device).repeat_interleave(cur_boards)
        boards = _sample_full_boards(row_count, device=device, generator=generator)
        ranks = _rank_hands(boards)
        allowed = board_allowed_hands(boards).contiguous()
        board_beliefs = beliefs.index_select(0, root_ids)
        board_beliefs.masked_fill_(~allowed[:, None, :], 0.0)
        board_beliefs = board_beliefs / board_beliefs.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-30)
        launch_start = time.perf_counter()
        payout_part, denom_part = allin_alias_tuple_score_cuda_into(
            workspace,
            board_beliefs=board_beliefs,
            hand_masks=combo_masks,
            hand_ranks=ranks,
            board_allowed=allowed.to(torch.int8),
            live_mask=live_mask_i8.index_select(0, root_ids),
            layer_amount=layer_amount_f32.index_select(0, root_ids),
            eligible=eligible_i8.index_select(0, root_ids),
            seed=(done_boards + 1) * 104_729,
        )
        kernel_launch_seconds += time.perf_counter() - launch_start
        # Rows are arange(B).repeat_interleave(cur_boards), so each root's boards
        # are contiguous: a segmented sum beats an atomic scatter-add here.
        payout_sum += payout_part.reshape(B, cur_boards, P, H).sum(dim=1)
        denom_sum += denom_part.reshape(B, cur_boards, P, H).sum(dim=1)
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
    folded_value = (
        batch.stacks_after - batch.starting_stacks
    ) / batch.scale[:, None].clamp_min(1.0)
    values = torch.where(batch.folded_mask[:, :, None], folded_value[:, :, None], values)
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
    """
    board_samples, tuple_samples = _resolve_sample_split(
        sample_count,
        board_samples,
        tuple_samples,
    )
    if tuple_tries <= 0:
        raise ValueError("tuple_tries must be positive")
    if board_chunk <= 0 or hand_chunk <= 0:
        raise ValueError("board_chunk and hand_chunk must be positive")

    start = time.perf_counter()
    device = batch.beliefs.device
    live_counts = batch.allin_mask.sum(dim=1)
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

    done_boards = 0
    while done_boards < board_samples:
        cur_boards = min(board_chunk, board_samples - done_boards)
        row_count = B * cur_boards
        root_ids = torch.arange(B, device=device).repeat_interleave(cur_boards)
        boards = _sample_full_boards(row_count, device=device, generator=generator)
        ranks = _rank_hands(boards)
        allowed = board_allowed_hands(boards)
        board_beliefs = beliefs.index_select(0, root_ids)
        board_beliefs.masked_fill_(~allowed[:, None, :], 0.0)
        board_beliefs = board_beliefs / board_beliefs.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-30)
        flat_beliefs = board_beliefs.reshape(row_count * P, H)
        live_rep = live_mask.index_select(0, root_ids)
        layer_amount_rep = layer_amount.index_select(0, root_ids)
        eligible_rep = eligible.index_select(0, root_ids)

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
            candidate_masks = combo_masks.index_select(0, candidates.reshape(-1)).reshape(
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
                alive = torch.zeros(row_count, cur_samples, dtype=torch.bool, device=device)
                selected = torch.zeros(row_count, P, cur_samples, dtype=torch.long, device=device)
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

                selected_masks = combo_masks.index_select(0, selected.reshape(-1)).reshape(
                    row_count,
                    P,
                    cur_samples,
                )
                used_mask = torch.zeros(row_count, cur_samples, dtype=torch.int64, device=device)
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
    folded_value = (
        batch.stacks_after - batch.starting_stacks
    ) / batch.scale[:, None].clamp_min(1.0)
    values = torch.where(batch.folded_mask[:, :, None], folded_value[:, :, None], values)
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
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return values, diagnostics
