from __future__ import annotations

import argparse
import functools
import itertools
import math
import time
from collections import Counter
from dataclasses import dataclass

import torch

from p2.env.card_utils import NUM_HANDS, board_allowed_hands, hand_combos_tensor
from p2.env.rules import rank_hands

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional benchmark dependency
    triton = None
    tl = None


@dataclass
class Estimate:
    equity: torch.Tensor
    stderr: torch.Tensor
    ess: float
    proposals: int
    seconds: float


@dataclass
class ExactResult:
    equity: torch.Tensor
    seconds: float


@dataclass
class PreparedShowdown:
    board: torch.Tensor
    beliefs: torch.Tensor
    hand_ranks: torch.Tensor
    combos: torch.Tensor
    hand_masks: torch.Tensor
    setup_seconds: float


@dataclass
class PreparedTritonSIS:
    beliefs: torch.Tensor
    hand_masks: torch.Tensor
    hand_ranks: torch.Tensor
    setup_seconds: float


@dataclass
class PreparedFastSISBoard:
    active_ids: torch.Tensor
    hand_cards0: torch.Tensor
    hand_cards1: torch.Tensor
    pair_to_active: torch.Tensor
    hand_masks: torch.Tensor
    hand_ranks: torch.Tensor
    setup_seconds: float


@dataclass
class PreparedFastSISBelief:
    board: PreparedFastSISBoard
    beliefs: torch.Tensor
    cdf: torch.Tensor
    card_mass: torch.Tensor
    pair_mass: torch.Tensor
    setup_seconds: float


@dataclass
class TritonRejectSISWorkspace:
    sampled: torch.Tensor
    failures: torch.Tensor
    used: torch.Tensor
    candidates: torch.Tensor
    weights: torch.Tensor
    shares: torch.Tensor
    accum: torch.Tensor
    equity: torch.Tensor
    stderr: torch.Tensor
    ess: torch.Tensor
    sample_count: int
    max_tries: int


@dataclass
class PreparedBatchedFastSISBelief:
    board: PreparedFastSISBoard
    beliefs: torch.Tensor
    card_mass: torch.Tensor
    pair_mass: torch.Tensor
    batch_size: int
    players: int
    setup_seconds: float


@dataclass
class BatchedTritonRejectSISWorkspace:
    candidates: torch.Tensor
    accum: torch.Tensor
    equity: torch.Tensor
    stderr: torch.Tensor
    ess: torch.Tensor
    sample_count: int
    max_tries: int


@dataclass
class BatchedAliasRejectSISWorkspace:
    alias_prob: torch.Tensor
    alias_idx: torch.Tensor
    candidates: torch.Tensor
    seed: torch.Tensor
    small_stack: torch.Tensor
    large_stack: torch.Tensor
    counts: torch.Tensor
    accum: torch.Tensor
    equity: torch.Tensor
    stderr: torch.Tensor
    ess: torch.Tensor
    sample_count: int
    max_tries: int
    setup_seconds: float


@dataclass(frozen=True)
class ExactPatternSpec:
    forced_by_opp: tuple[tuple[int, ...], ...]
    variable_count: int
    multiplicity: int
    sign: int


def random_beliefs(
    board: torch.Tensor,
    num_players: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    weights = torch.empty(
        board.shape[0],
        num_players,
        NUM_HANDS,
        dtype=torch.float32,
        device=board.device,
    ).exponential_(generator=generator)
    allowed = board_allowed_hands(board)
    weights.masked_fill_(~allowed[:, None, :], 0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def random_full_board(
    batch_size: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    scores = torch.rand(batch_size, 52, device=device, generator=generator)
    return torch.topk(scores, 5, dim=1).indices


def prepare_showdown(
    *,
    batch_size: int,
    num_players: int,
    device: torch.device,
    generator: torch.Generator,
) -> PreparedShowdown:
    start = time.perf_counter()
    board = random_full_board(batch_size, device=device, generator=generator)
    beliefs = random_beliefs(board, num_players, generator=generator)
    hand_ranks, _ = rank_hands(board)
    combos = hand_combos_tensor(device=device).long()
    hand_masks = (1 << combos[:, 0]) | (1 << combos[:, 1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return PreparedShowdown(
        board=board,
        beliefs=beliefs,
        hand_ranks=hand_ranks,
        combos=combos,
        hand_masks=hand_masks.long().contiguous(),
        setup_seconds=time.perf_counter() - start,
    )


def _sample_independent_hands(
    beliefs: torch.Tensor,
    sample_count: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size, players, hands = beliefs.shape
    samples = torch.multinomial(
        beliefs.reshape(batch_size * players, hands),
        sample_count,
        replacement=True,
        generator=generator,
    )
    return samples.view(batch_size, players, sample_count).transpose(1, 2).contiguous()


def _collision_free_mask(sampled_hands: torch.Tensor, combos: torch.Tensor) -> torch.Tensor:
    cards = combos[sampled_hands].flatten(start_dim=2)
    sorted_cards = cards.sort(dim=2).values
    return (sorted_cards[:, :, 1:] != sorted_cards[:, :, :-1]).all(dim=2)


def _showdown_shares(
    sampled_hands: torch.Tensor,
    hand_ranks: torch.Tensor,
) -> torch.Tensor:
    batch_size, sample_count, players = sampled_hands.shape
    ranks = hand_ranks[:, None, :].expand(-1, sample_count, -1).gather(2, sampled_hands)
    max_rank = ranks.max(dim=2, keepdim=True).values
    winners = ranks == max_rank
    return winners.to(torch.float32) / winners.sum(dim=2, keepdim=True).clamp_min(1)


def naive_rejection_chunk(
    beliefs: torch.Tensor,
    hand_ranks: torch.Tensor,
    combos: torch.Tensor,
    sample_count: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_hands = _sample_independent_hands(
        beliefs,
        sample_count,
        generator=generator,
    )
    valid = _collision_free_mask(sampled_hands, combos)
    shares = _showdown_shares(sampled_hands, hand_ranks)
    return shares, valid


def naive_rejection_until_ess(
    prepared: PreparedShowdown,
    *,
    target_ess: int,
    chunk_size: int,
    generator: torch.Generator,
) -> Estimate:
    start = time.perf_counter()
    beliefs = prepared.beliefs
    hand_ranks = prepared.hand_ranks
    batch_size, players = beliefs.shape[:2]
    share_sum = torch.zeros(batch_size, players, device=beliefs.device)
    share_sq_sum = torch.zeros_like(share_sum)
    accepted = torch.zeros(batch_size, device=beliefs.device)
    proposals = 0

    while float(accepted.min().item()) < target_ess:
        shares, valid = naive_rejection_chunk(
            beliefs,
            hand_ranks,
            prepared.combos,
            chunk_size,
            generator=generator,
        )
        valid_f = valid.to(torch.float32)
        share_sum += (shares * valid_f[:, :, None]).sum(dim=1)
        share_sq_sum += (shares.square() * valid_f[:, :, None]).sum(dim=1)
        accepted += valid_f.sum(dim=1)
        proposals += chunk_size

    equity = share_sum / accepted[:, None].clamp_min(1.0)
    second_moment = share_sq_sum / accepted[:, None].clamp_min(1.0)
    variance = (second_moment - equity.square()).clamp_min(0.0)
    stderr = (variance / accepted[:, None].clamp_min(1.0)).sqrt()
    if beliefs.device.type == "cuda":
        torch.cuda.synchronize(beliefs.device)
    return Estimate(
        equity=equity,
        stderr=stderr,
        ess=float(accepted.min().item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def sis_chunk(
    beliefs: torch.Tensor,
    board: torch.Tensor,
    hand_ranks: torch.Tensor,
    combos: torch.Tensor,
    sample_count: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = beliefs.device
    batch_size, players, hands = beliefs.shape
    sampled_hands = torch.empty(
        batch_size,
        sample_count,
        players,
        dtype=torch.long,
        device=device,
    )
    available = torch.ones(batch_size, sample_count, 52, dtype=torch.bool, device=device)
    board_cards = board[:, None, :].expand(-1, sample_count, -1)
    available.scatter_(2, board_cards, False)
    weights = torch.ones(batch_size, sample_count, dtype=torch.float32, device=device)

    for player in range(players):
        hand_ok = available[:, :, combos[:, 0]] & available[:, :, combos[:, 1]]
        proposal = beliefs[:, player, None, :] * hand_ok.to(torch.float32)
        normalizer = proposal.sum(dim=2).clamp_min(1e-12)
        weights *= normalizer
        sampled = torch.multinomial(
            (proposal / normalizer[:, :, None]).reshape(batch_size * sample_count, hands),
            1,
            replacement=True,
            generator=generator,
        ).view(batch_size, sample_count)
        sampled_hands[:, :, player] = sampled
        sampled_cards = combos[sampled]
        available.scatter_(2, sampled_cards, False)

    shares = _showdown_shares(sampled_hands, hand_ranks)
    return shares, weights


def sis_until_ess(
    prepared: PreparedShowdown,
    *,
    target_ess: int,
    chunk_size: int,
    generator: torch.Generator,
) -> Estimate:
    start = time.perf_counter()
    beliefs = prepared.beliefs
    board = prepared.board
    hand_ranks = prepared.hand_ranks
    batch_size, players = beliefs.shape[:2]
    sum_w = torch.zeros(batch_size, device=beliefs.device)
    sum_w2 = torch.zeros_like(sum_w)
    sum_wx = torch.zeros(batch_size, players, device=beliefs.device)
    sum_wx2 = torch.zeros_like(sum_wx)
    proposals = 0

    while True:
        shares, weights = sis_chunk(
            beliefs,
            board,
            hand_ranks,
            prepared.combos,
            chunk_size,
            generator=generator,
        )
        sum_w += weights.sum(dim=1)
        sum_w2 += weights.square().sum(dim=1)
        sum_wx += (shares * weights[:, :, None]).sum(dim=1)
        sum_wx2 += (shares.square() * weights[:, :, None]).sum(dim=1)
        proposals += chunk_size
        ess = sum_w.square() / sum_w2.clamp_min(1e-30)
        if float(ess.min().item()) >= target_ess:
            break

    equity = sum_wx / sum_w[:, None].clamp_min(1e-30)
    second_moment = sum_wx2 / sum_w[:, None].clamp_min(1e-30)
    variance = (second_moment - equity.square()).clamp_min(0.0)
    ess = sum_w.square() / sum_w2.clamp_min(1e-30)
    stderr = (variance / ess[:, None].clamp_min(1.0)).sqrt()
    if beliefs.device.type == "cuda":
        torch.cuda.synchronize(beliefs.device)
    return Estimate(
        equity=equity,
        stderr=stderr,
        ess=float(ess.min().item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def _threeway_ie_for_hero(
    beliefs: torch.Tensor,
    hand_ranks: torch.Tensor,
    hand_masks: torch.Tensor,
    combos: torch.Tensor,
    *,
    hero: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if beliefs.shape[0] != 3:
        raise ValueError("three-way exact IE expects exactly three players")
    opponents = [player for player in range(3) if player != hero]
    device = beliefs.device
    dtype = torch.float64
    ranks = hand_ranks.reshape(NUM_HANDS)
    masks = hand_masks.reshape(NUM_HANDS)
    hero_masks = masks[:, None]
    opp_masks = masks[None, :]
    disjoint = (hero_masks & opp_masks) == 0
    worse = disjoint & (ranks[None, :] < ranks[:, None])
    tied = disjoint & (ranks[None, :] == ranks[:, None])

    contains = torch.zeros(NUM_HANDS, 52, dtype=dtype, device=device)
    hand_ids = torch.arange(NUM_HANDS, device=device)
    contains[hand_ids, combos[:, 0]] = 1.0
    contains[hand_ids, combos[:, 1]] = 1.0

    def stats(player: int) -> tuple[torch.Tensor, ...]:
        belief = beliefs[player].to(dtype)
        alpha = worse.to(dtype).matmul(belief)
        beta = tied.to(dtype).matmul(belief)
        z = disjoint.to(dtype).matmul(belief)
        weighted_contains = belief[:, None] * contains
        alpha_card = worse.to(dtype).matmul(weighted_contains)
        beta_card = tied.to(dtype).matmul(weighted_contains)
        z_card = disjoint.to(dtype).matmul(weighted_contains)
        return alpha, beta, z, alpha_card, beta_card, z_card

    a_alpha, a_beta, a_z, a_alpha_c, a_beta_c, a_z_c = stats(opponents[0])
    b_alpha, b_beta, b_z, b_alpha_c, b_beta_c, b_z_c = stats(opponents[1])

    # Empty pattern: both opponents are already restricted away from hero cards.
    numerator = (
        a_alpha * b_alpha
        + 0.5 * (a_beta * b_alpha + a_alpha * b_beta)
        + (a_beta * b_beta) / 3.0
    )
    denominator = a_z * b_z

    # One shared-card collision pattern, sign -1.
    numerator -= (
        (a_alpha_c * b_alpha_c).sum(dim=1)
        + 0.5
        * (
            (a_beta_c * b_alpha_c).sum(dim=1)
            + (a_alpha_c * b_beta_c).sum(dim=1)
        )
        + (a_beta_c * b_beta_c).sum(dim=1) / 3.0
    )
    denominator -= (a_z_c * b_z_c).sum(dim=1)

    # Two shared-card/same-hand collision pattern, sign +1. Mixed tie/worse
    # terms are impossible for the same exact opponent hand.
    same_belief = beliefs[opponents[0]].to(dtype) * beliefs[opponents[1]].to(dtype)
    numerator += worse.to(dtype).matmul(same_belief)
    numerator += tied.to(dtype).matmul(same_belief) / 3.0
    denominator += disjoint.to(dtype).matmul(same_belief)

    equity_by_hand = numerator / denominator.clamp_min(1e-30)
    hero_belief = beliefs[hero].to(dtype)
    aggregate = (hero_belief * numerator).sum() / (hero_belief * denominator).sum().clamp_min(
        1e-30
    )
    return equity_by_hand, numerator, aggregate


def exact_threeway_ie(prepared: PreparedShowdown) -> ExactResult:
    if prepared.beliefs.shape != (1, 3, NUM_HANDS):
        raise ValueError("exact_threeway_ie expects one 3-player board")
    start = time.perf_counter()
    beliefs = prepared.beliefs[0]
    equities = []
    for hero in range(3):
        _, _, aggregate = _threeway_ie_for_hero(
            beliefs,
            prepared.hand_ranks,
            prepared.hand_masks,
            prepared.combos,
            hero=hero,
        )
        equities.append(aggregate)
    result = torch.stack(equities).to(torch.float32)[None, :]
    if prepared.beliefs.device.type == "cuda":
        torch.cuda.synchronize(prepared.beliefs.device)
    return ExactResult(equity=result, seconds=time.perf_counter() - start)


def _collision_patterns(num_opponents: int) -> list[tuple[tuple[int, ...], int]]:
    subset_masks = [
        mask
        for mask in range(1, 1 << num_opponents)
        if int(mask).bit_count() >= 2
    ]
    patterns: list[tuple[tuple[int, ...], int]] = []

    def rec(index: int, degrees: list[int], masks: list[int]) -> None:
        if index == len(subset_masks):
            if masks:
                sign = -1 if len(masks) % 2 else 1
                for mask in masks:
                    degree = int(mask).bit_count()
                    sign *= (-1 if degree % 2 else 1) * (degree - 1)
                patterns.append((tuple(masks), sign))
            return

        mask = subset_masks[index]
        players = [p for p in range(num_opponents) if (mask >> p) & 1]
        max_count = min(2 - degrees[p] for p in players)
        for count in range(max_count + 1):
            next_degrees = degrees.copy()
            for player in players:
                next_degrees[player] += count
            rec(index + 1, next_degrees, masks + [mask] * count)

    rec(0, [0] * num_opponents, [])
    return patterns


def _pattern_label_chunks(
    masks: tuple[int, ...],
    active_cards: list[int],
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if not masks:
        return (torch.empty(0, 0, dtype=torch.long, device=device),)

    chunks: list[torch.Tensor] = []
    current: list[tuple[int, ...]] = []
    for labels in itertools.permutations(active_cards, len(masks)):
        canonical = True
        for left in range(len(masks)):
            for right in range(left + 1, len(masks)):
                if masks[left] == masks[right] and labels[left] > labels[right]:
                    canonical = False
                    break
            if not canonical:
                break
        if not canonical:
            continue
        current.append(labels)
        if len(current) == chunk_size:
            chunks.append(torch.tensor(current, dtype=torch.long, device=device))
            current = []
    if current:
        chunks.append(torch.tensor(current, dtype=torch.long, device=device))
    return tuple(chunks)


def _pair_lookup(device: torch.device) -> torch.Tensor:
    combos = hand_combos_tensor(device=device).long()
    lookup = torch.full((52, 52), -1, dtype=torch.long, device=device)
    ids = torch.arange(NUM_HANDS, device=device)
    lookup[combos[:, 0], combos[:, 1]] = ids
    lookup[combos[:, 1], combos[:, 0]] = ids
    return lookup


@functools.cache
def _set_partitions_with_mobius(
    item_count: int,
) -> tuple[tuple[tuple[tuple[int, ...], ...], int], ...]:
    partitions: list[tuple[tuple[int, ...], ...]] = []

    def rec(item: int, blocks: list[list[int]]) -> None:
        if item == item_count:
            partitions.append(tuple(tuple(block) for block in blocks))
            return
        for block in blocks:
            block.append(item)
            rec(item + 1, blocks)
            block.pop()
        blocks.append([item])
        rec(item + 1, blocks)
        blocks.pop()

    rec(0, [])
    result: list[tuple[tuple[tuple[int, ...], ...], int]] = []
    for partition in partitions:
        coefficient = 1
        for block in partition:
            size = len(block)
            coefficient *= (-1) ** (size - 1) * math.factorial(size - 1)
        result.append((partition, coefficient))
    return tuple(result)


def _ordered_label_multiplicity(pattern_masks: tuple[int, ...]) -> int:
    multiplicity = 1
    for count in Counter(pattern_masks).values():
        multiplicity *= math.factorial(count)
    return multiplicity


@functools.cache
def _exact_pattern_specs(num_opponents: int) -> tuple[ExactPatternSpec, ...]:
    specs = []
    for pattern_masks, sign in _collision_patterns(num_opponents):
        forced_by_opp = tuple(
            tuple(var for var, mask in enumerate(pattern_masks) if (mask >> opp_idx) & 1)
            for opp_idx in range(num_opponents)
        )
        specs.append(
            ExactPatternSpec(
                forced_by_opp=forced_by_opp,
                variable_count=len(pattern_masks),
                multiplicity=_ordered_label_multiplicity(pattern_masks),
                sign=sign,
            )
        )
    return tuple(specs)


@functools.cache
def _subset_mode_data_cpu(num_opponents: int) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
    rows = []
    weights = []
    for subset in range(1 << num_opponents):
        modes = []
        ties = 0
        for opp_idx in range(num_opponents):
            mode = 1 if (subset >> opp_idx) & 1 else 0
            ties += mode
            modes.append(mode)
        rows.append(tuple(modes))
        weights.append(1.0 / float(ties + 1))
    return tuple(rows), tuple(weights)


def _factor_sum_all_distinct(
    factors: list[tuple[tuple[int, ...], torch.Tensor]],
    variable_count: int,
    multiplicity: int,
    *,
    hand_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    letters = "abcd"
    total = torch.zeros(hand_count, dtype=dtype, device=device)
    for partition, coefficient in _set_partitions_with_mobius(variable_count):
        variable_to_block = [0] * variable_count
        for block_idx, block in enumerate(partition):
            for variable in block:
                variable_to_block[variable] = block_idx

        scalar = torch.ones(hand_count, dtype=dtype, device=device)
        operands: list[torch.Tensor] = []
        subscripts: list[str] = []
        for scope, tensor in factors:
            if not scope:
                scalar = scalar * tensor
                continue
            collapsed = tuple(variable_to_block[variable] for variable in scope)
            if len(scope) == 1:
                operands.append(tensor)
                subscripts.append("h" + letters[collapsed[0]])
            elif collapsed[0] == collapsed[1]:
                operands.append(tensor.diagonal(dim1=1, dim2=2))
                subscripts.append("h" + letters[collapsed[0]])
            else:
                operands.append(tensor)
                subscripts.append("h" + letters[collapsed[0]] + letters[collapsed[1]])

        if operands:
            equation = ",".join(["h", *subscripts]) + "->h"
            value = torch.einsum(equation, scalar, *operands)
        else:
            value = scalar
        total += float(coefficient) * value

    return total / float(multiplicity)


def _unary4_product_sum_triton(
    scalar: torch.Tensor,
    operands: list[torch.Tensor],
    *,
    card_count: int,
) -> torch.Tensor | None:
    if triton is None or scalar.device.type != "cuda" or len(operands) != 4:
        return None
    out = torch.empty_like(scalar)
    block_c = triton.next_power_of_2(card_count)
    grid = (triton.cdiv(scalar.shape[0], 16),)
    _unary4_product_sum_kernel[grid](
        scalar,
        operands[0],
        operands[1],
        operands[2],
        operands[3],
        out,
        scalar.shape[0],
        card_count,
        BLOCK_H=16,
        BLOCK_C=block_c,
        num_warps=2,
    )
    return out


def _batched_unary_product_sum_triton(
    scalar: torch.Tensor,
    operands: list[torch.Tensor],
    *,
    card_count: int,
) -> torch.Tensor | None:
    if (
        triton is None
        or scalar.device.type != "cuda"
        or len(operands) < 2
        or len(operands) > 4
    ):
        return None
    batch_size, hand_count = scalar.shape
    out = torch.empty_like(scalar)
    block_c = triton.next_power_of_2(card_count)
    grid = (triton.cdiv(batch_size * hand_count, 128),)
    empty = operands[0]
    _batched_unary_product_sum_kernel[grid](
        scalar,
        operands[0],
        operands[1],
        operands[2] if len(operands) > 2 else empty,
        operands[3] if len(operands) > 3 else empty,
        out,
        batch_size * hand_count,
        card_count,
        COUNT=len(operands),
        BLOCK_N=128,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def _factor_sum_all_distinct_chunked_f32(
    factors: list[tuple[tuple[int, ...], torch.Tensor]],
    variable_count: int,
    multiplicity: int,
    *,
    hand_count: int,
    hand_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    letters = "abcd"
    total = torch.zeros(hand_count, dtype=torch.float32, device=device)
    card_count = 0
    for scope, tensor in factors:
        if scope:
            card_count = tensor.shape[1]
            break

    for partition, coefficient in _set_partitions_with_mobius(variable_count):
        variable_to_block = [0] * variable_count
        for block_idx, block in enumerate(partition):
            for variable in block:
                variable_to_block[variable] = block_idx

        scalar = torch.ones(hand_count, dtype=torch.float32, device=device)
        operands: list[torch.Tensor] = []
        subscripts: list[str] = []
        operand_scopes: list[tuple[int, ...]] = []
        for scope, tensor in factors:
            tensor = tensor.to(torch.float32)
            if not scope:
                scalar = scalar * tensor
                continue
            collapsed = tuple(variable_to_block[variable] for variable in scope)
            if len(scope) == 1:
                operands.append(tensor)
                subscripts.append("h" + letters[collapsed[0]])
                operand_scopes.append((collapsed[0],))
            elif collapsed[0] == collapsed[1]:
                operands.append(tensor.diagonal(dim1=1, dim2=2))
                subscripts.append("h" + letters[collapsed[0]])
                operand_scopes.append((collapsed[0],))
            else:
                operands.append(tensor)
                subscripts.append("h" + letters[collapsed[0]] + letters[collapsed[1]])
                operand_scopes.append((collapsed[0], collapsed[1]))

        value: torch.Tensor | None = None
        if (
            len(operands) == 4
            and all(len(scope) == 1 for scope in operand_scopes)
            and len({scope[0] for scope in operand_scopes}) == 1
            and card_count > 0
        ):
            value = _unary4_product_sum_triton(
                scalar.contiguous(),
                [operand.contiguous() for operand in operands],
                card_count=card_count,
            )

        if value is None:
            del subscripts
            value = _contract_collapsed_factors_f32(
                scalar,
                operand_scopes,
                operands,
                hand_count=hand_count,
                hand_chunk=hand_chunk,
                card_count=card_count,
                device=device,
            )

        total += float(coefficient) * value

    return total / float(multiplicity)


def _align_factor_to_scope(
    tensor: torch.Tensor,
    scope: tuple[int, ...],
    union_scope: tuple[int, ...],
) -> torch.Tensor:
    ordered_existing = tuple(var for var in union_scope if var in scope)
    if ordered_existing:
        perm = [0, *(1 + scope.index(var) for var in ordered_existing)]
        tensor = tensor.permute(perm)
    shape = [tensor.shape[0]]
    for var in union_scope:
        shape.append(tensor.shape[1 + ordered_existing.index(var)] if var in scope else 1)
    return tensor.reshape(shape)


def _multiply_and_sum_factor_chunk(
    involved: list[tuple[tuple[int, ...], torch.Tensor]],
    union_scope: tuple[int, ...],
    eliminate_var: int,
    *,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    product: torch.Tensor | None = None
    for scope, tensor in involved:
        aligned = _align_factor_to_scope(
            tensor[start_idx:end_idx],
            scope,
            union_scope,
        )
        product = aligned if product is None else product * aligned
    if product is None:
        raise RuntimeError("cannot eliminate variable without involved factors")
    sum_axis = 1 + union_scope.index(eliminate_var)
    return product.sum(dim=sum_axis)


def _contract_collapsed_factors_f32(
    scalar: torch.Tensor,
    operand_scopes: list[tuple[int, ...]],
    operands: list[torch.Tensor],
    *,
    hand_count: int,
    hand_chunk: int,
    card_count: int,
    device: torch.device,
) -> torch.Tensor:
    factors = [
        (scope, operand)
        for scope, operand in zip(operand_scopes, operands, strict=True)
        if scope
    ]

    while True:
        variables = sorted({var for scope, _ in factors for var in scope})
        if not variables:
            result = scalar
            for _, tensor in factors:
                result = result * tensor
            return result

        def elimination_cost(var: int) -> tuple[int, int]:
            involved_scopes = [scope for scope, _ in factors if var in scope]
            union_size = len({item for scope in involved_scopes for item in scope})
            return union_size, len(involved_scopes)

        eliminate_var = min(variables, key=elimination_cost)
        involved = [(scope, tensor) for scope, tensor in factors if eliminate_var in scope]
        remaining = [(scope, tensor) for scope, tensor in factors if eliminate_var not in scope]
        union_scope = tuple(sorted({var for scope, _ in involved for var in scope}))
        new_scope = tuple(var for var in union_scope if var != eliminate_var)
        output_shape = (hand_count, *([card_count] * len(new_scope)))
        if len(union_scope) >= 3:
            new_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
            for start_idx in range(0, hand_count, hand_chunk):
                end_idx = min(start_idx + hand_chunk, hand_count)
                new_tensor[start_idx:end_idx] = _multiply_and_sum_factor_chunk(
                    involved,
                    union_scope,
                    eliminate_var,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
        else:
            new_tensor = _multiply_and_sum_factor_chunk(
                involved,
                union_scope,
                eliminate_var,
                start_idx=0,
                end_idx=hand_count,
            )
        factors = remaining + [(new_scope, new_tensor)]


def _align_batched_factor_to_scope(
    tensor: torch.Tensor,
    scope: tuple[int, ...],
    union_scope: tuple[int, ...],
) -> torch.Tensor:
    ordered_existing = tuple(var for var in union_scope if var in scope)
    if ordered_existing:
        perm = [0, 1, *(2 + scope.index(var) for var in ordered_existing)]
        tensor = tensor.permute(perm)
    shape = [tensor.shape[0], tensor.shape[1]]
    for var in union_scope:
        shape.append(tensor.shape[2 + ordered_existing.index(var)] if var in scope else 1)
    return tensor.reshape(shape)


def _multiply_and_sum_batched_factor_chunk(
    involved: list[tuple[tuple[int, ...], torch.Tensor]],
    union_scope: tuple[int, ...],
    eliminate_var: int,
    *,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    product: torch.Tensor | None = None
    for scope, tensor in involved:
        aligned = _align_batched_factor_to_scope(
            tensor[:, start_idx:end_idx],
            scope,
            union_scope,
        )
        product = aligned if product is None else product * aligned
    if product is None:
        raise RuntimeError("cannot eliminate variable without involved factors")
    sum_axis = 2 + union_scope.index(eliminate_var)
    return product.sum(dim=sum_axis)


def _contract_collapsed_factors_batched_f32(
    scalar: torch.Tensor,
    operand_scopes: list[tuple[int, ...]],
    operands: list[torch.Tensor],
    *,
    hand_count: int,
    hand_chunk: int,
    card_count: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = scalar.shape[0]
    variables = sorted({var for scope in operand_scopes for var in scope})
    cycle_value = _cycle4_contract_bmm_f32(
        scalar,
        operand_scopes,
        operands,
        variables=variables,
        hand_count=hand_count,
    )
    if cycle_value is not None:
        return cycle_value
    triangle_value = _triangle_contract_bmm_f32(
        scalar,
        operand_scopes,
        operands,
        variables=variables,
        hand_count=hand_count,
    )
    if triangle_value is not None:
        return triangle_value

    factors = [
        (scope, operand)
        for scope, operand in zip(operand_scopes, operands, strict=True)
        if scope
    ]

    while True:
        variables = sorted({var for scope, _ in factors for var in scope})
        if not variables:
            result = scalar
            for _, tensor in factors:
                result = result * tensor
            return result

        def elimination_cost(var: int) -> tuple[int, int]:
            involved_scopes = [scope for scope, _ in factors if var in scope]
            union_size = len({item for scope in involved_scopes for item in scope})
            return union_size, len(involved_scopes)

        eliminate_var = min(variables, key=elimination_cost)
        involved = [(scope, tensor) for scope, tensor in factors if eliminate_var in scope]
        remaining = [(scope, tensor) for scope, tensor in factors if eliminate_var not in scope]
        union_scope = tuple(sorted({var for scope, _ in involved for var in scope}))
        new_scope = tuple(var for var in union_scope if var != eliminate_var)
        output_shape = (batch_size, hand_count, *([card_count] * len(new_scope)))
        if len(union_scope) >= 3:
            new_tensor = torch.empty(output_shape, dtype=torch.float32, device=device)
            for start_idx in range(0, hand_count, hand_chunk):
                end_idx = min(start_idx + hand_chunk, hand_count)
                new_tensor[:, start_idx:end_idx] = _multiply_and_sum_batched_factor_chunk(
                    involved,
                    union_scope,
                    eliminate_var,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
        else:
            new_tensor = _multiply_and_sum_batched_factor_chunk(
                involved,
                union_scope,
                eliminate_var,
                start_idx=0,
                end_idx=hand_count,
            )
        factors = remaining + [(new_scope, new_tensor)]


def _cycle4_contract_bmm_f32(
    scalar: torch.Tensor,
    operand_scopes: list[tuple[int, ...]],
    operands: list[torch.Tensor],
    *,
    variables: list[int],
    hand_count: int,
) -> torch.Tensor | None:
    if len(variables) != 4:
        return None
    if len(operand_scopes) != 4 or any(len(scope) != 2 for scope in operand_scopes):
        return None

    edge_by_pair: dict[tuple[int, int], torch.Tensor] = {}
    for scope, tensor in zip(operand_scopes, operands, strict=True):
        pair = tuple(sorted(scope))
        edge = tensor if scope == pair else tensor.transpose(-1, -2)
        if pair in edge_by_pair:
            return None
        edge_by_pair[pair] = edge
    edge_set = set(edge_by_pair)

    for x, y, z, w in itertools.permutations(variables, 4):
        needed = {
            tuple(sorted((x, y))),
            tuple(sorted((x, z))),
            tuple(sorted((y, w))),
            tuple(sorted((z, w))),
        }
        if needed != edge_set:
            continue
        xy = _edge_for_vars(edge_by_pair, x, y)
        yw = _edge_for_vars(edge_by_pair, y, w)
        xz = _edge_for_vars(edge_by_pair, x, z)
        zw = _edge_for_vars(edge_by_pair, z, w)
        batch_size = scalar.shape[0]
        flat = batch_size * hand_count
        left = torch.bmm(
            xy.reshape(flat, xy.shape[-2], xy.shape[-1]),
            yw.reshape(flat, yw.shape[-2], yw.shape[-1]),
        )
        right = torch.bmm(
            xz.reshape(flat, xz.shape[-2], xz.shape[-1]),
            zw.reshape(flat, zw.shape[-2], zw.shape[-1]),
        )
        return scalar * (left.reshape_as(xy) * right.reshape_as(xy)).sum(dim=(-1, -2))
    return None


def _edge_for_vars(
    edge_by_pair: dict[tuple[int, int], torch.Tensor],
    left: int,
    right: int,
) -> torch.Tensor:
    pair = tuple(sorted((left, right)))
    edge = edge_by_pair[pair]
    return edge if (left, right) == pair else edge.transpose(-1, -2)


def _triangle_components_f32(
    scalar: torch.Tensor,
    operand_scopes: list[tuple[int, ...]],
    operands: list[torch.Tensor],
    *,
    variables: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if len(variables) != 3:
        return None
    edge_scopes = [scope for scope in operand_scopes if len(scope) == 2]
    if len(edge_scopes) < 3:
        return None
    expected_edges = {
        (variables[0], variables[1]),
        (variables[0], variables[2]),
        (variables[1], variables[2]),
    }
    if {tuple(sorted(scope)) for scope in edge_scopes} != expected_edges:
        return None
    if any(len(scope) not in (1, 2) for scope in operand_scopes):
        return None

    unary_by_var: dict[int, torch.Tensor] = {}
    edge_by_pair: dict[tuple[int, int], torch.Tensor] = {}
    for scope, tensor in zip(operand_scopes, operands, strict=True):
        if len(scope) == 1:
            var = scope[0]
            unary_by_var[var] = unary_by_var.get(var, 1.0) * tensor
        else:
            a, b = scope
            pair = tuple(sorted(scope))
            edge = tensor if (a, b) == pair else tensor.transpose(-1, -2)
            if pair in edge_by_pair:
                edge_by_pair[pair] = edge_by_pair[pair] * edge
            else:
                edge_by_pair[pair] = edge

    a, b, c = variables
    ab = edge_by_pair[(a, b)]
    ac = edge_by_pair[(a, c)]
    bc = edge_by_pair[(b, c)]
    if a in unary_by_var:
        ab = ab * unary_by_var[a].unsqueeze(-1)
    if b in unary_by_var:
        ab = ab * unary_by_var[b].unsqueeze(-2)
    if c in unary_by_var:
        ac = ac * unary_by_var[c].unsqueeze(-2)
    return scalar, ab, ac, bc


def _triangle_components_contract_bmm_f32(
    components: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    hand_count: int,
) -> torch.Tensor:
    scalar, ab, ac, bc = components[0]
    if len(components) == 1:
        batch_size = scalar.shape[0]
        flat = batch_size * hand_count
        ac_flat = ac.reshape(flat, ac.shape[-2], ac.shape[-1])
        bc_t_flat = bc.transpose(-1, -2).reshape(flat, bc.shape[-1], bc.shape[-2])
        summed_c = torch.bmm(ac_flat, bc_t_flat).reshape_as(ab)
        return scalar * (ab * summed_c).sum(dim=(-1, -2))

    scalars = torch.stack([item[0] for item in components], dim=0)
    abs_ = torch.stack([item[1] for item in components], dim=0)
    acs = torch.stack([item[2] for item in components], dim=0)
    bcs = torch.stack([item[3] for item in components], dim=0)
    term_count, batch_size = scalars.shape[:2]
    flat = term_count * batch_size * hand_count
    ac_flat = acs.reshape(flat, acs.shape[-2], acs.shape[-1])
    bc_t_flat = bcs.transpose(-1, -2).reshape(flat, bcs.shape[-1], bcs.shape[-2])
    summed_c = torch.bmm(ac_flat, bc_t_flat).reshape_as(abs_)
    return scalars * (abs_ * summed_c).sum(dim=(-1, -2))


def _triangle_contract_bmm_f32(
    scalar: torch.Tensor,
    operand_scopes: list[tuple[int, ...]],
    operands: list[torch.Tensor],
    *,
    variables: list[int],
    hand_count: int,
) -> torch.Tensor | None:
    components = _triangle_components_f32(
        scalar,
        operand_scopes,
        operands,
        variables=variables,
    )
    if components is None:
        return None
    return _triangle_components_contract_bmm_f32([components], hand_count=hand_count)


def _direct_batched_einsum_chunked_f32(
    scalar: torch.Tensor,
    operand_scopes: list[tuple[int, ...]],
    operands: list[torch.Tensor],
    *,
    hand_count: int,
    hand_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    letters = "abcd"
    variables = sorted({var for scope in operand_scopes for var in scope})
    variable_to_letter = {var: letters[idx] for idx, var in enumerate(variables)}
    subscripts = [
        "nh" + "".join(variable_to_letter[var] for var in scope)
        for scope in operand_scopes
    ]
    equation = ",".join(["nh", *subscripts]) + "->nh"
    if len(variables) == 3:
        return torch.einsum(equation, scalar, *operands)
    output = torch.empty_like(scalar, device=device)
    for start_idx in range(0, hand_count, hand_chunk):
        end_idx = min(start_idx + hand_chunk, hand_count)
        output[:, start_idx:end_idx] = torch.einsum(
            equation,
            scalar[:, start_idx:end_idx],
            *[operand[:, start_idx:end_idx] for operand in operands],
        )
    return output


def _factor_sum_all_distinct_batched_f32(
    factors: list[tuple[tuple[int, ...], torch.Tensor]],
    variable_count: int,
    multiplicity: int,
    *,
    hand_count: int,
    hand_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = factors[0][1].shape[0]
    total = torch.zeros(batch_size, hand_count, dtype=torch.float32, device=device)
    card_count = 0
    for scope, tensor in factors:
        if scope:
            card_count = tensor.shape[2]
            break

    triangle_terms: list[
        tuple[
            float,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ] = []
    for partition, coefficient in _set_partitions_with_mobius(variable_count):
        variable_to_block = [0] * variable_count
        for block_idx, block in enumerate(partition):
            for variable in block:
                variable_to_block[variable] = block_idx

        scalar = torch.ones(batch_size, hand_count, dtype=torch.float32, device=device)
        operands: list[torch.Tensor] = []
        operand_scopes: list[tuple[int, ...]] = []
        for scope, tensor in factors:
            tensor = tensor.to(torch.float32)
            if not scope:
                scalar = scalar * tensor
                continue
            collapsed = tuple(variable_to_block[variable] for variable in scope)
            if len(scope) == 1:
                operands.append(tensor)
                operand_scopes.append((collapsed[0],))
            elif collapsed[0] == collapsed[1]:
                operands.append(tensor.diagonal(dim1=2, dim2=3))
                operand_scopes.append((collapsed[0],))
            else:
                operands.append(tensor)
                operand_scopes.append((collapsed[0], collapsed[1]))

        value: torch.Tensor | None = None
        variables = sorted({var for scope in operand_scopes for var in scope})
        triangle_components = _triangle_components_f32(
            scalar,
            operand_scopes,
            operands,
            variables=variables,
        )
        if triangle_components is not None:
            triangle_terms.append((float(coefficient), triangle_components))
            continue
        if (
            operands
            and len(operands) <= 4
            and all(len(scope) == 1 for scope in operand_scopes)
            and len({scope[0] for scope in operand_scopes}) == 1
        ):
            value = _batched_unary_product_sum_triton(
                scalar.contiguous(),
                [operand.contiguous() for operand in operands],
                card_count=card_count,
            )
        if value is None:
            value = _contract_collapsed_factors_batched_f32(
                scalar,
                operand_scopes,
                operands,
                hand_count=hand_count,
                hand_chunk=hand_chunk,
                card_count=card_count,
                device=device,
            )
        total += float(coefficient) * value

    triangle_batch = 8
    for start_idx in range(0, len(triangle_terms), triangle_batch):
        batch = triangle_terms[start_idx : start_idx + triangle_batch]
        values = _triangle_components_contract_bmm_f32(
            [components for _, components in batch],
            hand_count=hand_count,
        )
        if len(batch) == 1:
            total += batch[0][0] * values
            continue
        coeffs = torch.tensor(
            [coefficient for coefficient, _ in batch],
            dtype=torch.float32,
            device=device,
        )
        total += (values * coeffs[:, None, None]).sum(dim=0)

    return total / float(multiplicity)


def _pattern_factor_sum_modes_chunked_f32(
    forced_by_opp: list[list[int]],
    mode_sets: torch.Tensor,
    mode_weights: torch.Tensor,
    scalar: torch.Tensor,
    card: torch.Tensor,
    pair: torch.Tensor,
    *,
    hand_count: int,
    variable_count: int,
    multiplicity: int,
    hand_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    values = _pattern_factor_values_modes_chunked_f32(
        forced_by_opp,
        mode_sets,
        scalar,
        card,
        pair,
        hand_count=hand_count,
        variable_count=variable_count,
        multiplicity=multiplicity,
        hand_chunk=hand_chunk,
        device=device,
    )
    return (values * mode_weights[:, None]).sum(dim=0)


def _pattern_factor_values_modes_chunked_f32(
    forced_by_opp: list[list[int]],
    mode_sets: torch.Tensor,
    scalar: torch.Tensor,
    card: torch.Tensor,
    pair: torch.Tensor,
    *,
    hand_count: int,
    variable_count: int,
    multiplicity: int,
    hand_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    factors: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for opp_idx, forced_vars in enumerate(forced_by_opp):
        modes = mode_sets[:, opp_idx]
        if not forced_vars:
            factors.append(((), scalar[opp_idx].index_select(0, modes)))
        elif len(forced_vars) == 1:
            factors.append(((forced_vars[0],), card[opp_idx].index_select(0, modes)))
        else:
            factors.append(
                (
                    (forced_vars[0], forced_vars[1]),
                    pair[opp_idx].index_select(0, modes),
                )
            )
    values = _factor_sum_all_distinct_batched_f32(
        factors,
        variable_count,
        multiplicity,
        hand_count=hand_count,
        hand_chunk=hand_chunk,
        device=device,
    )
    return values


def _pattern_factor_sum_fast(
    forced_by_opp: list[list[int]],
    modes: list[int],
    scalar: torch.Tensor,
    card: torch.Tensor,
    pair: torch.Tensor,
    *,
    hand_count: int,
    variable_count: int,
    multiplicity: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    factors: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for opp_idx, mode in enumerate(modes):
        forced_vars = forced_by_opp[opp_idx]
        if not forced_vars:
            factors.append(((), scalar[opp_idx, mode]))
        elif len(forced_vars) == 1:
            factors.append(((forced_vars[0],), card[opp_idx, mode]))
        else:
            factors.append(((forced_vars[0], forced_vars[1]), pair[opp_idx, mode]))
    return _factor_sum_all_distinct(
        factors,
        variable_count,
        multiplicity,
        hand_count=hand_count,
        dtype=dtype,
        device=device,
    )


def _pattern_factor_sum_chunked_f32(
    forced_by_opp: list[list[int]],
    modes: list[int],
    scalar: torch.Tensor,
    card: torch.Tensor,
    pair: torch.Tensor,
    *,
    hand_count: int,
    variable_count: int,
    multiplicity: int,
    hand_chunk: int,
    device: torch.device,
) -> torch.Tensor:
    factors: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for opp_idx, mode in enumerate(modes):
        forced_vars = forced_by_opp[opp_idx]
        if not forced_vars:
            factors.append(((), scalar[opp_idx, mode]))
        elif len(forced_vars) == 1:
            factors.append(((forced_vars[0],), card[opp_idx, mode]))
        else:
            factors.append(((forced_vars[0], forced_vars[1]), pair[opp_idx, mode]))
    return _factor_sum_all_distinct_chunked_f32(
        factors,
        variable_count,
        multiplicity,
        hand_count=hand_count,
        hand_chunk=hand_chunk,
        device=device,
    )


def exact_nway_ie(
    prepared: PreparedShowdown,
    *,
    pattern_chunk: int,
    dtype: torch.dtype = torch.float64,
    chunked_f32: bool = False,
) -> ExactResult:
    if prepared.beliefs.shape[0] != 1:
        raise ValueError("exact_nway_ie expects one board")
    players = prepared.beliefs.shape[1]
    if players < 2:
        raise ValueError("exact_nway_ie expects at least two players")
    if players > 6:
        raise ValueError("exact_nway_ie is currently intended for up to 6-way validation")

    start = time.perf_counter()
    device = prepared.beliefs.device
    if False and chunked_f32:
        dtype = torch.float32
        pattern_chunk = max(1, pattern_chunk)
    beliefs = prepared.beliefs[0].to(dtype)
    ranks = prepared.hand_ranks.reshape(NUM_HANDS)
    masks = prepared.hand_masks.reshape(NUM_HANDS)
    combos = prepared.combos
    pair_lookup = _pair_lookup(device)
    active_card_mask = torch.ones(52, dtype=torch.bool, device=device)
    active_card_mask[prepared.board.reshape(-1).long()] = False
    active_cards = torch.nonzero(active_card_mask, as_tuple=False).flatten()
    hand_ids = torch.arange(NUM_HANDS, device=device)
    contains = torch.zeros(NUM_HANDS, 52, dtype=dtype, device=device)
    contains[hand_ids, combos[:, 0]] = 1.0
    contains[hand_ids, combos[:, 1]] = 1.0
    active_contains = contains[:, active_cards]
    pair_ids = pair_lookup[active_cards[:, None], active_cards[None, :]]
    pair_valid = pair_ids >= 0
    safe_pair_ids = pair_ids.clamp_min(0)
    relation_mats: list[torch.Tensor] | None = None
    scalar_by_player: torch.Tensor | None = None
    card_by_player: torch.Tensor | None = None
    pair_by_player: torch.Tensor | None = None

    if False and chunked_f32:
        hero_masks = masks[:, None]
        opp_masks = masks[None, :]
        disjoint = (hero_masks & opp_masks) == 0
        worse = disjoint & (ranks[None, :] < ranks[:, None])
        tied = disjoint & (ranks[None, :] == ranks[:, None])
        relation_mats = [
            worse.to(dtype),
            tied.to(dtype),
            disjoint.to(dtype),
        ]
        scalar_by_player = torch.empty(players, 3, NUM_HANDS, dtype=dtype, device=device)
        card_by_player = torch.empty(
            players,
            3,
            NUM_HANDS,
            active_cards.numel(),
            dtype=dtype,
            device=device,
        )
        pair_by_player = torch.empty(
            players,
            3,
            NUM_HANDS,
            active_cards.numel(),
            active_cards.numel(),
            dtype=dtype,
            device=device,
        )
        pair_masks = masks[safe_pair_ids]
        pair_disjoint = ((masks[:, None, None] & pair_masks[None, :, :]) == 0) & pair_valid[
            None,
            :,
            :,
        ]
        pair_ranks = ranks[safe_pair_ids]
        for player in range(players):
            belief = beliefs[player]
            weighted_contains = belief[:, None] * active_contains
            pair_mass = belief[safe_pair_ids] * pair_valid.to(dtype)
            for mode, relation in enumerate(relation_mats):
                scalar_by_player[player, mode] = relation.matmul(belief)
                card_by_player[player, mode] = relation.matmul(weighted_contains)
                if mode == 0:
                    pair_rel = pair_disjoint & (pair_ranks[None, :, :] < ranks[:, None, None])
                elif mode == 1:
                    pair_rel = pair_disjoint & (pair_ranks[None, :, :] == ranks[:, None, None])
                else:
                    pair_rel = pair_disjoint
                pair_by_player[player, mode] = pair_rel.to(dtype) * pair_mass[None, :, :]

    equities: list[torch.Tensor] = []
    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        num_opponents = len(opponents)
        if False and chunked_f32:
            opponent_ids = torch.tensor(opponents, dtype=torch.long, device=device)
            scalar = scalar_by_player.index_select(0, opponent_ids)
            card = card_by_player.index_select(0, opponent_ids)
            pair = pair_by_player.index_select(0, opponent_ids)
        else:
            hero_masks = masks[:, None]
            opp_masks = masks[None, :]
            disjoint = (hero_masks & opp_masks) == 0
            worse = disjoint & (ranks[None, :] < ranks[:, None])
            tied = disjoint & (ranks[None, :] == ranks[:, None])
            relation_mats = [
                worse.to(dtype),
                tied.to(dtype),
                disjoint.to(dtype),
            ]

            scalar = torch.empty(num_opponents, 3, NUM_HANDS, dtype=dtype, device=device)
            card = torch.empty(
                num_opponents,
                3,
                NUM_HANDS,
                active_cards.numel(),
                dtype=dtype,
                device=device,
            )
            pair = torch.empty(
                num_opponents,
                3,
                NUM_HANDS,
                active_cards.numel(),
                active_cards.numel(),
                dtype=dtype,
                device=device,
            )
            pair_masks = masks[safe_pair_ids]
            pair_disjoint = ((masks[:, None, None] & pair_masks[None, :, :]) == 0) & pair_valid[
                None,
                :,
                :,
            ]
            pair_ranks = ranks[safe_pair_ids]
            for opp_idx, player in enumerate(opponents):
                belief = beliefs[player]
                weighted_contains = belief[:, None] * active_contains
                pair_mass = belief[safe_pair_ids] * pair_valid.to(dtype)
                for mode, relation in enumerate(relation_mats):
                    scalar[opp_idx, mode] = relation.matmul(belief)
                    card[opp_idx, mode] = relation.matmul(weighted_contains)
                    if mode == 0:
                        pair_rel = pair_disjoint & (pair_ranks[None, :, :] < ranks[:, None, None])
                    elif mode == 1:
                        pair_rel = pair_disjoint & (
                            pair_ranks[None, :, :] == ranks[:, None, None]
                        )
                    else:
                        pair_rel = pair_disjoint
                    pair[opp_idx, mode] = pair_rel.to(dtype) * pair_mass[None, :, :]

        numerator = torch.zeros(NUM_HANDS, dtype=dtype, device=device)
        denominator = torch.ones(NUM_HANDS, dtype=dtype, device=device)
        for opp_idx in range(num_opponents):
            denominator *= scalar[opp_idx, 2]

        for subset in range(1 << num_opponents):
            prod = torch.ones(NUM_HANDS, dtype=dtype, device=device)
            ties = 0
            for opp_idx in range(num_opponents):
                mode = 1 if (subset >> opp_idx) & 1 else 0
                ties += mode
                prod *= scalar[opp_idx, mode]
            numerator += prod / float(ties + 1)

        if chunked_f32:
            subset_rows, subset_weights_cpu = _subset_mode_data_cpu(num_opponents)
            denom_mode_sets = torch.full(
                (1, num_opponents),
                2,
                dtype=torch.long,
                device=device,
            )
            denom_mode_weights = torch.ones(1, dtype=torch.float32, device=device)
            subset_mode_sets = torch.tensor(
                subset_rows,
                dtype=torch.long,
                device=device,
            )
            subset_mode_weights = torch.tensor(
                subset_weights_cpu,
                dtype=torch.float32,
                device=device,
            )
            combined_mode_sets = torch.cat([denom_mode_sets, subset_mode_sets], dim=0)
            combined_mode_weights = torch.cat(
                [denom_mode_weights, subset_mode_weights],
                dim=0,
            )

        for pattern_spec in _exact_pattern_specs(num_opponents):
            forced_by_opp = [list(forced) for forced in pattern_spec.forced_by_opp]
            variable_count = pattern_spec.variable_count
            multiplicity = pattern_spec.multiplicity
            sign = pattern_spec.sign
            if chunked_f32:
                combined_values = _pattern_factor_values_modes_chunked_f32(
                    forced_by_opp,
                    combined_mode_sets,
                    scalar,
                    card,
                    pair,
                    hand_count=NUM_HANDS,
                    variable_count=variable_count,
                    multiplicity=multiplicity,
                    hand_chunk=pattern_chunk,
                    device=device,
                )
                denominator += float(sign) * combined_values[0]
                pattern_num = (combined_values[1:] * combined_mode_weights[1:, None]).sum(
                    dim=0,
                )
            else:
                denominator += float(sign) * _pattern_factor_sum_fast(
                    forced_by_opp,
                    [2] * num_opponents,
                    scalar,
                    card,
                    pair,
                    hand_count=NUM_HANDS,
                    variable_count=variable_count,
                    multiplicity=multiplicity,
                    dtype=dtype,
                    device=device,
                )

            if not chunked_f32:
                pattern_num = torch.zeros(NUM_HANDS, dtype=dtype, device=device)
                for subset in range(1 << num_opponents):
                    modes = []
                    ties = 0
                    for opp_idx in range(num_opponents):
                        mode = 1 if (subset >> opp_idx) & 1 else 0
                        ties += mode
                        modes.append(mode)
                    pattern_num += _pattern_factor_sum_fast(
                        forced_by_opp,
                        modes,
                        scalar,
                        card,
                        pair,
                        hand_count=NUM_HANDS,
                        variable_count=variable_count,
                        multiplicity=multiplicity,
                        dtype=dtype,
                        device=device,
                    ) / float(ties + 1)
            numerator += float(sign) * pattern_num

        hero_belief = beliefs[hero]
        aggregate = (hero_belief * numerator).sum() / (
            hero_belief * denominator
        ).sum().clamp_min(1e-30)
        equities.append(aggregate)

    result = torch.stack(equities).to(torch.float32)[None, :]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return ExactResult(equity=result, seconds=time.perf_counter() - start)


def exact_nway_ie_chunked_f32(
    prepared: PreparedShowdown,
    *,
    pattern_chunk: int,
) -> ExactResult:
    return exact_nway_ie(
        prepared,
        pattern_chunk=pattern_chunk,
        dtype=torch.float32,
        chunked_f32=True,
    )


def restrict_beliefs_to_topk(
    prepared: PreparedShowdown,
    *,
    topk: int,
) -> PreparedShowdown:
    beliefs = torch.zeros_like(prepared.beliefs)
    allowed = board_allowed_hands(prepared.board)
    for player in range(prepared.beliefs.shape[1]):
        scores = prepared.beliefs[0, player].masked_fill(~allowed[0], 0.0)
        ids = torch.topk(scores, min(topk, int(allowed[0].sum().item()))).indices
        beliefs[0, player, ids] = prepared.beliefs[0, player, ids]
        beliefs[0, player] /= beliefs[0, player].sum().clamp_min(1e-30)
    return PreparedShowdown(
        board=prepared.board,
        beliefs=beliefs,
        hand_ranks=prepared.hand_ranks,
        combos=prepared.combos,
        hand_masks=prepared.hand_masks,
        setup_seconds=prepared.setup_seconds,
    )


def brute_force_nway(
    prepared: PreparedShowdown,
    *,
    chunk_size: int,
) -> ExactResult:
    if prepared.beliefs.shape[0] != 1:
        raise ValueError("brute_force_nway expects one board")
    start = time.perf_counter()
    beliefs = prepared.beliefs[0].to(torch.float64)
    players = beliefs.shape[0]
    support = [
        torch.nonzero(beliefs[player] > 0, as_tuple=False).flatten()
        for player in range(players)
    ]
    support_sizes = torch.tensor([ids.numel() for ids in support], device=beliefs.device)
    total = int(support_sizes.prod().item())
    ranks = prepared.hand_ranks.reshape(NUM_HANDS)
    denom = torch.zeros((), dtype=torch.float64, device=beliefs.device)
    win_sum = torch.zeros(players, dtype=torch.float64, device=beliefs.device)

    for start_idx in range(0, total, chunk_size):
        count = min(chunk_size, total - start_idx)
        linear = torch.arange(start_idx, start_idx + count, device=beliefs.device)
        combo_hands = []
        divisor = 1
        for player, ids in enumerate(support):
            local = (linear // divisor) % ids.numel()
            combo_hands.append(ids[local])
            divisor *= ids.numel()
        sampled = torch.stack(combo_hands, dim=1)
        sampled_masks = prepared.hand_masks[sampled]
        valid = torch.ones(count, dtype=torch.bool, device=beliefs.device)
        for left in range(players):
            for right in range(left + 1, players):
                valid &= (sampled_masks[:, left] & sampled_masks[:, right]) == 0
        weight = torch.ones(count, dtype=torch.float64, device=beliefs.device)
        for player in range(players):
            weight *= beliefs[player, sampled[:, player]]
        weight *= valid.to(torch.float64)
        sampled_ranks = ranks[sampled]
        max_rank = sampled_ranks.max(dim=1, keepdim=True).values
        winners = sampled_ranks == max_rank
        shares = winners.to(torch.float64) / winners.sum(dim=1, keepdim=True).clamp_min(1)
        denom += weight.sum()
        win_sum += (shares * weight[:, None]).sum(dim=0)

    equity = (win_sum / denom.clamp_min(1e-30)).to(torch.float32)[None, :]
    if prepared.beliefs.device.type == "cuda":
        torch.cuda.synchronize(prepared.beliefs.device)
    return ExactResult(equity=equity, seconds=time.perf_counter() - start)


def prepare_fast_sis_board(prepared: PreparedShowdown) -> PreparedFastSISBoard:
    if prepared.beliefs.device.type != "cuda":
        raise ValueError("fast Triton SIS requires CUDA tensors")
    start = time.perf_counter()
    allowed = board_allowed_hands(prepared.board)[0]
    active_ids = torch.nonzero(allowed, as_tuple=False).flatten().contiguous()
    active_combos = prepared.combos[active_ids]
    hand_ranks = prepared.hand_ranks.reshape(NUM_HANDS)[active_ids].contiguous()
    hand_masks = prepared.hand_masks[active_ids].contiguous()
    pair_to_active = torch.full((52, 52), -1, dtype=torch.int32, device=prepared.beliefs.device)
    active_pos = torch.arange(active_ids.numel(), dtype=torch.int32, device=prepared.beliefs.device)
    pair_to_active[active_combos[:, 0], active_combos[:, 1]] = active_pos
    pair_to_active[active_combos[:, 1], active_combos[:, 0]] = active_pos
    torch.cuda.synchronize(prepared.beliefs.device)
    return PreparedFastSISBoard(
        active_ids=active_ids,
        hand_cards0=active_combos[:, 0].to(torch.int32).contiguous(),
        hand_cards1=active_combos[:, 1].to(torch.int32).contiguous(),
        pair_to_active=pair_to_active.contiguous(),
        hand_masks=hand_masks,
        hand_ranks=hand_ranks,
        setup_seconds=time.perf_counter() - start,
    )


def prepare_fast_sis_belief(
    board: PreparedFastSISBoard,
    beliefs: torch.Tensor,
    *,
    build_cdf: bool = True,
    build_pair_mass: bool = True,
) -> PreparedFastSISBelief:
    if beliefs.shape[0] != 1:
        raise ValueError("fast Triton SIS expects one board")
    if beliefs.device.type != "cuda":
        raise ValueError("fast Triton SIS requires CUDA tensors")
    start = time.perf_counter()
    active_beliefs = beliefs[0, :, board.active_ids].to(torch.float32).contiguous()
    active_beliefs /= active_beliefs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if build_cdf:
        cdf = active_beliefs.cumsum(dim=1)
        cdf[:, -1] = 1.0
        cdf = cdf.contiguous()
    else:
        cdf = torch.empty(0, dtype=torch.float32, device=beliefs.device)

    players, active_hands = active_beliefs.shape
    c0 = board.hand_cards0.to(torch.long)
    c1 = board.hand_cards1.to(torch.long)
    card_mass = torch.zeros(players, 52, dtype=torch.float32, device=beliefs.device)
    card_mass.scatter_add_(1, c0[None, :].expand(players, -1), active_beliefs)
    card_mass.scatter_add_(1, c1[None, :].expand(players, -1), active_beliefs)

    if build_pair_mass:
        pair_mass = torch.zeros(players, 52, 52, dtype=torch.float32, device=beliefs.device)
        player_idx = torch.arange(players, device=beliefs.device)[:, None].expand(
            players,
            active_hands,
        )
        c0_idx = c0[None, :].expand(players, -1)
        c1_idx = c1[None, :].expand(players, -1)
        pair_mass[player_idx, c0_idx, c1_idx] = active_beliefs
        pair_mass[player_idx, c1_idx, c0_idx] = active_beliefs
    else:
        pair_mass = torch.empty(0, dtype=torch.float32, device=beliefs.device)

    torch.cuda.synchronize(beliefs.device)
    return PreparedFastSISBelief(
        board=board,
        beliefs=active_beliefs,
        cdf=cdf,
        card_mass=card_mass.contiguous(),
        pair_mass=pair_mass.contiguous(),
        setup_seconds=time.perf_counter() - start,
    )


def make_fast_sis_belief_workspace(
    board: PreparedFastSISBoard,
    *,
    players: int,
    build_cdf: bool = True,
    build_pair_mass: bool = True,
) -> PreparedFastSISBelief:
    device = board.active_ids.device
    active_hands = board.active_ids.numel()
    cdf = (
        torch.empty(players, active_hands, dtype=torch.float32, device=device)
        if build_cdf
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    pair_mass = (
        torch.empty(players, 52, 52, dtype=torch.float32, device=device)
        if build_pair_mass
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    return PreparedFastSISBelief(
        board=board,
        beliefs=torch.empty(players, active_hands, dtype=torch.float32, device=device),
        cdf=cdf,
        card_mass=torch.empty(players, 52, dtype=torch.float32, device=device),
        pair_mass=pair_mass,
        setup_seconds=0.0,
    )


def prepare_fast_sis_belief_into(
    workspace: PreparedFastSISBelief,
    beliefs: torch.Tensor,
) -> PreparedFastSISBelief:
    if beliefs.shape[0] != 1:
        raise ValueError("fast Triton SIS expects one board")
    if beliefs.device.type != "cuda":
        raise ValueError("fast Triton SIS requires CUDA tensors")
    start = time.perf_counter()
    board = workspace.board
    torch.index_select(beliefs[0], 1, board.active_ids, out=workspace.beliefs)
    workspace.beliefs /= workspace.beliefs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    if workspace.cdf.numel():
        torch.cumsum(workspace.beliefs, dim=1, out=workspace.cdf)
        workspace.cdf[:, -1] = 1.0

    players, active_hands = workspace.beliefs.shape
    c0 = board.hand_cards0.to(torch.long)
    c1 = board.hand_cards1.to(torch.long)
    workspace.card_mass.zero_()
    workspace.card_mass.scatter_add_(1, c0[None, :].expand(players, -1), workspace.beliefs)
    workspace.card_mass.scatter_add_(1, c1[None, :].expand(players, -1), workspace.beliefs)

    if workspace.pair_mass.numel():
        workspace.pair_mass.zero_()
        player_idx = torch.arange(players, device=beliefs.device)[:, None].expand(
            players,
            active_hands,
        )
        c0_idx = c0[None, :].expand(players, -1)
        c1_idx = c1[None, :].expand(players, -1)
        workspace.pair_mass[player_idx, c0_idx, c1_idx] = workspace.beliefs
        workspace.pair_mass[player_idx, c1_idx, c0_idx] = workspace.beliefs

    torch.cuda.synchronize(beliefs.device)
    workspace.setup_seconds = time.perf_counter() - start
    return workspace


def prepare_fast_sis_belief_no_pair_triton_into(
    workspace: PreparedFastSISBelief,
    beliefs: torch.Tensor,
    *,
    normalize: bool = True,
    synchronize: bool = True,
) -> PreparedFastSISBelief:
    if triton is None:
        raise RuntimeError("triton is not available")
    if beliefs.shape[0] != 1:
        raise ValueError("fast Triton SIS expects one board")
    if beliefs.device.type != "cuda":
        raise ValueError("fast Triton SIS requires CUDA tensors")
    if workspace.cdf.numel() or workspace.pair_mass.numel():
        raise ValueError("Triton no-pair belief prep expects empty cdf and pair_mass")
    start = time.perf_counter()
    active_hands = workspace.beliefs.shape[1]
    block_h = triton.next_power_of_2(active_hands)
    _prepare_belief_no_pair_kernel[(workspace.beliefs.shape[0],)](
        beliefs[0].contiguous(),
        workspace.board.active_ids,
        workspace.board.hand_cards0,
        workspace.board.hand_cards1,
        workspace.beliefs,
        workspace.card_mass,
        active_hands=active_hands,
        NORMALIZE=normalize,
        BLOCK_H=block_h,
        BLOCK_C=64,
        num_warps=8,
    )
    if synchronize:
        torch.cuda.synchronize(beliefs.device)
    workspace.setup_seconds = time.perf_counter() - start
    return workspace


def make_batched_fast_sis_belief_workspace(
    board: PreparedFastSISBoard,
    *,
    batch_size: int,
    players: int,
) -> PreparedBatchedFastSISBelief:
    device = board.active_ids.device
    active_hands = board.active_ids.numel()
    rows = batch_size * players
    return PreparedBatchedFastSISBelief(
        board=board,
        beliefs=torch.empty(rows, active_hands, dtype=torch.float32, device=device),
        card_mass=torch.empty(rows, 52, dtype=torch.float32, device=device),
        pair_mass=torch.empty(rows, 52, 52, dtype=torch.float16, device=device),
        batch_size=batch_size,
        players=players,
        setup_seconds=0.0,
    )


def prepare_batched_fast_sis_belief_no_pair_triton_into(
    workspace: PreparedBatchedFastSISBelief,
    beliefs: torch.Tensor,
    *,
    normalize: bool = True,
    synchronize: bool = True,
) -> PreparedBatchedFastSISBelief:
    if triton is None:
        raise RuntimeError("triton is not available")
    if beliefs.shape[:2] != (workspace.batch_size, workspace.players):
        raise ValueError("belief batch shape does not match workspace")
    if beliefs.device.type != "cuda":
        raise ValueError("batched fast Triton SIS requires CUDA tensors")
    start = time.perf_counter()
    active_hands = workspace.beliefs.shape[1]
    block_h = triton.next_power_of_2(active_hands)
    _prepare_batched_belief_no_pair_kernel[(workspace.beliefs.shape[0],)](
        beliefs.contiguous(),
        workspace.board.active_ids,
        workspace.board.hand_cards0,
        workspace.board.hand_cards1,
        workspace.beliefs,
        workspace.card_mass,
        workspace.pair_mass,
        active_hands=active_hands,
        PLAYERS=workspace.players,
        NORMALIZE=normalize,
        BLOCK_H=block_h,
        BLOCK_C=64,
        num_warps=8,
    )
    if synchronize:
        torch.cuda.synchronize(beliefs.device)
    workspace.setup_seconds = time.perf_counter() - start
    return workspace


if triton is not None:

    @triton.jit
    def _prepare_belief_no_pair_kernel(
        full_beliefs_ptr,
        active_ids_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        active_beliefs_ptr,
        card_mass_ptr,
        active_hands: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        player = tl.program_id(0)
        hand_offsets = tl.arange(0, BLOCK_H)
        hand_mask = hand_offsets < active_hands
        active_ids = tl.load(active_ids_ptr + hand_offsets, mask=hand_mask, other=0)
        values = tl.load(
            full_beliefs_ptr + player * 1326 + active_ids,
            mask=hand_mask,
            other=0.0,
        )
        if NORMALIZE:
            normalizer = tl.maximum(tl.sum(values, axis=0), 1.0e-12)
            values = values / normalizer
        tl.store(
            active_beliefs_ptr + player * active_hands + hand_offsets,
            values,
            mask=hand_mask,
        )

        card_offsets = tl.arange(0, BLOCK_C)
        tl.store(
            card_mass_ptr + player * 52 + card_offsets,
            tl.zeros((BLOCK_C,), tl.float32),
            mask=card_offsets < 52,
        )
        c0 = tl.load(hand_cards0_ptr + hand_offsets, mask=hand_mask, other=0)
        c1 = tl.load(hand_cards1_ptr + hand_offsets, mask=hand_mask, other=0)
        tl.atomic_add(card_mass_ptr + player * 52 + c0, values, sem="relaxed", mask=hand_mask)
        tl.atomic_add(card_mass_ptr + player * 52 + c1, values, sem="relaxed", mask=hand_mask)

    @triton.jit
    def _prepare_batched_belief_no_pair_kernel(
        full_beliefs_ptr,
        active_ids_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        active_beliefs_ptr,
        card_mass_ptr,
        pair_mass_ptr,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        hand_offsets = tl.arange(0, BLOCK_H)
        hand_mask = hand_offsets < active_hands
        active_ids = tl.load(active_ids_ptr + hand_offsets, mask=hand_mask, other=0)
        values = tl.load(
            full_beliefs_ptr + row * 1326 + active_ids,
            mask=hand_mask,
            other=0.0,
        )
        if NORMALIZE:
            normalizer = tl.maximum(tl.sum(values, axis=0), 1.0e-12)
            values = values / normalizer
        tl.store(
            active_beliefs_ptr + row * active_hands + hand_offsets,
            values,
            mask=hand_mask,
        )

        card_offsets = tl.arange(0, BLOCK_C)
        tl.store(
            card_mass_ptr + row * 52 + card_offsets,
            tl.zeros((BLOCK_C,), tl.float32),
            mask=card_offsets < 52,
        )
        c0 = tl.load(hand_cards0_ptr + hand_offsets, mask=hand_mask, other=0)
        c1 = tl.load(hand_cards1_ptr + hand_offsets, mask=hand_mask, other=0)
        tl.atomic_add(card_mass_ptr + row * 52 + c0, values, sem="relaxed", mask=hand_mask)
        tl.atomic_add(card_mass_ptr + row * 52 + c1, values, sem="relaxed", mask=hand_mask)
        pair_base = row * 2704
        tl.store(pair_mass_ptr + pair_base + c0 * 52 + c1, values, mask=hand_mask)
        tl.store(pair_mass_ptr + pair_base + c1 * 52 + c0, values, mask=hand_mask)

    @triton.jit
    def _unary4_product_sum_kernel(
        scalar_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        out_ptr,
        hand_count,
        card_count: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        hand_offsets = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
        card_offsets = tl.arange(0, BLOCK_C)
        active_h = hand_offsets < hand_count
        active_c = card_offsets < card_count
        offsets = hand_offsets[:, None] * card_count + card_offsets[None, :]
        values = (
            tl.load(a_ptr + offsets, mask=active_h[:, None] & active_c[None, :], other=0.0)
            * tl.load(
                b_ptr + offsets,
                mask=active_h[:, None] & active_c[None, :],
                other=0.0,
            )
            * tl.load(
                c_ptr + offsets,
                mask=active_h[:, None] & active_c[None, :],
                other=0.0,
            )
            * tl.load(
                d_ptr + offsets,
                mask=active_h[:, None] & active_c[None, :],
                other=0.0,
            )
        )
        summed = tl.sum(values, axis=1)
        scalar = tl.load(scalar_ptr + hand_offsets, mask=active_h, other=0.0)
        tl.store(out_ptr + hand_offsets, scalar * summed, mask=active_h)

    @triton.jit
    def _batched_unary_product_sum_kernel(
        scalar_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        out_ptr,
        item_count,
        card_count: tl.constexpr,
        COUNT: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        item_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        card_offsets = tl.arange(0, BLOCK_C)
        active_n = item_offsets < item_count
        active_c = card_offsets < card_count
        offsets = item_offsets[:, None] * card_count + card_offsets[None, :]
        values = tl.load(
            a_ptr + offsets,
            mask=active_n[:, None] & active_c[None, :],
            other=0.0,
        ) * tl.load(
            b_ptr + offsets,
            mask=active_n[:, None] & active_c[None, :],
            other=0.0,
        )
        if COUNT > 2:
            values *= tl.load(
                c_ptr + offsets,
                mask=active_n[:, None] & active_c[None, :],
                other=0.0,
            )
        if COUNT > 3:
            values *= tl.load(
                d_ptr + offsets,
                mask=active_n[:, None] & active_c[None, :],
                other=0.0,
            )
        summed = tl.sum(values, axis=1)
        scalar = tl.load(scalar_ptr + item_offsets, mask=active_n, other=0.0)
        tl.store(out_ptr + item_offsets, scalar * summed, mask=active_n)

    @triton.jit
    def _slot12(
        idx: tl.constexpr,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
    ):
        if idx == 0:
            return c0
        if idx == 1:
            return c1
        if idx == 2:
            return c2
        if idx == 3:
            return c3
        if idx == 4:
            return c4
        if idx == 5:
            return c5
        if idx == 6:
            return c6
        if idx == 7:
            return c7
        if idx == 8:
            return c8
        if idx == 9:
            return c9
        if idx == 10:
            return c10
        return c11

    @triton.jit
    def _active_normalizer(
        card_mass_ptr,
        pair_mass_ptr,
        player: tl.constexpr,
        used_count: tl.constexpr,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        MAX_USED: tl.constexpr,
    ):
        blocked = c0.to(tl.float32) * 0.0
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                blocked += tl.load(card_mass_ptr + player * 52 + ci)
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                for j in tl.static_range(i + 1, MAX_USED):
                    if j < used_count:
                        cj = _slot12(
                            j,
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                            c8,
                            c9,
                            c10,
                            c11,
                        )
                        blocked -= tl.load(pair_mass_ptr + player * 2704 + ci * 52 + cj)
        return tl.maximum(1.0 - blocked, 1.0e-12)

    @triton.jit
    def _active_normalizer_pair_lookup(
        beliefs_ptr,
        card_mass_ptr,
        pair_to_active_ptr,
        player: tl.constexpr,
        active_hands: tl.constexpr,
        used_count: tl.constexpr,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        MAX_USED: tl.constexpr,
    ):
        blocked = c0.to(tl.float32) * 0.0
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                blocked += tl.load(card_mass_ptr + player * 52 + ci)
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                for j in tl.static_range(i + 1, MAX_USED):
                    if j < used_count:
                        cj = _slot12(
                            j,
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                            c8,
                            c9,
                            c10,
                            c11,
                        )
                        pair_idx = tl.load(pair_to_active_ptr + ci * 52 + cj)
                        pair_value = tl.load(
                            beliefs_ptr + player * active_hands + pair_idx,
                            mask=pair_idx >= 0,
                            other=0.0,
                        )
                        blocked -= pair_value
        return tl.maximum(1.0 - blocked, 1.0e-12)

    @triton.jit
    def _active_normalizer_pair_lookup_batched(
        beliefs_ptr,
        card_mass_ptr,
        pair_to_active_ptr,
        belief_row,
        active_hands: tl.constexpr,
        used_count: tl.constexpr,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        MAX_USED: tl.constexpr,
    ):
        blocked = c0.to(tl.float32) * 0.0
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                blocked += tl.load(card_mass_ptr + belief_row * 52 + ci)
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                for j in tl.static_range(i + 1, MAX_USED):
                    if j < used_count:
                        cj = _slot12(
                            j,
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                            c8,
                            c9,
                            c10,
                            c11,
                        )
                        pair_idx = tl.load(pair_to_active_ptr + ci * 52 + cj)
                        pair_value = tl.load(
                            beliefs_ptr + belief_row * active_hands + pair_idx,
                            mask=pair_idx >= 0,
                            other=0.0,
                        )
                        blocked -= pair_value
        return tl.maximum(1.0 - blocked, 1.0e-12)

    @triton.jit
    def _active_normalizer_pair_mass_batched(
        card_mass_ptr,
        pair_mass_ptr,
        belief_row,
        used_count: tl.constexpr,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        MAX_USED: tl.constexpr,
    ):
        blocked = c0.to(tl.float32) * 0.0
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                blocked += tl.load(card_mass_ptr + belief_row * 52 + ci)
        for i in tl.static_range(0, MAX_USED):
            if i < used_count:
                ci = _slot12(i, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
                for j in tl.static_range(i + 1, MAX_USED):
                    if j < used_count:
                        cj = _slot12(
                            j,
                            c0,
                            c1,
                            c2,
                            c3,
                            c4,
                            c5,
                            c6,
                            c7,
                            c8,
                            c9,
                            c10,
                            c11,
                        )
                        blocked -= tl.load(pair_mass_ptr + belief_row * 2704 + ci * 52 + cj)
        return tl.maximum(1.0 - blocked, 1.0e-12)

    @triton.jit
    def _cdf_draw(
        cdf_ptr,
        player: tl.constexpr,
        u,
        active,
        H: tl.constexpr,
        STEPS: tl.constexpr,
    ):
        lo = u.to(tl.int32) * 0
        hi = lo + (H - 1)
        for _ in tl.static_range(0, STEPS):
            mid = (lo + hi) // 2
            value = tl.load(cdf_ptr + player * H + mid, mask=active, other=1.0)
            left = u <= value
            hi = tl.where(left, mid, hi)
            lo = tl.where(left, lo, mid + 1)
        return tl.minimum(lo, H - 1)

    @triton.jit
    def _fast_sis_kernel(
        cdf_ptr,
        card_mass_ptr,
        pair_mass_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        hand_masks_ptr,
        hand_ranks_ptr,
        weights_ptr,
        shares_ptr,
        fail_ptr,
        seed: tl.constexpr,
        sample_count,
        H: tl.constexpr,
        PLAYERS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_TRIES: tl.constexpr,
        MAX_USED: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        active = sample_idx < sample_count
        used = tl.full((BLOCK_S,), 0, tl.int64)
        weight = tl.full((BLOCK_S,), 1.0, tl.float32)
        alive = active
        c0 = tl.full((BLOCK_S,), 0, tl.int32)
        c1 = tl.full((BLOCK_S,), 0, tl.int32)
        c2 = tl.full((BLOCK_S,), 0, tl.int32)
        c3 = tl.full((BLOCK_S,), 0, tl.int32)
        c4 = tl.full((BLOCK_S,), 0, tl.int32)
        c5 = tl.full((BLOCK_S,), 0, tl.int32)
        c6 = tl.full((BLOCK_S,), 0, tl.int32)
        c7 = tl.full((BLOCK_S,), 0, tl.int32)
        c8 = tl.full((BLOCK_S,), 0, tl.int32)
        c9 = tl.full((BLOCK_S,), 0, tl.int32)
        c10 = tl.full((BLOCK_S,), 0, tl.int32)
        c11 = tl.full((BLOCK_S,), 0, tl.int32)
        r0 = tl.full((BLOCK_S,), -1, tl.int32)
        r1 = tl.full((BLOCK_S,), -1, tl.int32)
        r2 = tl.full((BLOCK_S,), -1, tl.int32)
        r3 = tl.full((BLOCK_S,), -1, tl.int32)
        r4 = tl.full((BLOCK_S,), -1, tl.int32)
        r5 = tl.full((BLOCK_S,), -1, tl.int32)

        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                norm = _active_normalizer(
                    card_mass_ptr,
                    pair_mass_ptr,
                    player,
                    player * 2,
                    c0,
                    c1,
                    c2,
                    c3,
                    c4,
                    c5,
                    c6,
                    c7,
                    c8,
                    c9,
                    c10,
                    c11,
                    MAX_USED,
                )
                weight *= norm
                found = tl.full((BLOCK_S,), False, tl.int1)
                selected = tl.full((BLOCK_S,), 0, tl.int32)
                for attempt in tl.static_range(0, MAX_TRIES):
                    u = tl.rand(seed, sample_idx * 64 + player * MAX_TRIES + attempt)
                    candidate = _cdf_draw(
                        cdf_ptr,
                        player,
                        u,
                        alive,
                        H,
                        SEARCH_STEPS,
                    )
                    candidate_mask = tl.load(hand_masks_ptr + candidate, mask=alive, other=0)
                    compatible = (candidate_mask & used) == 0
                    take = alive & compatible & ~found
                    selected = tl.where(take, candidate, selected)
                    found = found | (alive & compatible)
                alive = alive & found
                selected_mask = tl.load(hand_masks_ptr + selected, mask=alive, other=0)
                used = used | selected_mask
                a = tl.load(hand_cards0_ptr + selected, mask=alive, other=0)
                b = tl.load(hand_cards1_ptr + selected, mask=alive, other=0)
                rank = tl.load(hand_ranks_ptr + selected, mask=alive, other=-1)
                if player == 0:
                    c0 = a
                    c1 = b
                    r0 = rank
                if player == 1:
                    c2 = a
                    c3 = b
                    r1 = rank
                if player == 2:
                    c4 = a
                    c5 = b
                    r2 = rank
                if player == 3:
                    c6 = a
                    c7 = b
                    r3 = rank
                if player == 4:
                    c8 = a
                    c9 = b
                    r4 = rank
                if player == 5:
                    c10 = a
                    c11 = b
                    r5 = rank

        max01 = tl.maximum(r0, r1)
        max_rank = max01
        if PLAYERS > 2:
            max_rank = tl.maximum(max_rank, r2)
        if PLAYERS > 3:
            max_rank = tl.maximum(max_rank, r3)
        if PLAYERS > 4:
            max_rank = tl.maximum(max_rank, r4)
        if PLAYERS > 5:
            max_rank = tl.maximum(max_rank, r5)
        w0 = r0 == max_rank
        w1 = r1 == max_rank
        w2 = (r2 == max_rank) & (PLAYERS > 2)
        w3 = (r3 == max_rank) & (PLAYERS > 3)
        w4 = (r4 == max_rank) & (PLAYERS > 4)
        w5 = (r5 == max_rank) & (PLAYERS > 5)
        tie_count = (
            w0.to(tl.float32)
            + w1.to(tl.float32)
            + w2.to(tl.float32)
            + w3.to(tl.float32)
            + w4.to(tl.float32)
            + w5.to(tl.float32)
        )
        inv_tie = 1.0 / tl.maximum(tie_count, 1.0)
        weight = tl.where(alive, weight, 0.0)
        tl.store(weights_ptr + sample_idx, weight, mask=active)
        base = sample_idx * PLAYERS
        tl.store(shares_ptr + base + 0, tl.where(alive & w0, inv_tie, 0.0), mask=active)
        tl.store(shares_ptr + base + 1, tl.where(alive & w1, inv_tie, 0.0), mask=active)
        if PLAYERS > 2:
            tl.store(shares_ptr + base + 2, tl.where(alive & w2, inv_tie, 0.0), mask=active)
        if PLAYERS > 3:
            tl.store(shares_ptr + base + 3, tl.where(alive & w3, inv_tie, 0.0), mask=active)
        if PLAYERS > 4:
            tl.store(shares_ptr + base + 4, tl.where(alive & w4, inv_tie, 0.0), mask=active)
        if PLAYERS > 5:
            tl.store(shares_ptr + base + 5, tl.where(alive & w5, inv_tie, 0.0), mask=active)
        tl.store(fail_ptr + sample_idx, (~alive).to(tl.int32), mask=active)

    @triton.jit
    def _fast_sis_sample_kernel(
        cdf_ptr,
        hand_masks_ptr,
        sample_ptr,
        fail_ptr,
        seed,
        sample_count,
        H: tl.constexpr,
        PLAYERS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_TRIES: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        active = sample_idx < sample_count
        used = tl.full((BLOCK_S,), 0, tl.int64)
        alive = active
        for player in tl.static_range(0, PLAYERS):
            found = tl.full((BLOCK_S,), False, tl.int1)
            selected = tl.full((BLOCK_S,), 0, tl.int32)
            for attempt in tl.static_range(0, MAX_TRIES):
                u = tl.rand(seed, sample_idx * 64 + player * MAX_TRIES + attempt)
                candidate = _cdf_draw(
                    cdf_ptr,
                    player,
                    u,
                    alive,
                    H,
                    SEARCH_STEPS,
                )
                candidate_mask = tl.load(hand_masks_ptr + candidate, mask=alive, other=0)
                compatible = (candidate_mask & used) == 0
                take = alive & compatible & ~found
                selected = tl.where(take, candidate, selected)
                found = found | (alive & compatible)
            alive = alive & found
            selected_mask = tl.load(hand_masks_ptr + selected, mask=alive, other=0)
            used = used | selected_mask
            tl.store(sample_ptr + sample_idx * PLAYERS + player, selected, mask=active)
        tl.store(fail_ptr + sample_idx, (~alive).to(tl.int32), mask=active)

    @triton.jit
    def _uniform_is_sample_kernel(
        hand_masks_ptr,
        sample_ptr,
        fail_ptr,
        seed: tl.constexpr,
        sample_count,
        H: tl.constexpr,
        PLAYERS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_TRIES: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        active = sample_idx < sample_count
        used = tl.full((BLOCK_S,), 0, tl.int64)
        alive = active
        for player in tl.static_range(0, PLAYERS):
            found = tl.full((BLOCK_S,), False, tl.int1)
            selected = tl.full((BLOCK_S,), 0, tl.int32)
            for attempt in tl.static_range(0, MAX_TRIES):
                u = tl.rand(seed, sample_idx * 64 + player * MAX_TRIES + attempt)
                candidate = tl.minimum((u * H).to(tl.int32), H - 1)
                candidate_mask = tl.load(hand_masks_ptr + candidate, mask=alive, other=0)
                compatible = (candidate_mask & used) == 0
                take = alive & compatible & ~found
                selected = tl.where(take, candidate, selected)
                found = found | (alive & compatible)
            alive = alive & found
            selected_mask = tl.load(hand_masks_ptr + selected, mask=alive, other=0)
            used = used | selected_mask
            tl.store(sample_ptr + sample_idx * PLAYERS + player, selected, mask=active)
        tl.store(fail_ptr + sample_idx, (~alive).to(tl.int32), mask=active)

    @triton.jit
    def _stage_cdf_sample_kernel(
        cdf_ptr,
        hand_masks_ptr,
        used_ptr,
        sample_ptr,
        fail_ptr,
        seed,
        sample_count,
        H: tl.constexpr,
        PLAYER: tl.constexpr,
        PLAYERS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_TRIES: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        active = (sample_idx < sample_count) & (tl.load(fail_ptr + sample_idx, mask=sample_idx < sample_count, other=1) == 0)
        used = tl.load(used_ptr + sample_idx, mask=sample_idx < sample_count, other=0)
        found = tl.full((BLOCK_S,), False, tl.int1)
        selected = tl.full((BLOCK_S,), 0, tl.int32)
        for attempt in tl.static_range(0, MAX_TRIES):
            u = tl.rand(seed, sample_idx * 64 + PLAYER * MAX_TRIES + attempt)
            candidate = _cdf_draw(
                cdf_ptr,
                PLAYER,
                u,
                active,
                H,
                SEARCH_STEPS,
            )
            candidate_mask = tl.load(hand_masks_ptr + candidate, mask=active, other=0)
            compatible = (candidate_mask & used) == 0
            take = active & compatible & ~found
            selected = tl.where(take, candidate, selected)
            found = found | (active & compatible)
        ok = active & found
        selected_mask = tl.load(hand_masks_ptr + selected, mask=ok, other=0)
        tl.store(used_ptr + sample_idx, used | selected_mask, mask=sample_idx < sample_count)
        tl.store(sample_ptr + sample_idx * PLAYERS + PLAYER, selected, mask=sample_idx < sample_count)
        tl.store(fail_ptr + sample_idx, tl.where(ok, 0, 1), mask=active)

    @triton.jit
    def _select_first_compatible_kernel(
        candidates_ptr,
        hand_masks_ptr,
        used_ptr,
        sample_ptr,
        fail_ptr,
        sample_count,
        PLAYER: tl.constexpr,
        PLAYERS: tl.constexpr,
        TRIES: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        in_range = sample_idx < sample_count
        prior_fail = tl.load(fail_ptr + sample_idx, mask=in_range, other=1)
        active = in_range & (prior_fail == 0)
        used = tl.load(used_ptr + sample_idx, mask=in_range, other=0)
        found = tl.full((BLOCK_S,), False, tl.int1)
        selected = tl.full((BLOCK_S,), 0, tl.int64)

        for attempt in tl.static_range(0, TRIES):
            candidate = tl.load(
                candidates_ptr + sample_idx * TRIES + attempt,
                mask=active,
                other=0,
            )
            candidate_mask = tl.load(hand_masks_ptr + candidate, mask=active, other=0)
            compatible = (candidate_mask & used) == 0
            take = active & compatible & ~found
            selected = tl.where(take, candidate, selected)
            found = found | (active & compatible)

        ok = active & found
        selected_mask = tl.load(hand_masks_ptr + selected, mask=ok, other=0)
        tl.store(used_ptr + sample_idx, used | selected_mask, mask=in_range)
        tl.store(
            sample_ptr + sample_idx * PLAYERS + PLAYER,
            tl.where(ok, selected, 0),
            mask=in_range,
        )
        tl.store(
            fail_ptr + sample_idx,
            prior_fail + (active & ~found).to(tl.int32),
            mask=in_range,
        )

    @triton.jit
    def _select_all_compatible_kernel(
        candidates_ptr,
        hand_masks_ptr,
        used_ptr,
        sample_ptr,
        fail_ptr,
        sample_count,
        PLAYERS: tl.constexpr,
        TRIES: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        in_range = sample_idx < sample_count
        used = tl.load(used_ptr + sample_idx, mask=in_range, other=0)
        fail = tl.load(fail_ptr + sample_idx, mask=in_range, other=0)

        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                active = in_range & (fail == 0)
                found = tl.full((BLOCK_S,), False, tl.int1)
                selected = tl.full((BLOCK_S,), 0, tl.int64)
                player_offset = player * sample_count * TRIES

                for attempt in tl.static_range(0, TRIES):
                    candidate = tl.load(
                        candidates_ptr + player_offset + sample_idx * TRIES + attempt,
                        mask=active,
                        other=0,
                    )
                    candidate_mask = tl.load(
                        hand_masks_ptr + candidate,
                        mask=active,
                        other=0,
                    )
                    compatible = (candidate_mask & used) == 0
                    take = active & compatible & ~found
                    selected = tl.where(take, candidate, selected)
                    found = found | (active & compatible)

                ok = active & found
                selected_mask = tl.load(hand_masks_ptr + selected, mask=ok, other=0)
                used = tl.where(ok, used | selected_mask, used)
                fail += (active & ~found).to(tl.int32)
                tl.store(
                    sample_ptr + sample_idx * PLAYERS + player,
                    tl.where(ok, selected, 0),
                    mask=in_range,
                )

        tl.store(used_ptr + sample_idx, used, mask=in_range)
        tl.store(fail_ptr + sample_idx, fail, mask=in_range)

    @triton.jit
    def _sample_weights_shares_kernel(
        sample_ptr,
        fail_ptr,
        card_mass_ptr,
        pair_mass_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        hand_ranks_ptr,
        weights_ptr,
        shares_ptr,
        sample_count,
        PLAYERS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_USED: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        in_range = sample_idx < sample_count
        alive = in_range & (tl.load(fail_ptr + sample_idx, mask=in_range, other=1) == 0)
        base = sample_idx * PLAYERS

        c0 = tl.full((BLOCK_S,), 0, tl.int32)
        c1 = tl.full((BLOCK_S,), 0, tl.int32)
        c2 = tl.full((BLOCK_S,), 0, tl.int32)
        c3 = tl.full((BLOCK_S,), 0, tl.int32)
        c4 = tl.full((BLOCK_S,), 0, tl.int32)
        c5 = tl.full((BLOCK_S,), 0, tl.int32)
        c6 = tl.full((BLOCK_S,), 0, tl.int32)
        c7 = tl.full((BLOCK_S,), 0, tl.int32)
        c8 = tl.full((BLOCK_S,), 0, tl.int32)
        c9 = tl.full((BLOCK_S,), 0, tl.int32)
        c10 = tl.full((BLOCK_S,), 0, tl.int32)
        c11 = tl.full((BLOCK_S,), 0, tl.int32)
        r0 = tl.full((BLOCK_S,), -1, tl.int32)
        r1 = tl.full((BLOCK_S,), -1, tl.int32)
        r2 = tl.full((BLOCK_S,), -1, tl.int32)
        r3 = tl.full((BLOCK_S,), -1, tl.int32)
        r4 = tl.full((BLOCK_S,), -1, tl.int32)
        r5 = tl.full((BLOCK_S,), -1, tl.int32)

        weight = tl.full((BLOCK_S,), 1.0, tl.float32)
        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                if player > 0:
                    weight *= _active_normalizer(
                        card_mass_ptr,
                        pair_mass_ptr,
                        player,
                        player * 2,
                        c0,
                        c1,
                        c2,
                        c3,
                        c4,
                        c5,
                        c6,
                        c7,
                        c8,
                        c9,
                        c10,
                        c11,
                        MAX_USED,
                    )
                hand = tl.load(sample_ptr + base + player, mask=in_range, other=0)
                a = tl.load(hand_cards0_ptr + hand, mask=alive, other=0)
                b = tl.load(hand_cards1_ptr + hand, mask=alive, other=0)
                rank = tl.load(hand_ranks_ptr + hand, mask=alive, other=-1)
                if player == 0:
                    c0 = a
                    c1 = b
                    r0 = rank
                if player == 1:
                    c2 = a
                    c3 = b
                    r1 = rank
                if player == 2:
                    c4 = a
                    c5 = b
                    r2 = rank
                if player == 3:
                    c6 = a
                    c7 = b
                    r3 = rank
                if player == 4:
                    c8 = a
                    c9 = b
                    r4 = rank
                if player == 5:
                    c10 = a
                    c11 = b
                    r5 = rank

        max_rank = tl.maximum(r0, r1)
        if PLAYERS > 2:
            max_rank = tl.maximum(max_rank, r2)
        if PLAYERS > 3:
            max_rank = tl.maximum(max_rank, r3)
        if PLAYERS > 4:
            max_rank = tl.maximum(max_rank, r4)
        if PLAYERS > 5:
            max_rank = tl.maximum(max_rank, r5)

        w0 = (r0 == max_rank) & (PLAYERS > 0)
        w1 = (r1 == max_rank) & (PLAYERS > 1)
        w2 = (r2 == max_rank) & (PLAYERS > 2)
        w3 = (r3 == max_rank) & (PLAYERS > 3)
        w4 = (r4 == max_rank) & (PLAYERS > 4)
        w5 = (r5 == max_rank) & (PLAYERS > 5)
        tie_count = (
            w0.to(tl.float32)
            + w1.to(tl.float32)
            + w2.to(tl.float32)
            + w3.to(tl.float32)
            + w4.to(tl.float32)
            + w5.to(tl.float32)
        )
        inv_tie = 1.0 / tl.maximum(tie_count, 1.0)
        weight = tl.where(alive, weight, 0.0)
        tl.store(weights_ptr + sample_idx, weight, mask=in_range)
        tl.store(shares_ptr + base + 0, tl.where(alive & w0, inv_tie, 0.0), mask=in_range)
        tl.store(shares_ptr + base + 1, tl.where(alive & w1, inv_tie, 0.0), mask=in_range)
        if PLAYERS > 2:
            tl.store(shares_ptr + base + 2, tl.where(alive & w2, inv_tie, 0.0), mask=in_range)
        if PLAYERS > 3:
            tl.store(shares_ptr + base + 3, tl.where(alive & w3, inv_tie, 0.0), mask=in_range)
        if PLAYERS > 4:
            tl.store(shares_ptr + base + 4, tl.where(alive & w4, inv_tie, 0.0), mask=in_range)
        if PLAYERS > 5:
            tl.store(shares_ptr + base + 5, tl.where(alive & w5, inv_tie, 0.0), mask=in_range)

    @triton.jit
    def _sample_accumulate_kernel(
        sample_ptr,
        fail_ptr,
        beliefs_ptr,
        card_mass_ptr,
        pair_to_active_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        hand_ranks_ptr,
        accum_ptr,
        sample_count,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_USED: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        in_range = sample_idx < sample_count
        alive = in_range & (tl.load(fail_ptr + sample_idx, mask=in_range, other=1) == 0)
        base = sample_idx * PLAYERS

        c0 = tl.full((BLOCK_S,), 0, tl.int32)
        c1 = tl.full((BLOCK_S,), 0, tl.int32)
        c2 = tl.full((BLOCK_S,), 0, tl.int32)
        c3 = tl.full((BLOCK_S,), 0, tl.int32)
        c4 = tl.full((BLOCK_S,), 0, tl.int32)
        c5 = tl.full((BLOCK_S,), 0, tl.int32)
        c6 = tl.full((BLOCK_S,), 0, tl.int32)
        c7 = tl.full((BLOCK_S,), 0, tl.int32)
        c8 = tl.full((BLOCK_S,), 0, tl.int32)
        c9 = tl.full((BLOCK_S,), 0, tl.int32)
        c10 = tl.full((BLOCK_S,), 0, tl.int32)
        c11 = tl.full((BLOCK_S,), 0, tl.int32)
        r0 = tl.full((BLOCK_S,), -1, tl.int32)
        r1 = tl.full((BLOCK_S,), -1, tl.int32)
        r2 = tl.full((BLOCK_S,), -1, tl.int32)
        r3 = tl.full((BLOCK_S,), -1, tl.int32)
        r4 = tl.full((BLOCK_S,), -1, tl.int32)
        r5 = tl.full((BLOCK_S,), -1, tl.int32)

        weight = tl.full((BLOCK_S,), 1.0, tl.float32)
        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                if player > 0:
                    weight *= _active_normalizer_pair_lookup(
                        beliefs_ptr,
                        card_mass_ptr,
                        pair_to_active_ptr,
                        player,
                        active_hands,
                        player * 2,
                        c0,
                        c1,
                        c2,
                        c3,
                        c4,
                        c5,
                        c6,
                        c7,
                        c8,
                        c9,
                        c10,
                        c11,
                        MAX_USED,
                    )
                hand = tl.load(sample_ptr + base + player, mask=in_range, other=0)
                a = tl.load(hand_cards0_ptr + hand, mask=alive, other=0)
                b = tl.load(hand_cards1_ptr + hand, mask=alive, other=0)
                rank = tl.load(hand_ranks_ptr + hand, mask=alive, other=-1)
                if player == 0:
                    c0 = a
                    c1 = b
                    r0 = rank
                if player == 1:
                    c2 = a
                    c3 = b
                    r1 = rank
                if player == 2:
                    c4 = a
                    c5 = b
                    r2 = rank
                if player == 3:
                    c6 = a
                    c7 = b
                    r3 = rank
                if player == 4:
                    c8 = a
                    c9 = b
                    r4 = rank
                if player == 5:
                    c10 = a
                    c11 = b
                    r5 = rank

        max_rank = tl.maximum(r0, r1)
        if PLAYERS > 2:
            max_rank = tl.maximum(max_rank, r2)
        if PLAYERS > 3:
            max_rank = tl.maximum(max_rank, r3)
        if PLAYERS > 4:
            max_rank = tl.maximum(max_rank, r4)
        if PLAYERS > 5:
            max_rank = tl.maximum(max_rank, r5)

        w0 = (r0 == max_rank) & (PLAYERS > 0)
        w1 = (r1 == max_rank) & (PLAYERS > 1)
        w2 = (r2 == max_rank) & (PLAYERS > 2)
        w3 = (r3 == max_rank) & (PLAYERS > 3)
        w4 = (r4 == max_rank) & (PLAYERS > 4)
        w5 = (r5 == max_rank) & (PLAYERS > 5)
        tie_count = (
            w0.to(tl.float32)
            + w1.to(tl.float32)
            + w2.to(tl.float32)
            + w3.to(tl.float32)
            + w4.to(tl.float32)
            + w5.to(tl.float32)
        )
        inv_tie = 1.0 / tl.maximum(tie_count, 1.0)
        weight = tl.where(alive, weight, 0.0)
        weight2 = weight * weight
        fail_count = (in_range & ~alive).to(tl.float32)

        tl.atomic_add(accum_ptr + 0, tl.sum(weight, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 1, tl.sum(weight2, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 2, tl.sum(fail_count, axis=0), sem="relaxed")

        share0 = tl.where(alive & w0, inv_tie, 0.0)
        share1 = tl.where(alive & w1, inv_tie, 0.0)
        tl.atomic_add(accum_ptr + 3, tl.sum(weight * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 4, tl.sum(weight * share1, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 3 + PLAYERS, tl.sum(weight * share0 * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 4 + PLAYERS, tl.sum(weight * share1 * share1, axis=0), sem="relaxed")
        if PLAYERS > 2:
            share2 = tl.where(alive & w2, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 5, tl.sum(weight * share2, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 5 + PLAYERS, tl.sum(weight * share2 * share2, axis=0), sem="relaxed")
        if PLAYERS > 3:
            share3 = tl.where(alive & w3, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 6, tl.sum(weight * share3, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 6 + PLAYERS, tl.sum(weight * share3 * share3, axis=0), sem="relaxed")
        if PLAYERS > 4:
            share4 = tl.where(alive & w4, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 7, tl.sum(weight * share4, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 7 + PLAYERS, tl.sum(weight * share4 * share4, axis=0), sem="relaxed")
        if PLAYERS > 5:
            share5 = tl.where(alive & w5, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 8, tl.sum(weight * share5, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 8 + PLAYERS, tl.sum(weight * share5 * share5, axis=0), sem="relaxed")

    @triton.jit
    def _select_accumulate_kernel(
        candidates_ptr,
        beliefs_ptr,
        card_mass_ptr,
        pair_to_active_ptr,
        hand_masks_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        hand_ranks_ptr,
        accum_ptr,
        sample_count,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        TRIES: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_USED: tl.constexpr,
    ):
        sample_idx = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        in_range = sample_idx < sample_count
        used = tl.full((BLOCK_S,), 0, tl.int64)
        alive = in_range
        c0 = tl.full((BLOCK_S,), 0, tl.int32)
        c1 = tl.full((BLOCK_S,), 0, tl.int32)
        c2 = tl.full((BLOCK_S,), 0, tl.int32)
        c3 = tl.full((BLOCK_S,), 0, tl.int32)
        c4 = tl.full((BLOCK_S,), 0, tl.int32)
        c5 = tl.full((BLOCK_S,), 0, tl.int32)
        c6 = tl.full((BLOCK_S,), 0, tl.int32)
        c7 = tl.full((BLOCK_S,), 0, tl.int32)
        c8 = tl.full((BLOCK_S,), 0, tl.int32)
        c9 = tl.full((BLOCK_S,), 0, tl.int32)
        c10 = tl.full((BLOCK_S,), 0, tl.int32)
        c11 = tl.full((BLOCK_S,), 0, tl.int32)
        r0 = tl.full((BLOCK_S,), -1, tl.int32)
        r1 = tl.full((BLOCK_S,), -1, tl.int32)
        r2 = tl.full((BLOCK_S,), -1, tl.int32)
        r3 = tl.full((BLOCK_S,), -1, tl.int32)
        r4 = tl.full((BLOCK_S,), -1, tl.int32)
        r5 = tl.full((BLOCK_S,), -1, tl.int32)
        weight = tl.full((BLOCK_S,), 1.0, tl.float32)

        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                if player > 0:
                    weight *= _active_normalizer_pair_lookup(
                        beliefs_ptr,
                        card_mass_ptr,
                        pair_to_active_ptr,
                        player,
                        active_hands,
                        player * 2,
                        c0,
                        c1,
                        c2,
                        c3,
                        c4,
                        c5,
                        c6,
                        c7,
                        c8,
                        c9,
                        c10,
                        c11,
                        MAX_USED,
                    )
                found = tl.full((BLOCK_S,), False, tl.int1)
                selected = tl.full((BLOCK_S,), 0, tl.int64)
                player_offset = player * sample_count * TRIES
                for attempt in tl.static_range(0, TRIES):
                    candidate = tl.load(
                        candidates_ptr + player_offset + sample_idx * TRIES + attempt,
                        mask=alive,
                        other=0,
                    )
                    candidate_mask = tl.load(hand_masks_ptr + candidate, mask=alive, other=0)
                    compatible = (candidate_mask & used) == 0
                    take = alive & compatible & ~found
                    selected = tl.where(take, candidate, selected)
                    found = found | (alive & compatible)

                ok = alive & found
                selected_mask = tl.load(hand_masks_ptr + selected, mask=ok, other=0)
                used = tl.where(ok, used | selected_mask, used)
                alive = alive & found
                a = tl.load(hand_cards0_ptr + selected, mask=ok, other=0)
                b = tl.load(hand_cards1_ptr + selected, mask=ok, other=0)
                rank = tl.load(hand_ranks_ptr + selected, mask=ok, other=-1)
                if player == 0:
                    c0 = a
                    c1 = b
                    r0 = rank
                if player == 1:
                    c2 = a
                    c3 = b
                    r1 = rank
                if player == 2:
                    c4 = a
                    c5 = b
                    r2 = rank
                if player == 3:
                    c6 = a
                    c7 = b
                    r3 = rank
                if player == 4:
                    c8 = a
                    c9 = b
                    r4 = rank
                if player == 5:
                    c10 = a
                    c11 = b
                    r5 = rank

        max_rank = tl.maximum(r0, r1)
        if PLAYERS > 2:
            max_rank = tl.maximum(max_rank, r2)
        if PLAYERS > 3:
            max_rank = tl.maximum(max_rank, r3)
        if PLAYERS > 4:
            max_rank = tl.maximum(max_rank, r4)
        if PLAYERS > 5:
            max_rank = tl.maximum(max_rank, r5)

        w0 = (r0 == max_rank) & (PLAYERS > 0)
        w1 = (r1 == max_rank) & (PLAYERS > 1)
        w2 = (r2 == max_rank) & (PLAYERS > 2)
        w3 = (r3 == max_rank) & (PLAYERS > 3)
        w4 = (r4 == max_rank) & (PLAYERS > 4)
        w5 = (r5 == max_rank) & (PLAYERS > 5)
        tie_count = (
            w0.to(tl.float32)
            + w1.to(tl.float32)
            + w2.to(tl.float32)
            + w3.to(tl.float32)
            + w4.to(tl.float32)
            + w5.to(tl.float32)
        )
        inv_tie = 1.0 / tl.maximum(tie_count, 1.0)
        weight = tl.where(alive, weight, 0.0)
        weight2 = weight * weight
        fail_count = (in_range & ~alive).to(tl.float32)

        tl.atomic_add(accum_ptr + 0, tl.sum(weight, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 1, tl.sum(weight2, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 2, tl.sum(fail_count, axis=0), sem="relaxed")
        share0 = tl.where(alive & w0, inv_tie, 0.0)
        share1 = tl.where(alive & w1, inv_tie, 0.0)
        tl.atomic_add(accum_ptr + 3, tl.sum(weight * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 4, tl.sum(weight * share1, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 3 + PLAYERS, tl.sum(weight * share0 * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + 4 + PLAYERS, tl.sum(weight * share1 * share1, axis=0), sem="relaxed")
        if PLAYERS > 2:
            share2 = tl.where(alive & w2, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 5, tl.sum(weight * share2, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 5 + PLAYERS, tl.sum(weight * share2 * share2, axis=0), sem="relaxed")
        if PLAYERS > 3:
            share3 = tl.where(alive & w3, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 6, tl.sum(weight * share3, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 6 + PLAYERS, tl.sum(weight * share3 * share3, axis=0), sem="relaxed")
        if PLAYERS > 4:
            share4 = tl.where(alive & w4, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 7, tl.sum(weight * share4, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 7 + PLAYERS, tl.sum(weight * share4 * share4, axis=0), sem="relaxed")
        if PLAYERS > 5:
            share5 = tl.where(alive & w5, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + 8, tl.sum(weight * share5, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + 8 + PLAYERS, tl.sum(weight * share5 * share5, axis=0), sem="relaxed")

    @triton.jit
    def _select_accumulate_batched_kernel(
        candidates_ptr,
        beliefs_ptr,
        card_mass_ptr,
        pair_to_active_ptr,
        hand_masks_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        hand_ranks_ptr,
        accum_ptr,
        sample_count,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        TRIES: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_USED: tl.constexpr,
    ):
        batch = tl.program_id(0)
        sample_idx = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
        in_range = sample_idx < sample_count
        used = tl.full((BLOCK_S,), 0, tl.int64)
        alive = in_range
        c0 = tl.full((BLOCK_S,), 0, tl.int32)
        c1 = tl.full((BLOCK_S,), 0, tl.int32)
        c2 = tl.full((BLOCK_S,), 0, tl.int32)
        c3 = tl.full((BLOCK_S,), 0, tl.int32)
        c4 = tl.full((BLOCK_S,), 0, tl.int32)
        c5 = tl.full((BLOCK_S,), 0, tl.int32)
        c6 = tl.full((BLOCK_S,), 0, tl.int32)
        c7 = tl.full((BLOCK_S,), 0, tl.int32)
        c8 = tl.full((BLOCK_S,), 0, tl.int32)
        c9 = tl.full((BLOCK_S,), 0, tl.int32)
        c10 = tl.full((BLOCK_S,), 0, tl.int32)
        c11 = tl.full((BLOCK_S,), 0, tl.int32)
        r0 = tl.full((BLOCK_S,), -1, tl.int32)
        r1 = tl.full((BLOCK_S,), -1, tl.int32)
        r2 = tl.full((BLOCK_S,), -1, tl.int32)
        r3 = tl.full((BLOCK_S,), -1, tl.int32)
        r4 = tl.full((BLOCK_S,), -1, tl.int32)
        r5 = tl.full((BLOCK_S,), -1, tl.int32)
        weight = tl.full((BLOCK_S,), 1.0, tl.float32)

        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                belief_row = batch * PLAYERS + player
                if player > 0:
                    weight *= _active_normalizer_pair_lookup_batched(
                        beliefs_ptr,
                        card_mass_ptr,
                        pair_to_active_ptr,
                        belief_row,
                        active_hands,
                        player * 2,
                        c0,
                        c1,
                        c2,
                        c3,
                        c4,
                        c5,
                        c6,
                        c7,
                        c8,
                        c9,
                        c10,
                        c11,
                        MAX_USED,
                    )
                found = tl.full((BLOCK_S,), False, tl.int1)
                selected = tl.full((BLOCK_S,), 0, tl.int64)
                row_offset = belief_row * sample_count * TRIES
                for attempt in tl.static_range(0, TRIES):
                    candidate = tl.load(
                        candidates_ptr + row_offset + sample_idx * TRIES + attempt,
                        mask=alive,
                        other=0,
                    )
                    candidate_mask = tl.load(hand_masks_ptr + candidate, mask=alive, other=0)
                    compatible = (candidate_mask & used) == 0
                    take = alive & compatible & ~found
                    selected = tl.where(take, candidate, selected)
                    found = found | (alive & compatible)

                ok = alive & found
                selected_mask = tl.load(hand_masks_ptr + selected, mask=ok, other=0)
                used = tl.where(ok, used | selected_mask, used)
                alive = alive & found
                a = tl.load(hand_cards0_ptr + selected, mask=ok, other=0)
                b = tl.load(hand_cards1_ptr + selected, mask=ok, other=0)
                rank = tl.load(hand_ranks_ptr + selected, mask=ok, other=-1)
                if player == 0:
                    c0 = a
                    c1 = b
                    r0 = rank
                if player == 1:
                    c2 = a
                    c3 = b
                    r1 = rank
                if player == 2:
                    c4 = a
                    c5 = b
                    r2 = rank
                if player == 3:
                    c6 = a
                    c7 = b
                    r3 = rank
                if player == 4:
                    c8 = a
                    c9 = b
                    r4 = rank
                if player == 5:
                    c10 = a
                    c11 = b
                    r5 = rank

        max_rank = tl.maximum(r0, r1)
        if PLAYERS > 2:
            max_rank = tl.maximum(max_rank, r2)
        if PLAYERS > 3:
            max_rank = tl.maximum(max_rank, r3)
        if PLAYERS > 4:
            max_rank = tl.maximum(max_rank, r4)
        if PLAYERS > 5:
            max_rank = tl.maximum(max_rank, r5)

        w0 = (r0 == max_rank) & (PLAYERS > 0)
        w1 = (r1 == max_rank) & (PLAYERS > 1)
        w2 = (r2 == max_rank) & (PLAYERS > 2)
        w3 = (r3 == max_rank) & (PLAYERS > 3)
        w4 = (r4 == max_rank) & (PLAYERS > 4)
        w5 = (r5 == max_rank) & (PLAYERS > 5)
        tie_count = (
            w0.to(tl.float32)
            + w1.to(tl.float32)
            + w2.to(tl.float32)
            + w3.to(tl.float32)
            + w4.to(tl.float32)
            + w5.to(tl.float32)
        )
        inv_tie = 1.0 / tl.maximum(tie_count, 1.0)
        weight = tl.where(alive, weight, 0.0)
        weight2 = weight * weight
        fail_count = (in_range & ~alive).to(tl.float32)
        accum_base = batch * (3 + 2 * PLAYERS)

        tl.atomic_add(accum_ptr + accum_base + 0, tl.sum(weight, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 1, tl.sum(weight2, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 2, tl.sum(fail_count, axis=0), sem="relaxed")
        share0 = tl.where(alive & w0, inv_tie, 0.0)
        share1 = tl.where(alive & w1, inv_tie, 0.0)
        tl.atomic_add(accum_ptr + accum_base + 3, tl.sum(weight * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 4, tl.sum(weight * share1, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 3 + PLAYERS, tl.sum(weight * share0 * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 4 + PLAYERS, tl.sum(weight * share1 * share1, axis=0), sem="relaxed")
        if PLAYERS > 2:
            share2 = tl.where(alive & w2, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 5, tl.sum(weight * share2, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 5 + PLAYERS, tl.sum(weight * share2 * share2, axis=0), sem="relaxed")
        if PLAYERS > 3:
            share3 = tl.where(alive & w3, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 6, tl.sum(weight * share3, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 6 + PLAYERS, tl.sum(weight * share3 * share3, axis=0), sem="relaxed")
        if PLAYERS > 4:
            share4 = tl.where(alive & w4, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 7, tl.sum(weight * share4, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 7 + PLAYERS, tl.sum(weight * share4 * share4, axis=0), sem="relaxed")
        if PLAYERS > 5:
            share5 = tl.where(alive & w5, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 8, tl.sum(weight * share5, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 8 + PLAYERS, tl.sum(weight * share5 * share5, axis=0), sem="relaxed")

    @triton.jit
    def _pcg_hash_u32(x):
        state = x * 747796405 + 2891336453
        shift = (state >> 28) + 4
        word = ((state >> shift) ^ state) * 277803737
        return (word >> 22) ^ word

    @triton.jit
    def _alias_partition_kernel(
        beliefs_ptr,
        alias_prob_ptr,
        alias_idx_ptr,
        small_stack_ptr,
        large_stack_ptr,
        counts_ptr,
        active_hands: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        active = offs < active_hands
        row_offset = row * active_hands
        scaled = tl.load(beliefs_ptr + row_offset + offs, mask=active, other=0.0)
        scaled = scaled * active_hands
        tl.store(alias_prob_ptr + row_offset + offs, scaled, mask=active)
        tl.store(alias_idx_ptr + row_offset + offs, offs, mask=active)

        small_mask = active & (scaled < 1.0)
        large_mask = active & ~small_mask
        small_rank = tl.cumsum(small_mask.to(tl.int32), 0) - 1
        large_rank = tl.cumsum(large_mask.to(tl.int32), 0) - 1
        small_count = tl.sum(small_mask.to(tl.int32), axis=0)
        large_count = tl.sum(large_mask.to(tl.int32), axis=0)
        tl.store(small_stack_ptr + row_offset + small_rank, offs, mask=small_mask)
        tl.store(large_stack_ptr + row_offset + large_rank, offs, mask=large_mask)
        tl.store(counts_ptr + row * 2, small_count)
        tl.store(counts_ptr + row * 2 + 1, large_count)

    @triton.jit
    def _alias_resolve_stacks_kernel(
        alias_prob_ptr,
        alias_idx_ptr,
        small_stack_ptr,
        large_stack_ptr,
        counts_ptr,
        active_hands: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_offset = row * active_hands
        small_count = tl.load(counts_ptr + row * 2)
        large_count = tl.load(counts_ptr + row * 2 + 1)

        while (small_count > 0) & (large_count > 0):
            small_count -= 1
            small_idx = tl.load(small_stack_ptr + row_offset + small_count)
            large_idx = tl.load(large_stack_ptr + row_offset + large_count - 1)
            small_value = tl.load(alias_prob_ptr + row_offset + small_idx)
            large_value = tl.load(alias_prob_ptr + row_offset + large_idx)
            tl.store(alias_prob_ptr + row_offset + small_idx, small_value)
            tl.store(alias_idx_ptr + row_offset + small_idx, large_idx)
            new_large_value = large_value + small_value - 1.0
            tl.store(alias_prob_ptr + row_offset + large_idx, new_large_value)
            if new_large_value < 1.0:
                large_count -= 1
                tl.store(small_stack_ptr + row_offset + small_count, large_idx)
                small_count += 1

        while large_count > 0:
            large_count -= 1
            idx = tl.load(large_stack_ptr + row_offset + large_count)
            tl.store(alias_prob_ptr + row_offset + idx, 1.0)
            tl.store(alias_idx_ptr + row_offset + idx, idx)

        while small_count > 0:
            small_count -= 1
            idx = tl.load(small_stack_ptr + row_offset + small_count)
            tl.store(alias_prob_ptr + row_offset + idx, 1.0)
            tl.store(alias_idx_ptr + row_offset + idx, idx)

    @triton.jit
    def _alias_sample_batched_kernel(
        alias_prob_ptr,
        alias_idx_ptr,
        candidates_ptr,
        seed,
        sample_count,
        active_hands: tl.constexpr,
        TRIES: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        sample_idx = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
        active = sample_idx < sample_count
        row_offset = row * active_hands
        candidate_row_offset = row * sample_count * TRIES
        seed_mix = seed * tl.full((), 0x9E3779B9, tl.uint32)
        for attempt in tl.static_range(0, TRIES):
            random_offset = (row * sample_count + sample_idx) * (TRIES * 2) + attempt * 2
            x0 = _pcg_hash_u32(random_offset.to(tl.uint32) + seed_mix)
            x1 = _pcg_hash_u32((random_offset + 1).to(tl.uint32) + seed_mix)
            u_col = x0.to(tl.float32) * 2.3283064365386963e-10
            u_pick = x1.to(tl.float32) * 2.3283064365386963e-10
            column = tl.minimum((u_col * active_hands).to(tl.int32), active_hands - 1)
            cutoff = tl.load(alias_prob_ptr + row_offset + column, mask=active, other=1.0)
            alias = tl.load(alias_idx_ptr + row_offset + column, mask=active, other=0)
            candidate = tl.where(u_pick < cutoff, column, alias)
            tl.store(
                candidates_ptr + candidate_row_offset + sample_idx * TRIES + attempt,
                candidate,
                mask=active,
            )

    @triton.jit
    def _alias_accumulate_batched_kernel(
        alias_prob_ptr,
        alias_idx_ptr,
        beliefs_ptr,
        card_mass_ptr,
        pair_mass_ptr,
        hand_masks_ptr,
        hand_cards0_ptr,
        hand_cards1_ptr,
        hand_ranks_ptr,
        accum_ptr,
        seed_ptr,
        sample_count,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        TRIES: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_USED: tl.constexpr,
    ):
        batch = tl.program_id(0)
        sample_idx = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)
        in_range = sample_idx < sample_count
        used = tl.full((BLOCK_S,), 0, tl.int64)
        alive = in_range
        c0 = tl.full((BLOCK_S,), 0, tl.int32)
        c1 = tl.full((BLOCK_S,), 0, tl.int32)
        c2 = tl.full((BLOCK_S,), 0, tl.int32)
        c3 = tl.full((BLOCK_S,), 0, tl.int32)
        c4 = tl.full((BLOCK_S,), 0, tl.int32)
        c5 = tl.full((BLOCK_S,), 0, tl.int32)
        c6 = tl.full((BLOCK_S,), 0, tl.int32)
        c7 = tl.full((BLOCK_S,), 0, tl.int32)
        c8 = tl.full((BLOCK_S,), 0, tl.int32)
        c9 = tl.full((BLOCK_S,), 0, tl.int32)
        c10 = tl.full((BLOCK_S,), 0, tl.int32)
        c11 = tl.full((BLOCK_S,), 0, tl.int32)
        r0 = tl.full((BLOCK_S,), -1, tl.int32)
        r1 = tl.full((BLOCK_S,), -1, tl.int32)
        r2 = tl.full((BLOCK_S,), -1, tl.int32)
        r3 = tl.full((BLOCK_S,), -1, tl.int32)
        r4 = tl.full((BLOCK_S,), -1, tl.int32)
        r5 = tl.full((BLOCK_S,), -1, tl.int32)
        weight = tl.full((BLOCK_S,), 1.0, tl.float32)
        base_sample = batch * sample_count + sample_idx
        seed_value = tl.load(seed_ptr).to(tl.uint32)
        seed_mix = seed_value * tl.full((), 0x9E3779B9, tl.uint32)

        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                belief_row = batch * PLAYERS + player
                if player > 0:
                    weight *= _active_normalizer_pair_mass_batched(
                        card_mass_ptr,
                        pair_mass_ptr,
                        belief_row,
                        player * 2,
                        c0,
                        c1,
                        c2,
                        c3,
                        c4,
                        c5,
                        c6,
                        c7,
                        c8,
                        c9,
                        c10,
                        c11,
                        MAX_USED,
                    )
                found = tl.full((BLOCK_S,), False, tl.int1)
                selected = tl.full((BLOCK_S,), 0, tl.int32)
                row_offset = belief_row * active_hands
                for attempt in tl.static_range(0, TRIES):
                    random_offset = (
                        base_sample * (PLAYERS * TRIES * 2)
                        + player * (TRIES * 2)
                        + attempt * 2
                    )
                    x0 = _pcg_hash_u32(random_offset.to(tl.uint32) + seed_mix)
                    x1 = _pcg_hash_u32((random_offset + 1).to(tl.uint32) + seed_mix)
                    u_col = x0.to(tl.float32) * 2.3283064365386963e-10
                    u_pick = x1.to(tl.float32) * 2.3283064365386963e-10
                    column = tl.minimum((u_col * active_hands).to(tl.int32), active_hands - 1)
                    searching = alive & ~found
                    cutoff = tl.load(alias_prob_ptr + row_offset + column, mask=searching, other=1.0)
                    alias = tl.load(alias_idx_ptr + row_offset + column, mask=searching, other=0)
                    candidate = tl.where(u_pick < cutoff, column, alias)
                    candidate_mask = tl.load(hand_masks_ptr + candidate, mask=searching, other=0)
                    compatible = (candidate_mask & used) == 0
                    take = searching & compatible
                    selected = tl.where(take, candidate, selected)
                    found = found | take

                ok = alive & found
                selected_mask = tl.load(hand_masks_ptr + selected, mask=ok, other=0)
                used = tl.where(ok, used | selected_mask, used)
                alive = alive & found
                a = tl.load(hand_cards0_ptr + selected, mask=ok, other=0)
                b = tl.load(hand_cards1_ptr + selected, mask=ok, other=0)
                rank = tl.load(hand_ranks_ptr + selected, mask=ok, other=-1)
                if player == 0:
                    c0 = a
                    c1 = b
                    r0 = rank
                if player == 1:
                    c2 = a
                    c3 = b
                    r1 = rank
                if player == 2:
                    c4 = a
                    c5 = b
                    r2 = rank
                if player == 3:
                    c6 = a
                    c7 = b
                    r3 = rank
                if player == 4:
                    c8 = a
                    c9 = b
                    r4 = rank
                if player == 5:
                    c10 = a
                    c11 = b
                    r5 = rank

        max_rank = tl.maximum(r0, r1)
        if PLAYERS > 2:
            max_rank = tl.maximum(max_rank, r2)
        if PLAYERS > 3:
            max_rank = tl.maximum(max_rank, r3)
        if PLAYERS > 4:
            max_rank = tl.maximum(max_rank, r4)
        if PLAYERS > 5:
            max_rank = tl.maximum(max_rank, r5)

        w0 = (r0 == max_rank) & (PLAYERS > 0)
        w1 = (r1 == max_rank) & (PLAYERS > 1)
        w2 = (r2 == max_rank) & (PLAYERS > 2)
        w3 = (r3 == max_rank) & (PLAYERS > 3)
        w4 = (r4 == max_rank) & (PLAYERS > 4)
        w5 = (r5 == max_rank) & (PLAYERS > 5)
        tie_count = (
            w0.to(tl.float32)
            + w1.to(tl.float32)
            + w2.to(tl.float32)
            + w3.to(tl.float32)
            + w4.to(tl.float32)
            + w5.to(tl.float32)
        )
        inv_tie = 1.0 / tl.maximum(tie_count, 1.0)
        weight = tl.where(alive, weight, 0.0)
        weight2 = weight * weight
        fail_count = (in_range & ~alive).to(tl.float32)
        accum_base = batch * (3 + 2 * PLAYERS)

        tl.atomic_add(accum_ptr + accum_base + 0, tl.sum(weight, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 1, tl.sum(weight2, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 2, tl.sum(fail_count, axis=0), sem="relaxed")
        share0 = tl.where(alive & w0, inv_tie, 0.0)
        share1 = tl.where(alive & w1, inv_tie, 0.0)
        tl.atomic_add(accum_ptr + accum_base + 3, tl.sum(weight * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 4, tl.sum(weight * share1, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 3 + PLAYERS, tl.sum(weight * share0 * share0, axis=0), sem="relaxed")
        tl.atomic_add(accum_ptr + accum_base + 4 + PLAYERS, tl.sum(weight * share1 * share1, axis=0), sem="relaxed")
        if PLAYERS > 2:
            share2 = tl.where(alive & w2, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 5, tl.sum(weight * share2, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 5 + PLAYERS, tl.sum(weight * share2 * share2, axis=0), sem="relaxed")
        if PLAYERS > 3:
            share3 = tl.where(alive & w3, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 6, tl.sum(weight * share3, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 6 + PLAYERS, tl.sum(weight * share3 * share3, axis=0), sem="relaxed")
        if PLAYERS > 4:
            share4 = tl.where(alive & w4, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 7, tl.sum(weight * share4, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 7 + PLAYERS, tl.sum(weight * share4 * share4, axis=0), sem="relaxed")
        if PLAYERS > 5:
            share5 = tl.where(alive & w5, inv_tie, 0.0)
            tl.atomic_add(accum_ptr + accum_base + 8, tl.sum(weight * share5, axis=0), sem="relaxed")
            tl.atomic_add(accum_ptr + accum_base + 8 + PLAYERS, tl.sum(weight * share5 * share5, axis=0), sem="relaxed")

    @triton.jit
    def _finalize_accum_kernel(
        accum_ptr,
        equity_ptr,
        stderr_ptr,
        ess_ptr,
        PLAYERS: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        offs = tl.arange(0, BLOCK_P)
        active = offs < PLAYERS
        sum_w = tl.load(accum_ptr + 0)
        sum_w2 = tl.load(accum_ptr + 1)
        ess = sum_w * sum_w / tl.maximum(sum_w2, 1.0e-30)
        sum_wx = tl.load(accum_ptr + 3 + offs, mask=active, other=0.0)
        sum_wx2 = tl.load(accum_ptr + 3 + PLAYERS + offs, mask=active, other=0.0)
        equity = sum_wx / tl.maximum(sum_w, 1.0e-30)
        second_moment = sum_wx2 / tl.maximum(sum_w, 1.0e-30)
        variance = tl.maximum(second_moment - equity * equity, 0.0)
        stderr = tl.sqrt(variance / tl.maximum(ess, 1.0))
        tl.store(equity_ptr + offs, equity, mask=active)
        tl.store(stderr_ptr + offs, stderr, mask=active)
        tl.store(ess_ptr, ess)

    @triton.jit
    def _finalize_batched_accum_kernel(
        accum_ptr,
        equity_ptr,
        stderr_ptr,
        ess_ptr,
        PLAYERS: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        batch = tl.program_id(0)
        offs = tl.arange(0, BLOCK_P)
        active = offs < PLAYERS
        accum_base = batch * (3 + 2 * PLAYERS)
        out_base = batch * PLAYERS
        sum_w = tl.load(accum_ptr + accum_base + 0)
        sum_w2 = tl.load(accum_ptr + accum_base + 1)
        ess = sum_w * sum_w / tl.maximum(sum_w2, 1.0e-30)
        sum_wx = tl.load(accum_ptr + accum_base + 3 + offs, mask=active, other=0.0)
        sum_wx2 = tl.load(
            accum_ptr + accum_base + 3 + PLAYERS + offs,
            mask=active,
            other=0.0,
        )
        equity = sum_wx / tl.maximum(sum_w, 1.0e-30)
        second_moment = sum_wx2 / tl.maximum(sum_w, 1.0e-30)
        variance = tl.maximum(second_moment - equity * equity, 0.0)
        stderr = tl.sqrt(variance / tl.maximum(ess, 1.0))
        tl.store(equity_ptr + out_base + offs, equity, mask=active)
        tl.store(stderr_ptr + out_base + offs, stderr, mask=active)
        tl.store(ess_ptr + batch, ess)

    @triton.jit
    def _sis6_sample_kernel(
        beliefs_ptr,
        hand_masks_ptr,
        hand_ranks_ptr,
        weights_ptr,
        shares_ptr,
        seed: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        sample_idx = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < H
        hmask = tl.load(hand_masks_ptr + offs, mask=mask, other=0)
        used = tl.full((), 0, tl.int64)
        weight = tl.full((), 1.0, tl.float32)
        rank0 = tl.full((), -1, tl.int32)
        rank1 = tl.full((), -1, tl.int32)
        rank2 = tl.full((), -1, tl.int32)
        rank3 = tl.full((), -1, tl.int32)
        rank4 = tl.full((), -1, tl.int32)
        rank5 = tl.full((), -1, tl.int32)

        for player in tl.static_range(0, 6):
            probs = tl.load(
                beliefs_ptr + player * H + offs,
                mask=mask,
                other=0.0,
            )
            compat = (hmask & used) == 0
            proposal = tl.where(mask & compat, probs, 0.0)
            normalizer = tl.sum(proposal, axis=0)
            weight *= normalizer
            draw = tl.rand(seed, sample_idx * 8 + player) * normalizer
            prefix = tl.cumsum(proposal, axis=0)
            selected = tl.min(tl.where(prefix >= draw, offs, BLOCK_H), axis=0)
            selected_mask = tl.load(hand_masks_ptr + selected)
            used = used | selected_mask
            selected_rank = tl.load(hand_ranks_ptr + selected)
            if player == 0:
                rank0 = selected_rank
            if player == 1:
                rank1 = selected_rank
            if player == 2:
                rank2 = selected_rank
            if player == 3:
                rank3 = selected_rank
            if player == 4:
                rank4 = selected_rank
            if player == 5:
                rank5 = selected_rank

        max01 = tl.maximum(rank0, rank1)
        max23 = tl.maximum(rank2, rank3)
        max45 = tl.maximum(rank4, rank5)
        max_rank = tl.maximum(tl.maximum(max01, max23), max45)
        win0 = rank0 == max_rank
        win1 = rank1 == max_rank
        win2 = rank2 == max_rank
        win3 = rank3 == max_rank
        win4 = rank4 == max_rank
        win5 = rank5 == max_rank
        tie_count = (
            win0.to(tl.float32)
            + win1.to(tl.float32)
            + win2.to(tl.float32)
            + win3.to(tl.float32)
            + win4.to(tl.float32)
            + win5.to(tl.float32)
        )
        inv_tie = 1.0 / tie_count
        tl.store(weights_ptr + sample_idx, weight)
        base = sample_idx * 6
        tl.store(shares_ptr + base + 0, tl.where(win0, inv_tie, 0.0))
        tl.store(shares_ptr + base + 1, tl.where(win1, inv_tie, 0.0))
        tl.store(shares_ptr + base + 2, tl.where(win2, inv_tie, 0.0))
        tl.store(shares_ptr + base + 3, tl.where(win3, inv_tie, 0.0))
        tl.store(shares_ptr + base + 4, tl.where(win4, inv_tie, 0.0))
        tl.store(shares_ptr + base + 5, tl.where(win5, inv_tie, 0.0))


def prepare_triton_sis6(prepared: PreparedShowdown) -> PreparedTritonSIS:
    if triton is None:
        raise RuntimeError("triton is not available")
    if prepared.beliefs.device.type != "cuda":
        raise ValueError("Triton SIS requires CUDA tensors")
    if prepared.beliefs.shape != (1, 6, NUM_HANDS):
        raise ValueError("Triton SIS currently expects one 6-way board")

    start = time.perf_counter()
    beliefs = prepared.beliefs.reshape(6, NUM_HANDS).contiguous()

    torch.cuda.synchronize(beliefs.device)
    return PreparedTritonSIS(
        beliefs=beliefs,
        hand_masks=prepared.hand_masks.contiguous(),
        hand_ranks=prepared.hand_ranks.reshape(NUM_HANDS).contiguous(),
        setup_seconds=time.perf_counter() - start,
    )


def fast_triton_sis_chunk(
    prepared: PreparedFastSISBelief,
    sample_count: int,
    *,
    seed: int,
    block_s: int = 128,
    max_tries: int = 12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    players, active_hands = prepared.beliefs.shape
    if players < 2 or players > 6:
        raise ValueError("fast Triton SIS supports 2 to 6 players")
    sampled = torch.empty(
        sample_count,
        players,
        dtype=torch.int32,
        device=prepared.cdf.device,
    )
    failures = torch.empty(sample_count, dtype=torch.int32, device=prepared.cdf.device)
    grid = (triton.cdiv(sample_count, block_s),)
    _uniform_is_sample_kernel[grid](
        prepared.board.hand_masks,
        sampled,
        failures,
        int(seed),
        sample_count,
        H=active_hands,
        PLAYERS=players,
        BLOCK_S=block_s,
        MAX_TRIES=max_tries,
        num_warps=4,
    )
    sampled_l = sampled.to(torch.long)
    weights = torch.ones(sample_count, dtype=torch.float32, device=prepared.cdf.device)
    active_cards = int((1 + (1 + 8 * active_hands) ** 0.5) / 2)
    for player in range(players):
        remaining = active_cards - 2 * player
        compatible_count = remaining * (remaining - 1) * 0.5
        weights *= prepared.beliefs[player, sampled_l[:, player]] * compatible_count
    weights = torch.where(failures == 0, weights, torch.zeros_like(weights))

    ranks = prepared.board.hand_ranks[sampled_l]
    max_rank = ranks.max(dim=1, keepdim=True).values
    winners = ranks == max_rank
    shares = winners.to(torch.float32) / winners.sum(dim=1, keepdim=True).clamp_min(1)
    return shares, weights, failures


def warmup_fast_triton_sis(
    prepared: PreparedFastSISBelief,
    *,
    sample_count: int,
    seed: int,
    block_s: int = 128,
    max_tries: int = 12,
) -> float:
    start = time.perf_counter()
    fast_triton_sis_chunk(
        prepared,
        sample_count,
        seed=seed,
        block_s=block_s,
        max_tries=max_tries,
    )
    torch.cuda.synchronize(prepared.cdf.device)
    return time.perf_counter() - start


def fast_triton_sis_until_ess(
    prepared: PreparedFastSISBelief,
    *,
    target_ess: int,
    chunk_size: int,
    seed: int,
    block_s: int = 128,
    max_tries: int = 12,
) -> Estimate:
    start = time.perf_counter()
    players = prepared.cdf.shape[0]
    sum_w = torch.zeros((), device=prepared.cdf.device)
    sum_w2 = torch.zeros_like(sum_w)
    sum_wx = torch.zeros(players, device=prepared.cdf.device)
    sum_wx2 = torch.zeros_like(sum_wx)
    failed = torch.zeros((), dtype=torch.int64, device=prepared.cdf.device)
    proposals = 0

    while True:
        shares, weights, failures = fast_triton_sis_chunk(
            prepared,
            chunk_size,
            seed=seed + proposals,
            block_s=block_s,
            max_tries=max_tries,
        )
        sum_w += weights.sum()
        sum_w2 += weights.square().sum()
        sum_wx += (shares * weights[:, None]).sum(dim=0)
        sum_wx2 += (shares.square() * weights[:, None]).sum(dim=0)
        failed += failures.to(torch.int64).sum()
        proposals += chunk_size
        ess = sum_w.square() / sum_w2.clamp_min(1e-30)
        if float(ess.item()) >= target_ess:
            break

    equity = sum_wx / sum_w.clamp_min(1e-30)
    second_moment = sum_wx2 / sum_w.clamp_min(1e-30)
    variance = (second_moment - equity.square()).clamp_min(0.0)
    ess = sum_w.square() / sum_w2.clamp_min(1e-30)
    stderr = (variance / ess.clamp_min(1.0)).sqrt()
    torch.cuda.synchronize(prepared.cdf.device)
    fail_count = int(failed.item())
    if fail_count:
        print(f"fast_triton_sis_failures={fail_count}/{proposals}")
    return Estimate(
        equity=equity[None, :],
        stderr=stderr[None, :],
        ess=float(ess.item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def _cdf_sis_weights_and_shares(
    prepared: PreparedFastSISBelief,
    sampled: torch.Tensor,
    failures: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    players = sampled.shape[1]
    cards0 = prepared.board.hand_cards0[sampled].to(torch.long)
    cards1 = prepared.board.hand_cards1[sampled].to(torch.long)
    weights = torch.ones(sampled.shape[0], dtype=torch.float32, device=sampled.device)
    for player in range(1, players):
        used_cards = torch.stack(
            [
                card
                for prev in range(player)
                for card in (cards0[:, prev], cards1[:, prev])
            ],
            dim=1,
        )
        blocked = prepared.card_mass[player, used_cards].sum(dim=1)
        pair_correction = torch.zeros_like(blocked)
        used_count = used_cards.shape[1]
        for left in range(used_count):
            for right in range(left + 1, used_count):
                pair_correction += prepared.pair_mass[
                    player,
                    used_cards[:, left],
                    used_cards[:, right],
                ]
        weights *= (1.0 - blocked + pair_correction).clamp_min(1e-12)

    weights = torch.where(failures == 0, weights, torch.zeros_like(weights))
    ranks = prepared.board.hand_ranks[sampled]
    max_rank = ranks.max(dim=1, keepdim=True).values
    winners = ranks == max_rank
    shares = winners.to(torch.float32) / winners.sum(dim=1, keepdim=True).clamp_min(1)
    return shares, weights


def _cdf_sis_weights_and_shares_triton(
    prepared: PreparedFastSISBelief,
    sampled: torch.Tensor,
    failures: torch.Tensor,
    *,
    block_s: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    sample_count, players = sampled.shape
    weights = torch.empty(sample_count, dtype=torch.float32, device=sampled.device)
    shares = torch.empty(sample_count, players, dtype=torch.float32, device=sampled.device)
    _cdf_sis_weights_and_shares_triton_into(
        prepared,
        sampled,
        failures,
        shares,
        weights,
        block_s=block_s,
    )
    return shares, weights


def _cdf_sis_weights_and_shares_triton_into(
    prepared: PreparedFastSISBelief,
    sampled: torch.Tensor,
    failures: torch.Tensor,
    shares: torch.Tensor,
    weights: torch.Tensor,
    *,
    block_s: int,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    sample_count, players = sampled.shape
    grid = (triton.cdiv(sample_count, block_s),)
    _sample_weights_shares_kernel[grid](
        sampled,
        failures,
        prepared.card_mass,
        prepared.pair_mass,
        prepared.board.hand_cards0,
        prepared.board.hand_cards1,
        prepared.board.hand_ranks,
        weights,
        shares,
        sample_count,
        PLAYERS=players,
        BLOCK_S=block_s,
        MAX_USED=12,
        num_warps=4,
    )


def stage_triton_cdf_sis_chunk(
    prepared: PreparedFastSISBelief,
    sample_count: int,
    *,
    seed: int,
    block_s: int = 128,
    max_tries: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    players, active_hands = prepared.beliefs.shape
    sampled = torch.empty(
        sample_count,
        players,
        dtype=torch.int32,
        device=prepared.beliefs.device,
    )
    failures = torch.zeros(sample_count, dtype=torch.int32, device=prepared.beliefs.device)
    used = torch.zeros(sample_count, dtype=torch.int64, device=prepared.beliefs.device)
    grid = (triton.cdiv(sample_count, block_s),)
    for player in range(players):
        _stage_cdf_sample_kernel[grid](
            prepared.cdf,
            prepared.board.hand_masks,
            used,
            sampled,
            failures,
            int(seed + player * 1_000_003),
            sample_count,
            H=active_hands,
            PLAYER=player,
            PLAYERS=players,
            BLOCK_S=block_s,
            MAX_TRIES=max_tries,
            SEARCH_STEPS=(active_hands - 1).bit_length(),
            num_warps=4,
        )
    shares, weights = _cdf_sis_weights_and_shares(
        prepared,
        sampled.to(torch.long),
        failures,
    )
    return shares, weights, failures


def stage_triton_cdf_sis_until_ess(
    prepared: PreparedFastSISBelief,
    *,
    target_ess: int,
    chunk_size: int,
    seed: int,
    block_s: int = 128,
    max_tries: int = 8,
) -> Estimate:
    start = time.perf_counter()
    players = prepared.beliefs.shape[0]
    sum_w = torch.zeros((), device=prepared.beliefs.device)
    sum_w2 = torch.zeros_like(sum_w)
    sum_wx = torch.zeros(players, device=prepared.beliefs.device)
    sum_wx2 = torch.zeros_like(sum_wx)
    failed = torch.zeros((), dtype=torch.int64, device=prepared.beliefs.device)
    proposals = 0
    while True:
        shares, weights, failures = stage_triton_cdf_sis_chunk(
            prepared,
            chunk_size,
            seed=seed + proposals,
            block_s=block_s,
            max_tries=max_tries,
        )
        sum_w += weights.sum()
        sum_w2 += weights.square().sum()
        sum_wx += (shares * weights[:, None]).sum(dim=0)
        sum_wx2 += (shares.square() * weights[:, None]).sum(dim=0)
        failed += failures.to(torch.int64).sum()
        proposals += chunk_size
        ess = sum_w.square() / sum_w2.clamp_min(1e-30)
        if float(ess.item()) >= target_ess:
            break

    equity = sum_wx / sum_w.clamp_min(1e-30)
    second_moment = sum_wx2 / sum_w.clamp_min(1e-30)
    variance = (second_moment - equity.square()).clamp_min(0.0)
    ess = sum_w.square() / sum_w2.clamp_min(1e-30)
    stderr = (variance / ess.clamp_min(1.0)).sqrt()
    torch.cuda.synchronize(prepared.beliefs.device)
    fail_count = int(failed.item())
    if fail_count:
        print(f"stage_triton_cdf_sis_failures={fail_count}/{proposals}")
    return Estimate(
        equity=equity[None, :],
        stderr=stderr[None, :],
        ess=float(ess.item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def torch_multinomial_reject_sis_chunk(
    prepared: PreparedFastSISBelief,
    sample_count: int,
    *,
    max_tries: int = 8,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    players, _ = prepared.beliefs.shape
    device = prepared.beliefs.device
    sampled = torch.zeros(sample_count, players, dtype=torch.long, device=device)
    failures = torch.zeros(sample_count, dtype=torch.int32, device=device)
    used = torch.zeros(sample_count, dtype=torch.int64, device=device)
    row_ids = torch.arange(sample_count, device=device)
    for player in range(players):
        candidates = torch.multinomial(
            prepared.beliefs[player],
            sample_count * max_tries,
            replacement=True,
            generator=generator,
        ).view(sample_count, max_tries)
        candidate_masks = prepared.board.hand_masks[candidates]
        compatible = (candidate_masks & used[:, None]) == 0
        found = compatible.any(dim=1)
        selected_pos = compatible.to(torch.int64).argmax(dim=1)
        selected = candidates[row_ids, selected_pos]
        failures += (~found).to(torch.int32)
        selected = torch.where(found, selected, torch.zeros_like(selected))
        sampled[:, player] = selected
        used |= torch.where(
            found,
            prepared.board.hand_masks[selected],
            torch.zeros_like(used),
        )
    shares, weights = _cdf_sis_weights_and_shares(prepared, sampled, failures)
    return shares, weights, failures


def torch_multinomial_reject_sis_until_ess(
    prepared: PreparedFastSISBelief,
    *,
    target_ess: int,
    chunk_size: int,
    max_tries: int = 8,
    generator: torch.Generator,
) -> Estimate:
    start = time.perf_counter()
    players = prepared.beliefs.shape[0]
    sum_w = torch.zeros((), device=prepared.beliefs.device)
    sum_w2 = torch.zeros_like(sum_w)
    sum_wx = torch.zeros(players, device=prepared.beliefs.device)
    sum_wx2 = torch.zeros_like(sum_wx)
    failed = torch.zeros((), dtype=torch.int64, device=prepared.beliefs.device)
    proposals = 0
    while True:
        shares, weights, failures = torch_multinomial_reject_sis_chunk(
            prepared,
            chunk_size,
            max_tries=max_tries,
            generator=generator,
        )
        sum_w += weights.sum()
        sum_w2 += weights.square().sum()
        sum_wx += (shares * weights[:, None]).sum(dim=0)
        sum_wx2 += (shares.square() * weights[:, None]).sum(dim=0)
        failed += failures.to(torch.int64).sum()
        proposals += chunk_size
        ess = sum_w.square() / sum_w2.clamp_min(1e-30)
        if float(ess.item()) >= target_ess:
            break

    equity = sum_wx / sum_w.clamp_min(1e-30)
    second_moment = sum_wx2 / sum_w.clamp_min(1e-30)
    variance = (second_moment - equity.square()).clamp_min(0.0)
    ess = sum_w.square() / sum_w2.clamp_min(1e-30)
    stderr = (variance / ess.clamp_min(1.0)).sqrt()
    torch.cuda.synchronize(prepared.beliefs.device)
    fail_count = int(failed.item())
    if fail_count:
        print(f"torch_multinomial_reject_sis_failures={fail_count}/{proposals}")
    return Estimate(
        equity=equity[None, :],
        stderr=stderr[None, :],
        ess=float(ess.item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def torch_multinomial_triton_reject_sis_chunk(
    prepared: PreparedFastSISBelief,
    sample_count: int,
    *,
    max_tries: int = 8,
    block_s: int = 256,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    players, _ = prepared.beliefs.shape
    device = prepared.beliefs.device
    sampled = torch.empty(sample_count, players, dtype=torch.long, device=device)
    failures = torch.zeros(sample_count, dtype=torch.int32, device=device)
    used = torch.zeros(sample_count, dtype=torch.int64, device=device)
    grid = (triton.cdiv(sample_count, block_s),)
    for player in range(players):
        candidates = torch.multinomial(
            prepared.beliefs[player],
            sample_count * max_tries,
            replacement=True,
            generator=generator,
        ).contiguous()
        _select_first_compatible_kernel[grid](
            candidates,
            prepared.board.hand_masks,
            used,
            sampled,
            failures,
            sample_count,
            PLAYER=player,
            PLAYERS=players,
            TRIES=max_tries,
            BLOCK_S=block_s,
            num_warps=4,
        )
    shares, weights = _cdf_sis_weights_and_shares_triton(
        prepared,
        sampled,
        failures,
        block_s=block_s,
    )
    return shares, weights, failures


def make_triton_reject_sis_workspace(
    prepared: PreparedFastSISBelief,
    *,
    sample_count: int,
    max_tries: int = 8,
) -> TritonRejectSISWorkspace:
    players, _ = prepared.beliefs.shape
    device = prepared.beliefs.device
    return TritonRejectSISWorkspace(
        sampled=torch.empty(sample_count, players, dtype=torch.long, device=device),
        failures=torch.empty(sample_count, dtype=torch.int32, device=device),
        used=torch.empty(sample_count, dtype=torch.int64, device=device),
        candidates=torch.empty(
            players,
            sample_count * max_tries,
            dtype=torch.long,
            device=device,
        ),
        weights=torch.empty(sample_count, dtype=torch.float32, device=device),
        shares=torch.empty(sample_count, players, dtype=torch.float32, device=device),
        accum=torch.empty(3 + 2 * players, dtype=torch.float32, device=device),
        equity=torch.empty(players, dtype=torch.float32, device=device),
        stderr=torch.empty(players, dtype=torch.float32, device=device),
        ess=torch.empty((), dtype=torch.float32, device=device),
        sample_count=sample_count,
        max_tries=max_tries,
    )


def torch_multinomial_triton_reject_sis_chunk_into(
    prepared: PreparedFastSISBelief,
    workspace: TritonRejectSISWorkspace,
    *,
    block_s: int = 256,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    players, _ = prepared.beliefs.shape
    if workspace.sampled.shape != (workspace.sample_count, players):
        raise ValueError("workspace shape does not match prepared belief")
    sample_count = workspace.sample_count
    max_tries = workspace.max_tries
    workspace.failures.zero_()
    workspace.used.zero_()
    grid = (triton.cdiv(sample_count, block_s),)
    torch.multinomial(
        prepared.beliefs,
        sample_count * max_tries,
        replacement=True,
        generator=generator,
        out=workspace.candidates,
    )
    _select_all_compatible_kernel[grid](
        workspace.candidates,
        prepared.board.hand_masks,
        workspace.used,
        workspace.sampled,
        workspace.failures,
        sample_count,
        PLAYERS=players,
        TRIES=max_tries,
        BLOCK_S=block_s,
        num_warps=4,
    )
    _cdf_sis_weights_and_shares_triton_into(
        prepared,
        workspace.sampled,
        workspace.failures,
        workspace.shares,
        workspace.weights,
        block_s=block_s,
    )
    return workspace.shares, workspace.weights, workspace.failures


def torch_multinomial_triton_reject_sis_accumulate_into(
    prepared: PreparedFastSISBelief,
    workspace: TritonRejectSISWorkspace,
    *,
    block_s: int = 256,
    generator: torch.Generator,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    players, _ = prepared.beliefs.shape
    if workspace.sampled.shape != (workspace.sample_count, players):
        raise ValueError("workspace shape does not match prepared belief")
    sample_count = workspace.sample_count
    max_tries = workspace.max_tries
    grid = (triton.cdiv(sample_count, block_s),)
    torch.multinomial(
        prepared.beliefs,
        sample_count * max_tries,
        replacement=True,
        generator=generator,
        out=workspace.candidates,
    )
    _select_accumulate_kernel[grid](
        workspace.candidates,
        prepared.beliefs,
        prepared.card_mass,
        prepared.board.pair_to_active,
        prepared.board.hand_masks,
        prepared.board.hand_cards0,
        prepared.board.hand_cards1,
        prepared.board.hand_ranks,
        workspace.accum,
        sample_count,
        active_hands=prepared.beliefs.shape[1],
        PLAYERS=players,
        TRIES=max_tries,
        BLOCK_S=block_s,
        MAX_USED=12,
        num_warps=4,
    )


def torch_multinomial_triton_reject_sis_until_ess(
    prepared: PreparedFastSISBelief,
    *,
    target_ess: int,
    chunk_size: int,
    max_tries: int = 8,
    block_s: int = 256,
    fixed_chunk: bool = False,
    generator: torch.Generator,
    workspace: TritonRejectSISWorkspace | None = None,
) -> Estimate:
    if workspace is None:
        workspace = make_triton_reject_sis_workspace(
            prepared,
            sample_count=chunk_size,
            max_tries=max_tries,
        )
    elif workspace.sample_count != chunk_size or workspace.max_tries != max_tries:
        raise ValueError("workspace shape does not match chunk_size/max_tries")
    start = time.perf_counter()
    players = prepared.beliefs.shape[0]
    workspace.accum.zero_()
    proposals = 0
    while True:
        torch_multinomial_triton_reject_sis_accumulate_into(
            prepared,
            workspace,
            block_s=block_s,
            generator=generator,
        )
        proposals += chunk_size
        if fixed_chunk:
            break
        ess = workspace.accum[0].square() / workspace.accum[1].clamp_min(1e-30)
        if float(ess.item()) >= target_ess:
            break

    if fixed_chunk:
        _finalize_accum_kernel[(1,)](
            workspace.accum,
            workspace.equity,
            workspace.stderr,
            workspace.ess,
            PLAYERS=players,
            BLOCK_P=triton.next_power_of_2(players),
            num_warps=1,
        )
        equity = workspace.equity
        stderr = workspace.stderr
        ess = workspace.ess
    else:
        sum_w = workspace.accum[0]
        sum_w2 = workspace.accum[1]
        sum_wx = workspace.accum[3 : 3 + players]
        sum_wx2 = workspace.accum[3 + players : 3 + 2 * players]
        equity = sum_wx / sum_w.clamp_min(1e-30)
        second_moment = sum_wx2 / sum_w.clamp_min(1e-30)
        variance = (second_moment - equity.square()).clamp_min(0.0)
        ess = sum_w.square() / sum_w2.clamp_min(1e-30)
        stderr = (variance / ess.clamp_min(1.0)).sqrt()
    torch.cuda.synchronize(prepared.beliefs.device)
    return Estimate(
        equity=equity[None, :],
        stderr=stderr[None, :],
        ess=float(ess.item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def make_batched_triton_reject_sis_workspace(
    prepared: PreparedBatchedFastSISBelief,
    *,
    sample_count: int,
    max_tries: int = 4,
) -> BatchedTritonRejectSISWorkspace:
    device = prepared.beliefs.device
    rows = prepared.batch_size * prepared.players
    return BatchedTritonRejectSISWorkspace(
        candidates=torch.empty(
            rows,
            sample_count * max_tries,
            dtype=torch.long,
            device=device,
        ),
        accum=torch.empty(
            prepared.batch_size,
            3 + 2 * prepared.players,
            dtype=torch.float32,
            device=device,
        ),
        equity=torch.empty(
            prepared.batch_size,
            prepared.players,
            dtype=torch.float32,
            device=device,
        ),
        stderr=torch.empty(
            prepared.batch_size,
            prepared.players,
            dtype=torch.float32,
            device=device,
        ),
        ess=torch.empty(prepared.batch_size, dtype=torch.float32, device=device),
        sample_count=sample_count,
        max_tries=max_tries,
    )


def make_batched_alias_reject_sis_workspace(
    prepared: PreparedBatchedFastSISBelief,
    *,
    sample_count: int,
    max_tries: int = 4,
) -> BatchedAliasRejectSISWorkspace:
    start = time.perf_counter()
    device = prepared.beliefs.device
    rows, active_hands = prepared.beliefs.shape
    return BatchedAliasRejectSISWorkspace(
        alias_prob=torch.empty(rows, active_hands, dtype=torch.float32, device=device),
        alias_idx=torch.empty(rows, active_hands, dtype=torch.int32, device=device),
        candidates=torch.empty(
            rows,
            sample_count * max_tries,
            dtype=torch.int32,
            device=device,
        ),
        seed=torch.empty((), dtype=torch.int64, device=device),
        small_stack=torch.empty(rows, active_hands, dtype=torch.int32, device=device),
        large_stack=torch.empty(rows, active_hands, dtype=torch.int32, device=device),
        counts=torch.empty(rows, 2, dtype=torch.int32, device=device),
        accum=torch.empty(
            prepared.batch_size,
            3 + 2 * prepared.players,
            dtype=torch.float32,
            device=device,
        ),
        equity=torch.empty(
            prepared.batch_size,
            prepared.players,
            dtype=torch.float32,
            device=device,
        ),
        stderr=torch.empty(
            prepared.batch_size,
            prepared.players,
            dtype=torch.float32,
            device=device,
        ),
        ess=torch.empty(prepared.batch_size, dtype=torch.float32, device=device),
        sample_count=sample_count,
        max_tries=max_tries,
        setup_seconds=time.perf_counter() - start,
    )


def build_batched_alias_tables_triton_into(
    prepared: PreparedBatchedFastSISBelief,
    workspace: BatchedAliasRejectSISWorkspace,
    *,
    synchronize: bool = False,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if workspace.alias_prob.shape != prepared.beliefs.shape:
        raise ValueError("alias workspace shape does not match prepared batch")
    active_hands = prepared.beliefs.shape[1]
    block_h = triton.next_power_of_2(active_hands)
    _alias_partition_kernel[(prepared.beliefs.shape[0],)](
        prepared.beliefs,
        workspace.alias_prob,
        workspace.alias_idx,
        workspace.small_stack,
        workspace.large_stack,
        workspace.counts,
        active_hands=active_hands,
        BLOCK_H=block_h,
        num_warps=8,
    )
    _alias_resolve_stacks_kernel[(prepared.beliefs.shape[0],)](
        workspace.alias_prob,
        workspace.alias_idx,
        workspace.small_stack,
        workspace.large_stack,
        workspace.counts,
        active_hands=active_hands,
    )
    if synchronize:
        torch.cuda.synchronize(prepared.beliefs.device)


def torch_multinomial_triton_reject_sis_batched_fixed_into(
    prepared: PreparedBatchedFastSISBelief,
    workspace: BatchedTritonRejectSISWorkspace,
    *,
    block_s: int = 256,
    generator: torch.Generator,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if workspace.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rows = prepared.batch_size * prepared.players
    if workspace.candidates.shape[0] != rows:
        raise ValueError("workspace shape does not match prepared batch")
    sample_count = workspace.sample_count
    max_tries = workspace.max_tries
    torch.multinomial(
        prepared.beliefs,
        sample_count * max_tries,
        replacement=True,
        generator=generator,
        out=workspace.candidates,
    )
    triton_reject_sis_batched_candidates_into(
        prepared,
        workspace,
        block_s=block_s,
    )


def triton_reject_sis_batched_candidates_into(
    prepared: PreparedBatchedFastSISBelief,
    workspace: BatchedTritonRejectSISWorkspace,
    *,
    block_s: int = 256,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if workspace.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rows = prepared.batch_size * prepared.players
    if workspace.candidates.shape[0] != rows:
        raise ValueError("workspace shape does not match prepared batch")
    sample_count = workspace.sample_count
    max_tries = workspace.max_tries
    workspace.accum.zero_()
    grid = (prepared.batch_size, triton.cdiv(sample_count, block_s))
    _select_accumulate_batched_kernel[grid](
        workspace.candidates,
        prepared.beliefs,
        prepared.card_mass,
        prepared.board.pair_to_active,
        prepared.board.hand_masks,
        prepared.board.hand_cards0,
        prepared.board.hand_cards1,
        prepared.board.hand_ranks,
        workspace.accum,
        sample_count,
        active_hands=prepared.beliefs.shape[1],
        PLAYERS=prepared.players,
        TRIES=max_tries,
        BLOCK_S=block_s,
        MAX_USED=12,
        num_warps=4,
    )
    _finalize_batched_accum_kernel[(prepared.batch_size,)](
        workspace.accum,
        workspace.equity,
        workspace.stderr,
        workspace.ess,
        PLAYERS=prepared.players,
        BLOCK_P=triton.next_power_of_2(prepared.players),
        num_warps=1,
    )


def alias_triton_reject_sis_batched_fixed_into(
    prepared: PreparedBatchedFastSISBelief,
    workspace: BatchedAliasRejectSISWorkspace,
    *,
    block_s: int = 256,
    seed: int,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if workspace.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if workspace.alias_prob.shape != prepared.beliefs.shape:
        raise ValueError("alias workspace shape does not match prepared batch")
    sample_count = workspace.sample_count
    max_tries = workspace.max_tries
    workspace.seed.fill_(seed)
    workspace.accum.zero_()
    grid = (prepared.batch_size, triton.cdiv(sample_count, block_s))
    _alias_accumulate_batched_kernel[grid](
        workspace.alias_prob,
        workspace.alias_idx,
        prepared.beliefs,
        prepared.card_mass,
        prepared.pair_mass,
        prepared.board.hand_masks,
        prepared.board.hand_cards0,
        prepared.board.hand_cards1,
        prepared.board.hand_ranks,
        workspace.accum,
        workspace.seed,
        sample_count,
        active_hands=prepared.beliefs.shape[1],
        PLAYERS=prepared.players,
        TRIES=max_tries,
        BLOCK_S=block_s,
        MAX_USED=12,
        num_warps=4,
    )
    _finalize_batched_accum_kernel[(prepared.batch_size,)](
        workspace.accum,
        workspace.equity,
        workspace.stderr,
        workspace.ess,
        PLAYERS=prepared.players,
        BLOCK_P=triton.next_power_of_2(prepared.players),
        num_warps=1,
    )


def torch_cdf_sis_chunk(
    prepared: PreparedFastSISBelief,
    sample_count: int,
    *,
    max_tries: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    players, _ = prepared.beliefs.shape
    device = prepared.beliefs.device
    sampled = torch.zeros(sample_count, players, dtype=torch.long, device=device)
    failures = torch.zeros(sample_count, dtype=torch.int32, device=device)
    weights = torch.ones(sample_count, dtype=torch.float32, device=device)
    used = torch.zeros(sample_count, dtype=torch.int64, device=device)
    sampled_cards0 = []
    sampled_cards1 = []
    for player in range(players):
        if player > 0:
            used_cards = torch.stack(
                [
                    card
                    for prev in range(player)
                    for card in (sampled_cards0[prev], sampled_cards1[prev])
                ],
                dim=1,
            )
            blocked = prepared.card_mass[player, used_cards].sum(dim=1)
            pair_correction = torch.zeros_like(blocked)
            used_count = used_cards.shape[1]
            for left in range(used_count):
                for right in range(left + 1, used_count):
                    pair_correction += prepared.pair_mass[
                        player,
                        used_cards[:, left],
                        used_cards[:, right],
                    ]
            weights *= (1.0 - blocked + pair_correction).clamp_min(1e-12)

        found = torch.zeros(sample_count, dtype=torch.bool, device=device)
        selected = torch.zeros(sample_count, dtype=torch.long, device=device)
        for _ in range(max_tries):
            draws = torch.rand(sample_count, dtype=torch.float32, device=device)
            candidate = torch.searchsorted(prepared.cdf[player], draws).clamp_max(
                prepared.cdf.shape[1] - 1
            )
            candidate_mask = prepared.board.hand_masks[candidate]
            compatible = (candidate_mask & used) == 0
            take = compatible & ~found
            selected = torch.where(take, candidate, selected)
            found |= compatible
        failures += (~found).to(torch.int32)
        selected_mask = prepared.board.hand_masks[selected]
        used |= selected_mask
        sampled[:, player] = selected
        sampled_cards0.append(prepared.board.hand_cards0[selected].to(torch.long))
        sampled_cards1.append(prepared.board.hand_cards1[selected].to(torch.long))

    weights = torch.where(failures == 0, weights, torch.zeros_like(weights))
    ranks = prepared.board.hand_ranks[sampled]
    max_rank = ranks.max(dim=1, keepdim=True).values
    winners = ranks == max_rank
    shares = winners.to(torch.float32) / winners.sum(dim=1, keepdim=True).clamp_min(1)
    return shares, weights, failures


def torch_cdf_sis_until_ess(
    prepared: PreparedFastSISBelief,
    *,
    target_ess: int,
    chunk_size: int,
    max_tries: int = 8,
) -> Estimate:
    start = time.perf_counter()
    players = prepared.beliefs.shape[0]
    sum_w = torch.zeros((), device=prepared.beliefs.device)
    sum_w2 = torch.zeros_like(sum_w)
    sum_wx = torch.zeros(players, device=prepared.beliefs.device)
    sum_wx2 = torch.zeros_like(sum_wx)
    failed = torch.zeros((), dtype=torch.int64, device=prepared.beliefs.device)
    proposals = 0
    while True:
        shares, weights, failures = torch_cdf_sis_chunk(
            prepared,
            chunk_size,
            max_tries=max_tries,
        )
        sum_w += weights.sum()
        sum_w2 += weights.square().sum()
        sum_wx += (shares * weights[:, None]).sum(dim=0)
        sum_wx2 += (shares.square() * weights[:, None]).sum(dim=0)
        failed += failures.to(torch.int64).sum()
        proposals += chunk_size
        ess = sum_w.square() / sum_w2.clamp_min(1e-30)
        if float(ess.item()) >= target_ess:
            break

    equity = sum_wx / sum_w.clamp_min(1e-30)
    second_moment = sum_wx2 / sum_w.clamp_min(1e-30)
    variance = (second_moment - equity.square()).clamp_min(0.0)
    ess = sum_w.square() / sum_w2.clamp_min(1e-30)
    stderr = (variance / ess.clamp_min(1.0)).sqrt()
    torch.cuda.synchronize(prepared.beliefs.device)
    fail_count = int(failed.item())
    if fail_count:
        print(f"torch_cdf_sis_failures={fail_count}/{proposals}")
    return Estimate(
        equity=equity[None, :],
        stderr=stderr[None, :],
        ess=float(ess.item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def triton_sis6_chunk(
    prepared: PreparedTritonSIS,
    sample_count: int,
    *,
    seed: int,
    block_h: int = 2048,
    block_s: int = 8,
    max_tries: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.empty(sample_count, dtype=torch.float32, device=prepared.beliefs.device)
    shares = torch.empty(sample_count, 6, dtype=torch.float32, device=prepared.beliefs.device)
    del block_s, max_tries
    grid = (sample_count,)
    _sis6_sample_kernel[grid](
        prepared.beliefs,
        prepared.hand_masks,
        prepared.hand_ranks,
        weights,
        shares,
        int(seed),
        H=NUM_HANDS,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return shares, weights


def warmup_triton_sis6(
    prepared: PreparedTritonSIS,
    *,
    sample_count: int,
    seed: int,
    block_h: int = 2048,
    block_s: int = 8,
    max_tries: int = 8,
) -> float:
    start = time.perf_counter()
    triton_sis6_chunk(
        prepared,
        sample_count,
        seed=seed,
        block_h=block_h,
        block_s=block_s,
        max_tries=max_tries,
    )
    torch.cuda.synchronize(prepared.beliefs.device)
    return time.perf_counter() - start


def triton_sis6_until_ess(
    prepared: PreparedTritonSIS,
    *,
    target_ess: int,
    chunk_size: int,
    seed: int,
    block_h: int = 2048,
    block_s: int = 8,
    max_tries: int = 8,
) -> Estimate:
    start = time.perf_counter()
    sum_w = torch.zeros((), device=prepared.beliefs.device)
    sum_w2 = torch.zeros_like(sum_w)
    sum_wx = torch.zeros(6, device=prepared.beliefs.device)
    sum_wx2 = torch.zeros_like(sum_wx)
    proposals = 0

    while True:
        shares, weights = triton_sis6_chunk(
            prepared,
            chunk_size,
            seed=seed + proposals,
            block_h=block_h,
            block_s=block_s,
            max_tries=max_tries,
        )
        sum_w += weights.sum()
        sum_w2 += weights.square().sum()
        sum_wx += (shares * weights[:, None]).sum(dim=0)
        sum_wx2 += (shares.square() * weights[:, None]).sum(dim=0)
        proposals += chunk_size
        ess = sum_w.square() / sum_w2.clamp_min(1e-30)
        if float(ess.item()) >= target_ess:
            break

    equity = sum_wx / sum_w.clamp_min(1e-30)
    second_moment = sum_wx2 / sum_w.clamp_min(1e-30)
    variance = (second_moment - equity.square()).clamp_min(0.0)
    ess = sum_w.square() / sum_w2.clamp_min(1e-30)
    stderr = (variance / ess.clamp_min(1.0)).sqrt()
    torch.cuda.synchronize(prepared.beliefs.device)
    return Estimate(
        equity=equity[None, :],
        stderr=stderr[None, :],
        ess=float(ess.item()),
        proposals=proposals,
        seconds=time.perf_counter() - start,
    )


def _format_vector(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def run_batched_belief_benchmark(
    args: argparse.Namespace,
    prepared: PreparedShowdown,
    generator: torch.Generator,
) -> None:
    if triton is None:
        raise RuntimeError("triton is not available")
    if prepared.beliefs.device.type != "cuda":
        raise ValueError("batched belief benchmark requires CUDA")

    device = prepared.beliefs.device
    batch_size = args.batch_belief_size
    iterations = args.batch_belief_iterations
    sample_count = args.batch_belief_samples
    max_tries = args.batch_belief_max_tries
    block_s = args.batch_belief_block_s

    fast_board = prepare_fast_sis_board(prepared)
    board_batch = prepared.board.expand(batch_size, -1).contiguous()
    full_beliefs = random_beliefs(board_batch, args.players, generator=generator)
    belief_workspace = make_batched_fast_sis_belief_workspace(
        fast_board,
        batch_size=batch_size,
        players=args.players,
    )
    prepare_batched_fast_sis_belief_no_pair_triton_into(
        belief_workspace,
        full_beliefs,
        normalize=False,
    )
    alias_setup_seconds = 0.0
    if args.batch_belief_sampler == "alias":
        reject_workspace = make_batched_alias_reject_sis_workspace(
            belief_workspace,
            sample_count=sample_count,
            max_tries=max_tries,
        )
        if args.batch_belief_reuse_alias:
            build_batched_alias_tables_triton_into(
                belief_workspace,
                reject_workspace,
                synchronize=False,
            )
        alias_setup_seconds = reject_workspace.setup_seconds
    else:
        reject_workspace = make_batched_triton_reject_sis_workspace(
            belief_workspace,
            sample_count=sample_count,
            max_tries=max_tries,
        )

    warmup_start = time.perf_counter()
    for _ in range(args.batch_belief_warmup_iterations):
        if args.batch_belief_include_prep:
            prepare_batched_fast_sis_belief_no_pair_triton_into(
                belief_workspace,
                full_beliefs,
                normalize=False,
                synchronize=False,
            )
        if args.batch_belief_sampler == "multinomial":
            torch_multinomial_triton_reject_sis_batched_fixed_into(
                belief_workspace,
                reject_workspace,
                block_s=block_s,
                generator=generator,
            )
        else:
            if not args.batch_belief_reuse_alias:
                build_batched_alias_tables_triton_into(
                    belief_workspace,
                    reject_workspace,
                    synchronize=False,
                )
            alias_triton_reject_sis_batched_fixed_into(
                belief_workspace,
                reject_workspace,
                block_s=block_s,
                seed=args.seed,
            )
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(iterations):
        if args.batch_belief_include_prep:
            prepare_batched_fast_sis_belief_no_pair_triton_into(
                belief_workspace,
                full_beliefs,
                normalize=False,
                synchronize=False,
            )
        if args.batch_belief_sampler == "multinomial":
            torch_multinomial_triton_reject_sis_batched_fixed_into(
                belief_workspace,
                reject_workspace,
                block_s=block_s,
                generator=generator,
            )
        else:
            if not args.batch_belief_reuse_alias:
                build_batched_alias_tables_triton_into(
                    belief_workspace,
                    reject_workspace,
                    synchronize=False,
                )
            alias_triton_reject_sis_batched_fixed_into(
                belief_workspace,
                reject_workspace,
                block_s=block_s,
                seed=args.seed + 100_000 + _,
            )
    end_event.record()
    end_event.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    device_seconds = start_event.elapsed_time(end_event) / 1000.0

    belief_updates = batch_size * iterations
    samples = belief_updates * sample_count
    proposals = samples * max_tries
    mean_ess = float(reject_workspace.ess.mean().item())
    min_ess = float(reject_workspace.ess.min().item())
    print(
        "batch_belief_benchmark "
        f"players={args.players} batch={batch_size} iterations={iterations} "
        f"samples_per_belief={sample_count} max_tries={max_tries} "
        f"sampler={args.batch_belief_sampler} "
        f"include_prep={args.batch_belief_include_prep} "
        f"board_setup={fast_board.setup_seconds:.6f} "
        f"alias_workspace_setup={alias_setup_seconds:.6f} "
        f"alias_build_in_loop={args.batch_belief_sampler == 'alias' and not args.batch_belief_reuse_alias} "
        f"warmup={warmup_seconds:.6f} "
        f"device_seconds={device_seconds:.6f} wall_seconds={wall_seconds:.6f} "
        f"ms_per_batch={device_seconds / max(iterations, 1) * 1_000.0:.6f} "
        f"us_per_belief={device_seconds / max(belief_updates, 1) * 1_000_000.0:.6f} "
        f"belief_updates_per_second={belief_updates / max(device_seconds, 1e-12):.3f} "
        f"samples_per_second={samples / max(device_seconds, 1e-12):.3f} "
        f"proposals_per_second={proposals / max(device_seconds, 1e-12):.3f} "
        f"mean_ess={mean_ess:.1f} min_ess={min_ess:.1f}"
    )


def run_validation(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prepared = prepare_showdown(
        batch_size=1,
        num_players=args.players,
        device=device,
        generator=generator,
    )
    if args.batch_belief_benchmark:
        run_batched_belief_benchmark(args, prepared, generator)
        return

    naive = naive_rejection_until_ess(
        prepared,
        target_ess=args.ess,
        chunk_size=args.naive_chunk,
        generator=generator,
    )
    sis = sis_until_ess(
        prepared,
        target_ess=args.ess,
        chunk_size=args.sis_chunk,
        generator=generator,
    )
    triton_sis = None
    triton_setup_seconds = None
    fast_sis = None
    fast_sis_setup_seconds = None
    torch_cdf_sis = None
    torch_cdf_sis_setup_seconds = None
    stage_triton_cdf_sis = None
    stage_triton_cdf_sis_setup_seconds = None
    torch_mr_sis = None
    torch_mr_sis_setup_seconds = None
    torch_mr_triton_sis = None
    torch_mr_triton_sis_setup_seconds = None
    torch_mr_triton_sis_board_seconds = None
    torch_mr_triton_sis_belief_seconds = None
    torch_mr_triton_sis_warmup_seconds = None
    if args.triton:
        if args.players != 6:
            raise ValueError("--triton currently expects --players 6")
        triton_prepared = prepare_triton_sis6(prepared)
        triton_setup_seconds = warmup_triton_sis6(
            triton_prepared,
            sample_count=args.triton_chunk,
            seed=args.seed + 99_999,
            block_h=args.triton_block_h,
            block_s=args.triton_block_s,
            max_tries=args.triton_max_tries,
        ) + triton_prepared.setup_seconds
        triton_sis = triton_sis6_until_ess(
            triton_prepared,
            target_ess=args.ess,
            chunk_size=args.triton_chunk,
            seed=args.seed + 100_000,
            block_h=args.triton_block_h,
            block_s=args.triton_block_s,
            max_tries=args.triton_max_tries,
        )
    if args.fast_triton_sis:
        if triton is None:
            raise RuntimeError("triton is not available")
        fast_board = prepare_fast_sis_board(prepared)
        fast_belief_workspace = make_fast_sis_belief_workspace(
            fast_board,
            players=args.players,
            build_cdf=False,
            build_pair_mass=False,
        )
        fast_belief = prepare_fast_sis_belief_no_pair_triton_into(
            fast_belief_workspace,
            prepared.beliefs,
        )
        fast_sis_setup_seconds = (
            fast_board.setup_seconds
            + fast_belief.setup_seconds
            + warmup_fast_triton_sis(
                fast_belief,
                sample_count=args.fast_triton_chunk,
                seed=args.seed + 199_999,
                block_s=args.fast_triton_block_s,
                max_tries=args.fast_triton_max_tries,
            )
        )
        fast_sis = fast_triton_sis_until_ess(
            fast_belief,
            target_ess=args.ess,
            chunk_size=args.fast_triton_chunk,
            seed=args.seed + 200_000,
            block_s=args.fast_triton_block_s,
            max_tries=args.fast_triton_max_tries,
        )
    if args.torch_cdf_sis:
        fast_board = prepare_fast_sis_board(prepared)
        fast_belief = prepare_fast_sis_belief(fast_board, prepared.beliefs)
        torch_cdf_sis_setup_seconds = fast_board.setup_seconds + fast_belief.setup_seconds
        torch_cdf_sis = torch_cdf_sis_until_ess(
            fast_belief,
            target_ess=args.ess,
            chunk_size=args.torch_cdf_chunk,
            max_tries=args.torch_cdf_max_tries,
        )
    if args.stage_triton_cdf_sis:
        if triton is None:
            raise RuntimeError("triton is not available")
        fast_board = prepare_fast_sis_board(prepared)
        fast_belief = prepare_fast_sis_belief(fast_board, prepared.beliefs)
        warmup_start = time.perf_counter()
        stage_triton_cdf_sis_chunk(
            fast_belief,
            args.stage_triton_cdf_chunk,
            seed=args.seed + 299_999,
            block_s=args.stage_triton_cdf_block_s,
            max_tries=args.stage_triton_cdf_max_tries,
        )
        torch.cuda.synchronize(device)
        stage_triton_cdf_sis_setup_seconds = (
            fast_board.setup_seconds
            + fast_belief.setup_seconds
            + time.perf_counter()
            - warmup_start
        )
        stage_triton_cdf_sis = stage_triton_cdf_sis_until_ess(
            fast_belief,
            target_ess=args.ess,
            chunk_size=args.stage_triton_cdf_chunk,
            seed=args.seed + 300_000,
            block_s=args.stage_triton_cdf_block_s,
            max_tries=args.stage_triton_cdf_max_tries,
        )
    if args.torch_multinomial_reject_sis:
        fast_board = prepare_fast_sis_board(prepared)
        fast_belief = prepare_fast_sis_belief(fast_board, prepared.beliefs)
        torch_mr_sis_setup_seconds = fast_board.setup_seconds + fast_belief.setup_seconds
        torch_mr_sis = torch_multinomial_reject_sis_until_ess(
            fast_belief,
            target_ess=args.ess,
            chunk_size=args.torch_multinomial_reject_chunk,
            max_tries=args.torch_multinomial_reject_max_tries,
            generator=generator,
        )
    if args.torch_multinomial_triton_reject_sis:
        if triton is None:
            raise RuntimeError("triton is not available")
        fast_board = prepare_fast_sis_board(prepared)
        fast_belief_workspace = make_fast_sis_belief_workspace(
            fast_board,
            players=args.players,
            build_cdf=False,
            build_pair_mass=False,
        )
        warmup_start = time.perf_counter()
        prepare_fast_sis_belief_no_pair_triton_into(
            fast_belief_workspace,
            prepared.beliefs,
            normalize=False,
            synchronize=False,
        )
        triton_reject_workspace = make_triton_reject_sis_workspace(
            fast_belief_workspace,
            sample_count=args.torch_multinomial_triton_reject_chunk,
            max_tries=args.torch_multinomial_triton_reject_max_tries,
        )
        _ = torch_multinomial_triton_reject_sis_until_ess(
            fast_belief_workspace,
            target_ess=args.ess,
            chunk_size=args.torch_multinomial_triton_reject_chunk,
            max_tries=args.torch_multinomial_triton_reject_max_tries,
            block_s=args.torch_multinomial_triton_reject_block_s,
            fixed_chunk=args.torch_multinomial_triton_reject_fixed_chunk,
            generator=torch.Generator(device=device).manual_seed(0),
            workspace=triton_reject_workspace,
        )
        torch.cuda.synchronize(device)
        fast_belief = prepare_fast_sis_belief_no_pair_triton_into(
            fast_belief_workspace,
            prepared.beliefs,
            normalize=False,
            synchronize=False,
        )
        torch_mr_triton_sis_board_seconds = fast_board.setup_seconds
        torch_mr_triton_sis_belief_seconds = fast_belief.setup_seconds
        torch_mr_triton_sis_warmup_seconds = time.perf_counter() - warmup_start
        torch_mr_triton_sis_setup_seconds = (
            fast_board.setup_seconds
            + fast_belief.setup_seconds
            + torch_mr_triton_sis_warmup_seconds
        )
        torch_mr_triton_sis = torch_multinomial_triton_reject_sis_until_ess(
            fast_belief,
            target_ess=args.ess,
            chunk_size=args.torch_multinomial_triton_reject_chunk,
            max_tries=args.torch_multinomial_triton_reject_max_tries,
            block_s=args.torch_multinomial_triton_reject_block_s,
            fixed_chunk=args.torch_multinomial_triton_reject_fixed_chunk,
            generator=generator,
            workspace=triton_reject_workspace,
        )
    exact = None
    exact_nway = None
    exact_nway_f32 = None
    if args.exact_threeway:
        if args.players != 3:
            raise ValueError("--exact-threeway expects --players 3")
        exact = exact_threeway_ie(prepared)
    if args.exact_nway:
        exact_nway = exact_nway_ie(prepared, pattern_chunk=args.pattern_chunk)
    if args.exact_nway_f32_chunked:
        exact_nway_f32 = exact_nway_ie_chunked_f32(
            prepared,
            pattern_chunk=args.exact_nway_f32_hand_chunk,
        )
    if args.brute_topk > 0:
        restricted = restrict_beliefs_to_topk(prepared, topk=args.brute_topk)
        brute = brute_force_nway(restricted, chunk_size=args.brute_chunk)
        print(
            "brute_topk "
            f"players={args.players} k={args.brute_topk} "
            f"chunk={args.brute_chunk} seconds={brute.seconds:.3f} "
            f"equity={_format_vector(brute.equity)}"
        )
        if args.players == 3:
            restricted_exact = exact_threeway_ie(restricted)
            restricted_exact_nway = exact_nway_ie(
                restricted,
                pattern_chunk=args.pattern_chunk,
            )
            print(
                "exact_topk "
                f"seconds={restricted_exact.seconds:.3f} "
                f"equity={_format_vector(restricted_exact.equity)} "
                f"max_abs_vs_brute="
                f"{float((restricted_exact.equity - brute.equity).abs().max().item()):.3e}"
            )
            print(
                "exact_nway_topk "
                f"seconds={restricted_exact_nway.seconds:.3f} "
                f"equity={_format_vector(restricted_exact_nway.equity)} "
                f"max_abs_vs_pie="
                f"{float((restricted_exact_nway.equity - restricted_exact.equity).abs().max().item()):.3e} "
                f"max_abs_vs_brute="
                f"{float((restricted_exact_nway.equity - brute.equity).abs().max().item()):.3e}"
            )
        elif args.exact_nway:
            restricted_exact_nway = exact_nway_ie(
                restricted,
                pattern_chunk=args.pattern_chunk,
            )
            print(
                "exact_nway_topk "
                f"seconds={restricted_exact_nway.seconds:.3f} "
                f"equity={_format_vector(restricted_exact_nway.equity)} "
                f"max_abs_vs_brute="
                f"{float((restricted_exact_nway.equity - brute.equity).abs().max().item()):.3e}"
            )
    reference = sis_until_ess(
        prepared,
        target_ess=args.reference_ess,
        chunk_size=args.sis_chunk,
        generator=generator,
    )

    combined_naive = (naive.stderr.square() + reference.stderr.square()).sqrt()
    combined_sis = (sis.stderr.square() + reference.stderr.square()).sqrt()
    naive_z = ((naive.equity - reference.equity).abs() / combined_naive.clamp_min(1e-9)).max()
    sis_z = ((sis.equity - reference.equity).abs() / combined_sis.clamp_min(1e-9)).max()
    triton_z = None
    if triton_sis is not None:
        combined_triton = (triton_sis.stderr.square() + reference.stderr.square()).sqrt()
        triton_z = (
            (triton_sis.equity - reference.equity).abs()
            / combined_triton.clamp_min(1e-9)
        ).max()
    fast_sis_z = None
    if fast_sis is not None:
        combined_fast = (fast_sis.stderr.square() + reference.stderr.square()).sqrt()
        fast_sis_z = (
            (fast_sis.equity - reference.equity).abs()
            / combined_fast.clamp_min(1e-9)
        ).max()
    torch_cdf_sis_z = None
    if torch_cdf_sis is not None:
        combined_torch_cdf = (
            torch_cdf_sis.stderr.square() + reference.stderr.square()
        ).sqrt()
        torch_cdf_sis_z = (
            (torch_cdf_sis.equity - reference.equity).abs()
            / combined_torch_cdf.clamp_min(1e-9)
        ).max()
    stage_triton_cdf_sis_z = None
    if stage_triton_cdf_sis is not None:
        combined_stage_cdf = (
            stage_triton_cdf_sis.stderr.square() + reference.stderr.square()
        ).sqrt()
        stage_triton_cdf_sis_z = (
            (stage_triton_cdf_sis.equity - reference.equity).abs()
            / combined_stage_cdf.clamp_min(1e-9)
        ).max()
    torch_mr_sis_z = None
    if torch_mr_sis is not None:
        combined_mr = (torch_mr_sis.stderr.square() + reference.stderr.square()).sqrt()
        torch_mr_sis_z = (
            (torch_mr_sis.equity - reference.equity).abs()
            / combined_mr.clamp_min(1e-9)
        ).max()
    torch_mr_triton_sis_z = None
    if torch_mr_triton_sis is not None:
        combined_mr_triton = (
            torch_mr_triton_sis.stderr.square() + reference.stderr.square()
        ).sqrt()
        torch_mr_triton_sis_z = (
            (torch_mr_triton_sis.equity - reference.equity).abs()
            / combined_mr_triton.clamp_min(1e-9)
        ).max()
    exact_z = None
    if exact is not None:
        exact_z = ((exact.equity - reference.equity).abs() / reference.stderr.clamp_min(1e-9)).max()
    exact_nway_z = None
    if exact_nway is not None:
        exact_nway_z = (
            (exact_nway.equity - reference.equity).abs()
            / reference.stderr.clamp_min(1e-9)
        ).max()
    exact_nway_f32_z = None
    if exact_nway_f32 is not None:
        exact_nway_f32_z = (
            (exact_nway_f32.equity - reference.equity).abs()
            / reference.stderr.clamp_min(1e-9)
        ).max()

    print(
        f"device={device} board={prepared.board.detach().cpu().tolist()[0]} "
        f"setup_seconds={prepared.setup_seconds:.3f}"
    )
    if triton_setup_seconds is not None:
        print(f"triton_setup_seconds={triton_setup_seconds:.3f}")
    if fast_sis_setup_seconds is not None:
        print(f"fast_triton_sis_setup_seconds={fast_sis_setup_seconds:.3f}")
    if torch_cdf_sis_setup_seconds is not None:
        print(f"torch_cdf_sis_setup_seconds={torch_cdf_sis_setup_seconds:.3f}")
    if stage_triton_cdf_sis_setup_seconds is not None:
        print(
            "stage_triton_cdf_sis_setup_seconds="
            f"{stage_triton_cdf_sis_setup_seconds:.3f}"
        )
    if torch_mr_sis_setup_seconds is not None:
        print(f"torch_multinomial_reject_sis_setup_seconds={torch_mr_sis_setup_seconds:.3f}")
    if torch_mr_triton_sis_setup_seconds is not None:
        print(
            "torch_multinomial_triton_reject_sis_setup_seconds="
            f"{torch_mr_triton_sis_setup_seconds:.3f}"
        )
    if torch_mr_triton_sis is not None:
        if (
            torch_mr_triton_sis_board_seconds is None
            or torch_mr_triton_sis_belief_seconds is None
            or torch_mr_triton_sis_warmup_seconds is None
        ):
            raise RuntimeError("missing split timing for torch multinomial Triton SIS")
        print(
            "torch_multinomial_triton_reject_sis_timing "
            f"board_setup={torch_mr_triton_sis_board_seconds:.6f} "
            f"belief_setup={torch_mr_triton_sis_belief_seconds:.6f} "
            f"warmup={torch_mr_triton_sis_warmup_seconds:.6f} "
            f"sample={torch_mr_triton_sis.seconds:.6f} "
            "belief_update_total="
            f"{torch_mr_triton_sis_belief_seconds + torch_mr_triton_sis.seconds:.6f}"
        )
    print(
        "naive "
        f"ess={naive.ess:.0f} proposals={naive.proposals} "
        f"seconds={naive.seconds:.3f} equity={_format_vector(naive.equity)} "
        f"stderr={_format_vector(naive.stderr)}"
    )
    print(
        "sis   "
        f"ess={sis.ess:.0f} proposals={sis.proposals} "
        f"seconds={sis.seconds:.3f} equity={_format_vector(sis.equity)} "
        f"stderr={_format_vector(sis.stderr)}"
    )
    if triton_sis is not None:
        print(
            "triton_sis "
            f"ess={triton_sis.ess:.0f} proposals={triton_sis.proposals} "
            f"seconds={triton_sis.seconds:.3f} equity={_format_vector(triton_sis.equity)} "
            f"stderr={_format_vector(triton_sis.stderr)}"
        )
    if fast_sis is not None:
        print(
            "fast_triton_sis "
            f"ess={fast_sis.ess:.0f} proposals={fast_sis.proposals} "
            f"seconds={fast_sis.seconds:.3f} equity={_format_vector(fast_sis.equity)} "
            f"stderr={_format_vector(fast_sis.stderr)}"
        )
    if torch_cdf_sis is not None:
        print(
            "torch_cdf_sis "
            f"ess={torch_cdf_sis.ess:.0f} proposals={torch_cdf_sis.proposals} "
            f"seconds={torch_cdf_sis.seconds:.3f} "
            f"equity={_format_vector(torch_cdf_sis.equity)} "
            f"stderr={_format_vector(torch_cdf_sis.stderr)}"
        )
    if stage_triton_cdf_sis is not None:
        print(
            "stage_triton_cdf_sis "
            f"ess={stage_triton_cdf_sis.ess:.0f} "
            f"proposals={stage_triton_cdf_sis.proposals} "
            f"seconds={stage_triton_cdf_sis.seconds:.3f} "
            f"equity={_format_vector(stage_triton_cdf_sis.equity)} "
            f"stderr={_format_vector(stage_triton_cdf_sis.stderr)}"
        )
    if torch_mr_sis is not None:
        print(
            "torch_multinomial_reject_sis "
            f"ess={torch_mr_sis.ess:.0f} proposals={torch_mr_sis.proposals} "
            f"seconds={torch_mr_sis.seconds:.3f} "
            f"equity={_format_vector(torch_mr_sis.equity)} "
            f"stderr={_format_vector(torch_mr_sis.stderr)}"
        )
    if torch_mr_triton_sis is not None:
        print(
            "torch_multinomial_triton_reject_sis "
            f"ess={torch_mr_triton_sis.ess:.0f} "
            f"proposals={torch_mr_triton_sis.proposals} "
            f"seconds={torch_mr_triton_sis.seconds:.3f} "
            f"equity={_format_vector(torch_mr_triton_sis.equity)} "
            f"stderr={_format_vector(torch_mr_triton_sis.stderr)}"
        )
    if exact is not None:
        print(
            "exact_ie "
            f"seconds={exact.seconds:.3f} equity={_format_vector(exact.equity)}"
        )
    if exact_nway is not None:
        line = (
            "exact_nway "
            f"seconds={exact_nway.seconds:.3f} "
            f"equity={_format_vector(exact_nway.equity)}"
        )
        if exact is not None:
            line += (
                f" max_abs_vs_pie="
                f"{float((exact_nway.equity - exact.equity).abs().max().item()):.3e}"
            )
        print(line)
    if exact_nway_f32 is not None:
        line = (
            "exact_nway_f32_chunked "
            f"seconds={exact_nway_f32.seconds:.3f} "
            f"equity={_format_vector(exact_nway_f32.equity)}"
        )
        if exact is not None:
            line += (
                " max_abs_vs_pie="
                f"{float((exact_nway_f32.equity - exact.equity).abs().max().item()):.3e}"
            )
        if exact_nway is not None:
            line += (
                " max_abs_vs_exact_nway="
                f"{float((exact_nway_f32.equity - exact_nway.equity).abs().max().item()):.3e}"
            )
        print(line)
    print(
        "ref   "
        f"ess={reference.ess:.0f} proposals={reference.proposals} "
        f"seconds={reference.seconds:.3f} equity={_format_vector(reference.equity)} "
        f"stderr={_format_vector(reference.stderr)}"
    )
    z_parts = [
        f"naive={float(naive_z.item()):.3f}",
        f"sis={float(sis_z.item()):.3f}",
    ]
    if triton_z is not None:
        z_parts.append(f"triton_sis={float(triton_z.item()):.3f}")
    if fast_sis_z is not None:
        z_parts.append(f"fast_triton_sis={float(fast_sis_z.item()):.3f}")
    if torch_cdf_sis_z is not None:
        z_parts.append(f"torch_cdf_sis={float(torch_cdf_sis_z.item()):.3f}")
    if stage_triton_cdf_sis_z is not None:
        z_parts.append(
            f"stage_triton_cdf_sis={float(stage_triton_cdf_sis_z.item()):.3f}"
        )
    if torch_mr_sis_z is not None:
        z_parts.append(
            f"torch_multinomial_reject_sis={float(torch_mr_sis_z.item()):.3f}"
        )
    if torch_mr_triton_sis_z is not None:
        z_parts.append(
            "torch_multinomial_triton_reject_sis="
            f"{float(torch_mr_triton_sis_z.item()):.3f}"
        )
    if exact_z is not None:
        z_parts.append(f"exact_ie={float(exact_z.item()):.3f}")
    if exact_nway_z is not None:
        z_parts.append(f"exact_nway={float(exact_nway_z.item()):.3f}")
    if exact_nway_f32_z is not None:
        z_parts.append(f"exact_nway_f32_chunked={float(exact_nway_f32_z.item()):.3f}")
    print("max_z_vs_ref " + " ".join(z_parts))
    bad_z = float(naive_z.item()) > args.max_z or float(sis_z.item()) > args.max_z
    if triton_z is not None:
        bad_z = bad_z or float(triton_z.item()) > args.max_z
    if fast_sis_z is not None:
        bad_z = bad_z or float(fast_sis_z.item()) > args.max_z
    if torch_cdf_sis_z is not None:
        bad_z = bad_z or float(torch_cdf_sis_z.item()) > args.max_z
    if stage_triton_cdf_sis_z is not None:
        bad_z = bad_z or float(stage_triton_cdf_sis_z.item()) > args.max_z
    if torch_mr_sis_z is not None:
        bad_z = bad_z or float(torch_mr_sis_z.item()) > args.max_z
    if torch_mr_triton_sis_z is not None:
        bad_z = bad_z or float(torch_mr_triton_sis_z.item()) > args.max_z
    if exact_z is not None:
        bad_z = bad_z or float(exact_z.item()) > args.max_z
    if exact_nway_z is not None:
        bad_z = bad_z or float(exact_nway_z.item()) > args.max_z
    if exact_nway_f32_z is not None:
        bad_z = bad_z or float(exact_nway_f32_z.item()) > args.max_z
    if bad_z:
        raise SystemExit(
            f"estimator error exceeded {args.max_z:.1f} combined standard errors"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--ess", type=int, default=10_000)
    parser.add_argument("--reference-ess", type=int, default=200_000)
    parser.add_argument("--naive-chunk", type=int, default=16_384)
    parser.add_argument("--sis-chunk", type=int, default=8_192)
    parser.add_argument("--triton", action="store_true")
    parser.add_argument("--triton-chunk", type=int, default=8_192)
    parser.add_argument("--triton-block-h", type=int, default=2048)
    parser.add_argument("--triton-block-s", type=int, default=8)
    parser.add_argument("--triton-max-tries", type=int, default=8)
    parser.add_argument("--fast-triton-sis", action="store_true")
    parser.add_argument("--fast-triton-chunk", type=int, default=16_384)
    parser.add_argument("--fast-triton-block-s", type=int, default=128)
    parser.add_argument("--fast-triton-max-tries", type=int, default=12)
    parser.add_argument("--torch-cdf-sis", action="store_true")
    parser.add_argument("--torch-cdf-chunk", type=int, default=8_192)
    parser.add_argument("--torch-cdf-max-tries", type=int, default=8)
    parser.add_argument("--stage-triton-cdf-sis", action="store_true")
    parser.add_argument("--stage-triton-cdf-chunk", type=int, default=8_192)
    parser.add_argument("--stage-triton-cdf-block-s", type=int, default=128)
    parser.add_argument("--stage-triton-cdf-max-tries", type=int, default=8)
    parser.add_argument("--torch-multinomial-reject-sis", action="store_true")
    parser.add_argument("--torch-multinomial-reject-chunk", type=int, default=16_384)
    parser.add_argument("--torch-multinomial-reject-max-tries", type=int, default=8)
    parser.add_argument("--torch-multinomial-triton-reject-sis", action="store_true")
    parser.add_argument(
        "--torch-multinomial-triton-reject-chunk",
        type=int,
        default=52_224,
    )
    parser.add_argument(
        "--torch-multinomial-triton-reject-block-s",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--torch-multinomial-triton-reject-max-tries",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--torch-multinomial-triton-reject-fixed-chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch-belief-benchmark", action="store_true")
    parser.add_argument(
        "--batch-belief-sampler",
        choices=("alias", "multinomial"),
        default="alias",
    )
    parser.add_argument("--batch-belief-size", type=int, default=512)
    parser.add_argument("--batch-belief-iterations", type=int, default=10_000)
    parser.add_argument("--batch-belief-warmup-iterations", type=int, default=3)
    parser.add_argument("--batch-belief-samples", type=int, default=10_240)
    parser.add_argument("--batch-belief-block-s", type=int, default=128)
    parser.add_argument("--batch-belief-max-tries", type=int, default=4)
    parser.add_argument("--batch-belief-reuse-alias", action="store_true")
    parser.add_argument(
        "--batch-belief-include-prep",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--exact-threeway", action="store_true")
    parser.add_argument("--exact-nway", action="store_true")
    parser.add_argument("--pattern-chunk", type=int, default=4096)
    parser.add_argument("--exact-nway-f32-chunked", action="store_true")
    parser.add_argument("--exact-nway-f32-hand-chunk", type=int, default=64)
    parser.add_argument("--brute-topk", type=int, default=0)
    parser.add_argument("--brute-chunk", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--max-z", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_validation(parse_args())
