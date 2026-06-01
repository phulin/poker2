from __future__ import annotations

from dataclasses import dataclass

import torch

from p2.env.card_utils import NUM_HANDS, board_allowed_hands, hand_combos_tensor
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.env.rules import rank_hands
from p2.search.cfr_evaluator import PublicBeliefState


STREET_TO_BOARD_CARDS = {1: 3, 2: 4, 3: 5}
CLOSED_STREET_TO_NEXT_STREET = {0: 1, 1: 2, 2: 3}
RANGE_MIXTURE_WEIGHTS = (0.25, 0.15, 0.20, 0.20, 0.15, 0.05)
RANGE_MIXTURE_NAMES = (
    "recursive_strength",
    "uniform",
    "exponential",
    "strength_biased",
    "polarized",
    "capped",
)
RANGE_MODE_RECURSIVE = 0
RANGE_MODE_UNIFORM = 1
RANGE_MODE_EXPONENTIAL = 2
RANGE_MODE_STRENGTH_BIASED = 3
RANGE_MODE_POLARIZED = 4
RANGE_MODE_CAPPED = 5
BOARD_TEXTURE_NAMES = ("paired", "monotone", "two_tone", "straight_heavy", "dry")
BOARD_TEXTURE_WEIGHTS = (0.20, 0.10, 0.25, 0.25, 0.20)
BOARD_TEXTURE_CANDIDATES = 24
BOARD_TEXTURE_PAIRED = 0
BOARD_TEXTURE_MONOTONE = 1
BOARD_TEXTURE_TWO_TONE = 2
BOARD_TEXTURE_STRAIGHT_HEAVY = 3
BOARD_TEXTURE_DRY = 4


@dataclass(frozen=True)
class ChanceRootSample:
    """Closed-street target root with post-chance context and pre-chance beliefs."""

    pbs: PublicBeliefState
    pre_chance_beliefs: torch.Tensor
    closed_street: int


def _sample_unique_cards(
    batch_size: int,
    num_cards: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    scores = torch.rand(batch_size, 52, device=device, generator=generator)
    return scores.argsort(dim=1)[:, :num_cards]


def _board_texture_target_matches(board: torch.Tensor) -> torch.Tensor:
    """Return board matches for paired/monotone/two-tone/connected/dry targets."""

    if board.shape[-1] != 5:
        padded = torch.full(
            (*board.shape[:-1], 5), -1, dtype=torch.long, device=board.device
        )
        padded[..., : board.shape[-1]] = board
        board = padded

    flat = board.reshape(-1, 5)
    valid = flat >= 0
    ranks = torch.where(valid, flat % 13, torch.zeros_like(flat))
    suits = torch.where(valid, flat // 13, torch.zeros_like(flat))
    rank_counts = torch.zeros(flat.shape[0], 13, dtype=torch.long, device=board.device)
    suit_counts = torch.zeros(flat.shape[0], 4, dtype=torch.long, device=board.device)
    rank_counts.scatter_add_(1, ranks, valid.to(torch.long))
    suit_counts.scatter_add_(1, suits, valid.to(torch.long))

    card_counts = valid.sum(dim=1)
    paired = rank_counts.amax(dim=1) >= 2
    monotone = suit_counts.amax(dim=1) >= torch.minimum(
        card_counts, torch.full_like(card_counts, 3)
    )
    two_tone = (suit_counts == 2).any(dim=1) & ~monotone
    rank_present = rank_counts > 0
    regular_windows = rank_present.unfold(1, 5, 1).sum(dim=2).amax(dim=1)
    wheel_window = rank_present[:, [12, 0, 1, 2, 3]].sum(dim=1)
    connected_count = torch.maximum(regular_windows, wheel_window)
    needed_connected = torch.where(
        card_counts <= 3,
        torch.full_like(card_counts, 3),
        torch.full_like(card_counts, 4),
    )
    straight_heavy = connected_count >= needed_connected
    dry = ~paired & ~monotone & ~straight_heavy

    matches = torch.stack(
        (
            paired,
            monotone & ~paired,
            two_tone & ~paired & ~monotone,
            straight_heavy & ~paired & ~monotone,
            dry,
        ),
        dim=1,
    )
    return matches.reshape(*board.shape[:-1], len(BOARD_TEXTURE_NAMES))


def _board_texture_labels(board: torch.Tensor) -> torch.Tensor:
    """Classify boards into a single prioritized public-texture bucket."""

    matches = _board_texture_target_matches(board)
    labels = torch.full(
        board.shape[:-1],
        BOARD_TEXTURE_TWO_TONE,
        dtype=torch.long,
        device=board.device,
    )
    labels = torch.where(matches[..., BOARD_TEXTURE_DRY], BOARD_TEXTURE_DRY, labels)
    labels = torch.where(
        matches[..., BOARD_TEXTURE_STRAIGHT_HEAVY],
        BOARD_TEXTURE_STRAIGHT_HEAVY,
        labels,
    )
    labels = torch.where(
        matches[..., BOARD_TEXTURE_MONOTONE], BOARD_TEXTURE_MONOTONE, labels
    )
    labels = torch.where(matches[..., BOARD_TEXTURE_PAIRED], BOARD_TEXTURE_PAIRED, labels)
    return labels


def _sample_texture_stratified_boards(
    batch_size: int,
    num_cards: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    weights = torch.tensor(BOARD_TEXTURE_WEIGHTS, dtype=torch.float32, device=device)
    targets = torch.multinomial(
        weights,
        batch_size,
        replacement=True,
        generator=generator,
    )
    candidates = _sample_unique_cards(
        batch_size * BOARD_TEXTURE_CANDIDATES,
        num_cards,
        device=device,
        generator=generator,
    ).reshape(BOARD_TEXTURE_CANDIDATES, batch_size, num_cards)
    target_matches = _board_texture_target_matches(candidates).reshape(
        BOARD_TEXTURE_CANDIDATES, batch_size, len(BOARD_TEXTURE_NAMES)
    )
    matches = target_matches.gather(
        2,
        targets[None, :, None].expand(BOARD_TEXTURE_CANDIDATES, -1, 1),
    ).squeeze(2)
    selected_attempts = matches.to(torch.long).argmax(dim=0)
    rows = torch.arange(batch_size, device=device)
    selected = candidates[selected_attempts, rows]
    return torch.where(matches.any(dim=0)[:, None], selected, candidates[0])


def postflop_spot_sampler_metadata() -> dict:
    return {
        "board_texture_stratified": True,
        "board_texture_weights": dict(zip(BOARD_TEXTURE_NAMES, BOARD_TEXTURE_WEIGHTS)),
        "board_texture_candidates": BOARD_TEXTURE_CANDIDATES,
        "belief_mixture_weights": dict(zip(RANGE_MIXTURE_NAMES, RANGE_MIXTURE_WEIGHTS)),
    }


def _uniform_board_legal_beliefs(
    board: torch.Tensor, *, num_players: int
) -> torch.Tensor:
    allowed = board_allowed_hands(board)
    beliefs = allowed[:, None, :].expand(-1, num_players, -1).to(torch.float32)
    return beliefs / beliefs.sum(dim=-1, keepdim=True).clamp(min=1.0)


def _rank_shape_scores(board: torch.Tensor) -> torch.Tensor:
    """Return a tensorized board-aware rank proxy for incomplete streets."""

    device = board.device
    combos = hand_combos_tensor(device=device)
    combo_ranks = combos % 13
    combo_suits = combos // 13

    high_rank = combo_ranks.max(dim=1).values.to(torch.float32) / 12.0
    low_rank = combo_ranks.min(dim=1).values.to(torch.float32) / 12.0
    pocket_pair = (combo_ranks[:, 0] == combo_ranks[:, 1]).to(torch.float32)
    suited = (combo_suits[:, 0] == combo_suits[:, 1]).to(torch.float32)

    board_valid = board >= 0
    board_ranks = torch.where(board_valid, board % 13, torch.full_like(board, -1))
    board_suits = torch.where(board_valid, board // 13, torch.full_like(board, -1))
    rank_hits = (
        combo_ranks[None, :, :, None] == board_ranks[:, None, None, :]
    ) & board_valid[:, None, None, :]
    suit_hits = (
        combo_suits[None, :, :, None] == board_suits[:, None, None, :]
    ) & board_valid[:, None, None, :]

    board_pairing = rank_hits.to(torch.float32).sum(dim=(2, 3)).clamp(max=3.0) / 3.0
    suit_pressure = suit_hits.to(torch.float32).sum(dim=3).amax(dim=2) / 5.0
    return (
        0.42 * high_rank[None, :]
        + 0.12 * low_rank[None, :]
        + 0.22 * pocket_pair[None, :]
        + 0.08 * suited[None, :]
        + 0.34 * board_pairing
        + 0.12 * suit_pressure
    )


def _strength_scores(board: torch.Tensor) -> torch.Tensor:
    """Return weaker-to-stronger sortable scores for each board/combo pair."""

    if (board >= 0).all():
        ranks, _ = rank_hands(board.long())
        return ranks.to(torch.float32)
    return _rank_shape_scores(board)


def _normalize_allowed_scores(
    scores: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    inf = torch.full_like(scores, float("inf"))
    ninf = torch.full_like(scores, float("-inf"))
    lo = torch.where(allowed, scores, inf).amin(dim=1, keepdim=True)
    hi = torch.where(allowed, scores, ninf).amax(dim=1, keepdim=True)
    return ((scores - lo) / (hi - lo).clamp(min=1.0)).clamp(0.0, 1.0)


def _recursive_strength_beliefs(
    board: torch.Tensor,
    allowed: torch.Tensor,
    *,
    num_players: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Tensorized DeepStack-style recursive range generator R(S, p)."""

    batch_size = board.shape[0]
    device = board.device
    scores = _strength_scores(board)
    sort_key = torch.where(
        allowed,
        scores,
        torch.full_like(scores, float("inf")),
    )
    order = torch.argsort(sort_key, dim=1, stable=True)
    legal_counts = allowed.sum(dim=1, dtype=torch.long)

    starts = torch.zeros(batch_size, num_players, 1, device=device, dtype=torch.long)
    lengths = legal_counts[:, None, None].expand(-1, num_players, -1).clone()
    masses = torch.ones(batch_size, num_players, 1, device=device, dtype=torch.float32)
    max_depth = (NUM_HANDS - 1).bit_length()
    for _ in range(max_depth):
        active = lengths > 1
        left_lengths = torch.where(active, lengths // 2, lengths)
        right_lengths = torch.where(
            active, lengths - left_lengths, torch.zeros_like(lengths)
        )
        split = torch.rand(masses.shape, device=device, generator=generator)
        left_masses = torch.where(active, masses * split, masses)
        right_masses = torch.where(active, masses - left_masses, torch.zeros_like(masses))

        starts = torch.stack((starts, starts + left_lengths), dim=-1).flatten(2)
        lengths = torch.stack((left_lengths, right_lengths), dim=-1).flatten(2)
        masses = torch.stack((left_masses, right_masses), dim=-1).flatten(2)

    sorted_mass = torch.zeros(
        batch_size, num_players, NUM_HANDS, device=device, dtype=torch.float32
    )
    valid = lengths > 0
    sorted_mass.scatter_add_(2, starts.clamp(max=NUM_HANDS - 1), masses * valid)
    beliefs = torch.zeros_like(sorted_mass)
    beliefs.scatter_(2, order[:, None, :].expand(-1, num_players, -1), sorted_mass)
    beliefs = beliefs * allowed[:, None, :].to(torch.float32)
    return beliefs / beliefs.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def _random_board_legal_beliefs(
    board: torch.Tensor,
    *,
    num_players: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    allowed = board_allowed_hands(board)
    allowed_f = allowed[:, None, :].expand(-1, num_players, -1).to(torch.float32)
    batch_size = board.shape[0]
    device = board.device
    strength = _normalize_allowed_scores(_strength_scores(board), allowed)
    strength = strength[:, None, :].expand(-1, num_players, -1)

    weights = torch.tensor(RANGE_MIXTURE_WEIGHTS, device=device, dtype=torch.float32)
    modes = torch.multinomial(
        weights,
        batch_size * num_players,
        replacement=True,
        generator=generator,
    ).reshape(batch_size, num_players)

    noise = -torch.rand(
        batch_size,
        num_players,
        NUM_HANDS,
        device=device,
        generator=generator,
    ).clamp_min(1e-12).log()
    concentration = torch.empty(
        batch_size, num_players, 1, device=device, dtype=torch.float32
    ).uniform_(1.5, 5.0, generator=generator)
    uniform_scores = allowed_f
    exponential_scores = noise + 0.05
    strength_scores = torch.exp((strength - 0.5) * concentration + 0.20 * noise)
    polarized_scores = torch.exp(
        (strength - 0.5).abs() * concentration + 0.20 * noise
    )
    capped_scores = torch.exp((1.0 - strength) * concentration + 0.20 * noise)
    score_bank = torch.stack(
        (
            uniform_scores,
            uniform_scores,
            exponential_scores,
            strength_scores,
            polarized_scores,
            capped_scores,
        ),
        dim=0,
    )
    gathered_scores = score_bank.gather(
        0,
        modes[None, :, :, None].expand(1, -1, -1, NUM_HANDS),
    ).squeeze(0)
    gathered_scores = (gathered_scores + 1e-6) * allowed_f
    shape_beliefs = gathered_scores / gathered_scores.sum(
        dim=-1, keepdim=True
    ).clamp(min=1e-12)

    recursive_beliefs = _recursive_strength_beliefs(
        board,
        allowed,
        num_players=num_players,
        generator=generator,
    )
    return torch.where(
        (modes == RANGE_MODE_RECURSIVE)[:, :, None],
        recursive_beliefs,
        shape_beliefs,
    )


def _apply_postflop_spot_templates(
    env: HUNLTensorEnv | PBSEnv,
    *,
    generator: torch.Generator | None,
) -> None:
    """Apply conservative street-start pot/SPR templates in-place."""

    batch_size = env.N
    device = env.device
    template = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    # Approximate DeepStack-style pot buckets on the env's stack scale:
    # single-raised, three-bet, short-stack low-SPR, and deep/high-SPR.
    low = torch.tensor([0.04, 0.10, 0.35, 0.02], device=device)
    high = torch.tensor([0.10, 0.28, 0.90, 0.06], device=device)
    span = high[template] - low[template]
    pot_frac = low[template] + span * torch.rand(
        batch_size, device=device, generator=generator
    )

    scale = env.scale.to(torch.float32).clamp(min=float(env.bb * 4))
    max_pot = (2.0 * (scale - float(env.bb))).clamp(min=float(env.bb * 2))
    pot = (pot_frac * scale).round().to(torch.long)
    pot = pot.clamp(min=int(env.bb * 2))
    pot = torch.minimum(pot, max_pot.to(torch.long))
    env.pot.copy_(pot)

    p0_contrib = pot // 2
    p1_contrib = pot - p0_contrib
    contributions = torch.stack((p0_contrib, p1_contrib), dim=1)
    stacks = (env.starting_stacks - contributions).clamp(min=0)
    env.stacks.copy_(stacks)
    env.chips_placed.copy_(contributions)
    env.committed.zero_()

    actions_last = torch.tensor([2, 3, 5, 2], device=device, dtype=torch.long)
    env.actions_last_round.copy_(actions_last[template])
    env.min_raise.fill_(env.bb)


@torch.no_grad()
def sample_postflop_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    street: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    """Sample legal-looking heads-up postflop street-start roots.

    The sampler constructs post-chance roots with no active bet, random
    button/to-act assignment, unique public boards, randomized board-legal
    beliefs, and conservative legal pot/SPR templates.
    """

    if int(env_proto.num_players) != 2:
        raise ValueError("sample_postflop_start_roots currently supports heads-up only")
    if street not in STREET_TO_BOARD_CARDS:
        raise ValueError(f"street must be one of {sorted(STREET_TO_BOARD_CARDS)}")

    device = env_proto.device
    board_cards = STREET_TO_BOARD_CARDS[street]
    if stratify_board_textures:
        board = _sample_texture_stratified_boards(
            batch_size, board_cards, device=device, generator=generator
        )
    else:
        board = _sample_unique_cards(
            batch_size, board_cards, device=device, generator=generator
        )
    board_padded = torch.full(
        (batch_size, 5), -1, dtype=torch.long, device=device
    )
    board_padded[:, :board_cards] = board
    if randomize_beliefs:
        beliefs = _random_board_legal_beliefs(
            board_padded, num_players=2, generator=generator
        )
    else:
        beliefs = _uniform_board_legal_beliefs(board_padded, num_players=2)
    beliefs = beliefs.to(device=device)
    pbs = PublicBeliefState.from_proto(
        env_proto=env_proto,
        beliefs=beliefs,
        num_envs=batch_size,
    )
    env = pbs.env
    env.reset()

    env.street.fill_(street)
    env.actions_this_round.zero_()
    env.actions_last_round.fill_(2)
    env.acted_since_reset.zero_()
    env.committed.zero_()
    env.has_folded.zero_()
    env.is_allin.zero_()
    env.done.zero_()
    env.winner.fill_(-1)
    env.min_raise.fill_(env.bb)
    if randomize_spots:
        _apply_postflop_spot_templates(env, generator=generator)

    button = torch.randint(
        0, 2, (batch_size,), device=device, generator=generator
    )
    env.button.copy_(button)
    env.to_act.copy_(1 - button)
    env.last_to_act.copy_(env.to_act)

    env.board_indices.fill_(-1)
    env.board_indices[:, :board_cards] = board
    env.last_board_indices.copy_(env.board_indices)
    if street > 1:
        env.last_board_indices[:, board_cards - 1] = -1
    else:
        env.last_board_indices.fill_(-1)
    env.board_onehot.zero_()
    env.board_onehot[:, :board_cards] = env.card_onehot_cache[board]

    pbs.beliefs = pbs.beliefs.reshape(batch_size, 2, NUM_HANDS)
    return pbs


@torch.no_grad()
def sample_postflop_legal_prefix_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    street: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    """Sample postflop roots reached by replaying one random legal action.

    This produces conservative in-street prefix states without independently
    mutating action, pot, stack, or commitment fields.
    """

    pbs = sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=street,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
        randomize_spots=randomize_spots,
        stratify_board_textures=stratify_board_textures,
    )
    env = pbs.env
    legal = env.legal_bins_mask()
    num_actions = legal.shape[1]
    allin_index = num_actions - 1
    action_ids = torch.arange(num_actions, device=env.device)
    bet_mask = legal & (action_ids[None, :] >= 2) & (action_ids[None, :] < allin_index)
    fallback_mask = legal & (action_ids[None, :] == 1)
    selectable = torch.where(bet_mask.any(dim=1, keepdim=True), bet_mask, fallback_mask)
    scores = torch.rand(
        batch_size,
        num_actions,
        device=env.device,
        generator=generator,
    )
    bin_indices = torch.where(selectable, scores, torch.full_like(scores, -1.0)).argmax(
        dim=1
    )
    env.step_bins(bin_indices)
    pbs.beliefs = pbs.beliefs.reshape(batch_size, 2, NUM_HANDS)
    return pbs


@torch.no_grad()
def sample_flop_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    return sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=1,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
        randomize_spots=randomize_spots,
        stratify_board_textures=stratify_board_textures,
    )


@torch.no_grad()
def sample_flop_legal_prefix_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    return sample_postflop_legal_prefix_roots(
        env_proto,
        batch_size=batch_size,
        street=1,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
        randomize_spots=randomize_spots,
        stratify_board_textures=stratify_board_textures,
    )


@torch.no_grad()
def sample_turn_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    return sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=2,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
        randomize_spots=randomize_spots,
        stratify_board_textures=stratify_board_textures,
    )


@torch.no_grad()
def sample_turn_legal_prefix_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    return sample_postflop_legal_prefix_roots(
        env_proto,
        batch_size=batch_size,
        street=2,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
        randomize_spots=randomize_spots,
        stratify_board_textures=stratify_board_textures,
    )


@torch.no_grad()
def sample_river_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    return sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=3,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
        randomize_spots=randomize_spots,
        stratify_board_textures=stratify_board_textures,
    )


@torch.no_grad()
def sample_river_legal_prefix_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
    randomize_spots: bool = True,
    stratify_board_textures: bool = True,
) -> PublicBeliefState:
    return sample_postflop_legal_prefix_roots(
        env_proto,
        batch_size=batch_size,
        street=3,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
        randomize_spots=randomize_spots,
        stratify_board_textures=stratify_board_textures,
    )


@torch.no_grad()
def sample_end_of_street_chance_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    closed_street: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
) -> ChanceRootSample:
    """Sample roots for end-of-street chance-value targets.

    Chance targets are represented with a post-chance next-street environment
    plus separate pre-chance beliefs over the previous board. For example,
    closed flop roots use a turn environment and flop-legal beliefs.
    """

    if closed_street not in CLOSED_STREET_TO_NEXT_STREET:
        raise ValueError(
            f"closed_street must be one of {sorted(CLOSED_STREET_TO_NEXT_STREET)}"
        )

    next_street = CLOSED_STREET_TO_NEXT_STREET[closed_street]
    pbs = sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=next_street,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
    )
    if randomize_beliefs:
        pre_chance_beliefs = _random_board_legal_beliefs(
            pbs.env.last_board_indices,
            num_players=2,
            generator=generator,
        )
    else:
        pre_chance_beliefs = _uniform_board_legal_beliefs(
            pbs.env.last_board_indices,
            num_players=2,
        )
    pre_chance_beliefs = pre_chance_beliefs.to(device=env_proto.device)
    return ChanceRootSample(
        pbs=pbs,
        pre_chance_beliefs=pre_chance_beliefs,
        closed_street=closed_street,
    )
