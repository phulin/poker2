from __future__ import annotations

from dataclasses import dataclass

import torch

from p2.env.card_utils import NUM_HANDS, board_allowed_hands
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.search.cfr_evaluator import PublicBeliefState


STREET_TO_BOARD_CARDS = {1: 3, 2: 4, 3: 5}
CLOSED_STREET_TO_NEXT_STREET = {0: 1, 1: 2, 2: 3}


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


def _uniform_board_legal_beliefs(
    board: torch.Tensor, *, num_players: int
) -> torch.Tensor:
    allowed = board_allowed_hands(board)
    beliefs = allowed[:, None, :].expand(-1, num_players, -1).to(torch.float32)
    return beliefs / beliefs.sum(dim=-1, keepdim=True).clamp(min=1.0)


def _random_board_legal_beliefs(
    board: torch.Tensor,
    *,
    num_players: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    allowed = board_allowed_hands(board)
    allowed_f = allowed[:, None, :].expand(-1, num_players, -1).to(torch.float32)
    scores = -torch.rand(
        board.shape[0],
        num_players,
        NUM_HANDS,
        device=board.device,
        generator=generator,
    ).clamp_min(1e-12).log()
    # Keep a small uniform component so every board-legal hand remains represented
    # while still generating asymmetric, non-uniform ranges.
    scores = scores + 0.05
    scores = scores * allowed_f
    return scores / scores.sum(dim=-1, keepdim=True).clamp(min=1.0)


@torch.no_grad()
def sample_postflop_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    street: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
) -> PublicBeliefState:
    """Sample legal-looking heads-up postflop street-start roots.

    The first implementation is intentionally conservative: it constructs
    post-chance roots with no active bet, random button/to-act assignment,
    unique public boards, and uniform board-legal beliefs. Betting history is
    summarized by the inherited pot/stack state from reset plus round counters.
    """

    if int(env_proto.num_players) != 2:
        raise ValueError("sample_postflop_start_roots currently supports heads-up only")
    if street not in STREET_TO_BOARD_CARDS:
        raise ValueError(f"street must be one of {sorted(STREET_TO_BOARD_CARDS)}")

    device = env_proto.device
    board_cards = STREET_TO_BOARD_CARDS[street]
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
def sample_flop_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
) -> PublicBeliefState:
    return sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=1,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
    )


@torch.no_grad()
def sample_turn_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
) -> PublicBeliefState:
    return sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=2,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
    )


@torch.no_grad()
def sample_river_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
    randomize_beliefs: bool = True,
) -> PublicBeliefState:
    return sample_postflop_start_roots(
        env_proto,
        batch_size=batch_size,
        street=3,
        generator=generator,
        randomize_beliefs=randomize_beliefs,
    )


@torch.no_grad()
def sample_end_of_street_chance_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    closed_street: int,
    generator: torch.Generator | None = None,
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
    )
    pre_chance_beliefs = _uniform_board_legal_beliefs(
        pbs.env.last_board_indices, num_players=2
    ).to(device=env_proto.device)
    return ChanceRootSample(
        pbs=pbs,
        pre_chance_beliefs=pre_chance_beliefs,
        closed_street=closed_street,
    )
