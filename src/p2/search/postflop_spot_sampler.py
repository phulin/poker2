from __future__ import annotations

import torch

from p2.env.card_utils import NUM_HANDS, board_allowed_hands
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.search.cfr_evaluator import PublicBeliefState


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


@torch.no_grad()
def sample_river_start_roots(
    env_proto: HUNLTensorEnv | PBSEnv,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> PublicBeliefState:
    """Sample legal-looking heads-up river street-start roots.

    The first implementation is intentionally conservative: it constructs
    post-chance river roots with no active bet, random button/to-act assignment,
    unique five-card boards, and uniform board-legal beliefs. Betting history is
    summarized by the inherited pot/stack state from reset plus round counters.
    """

    if int(env_proto.num_players) != 2:
        raise ValueError("sample_river_start_roots currently supports heads-up only")

    device = env_proto.device
    board = _sample_unique_cards(
        batch_size, 5, device=device, generator=generator
    )
    beliefs = _uniform_board_legal_beliefs(board, num_players=2).to(device=device)
    pbs = PublicBeliefState.from_proto(
        env_proto=env_proto,
        beliefs=beliefs,
        num_envs=batch_size,
    )
    env = pbs.env
    env.reset()

    env.street.fill_(3)
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
    env.board_indices[:, :5] = board
    env.last_board_indices.copy_(env.board_indices)
    env.last_board_indices[:, 4] = -1
    env.board_onehot.zero_()
    env.board_onehot[:, :5] = env.card_onehot_cache[board]

    pbs.beliefs = pbs.beliefs.reshape(batch_size, 2, NUM_HANDS)
    return pbs
