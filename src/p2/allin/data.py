from __future__ import annotations

from dataclasses import dataclass

import torch

from p2.env.card_utils import NUM_HANDS


@dataclass
class PreflopAllInBatch:
    beliefs: torch.Tensor
    starting_stacks: torch.Tensor
    committed: torch.Tensor
    stacks_after: torch.Tensor
    allin_mask: torch.Tensor
    folded_mask: torch.Tensor
    scale: torch.Tensor

    def to(self, device: torch.device | str) -> "PreflopAllInBatch":
        device = torch.device(device)
        return PreflopAllInBatch(
            beliefs=self.beliefs.to(device),
            starting_stacks=self.starting_stacks.to(device),
            committed=self.committed.to(device),
            stacks_after=self.stacks_after.to(device),
            allin_mask=self.allin_mask.to(device),
            folded_mask=self.folded_mask.to(device),
            scale=self.scale.to(device),
        )


def sample_weighted_uniform_stacks(
    batch_size: int,
    players: int,
    *,
    bb: int,
    min_stack_bb: int = 10,
    mid_stack_bb: int = 200,
    max_stack_bb: int = 400,
    high_stack_mass_ratio: float = 1.0 / 3.0,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Sample independent per-player stacks with the weighted-uniform BB law."""
    min_chips = max(int(round(min_stack_bb * bb)), bb)
    mid_chips = max(int(round(mid_stack_bb * bb)), min_chips)
    max_chips = max(int(round(max_stack_bb * bb)), mid_chips)
    high_prob = high_stack_mass_ratio / (1.0 + high_stack_mass_ratio)

    choose_high = (
        torch.rand(batch_size, players, device=device, generator=generator) < high_prob
    )
    rand = torch.rand(batch_size, players, device=device, generator=generator)
    low_width = mid_chips - min_chips + 1
    high_width = max_chips - mid_chips + 1
    low = min_chips + torch.floor(rand * low_width)
    high = mid_chips + torch.floor(rand * high_width)
    return torch.where(choose_high, high, low).to(torch.float32)


def _sample_status_masks(
    batch_size: int,
    players: int,
    *,
    min_allin_players: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if min_allin_players < 2:
        raise ValueError("min_allin_players must be at least 2")
    if min_allin_players > players:
        raise ValueError("min_allin_players cannot exceed players")

    scores = torch.rand(batch_size, players, device=device, generator=generator)
    order = scores.argsort(dim=1)
    extra = torch.randint(
        0,
        players - min_allin_players + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )
    live_count = min_allin_players + extra
    ranks = torch.empty_like(order)
    rank_src = torch.arange(players, device=device)[None, :].expand(batch_size, -1)
    ranks.scatter_(1, order, rank_src)
    allin_mask = ranks < live_count[:, None]
    folded_mask = ~allin_mask
    return allin_mask, folded_mask


def make_random_preflop_allin_batch(
    batch_size: int,
    players: int = 4,
    *,
    bb: int = 100,
    min_stack_bb: int = 10,
    mid_stack_bb: int = 200,
    max_stack_bb: int = 400,
    high_stack_mass_ratio: float = 1.0 / 3.0,
    concentration: float = 1.0,
    min_allin_players: int = 2,
    folded_commit_max_frac: float = 0.35,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> PreflopAllInBatch:
    """Create random preflop all-in training features.

    Live seats are marked all-in; non-live seats are marked folded. All-in seats
    commit their full starting stack. Folded seats contribute random dead money
    up to ``folded_commit_max_frac`` of their starting stack.
    """
    if players < 2:
        raise ValueError("players must be at least 2")
    if concentration <= 0:
        raise ValueError("concentration must be positive")
    if not 0.0 <= folded_commit_max_frac <= 1.0:
        raise ValueError("folded_commit_max_frac must be in [0, 1]")

    device = torch.device(device) if device is not None else torch.device("cpu")
    weights = torch.empty(
        batch_size,
        players,
        NUM_HANDS,
        device=device,
        dtype=torch.float32,
    ).exponential_(generator=generator)
    if concentration != 1.0:
        weights = weights.pow(1.0 / concentration)
    beliefs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)

    starting = sample_weighted_uniform_stacks(
        batch_size,
        players,
        bb=bb,
        min_stack_bb=min_stack_bb,
        mid_stack_bb=mid_stack_bb,
        max_stack_bb=max_stack_bb,
        high_stack_mass_ratio=high_stack_mass_ratio,
        device=device,
        generator=generator,
    )
    allin_mask, folded_mask = _sample_status_masks(
        batch_size,
        players,
        min_allin_players=min_allin_players,
        device=device,
        generator=generator,
    )

    folded_frac = (
        torch.rand(batch_size, players, device=device, generator=generator)
        * folded_commit_max_frac
    )
    committed = torch.where(allin_mask, starting, starting * folded_frac)
    committed = torch.where(folded_mask, committed, starting)
    stacks_after = (starting - committed).clamp_min(0.0)
    scale = starting.mean(dim=1).clamp_min(1.0)

    return PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=starting,
        committed=committed,
        stacks_after=stacks_after,
        allin_mask=allin_mask,
        folded_mask=folded_mask,
        scale=scale,
    )
