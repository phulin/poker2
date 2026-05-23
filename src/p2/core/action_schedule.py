from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_BET_BINS_BY_DEPTH: tuple[tuple[float, ...], ...] = (
    (0.25, 0.5, 0.75, 1.0, 1.5),
    (0.5, 1.0),
    (1.0,),
    (1.0,),
    (1.0,),
    (),
)


@dataclass(frozen=True)
class ActionSchedule:
    bet_bins: tuple[float, ...]
    bet_bins_by_depth: tuple[tuple[float, ...], ...] | None
    allin_by_depth: tuple[bool, ...] | None

    @property
    def num_actions(self) -> int:
        return len(self.bet_bins) + 3

    @property
    def num_depths(self) -> int | None:
        if self.bet_bins_by_depth is None:
            return None
        return len(self.bet_bins_by_depth)


def _normalize_depth_bins(
    bet_bins_by_depth: list[list[float]] | tuple[tuple[float, ...], ...] | None,
) -> tuple[tuple[float, ...], ...] | None:
    if bet_bins_by_depth is None:
        return None
    return tuple(tuple(float(x) for x in depth_bins) for depth_bins in bet_bins_by_depth)


def _normalize_allin_by_depth(
    allin_by_depth: list[bool] | tuple[bool, ...] | None,
    bet_bins_by_depth: tuple[tuple[float, ...], ...] | None,
) -> tuple[bool, ...] | None:
    if allin_by_depth is not None:
        values = tuple(bool(x) for x in allin_by_depth)
        if bet_bins_by_depth is not None and len(values) != len(bet_bins_by_depth):
            raise ValueError(
                "search.allin_by_depth must have the same length as "
                "search.bet_bins_by_depth"
            )
        return values
    if bet_bins_by_depth is None:
        return None

    values = [True] * len(bet_bins_by_depth)
    if values and len(bet_bins_by_depth[-1]) == 0:
        values[-1] = False
    return tuple(values)


def make_action_schedule(
    env_bet_bins: list[float] | tuple[float, ...],
    bet_bins_by_depth: list[list[float]] | tuple[tuple[float, ...], ...] | None = None,
    allin_by_depth: list[bool] | tuple[bool, ...] | None = None,
) -> ActionSchedule:
    depth_bins = _normalize_depth_bins(bet_bins_by_depth)
    if depth_bins is None:
        global_bins = tuple(float(x) for x in env_bet_bins)
    else:
        global_bins = tuple(sorted({b for depth in depth_bins for b in depth}))
    allin = _normalize_allin_by_depth(allin_by_depth, depth_bins)
    return ActionSchedule(
        bet_bins=global_bins,
        bet_bins_by_depth=depth_bins,
        allin_by_depth=allin,
    )


def apply_action_schedule_to_config(cfg) -> ActionSchedule:
    schedule = make_action_schedule(
        cfg.env.bet_bins,
        cfg.search.bet_bins_by_depth,
        cfg.search.allin_by_depth,
    )
    cfg.env.bet_bins = list(schedule.bet_bins)
    cfg.model.num_actions = schedule.num_actions
    if schedule.num_depths is not None:
        cfg.search.depth = schedule.num_depths
    return schedule


def legal_action_mask_for_depth(
    schedule: ActionSchedule,
    depth: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(schedule.num_actions, dtype=torch.bool, device=device)
    mask[:2] = True
    if schedule.bet_bins_by_depth is None:
        mask[2:-1] = True
        mask[-1] = True
        return mask

    depth_bins = schedule.bet_bins_by_depth[min(depth, len(schedule.bet_bins_by_depth) - 1)]
    bin_to_action = {bet_bin: i + 2 for i, bet_bin in enumerate(schedule.bet_bins)}
    for bet_bin in depth_bins:
        mask[bin_to_action[bet_bin]] = True
    assert schedule.allin_by_depth is not None
    mask[-1] = schedule.allin_by_depth[
        min(depth, len(schedule.allin_by_depth) - 1)
    ]
    return mask


def legal_action_masks_by_depth(
    schedule: ActionSchedule,
    *,
    device: torch.device,
) -> torch.Tensor:
    depth_count = schedule.num_depths or 1
    return torch.stack(
        [
            legal_action_mask_for_depth(schedule, depth, device=device)
            for depth in range(depth_count)
        ],
        dim=0,
    )
